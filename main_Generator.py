# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import pytz
import time
import json
import argparse
import datetime
import subprocess
import numpy as np
from pathlib import Path
import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from transformers import GPT2Tokenizer

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

from mmengine.config import DictAction

# import models
from models.model_gen import MRM_GEN
from models.model_mask import MRM_MASK
import utils.misc as misc
from utils import DATASET_DICT
from utils.pretrain_datasets import MultimodalBertDataset_Img
from engine import train_one_epoch, test_one_epoch, train_one_epoch_generator
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_output_dir(args):

    tz_beijing = pytz.timezone('Asia/Shanghai') 
    datetime_NY = datetime.datetime.now(tz_beijing)
    if args.output_dir:
        save_folder_name = f'exp/exp-{args.output_dir}-{datetime_NY.strftime("%Y%m%d%H%M%S")}'
    else:
        save_folder_name = f'exp/exp-debug-{datetime_NY.strftime("%Y%m%d%H%M%S")}'

    args.output_dir = save_folder_name
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE_Generator pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mrm', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # parser.add_argument('--lr', type=float, default=None, metavar='LR',
    #                     help='learning rate (absolute lr)')
    # parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
    #                     help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--lr_ve', type=float, default=None, metavar='LR',
                        help='learning rate for visual extractor')
    parser.add_argument('--lr_ed', type=float, default=None, metavar='LR',
                        help='learning rate for language model')
    
    
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/dataset/mimic_cxr_ap-pa_dataset', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # add the mmengine config
    parser.add_argument('--aliyunpan_path', default='/exp_cross_modal_medical_retrieval', type=str)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    return parser

def load_mask_model(device, checkpoints='/workspace/prototype/checkpoint_retrieval/mcr_pretrained.pth'):
    model_mask = MRM_MASK(nor_pix_loss=args.norm_pix_loss)
    checkpint_weights = torch.load(checkpoints, map_location='cpu')['model']
    model_state_dict = model_mask.state_dict() 

    for name, param in checkpint_weights.items():
        if name in model_state_dict:   
            model_state_dict[name] = param  

    model_mask.load_state_dict(model_state_dict,strict=False)

    model_mask.to(device)
    model_mask.eval()
    
    return model_mask

def get_tokenizer():
    checkpoint = "/workspace/prototype/gpt-2-pubmed-medium" 
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def main(args):

    misc.init_distributed_mode(args)

    if misc.is_main_process():

        set_output_dir(args)

        with open(f"{args.output_dir}/config.json", mode="w") as f:
            json.dump(args.__dict__, f, indent=4)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    max_caption_length = 100 
    choose_mask_ratio = 0.8 
    valid_batch_size = 6
    test_batch_size = 6

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_resolution if hasattr(args, 'input_resolution') else 224 , scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
        transforms.RandomAffine(degrees=(-10.0, 10.0), translate=(0.01, 0.05), scale=(0.95, 1.05)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4978], std=[0.2449])])
    dataset_train = MultimodalBertDataset_Img('./dataset/mimic-cxr/official_protocol_train_processed_2.csv',  # processed 去掉侧面胸片
                                               os.path.join(args.data_path), 
                                               transform=transform_train,
                                               max_caption_length=max_caption_length,
                                               token_name=args.report_token_name if hasattr(args, 'report_token_name') else "/workspace/prototype/gpt-2-pubmed-medium")

    print(dataset_train)


    transform_validation = transforms.Compose([
        transforms.Resize([args.input_resolution if hasattr(args, 'input_resolution') else 224, 
                           args.input_resolution if hasattr(args, 'input_resolution') else 224], interpolation=InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4978], std=[0.2449])])
    dataset_validation = MultimodalBertDataset_Img('./dataset/mimic-cxr/official_protocol_validate_processed_2.csv',
                                                    os.path.join(args.data_path),
                                                    transform=transform_validation,
                                                    max_caption_length=max_caption_length,
                                                    token_name=args.report_token_name if hasattr(args, 'report_token_name') else "/workspace/prototype/gpt-2-pubmed-medium")
    
    print(dataset_validation)


    # # test
    # transform_test = transforms.Compose([
    #     transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC),
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.4978], std=[0.2449])])

    # dataset_test = MultimodalBertDataset_Img('./dataset/mimic-cxr/official_protocol_test_processed_2.csv', 
    #                                         '/dataset/mimic_cxr_ap-pa_dataset/files',
    #                                         transform=transform_test,
    #                                         max_caption_length=max_caption_length,
    #                                         token_name="/workspace/prototype/gpt-2-pubmed-medium")
    
    # print(dataset_test)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_validation = torch.utils.data.DistributedSampler(
            dataset_validation, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_validation = %s" % str(sampler_validation))

        # sampler_test = torch.utils.data.DistributedSampler(
        #     dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        # print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=f'{args.output_dir}/runs')
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dataset_train.collate_fn
    )

    data_loader_validation = torch.utils.data.DataLoader(
        dataset_validation, sampler=sampler_validation,
        batch_size=valid_batch_size,   # 根据显存大小确定
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=dataset_validation.collate_fn, 
        shuffle=False
    )

    # # test contrast
    # data_loader_test = torch.utils.data.DataLoader(dataset_test,
    #                                                sampler=sampler_test,
    #                                                     batch_size=test_batch_size,   # 根据显存大小确定
    #                                                     num_workers=0,
    #                                                     pin_memory=True,
    #                                                     collate_fn=dataset_test.collate_fn, 
    #                                                     shuffle=False)

    tokenizer = get_tokenizer()

    model_mask = load_mask_model(device=device)

    model = MRM_GEN( tokenizer=tokenizer, max_caption_length=max_caption_length, nor_pix_loss=args.norm_pix_loss)
    
    model.to(device)
    
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print("accumulate grad iterations: %d" % args.accum_iter)
    # print("effective batch size: %d" % eff_batch_size)
    print("learning rate for visual extractor: %.2e" % args.lr_ve)
    print("learning rate for language model: %.2e" % args.lr_ed)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model._set_static_graph()
        model_without_ddp = model.module
    
    optimizer = torch.optim.AdamW(model.module.get_parameter_group(args.lr_ve, args.lr_ed), amsgrad=True, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler() 

    misc.load_model_new(args=args, model_without_ddp=model_without_ddp)

    print(f"Training for {args.epochs} epochs !")
    print("args.distributed:",args.distributed)  # True

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, 
            model_mask,         
            choose_mask_ratio,   
            data_loader_train,
            tokenizer,   
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            max_caption_length = max_caption_length
        )
        
        if misc.is_main_process():
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    torch.distributed.barrier()

if __name__ == '__main__':

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    args = get_args_parser()
    args = args.parse_args()

    if args.cfg_options is not None:
        for key in args.cfg_options.keys():
            if 'loss' == key:
                args.__setattr__('loss', {item[0]: float(item[1]) for item in args.cfg_options['loss']})
            else:
                args.__setattr__(key, args.cfg_options[key])
        
        args.__delattr__('cfg_options')

    main(args)

