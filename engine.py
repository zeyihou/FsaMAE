import wandb
import torch
import numpy as np

import torch.nn as nn


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import utils.misc as misc
import utils.lr_sched as lr_sched

# from utils.cider import Cider

from utils.CheXbert.src.models.bert_labeler import bert_labeler
from utils.CheXbert.src.label import label

from tqdm import tqdm
from collections import defaultdict
import io

import re

import spacy
import evaluate

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


from typing import List, Dict

import tempfile
import os
import csv

from transformers import GPT2Tokenizer

def train_one_epoch(model, model_mask, choose_mask_ratio, data_loader, tokenizer, optimizer, device, epoch, 
                    loss_scaler, log_writer=None, args=None, max_caption_length=300):
        
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_ve', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_ed', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    mask_ratio = args.mask_ratio
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header), start=1):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.no_grad():
            output_mask = model_mask(batch, prior_mask_ratio=choose_mask_ratio, is_training=False) 

            guide_mask = output_mask['MASK']['mask'] 
        
        with torch.cuda.amp.autocast():

            loss = model(batch, guide_mask=guide_mask, mask_ratio=mask_ratio, is_training=True, loss_keys=list(args.loss.keys()), max_caption_length=max_caption_length)

            loss_value = {key: loss[key].item() for key in loss.keys()} 

            loss = sum([args.loss[key] * loss[key] for key in args.loss.keys()])
            
            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if data_iter_step % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(**loss_value)
            metric_logger.update(lr_ve=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr_ed=optimizer.param_groups[1]["lr"])

            loss_value_reduce = {key: misc.all_reduce_mean(loss_value[key]) for key in loss_value.keys()}
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and (data_iter_step + 1) % 100:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(((data_iter_step + 1) / len(data_loader) + epoch) * 1000)

                for key in loss_value_reduce.keys():
                    log_writer.add_scalar(f'train/{key}_loss', loss_value_reduce[key], epoch_1000x)

                # log_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch_1000x)
                log_writer.add_scalar('lr_ve', optimizer.param_groups[0]["lr"], epoch_1000x)
                log_writer.add_scalar('lr_ed', optimizer.param_groups[1]["lr"], epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
