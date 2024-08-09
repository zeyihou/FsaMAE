import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from .balanced_sampler import MultilabelBalancedRandomSampler
from torchvision.transforms.functional import InterpolationMode

import utils.misc as misc

class R2DataLoader(DataLoader):
    def __init__(self, batch_size, num_workers, split, max_caption_length, vis = False):
        self.dataset_name = "iu_xray"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.drop_last = True if split =='train' else False
        self.vis = vis

        self.randaug = True
        self.balanced = True

        if split == 'train':
            if self.randaug:
                print('Random applied transformation is utilized for ' + split +' dataset.')
                self.transform = transforms.Compose([
                                transforms.RandomResizedCrop( 224 , scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
                                transforms.RandomAffine(degrees=(-10.0, 10.0), translate=(0.01, 0.05), scale=(0.95, 1.05)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4978], std=[0.2449])])

            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
        else:
            self.transform = transform_test = transforms.Compose([
                    transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4978], std=[0.2449])])

        self.dataset = IuxrayMultiImageDataset(self.split, max_caption_length, transform=self.transform)

        if True:  # args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                self.dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)


        # if self.balanced:
        #     if split == 'train' and not self.vis:
        #         print('Balanced sampler is established for ' + split +' dataset.')
        #         self.sampler = MultilabelBalancedRandomSampler(np.array(self.dataset._labels))
        #         self.init_kwargs = {
        #             'dataset': self.dataset,
        #             'batch_size': self.batch_size,
        #             'sampler': self.sampler,
        #             'num_workers': self.num_workers,
        #             'pin_memory': True,
        #             'drop_last': self.drop_last,
        #             #'collate_fn': self.collate_fn,
        #             'prefetch_factor': self.batch_size // self.num_workers * 2
        #         }
        #     else:
        #         self.init_kwargs = {
        #             'dataset': self.dataset,
        #             # 'sampler': self.sampler,
        #             'batch_size': self.batch_size,
        #             'shuffle': shuffle,
        #             'num_workers': self.num_workers,
        #             'pin_memory': True,
        #             'drop_last': self.drop_last,
        #             #'collate_fn': self.collate_fn,
        #             'prefetch_factor': self.batch_size // self.num_workers * 2
        #         }

        # else:
        self.init_kwargs = {
                'dataset': self.dataset,
                'sampler': sampler_train,
                'batch_size': self.batch_size,
                # 'shuffle':shuffle,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers,
                'pin_memory': True,
                'drop_last': self.drop_last,
                'prefetch_factor': 2
            }


        # num_tasks = dist.get_world_size()
        # global_rank = dist.get_rank()
        #
        # self.sampler = DistributedSampler(self.dataset, num_replicas=num_tasks,
        #                                   rank=global_rank, shuffle=self.shuffle)

        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        # images_id, index, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        images_id, index, img, reports_ids, reports_masks, seq_lengths = zip(*data)

        # images = torch.stack(images, 0)

        img = torch.stack(img, 0)


        reports_ids = torch.stack(reports_ids).squeeze()
        reports_masks = torch.stack(reports_masks).squeeze()



        # max_seq_length = max(seq_lengths)

        # targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        # targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        # for i, report_ids in enumerate(reports_ids):
        #     targets[i, :len(report_ids)] = report_ids

        # for i, report_masks in enumerate(reports_masks):
        #     targets_masks[i, :len(report_masks)] = report_masks

        # return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)
        return images_id, index, img, reports_ids, reports_masks

