import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as tfs
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, GPT2Tokenizer


class BaseDataset(Dataset):
    def __init__(self, split, max_caption_length, transform=None):
        self.max_seq_length = max_caption_length
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join("/workspace", "iu_xray-PA",  'images')
        self.ann_path = os.path.join("/workspace", "iu_xray-PA", 'annotation.json')
        self.ann = json.loads(open(self.ann_path, 'r').read())
        # self.labels_path = os.path.join("data", "iu_xray", "labels/labels.json")
        # self.labels = json.loads(open(self.labels_path, 'r').read())

        self.dataset_name = 'iu_xray'

        self.examples = self.ann[self.split]
        # if self.dataset_name == 'iu_xray':
        #     self._labels = []
        #     for e in self.examples:
        #         img_id = e['id']
        #         array = img_id.split('-')
        #         modified_id = array[0] + '-' + array[1]
        #         self._labels.append(self.labels[modified_id])

        self.tokenizer = GPT2Tokenizer.from_pretrained("/workspace/prototype/gpt-2-pubmed-medium", max_length=max_caption_length)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"

        for i in range(len(self.examples)):
            # self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            phrases_with_special_tokens = self.bos_token + self.examples[i]['report'] + self.eos_token ## 添加头尾特殊字符
            phrases_outputs = self.tokenizer(phrases_with_special_tokens, padding="max_length", truncation=True, return_tensors="pt", max_length=max_caption_length)
            self.examples[i]['ids'] = phrases_outputs['input_ids']  

            # self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            self.examples[i]['mask'] = phrases_outputs['attention_mask']

            

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        # array = image_id.split('-')
        # modified_id = array[0] + '-' + array[1]
        # label = np.array(self.labels[modified_id]).astype(np.float32)
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        # image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            # image_2 = self.transform(image_2)
        # image = torch.stack((image_1, image_2), 0)
        image = image_1   #  只取正面胸片
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, idx, image, report_ids, report_masks, seq_length)
        return sample
