from copy import deepcopy
import os
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, GPT2Tokenizer
import tokenizers
import random
import nltk

from utils.report_encoder import ReportJointSentenceBatchCollator


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    

class MultimodalBertDataset(Dataset):
    def __init__(
        self,
        csv_file,
        data_root,
        transform,
        mask_prob: float = 0.5,
        max_caption_length: int = 100,
        token_name: str = "microsoft/BiomedVLP-CXR-BERT-general",
    ):
        self.csv_file = csv_file
        self.data_root = data_root
        self.transform = transform
        self.images_list, self.report_list = self.read_csv()
        
        self.max_caption_length = max_caption_length
        self.tokenizer = AutoTokenizer.from_pretrained(token_name, max_length=self.max_caption_length)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        # TODO: add special tokens for itm task (blip/blip2/dcl)
        # self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        # self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]
        
        self.pad_id = self.tokenizer.get_vocab()['[PAD]']
        self.mask_id = self.tokenizer.get_vocab()['[MASK]']
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.images_list)
    
    def _random_mask(self,tokens):
        
        masked_tokens = deepcopy(tokens)
        for i in range(1, masked_tokens.shape[1]-1):
            if masked_tokens[0][i] == self.pad_id:
                break
            
            if masked_tokens[0][i-1] == self.mask_id and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                masked_tokens[0][i] = self.mask_id
                continue
            
            if masked_tokens[0][i-1] != self.mask_id and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                continue

            prob = random.random()
            if prob < self.mask_prob:
                masked_tokens[0][i] = self.mask_id

        return masked_tokens

    def __getitem__(self, index):
        
        if 'train' in self.csv_file:
            image_path = os.path.join(self.data_root, random.choice(self.images_list[index]))
        else:
            image_path = os.path.join(self.data_root, self.images_list[index])

        if self.transform.transforms[0].size[0] > 224:
            image_path = image_path.replace("_small", "")

        image = Image.open(open(image_path, 'rb')).convert('RGB')
        image = self.transform(image)

        sentences = self.report_list[index]
        # random shuffle the sentences order
        if 'train' in self.csv_file and random.random() < 0.5:
            random.shuffle(sentences)
        
        sentences = ' '.join(sentences)
        encoded = self.tokenizer([sentences], padding="max_length", 
                                 truncation=True, return_tensors="pt", 
                                 max_length=self.max_caption_length)

        ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        type_ids = encoded['token_type_ids']
        masked_ids = self._random_mask(ids)   
        return index, image, ids, attention_mask, type_ids, masked_ids
    
    def read_csv(self):

        data = pd.read_csv(self.csv_file, sep=',')

        image_path = data["image_path"]
        sentences = [nltk.sent_tokenize(report) for report in data['report']]

        if 'train' in self.csv_file:
            datas = {}
            for _image_path, sentence in zip(image_path, sentences):
                if '-'.join(_image_path.split('/')[:-1]) not in datas.keys():
                    datas['-'.join(_image_path.split('/')[:-1])] = {'images': [_image_path], 'report': sentence}
                else:
                    datas['-'.join(_image_path.split('/')[:-1])]['images'].append(_image_path)

            return [datas[key]['images'] for key in datas.keys()], [datas[key]['report'] for key in datas.keys()]
        else:
            return image_path, sentences

    def collate_fn(self, instances: List[Tuple]):
        index_list, image_list, ids_list, attention_mask_list, type_ids_list, masked_ids_list = [], [], [], [], [], []
        # flattern
        for b in instances:
            index, image, ids, attention_mask, type_ids, masked_ids = b
            index_list.append(index)
            image_list.append(image)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)
            type_ids_list.append(type_ids)
            masked_ids_list.append(masked_ids)

        # stack
        image_stack = torch.stack(image_list)
        ids_stack = torch.stack(ids_list).squeeze()
        attention_mask_stack = torch.stack(attention_mask_list).squeeze()
        type_ids_stack = torch.stack(type_ids_list).squeeze()
        masked_ids_stack = torch.stack(masked_ids_list).squeeze()

        # sort and add to dictionary
        return_dict = {
            "index": index_list,
            "image": image_stack,
            "labels": ids_stack,
            "attention_mask": attention_mask_stack,
            "type_ids": type_ids_stack,
            "ids": masked_ids_stack
        }

        return return_dict
    



class MultimodalBertDataset_Img(Dataset):
    def __init__(
        self,
        csv_file,
        data_root,
        transform,
        max_caption_length: int = 100,
        token_name: str = "/workspace/prototype/gpt-2-pubmed-medium",
    ):
        self.csv_file = csv_file
        self.data_root = data_root
        self.transform = transform
        self.images_list, self.report_list = self.read_csv()
        
        self.max_caption_length = max_caption_length

        self.tokenizer = GPT2Tokenizer.from_pretrained(token_name, max_length=self.max_caption_length)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # TODO: add special tokens
        self.bos_token = "<|endoftext|>" 
        self.eos_token = "<|endoftext|>"

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):

        image_path = os.path.join(self.data_root, self.images_list[index])

        if self.transform.transforms[0].size[0] > 224:
            image_path = image_path.replace("_small", "")

        image = Image.open(open(image_path, 'rb')).convert('RGB')
        image = self.transform(image)

        report = self.report_list[index]

        try:
            phrases_with_special_tokens = self.bos_token + str(report) + self.eos_token 
        except TypeError:
            print(image_path)
            print(report)

        phrases_outputs = self.tokenizer(phrases_with_special_tokens,padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_caption_length)
        
        ids = phrases_outputs['input_ids']  
        attention_mask = phrases_outputs['attention_mask']
       

        return index, image_path, image, ids, attention_mask
    
    def read_csv(self):

        data = pd.read_csv(self.csv_file, sep=',')

        image_path = data["image_path"]

        # [,,, ... ,,]
        reports = data['report']   
        
        return image_path, reports

    def collate_fn(self, instances: List[Tuple]):
        index_list, image_path_list, image_list, ids_list, attention_mask_list = [], [], [], [], []

        # flattern
        for b in instances:
            index, image_path, image, ids, attention_mask = b
            index_list.append(index)
            image_path_list.append(image_path)
            image_list.append(image)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)

        # stack
        image_stack = torch.stack(image_list)
        ids_stack = torch.stack(ids_list).squeeze()
        attention_mask_stack = torch.stack(attention_mask_list).squeeze()

        # sort and add to dictionary
        return_dict = {
            "index": index_list,
            "image_path": image_path_list,
            "image": image_stack,
            "ids": ids_stack,
            "attention_mask": attention_mask_stack,
        }

        return return_dict