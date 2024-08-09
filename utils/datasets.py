import os
import random
from copy import deepcopy
from dataclasses import dataclass

import nltk
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

CSV_KEYS = {'mimic_cxr': {'image': 'image_path', 'text': 'report', 'indicator': lambda x:'-'.join(x.split('/')[:-1])}}

class TensorDataclassMixin:
    def __init__(self):
        super(TensorDataclassMixin, self).__init__()
        assert dataclasses.is_dataclass(self), f'{type(self)} has to be a dataclass to use TensorDataclassMixin'

    def apply(self, tensor_fn, ignore=None):
        def apply_to_value(value):
            if value is None:
                return None
            elif isinstance(value, torch.Tensor):
                return tensor_fn(value)
            elif isinstance(value, list):
                return [apply_to_value(el) for el in value]
            elif isinstance(value, tuple):
                return tuple(apply_to_value(el) for el in value)
            elif isinstance(value, dict):
                return {key: apply_to_value(el) for key, el in value.items()}
            elif isinstance(value, TensorDataclassMixin):
                return value.apply(tensor_fn)
            else:
                return value

        def apply_to_field(field: dataclasses.Field):
            value = getattr(self, field.name)
            if ignore is not None and field.name in ignore:
                return value
            else:
                return apply_to_value(value)

        return self.__class__(**{field.name: apply_to_field(field) for field in dataclasses.fields(self)})

    def to(self, device, *args, non_blocking=True, **kwargs):
        return self.apply(lambda x: x.to(device, *args, non_blocking=non_blocking, **kwargs))

    def view(self, *args):
        return self.apply(lambda x: x.view(*args))

    def detach(self):
        return self.apply(lambda x: x.detach())
    
    def unsqueeze(self, dim):
        return self.apply(lambda x: x.unsqueeze(dim))
    
    def squeeze(self, dim):
        return self.apply(lambda x: x.squeeze(dim))

    def __getitem__(self, *args):
        return self.apply(lambda x: x.__getitem__(*args))

    def to_dict(self):
        return dataclasses.asdict(self)

@dataclass
class BatchSentenceSplitsInfo(TensorDataclassMixin):
    batch_size: int
    max_sentences_per_sample: int
    max_tokens_per_sentence: int
    sentence_token_mask: torch.Tensor 
    sentence_mask: torch.Tensor 
    token_same_sentence_mask: torch.Tensor 

    @staticmethod
    def compute_for_batch(sentences_start_positions_batch, sentences_lengths_batch):
        B = len(sentences_start_positions_batch)
        max_sentences_per_sample = max(len(sample) for sample in sentences_lengths_batch)
        max_tokens_per_sentence = max(sentence_length for sample in sentences_lengths_batch for sentence_length in sample)

        sentence_mask = torch.zeros((B, max_sentences_per_sample), dtype=torch.bool)
        sentence_token_mask = torch.zeros((B, max_sentences_per_sample, max_tokens_per_sentence), dtype=torch.bool)
        max_tokens = max(sum(sample_sentence_lengths) for sample_sentence_lengths in sentences_lengths_batch)
        token_same_sentence_mask = torch.zeros((B, max_tokens, max_tokens))

        for sample_index, (sentences_start_positions, sentences_lengths) \
                in enumerate(zip(sentences_start_positions_batch, sentences_lengths_batch)):
            num_sentences = len(sentences_start_positions)

            sentence_mask[sample_index, :num_sentences] = True
            for sentence_index, (sentences_start, sentences_length) \
                    in enumerate(zip(sentences_start_positions, sentences_lengths)):
                sentence_token_mask[sample_index, sentence_index, 0:sentences_length] = True
                token_same_sentence_mask[
                    sample_index,
                    sentences_start:sentences_start+sentences_length,
                    sentences_start:sentences_start+sentences_length
                ] = True

        return BatchSentenceSplitsInfo(batch_size=B,
                                       max_sentences_per_sample=max_sentences_per_sample,
                                       max_tokens_per_sentence=max_tokens_per_sentence,
                                       sentence_token_mask=sentence_token_mask,
                                       sentence_mask=sentence_mask,
                                       token_same_sentence_mask=token_same_sentence_mask)


class Multi_Modal_Dataset(Dataset):

    def __init__(self, csv_file, data_root, transform, 
                 dataset_name='mimic_cxr', mask_prob=0.15, 
                 max_caption_length=100, granularity='sentence',
                 token_name='Bio_ClinicalBERT'):
        
        self.csv_file = csv_file
        self.data_root = data_root
        self.transform = transform
        self.granularity = granularity
        self.dataset_name = dataset_name
        self.images_list, self.report_list = self.read_csv()

        self.mask_prob = mask_prob
        self.max_caption_length = max_caption_length

        self.tokenizer = AutoTokenizer.from_pretrained(token_name, max_length=self.max_caption_length)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.pad_id, self.mask_id = self.tokenizer.get_vocab()['[PAD]'], self.tokenizer.get_vocab()['[MASK]']
    
    def read_csv(self):

        data = pd.read_csv(self.csv_file, sep=',')

        image_path = data[CSV_KEYS[self.dataset_name]['image']]
        sentences = [nltk.sent_tokenize(report) for report in data[CSV_KEYS[self.dataset_name]['text']]]

        if 'train' in self.csv_file:
            datas = {}
            for _image_path, sentence in zip(image_path, sentences):
                if CSV_KEYS[self.dataset_name]['indicator'](_image_path) not in datas.keys():
                    datas[CSV_KEYS[self.dataset_name]['indicator'](_image_path)] = {'images': [_image_path], 'report': sentence}
                else:
                    datas[CSV_KEYS[self.dataset_name]['indicator'](_image_path)]['images'].append(_image_path)

            return [datas[key]['images'] for key in datas.keys()], [datas[key]['report'] for key in datas.keys()]
        else:
            return image_path, sentences
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):

        if 'train' in self.csv_file:
            image_path = os.path.join(self.data_root, random.choice(self.images_list[index]))
        else:
            image_path = os.path.join(self.data_root, self.images_list[index])
        image = Image.open(open(image_path, 'rb')).convert('RGB')
        image = self.transform(image)

        sentences = self.report_list[index]
        if 'train' in self.csv_file and random.random() < 0.5:
            random.shuffle(sentences)
        
        if self.granularity == 'report':
            sentences = [' '.join(sentences)]
        elif self.granularity == 'sentence':
            sentences = sentences
        
        return index, image, sentences

    def _random_mask(self,tokens):
        
        masked_tokens = deepcopy(tokens)
        for i in range(1, len(masked_tokens)):
            if masked_tokens[i] == self.pad_id:
                break
            
            if masked_tokens[i-1] == self.mask_id and self.idxtoword[masked_tokens[i]][0:2] == '##':
                masked_tokens[i] = self.mask_id
                continue
            
            if masked_tokens[i-1] != self.mask_id and self.idxtoword[masked_tokens[i]][0:2] == '##':
                continue
            prob = random.random()
            if prob < self.mask_prob:
                masked_tokens[i] = self.mask_id

        return masked_tokens

    def collate_fn(self, instances):

        index_list, image_list, sentences_list = [], [], []
        max_sentence_length = self.max_caption_length - self.tokenizer.num_special_tokens_to_add(pair=False)

        for instance in instances:
            index, image, sentences = instance
            index_list.append(index)
            image_list.append(image)
            sentences_list.append(sentences)
        
        all_sentences = [sentence for sentences in sentences_list for sentence in sentences]

        sentences_slices, start_index = [], 0
        for sentences in sentences_list:
            num_sentences = len(sentences)
            sentences_slices.append(slice(start_index, start_index + num_sentences))
            start_index += num_sentences
        
        all_sentences = self.tokenizer.batch_encode_plus(
            all_sentences,
            max_length=max_sentence_length,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False)

        sentences_start_positions_batch, sentences_lengths_batch = [], []
        input_ids_batch, masked_input_ids_batch, special_tokens_mask_batch = [], [], []
        for sentence_slice in sentences_slices:
            
            input_ids, masked_input_ids = [], []
            sentences_lengths, sentences_start_positions = [], []
            sentences = all_sentences['input_ids'][sentence_slice]

            current_start_index = 0
            for sentence in sentences:
                if current_start_index + len(sentence) <= max_sentence_length:
                    input_ids.extend(sentence)
                    masked_input_ids.extend(self._random_mask(sentence))
                    sentences_start_positions.append(current_start_index)
                    sentences_lengths.append(len(sentence))
                    current_start_index += len(sentence)
                elif current_start_index == 0:
                    input_ids.extend(sentence[:max_sentence_length])
                    masked_input_ids.extend(self._random_mask(sentence))
                    sentences_start_positions.append(0)
                    sentences_lengths.append(max_sentence_length)
                    break
                else:
                    break
            
            input_ids_with_special_tokens = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            masked_input_ids_with_special_tokens = self.tokenizer.build_inputs_with_special_tokens(masked_input_ids)

            input_ids_batch.append(input_ids_with_special_tokens)
            masked_input_ids_batch.append(masked_input_ids_with_special_tokens)
            special_tokens_mask_batch.append(
                self.tokenizer.get_special_tokens_mask(input_ids_with_special_tokens,
                                                       already_has_special_tokens=True))
            sentences_start_positions_batch.append(sentences_start_positions)
            sentences_lengths_batch.append(sentences_lengths)
        
        batch_sentences = self.tokenizer.pad({'input_ids': input_ids_batch, 
                                              'special_tokens_mask': special_tokens_mask_batch},
                                            #   padding='max_length', max_length=self.max_caption_length,
                                              return_attention_mask=True, return_tensors='pt')
        masked_batch_sentences = self.tokenizer.pad({'input_ids': masked_input_ids_batch, 
                                                     'special_tokens_mask': special_tokens_mask_batch},
                                                    #  padding='max_length', max_length=self.max_caption_length,
                                                     return_attention_mask=True, return_tensors='pt')

        image_stack = torch.stack(image_list)
        ids_stack = batch_sentences['input_ids']
        masked_ids_stack = masked_batch_sentences['input_ids']
        type_ids_stack = batch_sentences['special_tokens_mask']
        attention_mask_stack = batch_sentences['attention_mask']

        return {
            "index": index_list,
            "image": image_stack,
            "labels": ids_stack,
            "attention_mask": attention_mask_stack,
            "type_ids": type_ids_stack,
            "ids": masked_ids_stack,
            "sentence_splits": BatchSentenceSplitsInfo.compute_for_batch(
                    sentences_start_positions_batch, sentences_lengths_batch)
        }