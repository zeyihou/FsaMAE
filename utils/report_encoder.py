from copy import deepcopy
from typing import Optional, Union, Tuple, Any

import random
import dataclasses
from dataclasses import dataclass

import torch

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
class AttentionMask(TensorDataclassMixin):
    binary_mask: torch.Tensor
    inverted_binary_mask: torch.Tensor
    additive_mask: torch.Tensor

    @staticmethod
    def from_binary_mask(binary_mask: torch.Tensor, dtype):
        if binary_mask is not None:
            binary_mask = binary_mask.bool()
        additive_mask = AttentionMask._compute_additive_attention_mask(binary_mask, dtype)
        return AttentionMask(binary_mask, ~binary_mask, additive_mask)

    @staticmethod
    def from_binary_mask_or_attention_mask(mask: Optional[Union['AttentionMask', torch.Tensor]], dtype):
        if mask is None or isinstance(mask, AttentionMask):
            return mask
        else:
            assert isinstance(mask, torch.Tensor) and (mask.dtype in (torch.bool, torch.uint8, torch.int64)), \
                (type(mask), mask.dtype)
            return AttentionMask.from_binary_mask(mask, dtype)

    @staticmethod
    def _compute_additive_attention_mask(binary_attention_mask: torch.Tensor, dtype):
        if binary_attention_mask is None:
            return None
        additive_attention_mask = torch.zeros_like(binary_attention_mask, dtype=dtype)
        additive_attention_mask.masked_fill_(~binary_attention_mask, float('-inf'))
        return additive_attention_mask

    @staticmethod
    def get_additive_mask(mask: Optional[Union['AttentionMask', torch.Tensor]], dtype):
        if mask is None:
            return None
        if isinstance(mask, AttentionMask):
            return mask.additive_mask
        elif mask.dtype == torch.bool or mask.dtype == torch.uint8:
            return AttentionMask._compute_additive_attention_mask(mask, dtype)
        else:
            return mask

    @staticmethod
    def get_additive_cross_attention_mask(mask_a: Optional['AttentionMask'] = None,
                                          mask_b: Optional['AttentionMask'] = None,
                                          mask_ab: Optional['AttentionMask'] = None):
        """

        :param mask_a: (B x N_a)
        :param mask_b: (B x N_b)
        :param mask_ab: (B x N_a x N_b)
        :return:
        """
        if mask_a is None and mask_b is None and mask_ab is None:
            return None
        else:
            mask = 0.
            if mask_ab is not None:
                mask = mask + mask_ab.additive_mask
            if mask_a is not None:
                mask = mask + mask_a.additive_mask[:, :, None]
            if mask_b is not None:
                mask = mask + mask_b.additive_mask[:, None, :]
            return mask

@dataclass
class BatchSentenceSplitsInfo():
    
    batch_size: int
    max_sentences_per_sample: int
    max_tokens_per_sentence: int
    sentence_token_mask: torch.Tensor  # (B x N_sent x N_sent_tok)
    sentence_mask: torch.Tensor  # (B x N_sent)
    token_same_sentence_mask: torch.Tensor  # (B x N_tok x N_tok)

    @staticmethod
    def compute_for_batch(sentences_start_positions_batch, sentences_lengths_batch):

        B = len(sentences_start_positions_batch)
        max_sentences_per_sample = max(len(sample) for sample in sentences_lengths_batch)
        max_tokens_per_sentence = max(sentence_length for sample in sentences_lengths_batch for sentence_length in sample)

        sentence_mask = torch.zeros((B, max_sentences_per_sample), dtype=torch.bool)
        sentence_token_mask = torch.zeros((B, max_sentences_per_sample, max_tokens_per_sentence), dtype=torch.bool)
        max_tokens = max(sum(sample_sentence_lengths) for sample_sentence_lengths in sentences_lengths_batch)
        token_same_sentence_mask = torch.zeros((B, max_tokens, max_tokens))

        for sample_index, (sentences_start_positions, sentences_lengths) in enumerate(zip(sentences_start_positions_batch, sentences_lengths_batch)):
            num_sentences = len(sentences_start_positions)

            sentence_mask[sample_index, :num_sentences] = True
            for sentence_index, (sentences_start, sentences_length) in enumerate(zip(sentences_start_positions, sentences_lengths)):
                sentence_token_mask[sample_index, sentence_index, :sentences_length] = True
                token_same_sentence_mask[
                    sample_index,
                    sentences_start:sentences_start + sentences_length,
                    sentences_start:sentences_start + sentences_length
                ]=True
        
        return BatchSentenceSplitsInfo(batch_size=B,
                                       max_sentences_per_sample=max_sentences_per_sample,
                                       max_tokens_per_sentence=max_tokens_per_sentence,
                                       sentence_token_mask=sentence_token_mask,
                                       sentence_mask=sentence_mask,
                                       token_same_sentence_mask=token_same_sentence_mask)


class ReportJointSentenceBatchCollator(object):

    def __init__(self, tokenizer, max_length, is_training=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    
    def _random_mask(self, tokens):
        masked_tokens = deepcopy(tokens)
        for j in range(masked_tokens.shape[0]):
            for i in range(1, masked_tokens.shape[1] - 1):
                if masked_tokens[j][i] == 3:
                    break
                
                if masked_tokens[j][i-1] == 4 and self.idxtoword[masked_tokens[j][i].item()][0:2] == '##':
                    masked_tokens[j][i] = 4
                    continue
                
                if masked_tokens[j][i-1] != 4 and self.idxtoword[masked_tokens[j][i].item()][0:2] == '##':
                    continue

                prob = random.random()
                if prob < 0.15:
                    masked_tokens[j][i] = 4

        return masked_tokens
    
    def __call__(self, batch):

        batch_sentences = [sample for sample in batch]
        all_sentences = [sent for sample in batch_sentences for sent in sample]

        sentences_slices_of_samples = []
        start_index = 0
        for sample_sentences in batch_sentences:
            num_sentences = len(sample_sentences)
            sentences_slices_of_samples.append(slice(start_index, start_index + num_sentences))
            start_index += num_sentences
        
        max_context_length = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)

        all_sentences = self.tokenizer.batch_encode_plus(
            all_sentences,
            max_length=max_context_length,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False,
        )

        sentences_start_positions_batch = []
        sentences_lengths_batch = []
        input_ids_batch = []
        special_tokens_mask_batch = []
        for sentence_slice_of_sample in sentences_slices_of_samples:

            sentences_of_sample = all_sentences['input_ids'][sentence_slice_of_sample]

            input_ids_of_sample = []
            current_start_index = 0
            sentences_start_positions = []
            sentences_lengths = []
            for sentence in sentences_of_sample:
                if current_start_index + len(sentence) <= max_context_length:
                    input_ids_of_sample.extend(sentence)
                    sentences_start_positions.append(current_start_index)
                    sentences_lengths.append(len(sentence))
                    current_start_index += len(sentence)
                elif current_start_index == 0:
                    input_ids_of_sample.extend(sentence[:max_context_length])
                    sentences_start_positions.append(0)
                    sentences_lengths.append(max_context_length)
                else:
                    break
            
            input_ids_with_special_tokens = self.tokenizer.build_inputs_with_special_tokens(input_ids_of_sample)
            input_ids_batch.append(input_ids_with_special_tokens)
            special_tokens_mask_batch.append(self.tokenizer.get_special_tokens_mask(input_ids_with_special_tokens, 
                                                                                    already_has_special_tokens=True))
            sentences_start_positions_batch.append(sentences_start_positions)
            sentences_lengths_batch.append(sentences_lengths)

        batch_sentences = self.tokenizer.pad(
            {'input_ids': input_ids_batch, 'special_tokens_mask': special_tokens_mask_batch},
            return_attention_mask=True,
            return_tensors='pt'
        )

        if self.is_training:
            batch_sentences['mask_ids'] = self._random_mask(batch_sentences['input_ids'])

        batch_sentences['sentence_splits'] = BatchSentenceSplitsInfo.compute_for_batch(sentences_start_positions_batch, 
                                                                                       sentences_lengths_batch)

        return batch_sentences