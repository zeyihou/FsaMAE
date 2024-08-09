from .pretrain_sentence_datasets import Multi_Modal_Multi_Sentence_Bert_Dataset
from .pretrain_datasets import MultimodalBertDataset
from .datasets import Multi_Modal_Dataset

DATASET_DICT = {
    'report': MultimodalBertDataset,
    'sentence': Multi_Modal_Dataset,
}