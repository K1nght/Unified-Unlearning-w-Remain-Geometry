import copy 
import numpy as np 
import torch
import torch.nn as nn 
from torch.utils.data import Dataset

from .pretrain_dataset import PretrainDataset


def replace_indexes(dataset, indexes):
    new_dataset = copy.deepcopy(dataset)
    new_dataset.data = dataset.data[indexes]
    try:
        new_dataset.targets = np.array(dataset.targets)[indexes]
    except:
        new_dataset.labels = np.array(dataset.labels)[indexes]
    return new_dataset

class UnLearnDataset(PretrainDataset):
    """
    Unlearning Dataset
    """
    def __init__(self, dataset_name, root, img_size, setting=None):
        super(UnLearnDataset, self).__init__(dataset_name, root, img_size, setting)
        self.forget_trainset = Dataset
        self.retain_trainset = Dataset
        self.forget_validset = Dataset
        self.retain_validset = Dataset

