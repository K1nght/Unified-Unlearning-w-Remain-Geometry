import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

class PretrainDataset:
    """
    Pretrain Dataset
    """
    def __init__(self, dataset_name, root, img_size, setting="Pretrain"): 
        super().__init__() 
        self.dataset_name = dataset_name 
        self.root = root 
        self.img_size = img_size 
        self.setting = setting 
        self.train_dataset = Dataset
        self.valid_dataset = Dataset
        self.transform_train = nn.Module
        self.transform_valid = nn.Module
