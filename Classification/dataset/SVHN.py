import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import SVHN

from .pretrain_dataset import PretrainDataset
from .unlearn_dataset import UnLearnDataset, replace_indexes

SVHN_MEAN = [0.43768447637557983, 0.4437686800956726, 0.47280409932136536]
SVHN_STD = [0.12008652836084366, 0.1231374442577362, 0.1052042692899704]

transform_train = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
]

transform_valid = [
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
]

class PretrainSVHN(PretrainDataset):
    def __init__(self, root, img_size=32):
        super().__init__("SVHN", root, img_size, "pretrain")
        self.transform_train = transforms.Compose(transform_train + [transforms.Resize(img_size)])
        self.transform_valid = transforms.Compose(transform_valid + [transforms.Resize(img_size)])

    def get_datasets(self):
        self.train_dataset = SVHN(root=self.root, split='train', download=True, transform=self.transform_train)
        self.valid_dataset = SVHN(root=self.root, split='test', download=True, transform=self.transform_valid)
        return self.train_dataset, self.valid_dataset


class FullClassUnlearnSVHN(UnLearnDataset):
    def __init__(self, root, img_size=32):
        super().__init__("SVHN", root, img_size, "fullclass")
        self.transform_train = transforms.Compose(transform_train + [transforms.Resize(img_size)])
        self.transform_valid = transforms.Compose(transform_valid + [transforms.Resize(img_size)])

    def get_datasets(self):
        self.train_dataset = SVHN(root=self.root, split='train', download=True, transform=self.transform_train)
        self.valid_dataset = SVHN(root=self.root, split='test', download=True, transform=self.transform_valid)
        return self.train_dataset, self.valid_dataset

    def full_class_split(self, forget_classes):
        forget_train_indexes = []
        forget_valid_indexes = []
        for fc in forget_classes:
            forget_train_indexes.append(np.array(self.train_dataset.labels) == fc)
            forget_valid_indexes.append(np.array(self.valid_dataset.labels) == fc)
        forget_train_indexes = np.concatenate(forget_train_indexes)
        forget_valid_indexes = np.concatenate(forget_valid_indexes)
        self.forget_trainset = replace_indexes(self.train_dataset, forget_train_indexes)
        self.retain_trainset = replace_indexes(self.train_dataset, np.logical_not(forget_train_indexes))
        self.forget_validset = replace_indexes(self.valid_dataset, forget_valid_indexes)
        self.retain_validset = replace_indexes(self.valid_dataset, np.logical_not(forget_valid_indexes))
        return self.forget_trainset, self.retain_trainset, self.forget_validset, self.retain_validset

class RandomUnlearnSVHN(UnLearnDataset):
    def __init__(self, root, img_size=32):
        super().__init__("SVHN", root, img_size, "random")
        self.transform_train = transforms.Compose(transform_train + [transforms.Resize(img_size)])
        self.transform_valid = transforms.Compose(transform_valid + [transforms.Resize(img_size)])

    def get_datasets(self):
        self.train_dataset = SVHN(root=self.root, split='train', download=True, transform=self.transform_train)
        self.valid_dataset = SVHN(root=self.root, split='test', download=True, transform=self.transform_valid)
        return self.train_dataset, self.valid_dataset

    def random_split(self, forget_perc, save_path):
        random_indexes_path = os.path.join(save_path, "random_idx.npy")
        random_indexes = self.get_random_indexes(random_indexes_path)
        print(random_indexes)
        forget_len = int(len(self.train_dataset) * forget_perc)
        forget_train_indexes = random_indexes[:forget_len]
        retain_train_indexes = random_indexes[forget_len:]
        self.forget_trainset = replace_indexes(self.train_dataset, forget_train_indexes)
        self.retain_trainset = replace_indexes(self.train_dataset, retain_train_indexes)
        
        return self.forget_trainset, self.retain_trainset
    
    def get_random_indexes(self, random_indexes_path):
        if os.path.exists(random_indexes_path):
            random_indexes = np.load(random_indexes_path)
            print(f"Load random indexes from {random_indexes_path}")
        else:
            train_len = len(self.train_dataset)
            random_indexes = list(range(train_len))
            random.shuffle(random_indexes)
            random_indexes = np.array(random_indexes)
            np.save(random_indexes_path, random_indexes)
            print(f"Save random indexes to {random_indexes_path}")
        return random_indexes
