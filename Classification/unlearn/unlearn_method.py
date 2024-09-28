import torch.nn as nn
from torch.utils.data import Dataset

class UnlearnMethod:
    def __init__(self, model, loss_function, save_path, args) -> None:
        self.unlearn_dataloaders = None
        self.model = model
        self.loss_function = loss_function
        self.save_path = save_path
        self.args = args 
        self.params = {}

    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders
        return

    def get_unlearned_model(self) -> nn.Module:
        return self.model 

    def get_params(self) -> dict:
        return self.params

class UnLearnDataset(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            data = self.forget_data[index]
            unlearn_label = 1
            return data, unlearn_label
        else:
            data = self.retain_data[index - self.forget_len]
            unlearn_label = 0
            return data, unlearn_label

class SimpleDataset(Dataset):
    def __init__(self, data, targets) -> None:
        super().__init__()
        self.data = data 
        self.targets = targets 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (self.data[index], self.targets[index])