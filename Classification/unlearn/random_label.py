import os
import time 
import random 
from copy import deepcopy
from datetime import datetime

import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import utils
from .unlearn_method import UnlearnMethod, UnLearnDataset
from trainer import validate


class RandomLabel(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.unlearning_data = None 
        self.unlearning_dataloader = None 
        self.seed = args.seed
        self.eval = True
        # params
        # TinyImageNet 
        # self.opt = 'adamw'
        # self.momentum = 0.9
        # self.weight_decay = 0.05
        # self.lr = 1e-4
        # self.epochs = 1
        # self.sched = 'cosine'
        # CIFAR10 
        self.opt = 'sgd'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.lr = 0.003
        self.epochs = 10
        self.sched = 'cosine'

    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders 
        unlearn_forget_train = deepcopy(self.unlearn_dataloaders['forget_train'].dataset)
        try:
            new_ys = []
            for y in unlearn_forget_train.targets:
                unlearn_labels = list(range(self.num_classes))
                unlearn_labels.remove(y)
                new_ys.append(random.choice(unlearn_labels))
            unlearn_forget_train.targets = new_ys
        except:
            new_ys = []
            for y in unlearn_forget_train.labels:
                unlearn_labels = list(range(self.num_classes))
                unlearn_labels.remove(y)
                new_ys.append(random.choice(unlearn_labels))
            unlearn_forget_train.labels = new_ys
        self.unlearning_data = UnLearnDataset(
            retain_data=self.unlearn_dataloaders['retain_train'].dataset, 
            forget_data=unlearn_forget_train)
        
        self.unlearn_dataloader = DataLoader(self.unlearning_data, batch_size=self.batch_size, shuffle=True, num_workers=4)


    def get_unlearned_model(self) -> nn.Module:

        if self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.opt == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        elif self.opt == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        if self.sched == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
        forget_trainloader = self.unlearn_dataloaders['forget_train']
        retain_validloader = self.unlearn_dataloaders['retain_valid']
        forget_validloader = self.unlearn_dataloaders['forget_valid'] 

        for epoch in range(1, self.epochs + 1):
            # train on retain train and forget train with random label
            start = time.time()
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            for i, (data, unlearn_labels) in enumerate(self.unlearn_dataloader):
                self.model.train() 
                images, labels = data
                labels = labels.cuda()
                images = images.cuda()

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                acc1 = utils.accuracy(outputs.data, labels)[0]
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))

            finish = time.time()
            lr = optimizer.param_groups[0]['lr']
            print(f"Train Epoch {epoch} Loss {losses.avg:.4f} Acc {top1.avg:.4f} LR {lr:6f} Time: {finish-start:.2f}s")
            
            if self.eval:
                # validata on retain valid, forget train, forget valid
                forget_train_metrics = validate(forget_trainloader, self.model, self.loss_function)
                if retain_validloader:
                    retain_valid_metrics = validate(retain_validloader, self.model, self.loss_function)
                else:
                    retain_valid_metrics = {k:None for k in forget_train_metrics.keys()}
                if forget_validloader:
                    forget_valid_metrics = validate(forget_validloader, self.model, self.loss_function)
                else:
                    forget_valid_metrics = {k:None for k in forget_train_metrics.keys()}
            scheduler.step(epoch)
        return self.model 

    def get_params(self) -> dict:
        self.params = {
            'opt' : self.opt,
            'momentum' : self.momentum,
            'weight_decay' : self.weight_decay,
            'lr' : self.lr,
            'epochs' : self.epochs,
            'sched' : self.sched,}
        return self.params

