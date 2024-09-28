import os
from datetime import datetime

import torch 
import torch.nn as nn

import utils
from .unlearn_method import UnlearnMethod
from trainer import train, validate


class Finetune(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)
        self.num_classes = args.num_classes
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
        self.lr = 0.01
        self.epochs = 10
        self.sched = 'cosine'

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
        retain_trainloader = self.unlearn_dataloaders['retain_train']
        forget_trainloader = self.unlearn_dataloaders['forget_train']
        retain_validloader = self.unlearn_dataloaders['retain_valid']
        forget_validloader = self.unlearn_dataloaders['forget_valid']

        for epoch in range(1, self.epochs + 1):
            # train on retain train
            train(epoch, retain_trainloader, self.model, self.loss_function, optimizer)
            if self.eval:
                # validata on retain valid, forget train, forget valid
                validate(forget_trainloader, self.model, self.loss_function, "Forget Train")
                if retain_validloader:
                    validate(retain_validloader, self.model, self.loss_function, "Retain Valid")
                if forget_validloader:
                    validate(forget_validloader, self.model, self.loss_function, "Forget Valid")
            scheduler.step(epoch)
        return self.model 

    def get_params(self) -> dict:
        self.params = {
            'opt' : self.opt,
            'momentum' : self.momentum,
            'weight_decay' : self.weight_decay,
            'lr' : self.lr,
            'epochs' : self.epochs,
            'sched' : self.sched}
        return self.params