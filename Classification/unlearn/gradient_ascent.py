import os
import time
from datetime import datetime

import torch 
import torch.nn as nn

import utils
from .unlearn_method import UnlearnMethod
from trainer import train, validate


class GradAscent(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)
        self.num_classes = args.num_classes
        self.seed = args.seed
        self.eval = False
        # params
        # TinyImageNet
        # self.opt = 'adamw'
        # self.momentum = 0.9
        # self.weight_decay = 0.05
        # self.lr = 1e-6
        # self.epochs = 9
        # self.max_norm = 0.1
        # CIFAR10 
        self.opt = 'sgd'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.lr = 0.0001
        self.epochs = 9
        self.max_norm = 0.1

    def get_unlearned_model(self) -> nn.Module:
        retain_trainloader = self.unlearn_dataloaders['retain_train']
        forget_trainloader = self.unlearn_dataloaders['forget_train']
        retain_validloader = self.unlearn_dataloaders['retain_valid']
        forget_validloader = self.unlearn_dataloaders['forget_valid']

        if self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.opt == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        elif self.opt == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        # if self.sched == 'cosine':
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         optimizer, T_max=self.epochs*len(forget_trainloader)
        #     )

        for epoch in range(1, self.epochs + 1):
            # gradient ascent on forget train
            start = time.time()
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            for i, (images, labels) in enumerate(forget_trainloader):
                self.model.eval() # important 
                labels = labels.cuda()
                images = images.cuda()

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = -1 * self.loss_function(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=self.max_norm)
                optimizer.step()

                acc1 = utils.accuracy(outputs.data, labels)[0]
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                # if scheduler is not None:
                #     scheduler.step()

            finish = time.time()
            lr = optimizer.param_groups[0]['lr']
            print(f"Train Epoch {epoch} Loss {losses.avg:.4f} Acc {top1.avg:.4f} LR {lr} Time: {finish-start:.2f}s")
            
            if self.eval:
                # validata on retain valid, retain train, forget valid
                validate(retain_trainloader, self.model, self.loss_function)
                if retain_validloader:
                    validate(retain_validloader, self.model, self.loss_function)
                if forget_validloader:
                    validate(forget_validloader, self.model, self.loss_function)
        return self.model 

    def get_params(self) -> dict:
        self.params = {
            'opt' : self.opt,
            'momentum' : self.momentum,
            'weight_decay' : self.weight_decay,
            'lr' : self.lr,
            'epochs' : self.epochs,
            # 'sched' : self.sched,
            'max_norm': self.max_norm}
        return self.params