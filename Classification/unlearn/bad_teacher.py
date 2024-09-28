import os
import time 
from copy import deepcopy
from datetime import datetime

import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import utils
from .unlearn_method import UnlearnMethod, UnLearnDataset
from models import create_model
from trainer import train, validate


def DistillLoss(
    outputs, unlearn_labels, full_teacher_logits, unlearning_teacher_logits, KL_temperature
):
    unlearn_labels = torch.unsqueeze(unlearn_labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearning_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = unlearn_labels * u_teacher_out + (1 - unlearn_labels) * f_teacher_out
    student_out = F.log_softmax(outputs / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)

class BadTeacher(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)
        self.model_name = args.model
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.unlearning_data = None 
        self.unlearning_dataloader = None 
        self.full_teacher = None
        self.unlearning_teacher = None
        self.seed = args.seed
        self.eval = False
        # params
        # TinyImageNet 
        # self.KL_temperature = 1
        # self.opt = 'adamw'
        # self.momentum = 0.9
        # self.weight_decay = 0.05
        # self.lr = 1e-4
        # self.epochs = 1
        # self.sched = 'cosine'
        # CIFAR10 
        self.KL_temperature = 1
        self.opt = 'sgd'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.lr = 0.02
        self.epochs = 10
        self.sched = 'cosine'

    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders 
        self.unlearning_data = UnLearnDataset(
            retain_data=self.unlearn_dataloaders['retain_train'].dataset, 
            forget_data=self.unlearn_dataloaders['forget_train'].dataset)
        self.unlearn_dataloader = DataLoader(self.unlearning_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.full_teacher = deepcopy(self.model)
        self.unlearning_teacher = create_model(model_name=self.model_name, num_classes=self.num_classes)
        self.unlearning_teacher.cuda()

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
            # train on retain train and forget train with bad teacher
            start = time.time()
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            for i, (data, unlearn_labels) in enumerate(self.unlearn_dataloader):
                self.model.train() 
                self.full_teacher.eval() 
                self.unlearning_teacher.eval() 
                images, labels = data
                labels = labels.cuda()
                images = images.cuda()
                unlearn_labels = unlearn_labels.cuda()
                with torch.no_grad():
                    full_teacher_logits = self.full_teacher(images)
                    unlearning_teacher_logits = self.unlearning_teacher(images)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = DistillLoss(outputs, 
                    unlearn_labels, 
                    full_teacher_logits, 
                    unlearning_teacher_logits,
                    self.KL_temperature)
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
            'sched' : self.sched,
            'KL_temperature': self.KL_temperature}
        return self.params

