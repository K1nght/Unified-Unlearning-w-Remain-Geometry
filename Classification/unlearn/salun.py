import os
import time 
import random 
from copy import deepcopy
from datetime import datetime
from typing import Dict, List

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from .unlearn_method import UnlearnMethod, UnLearnDataset
from trainer import train, validate


class SalUn(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.mask = self.zerolike_params_dict(self.model)
        self.unlearning_data = None 
        self.unlearning_dataloader = None 
        self.eval = True
        # params
        # TinyImageNet 
        # self.opt = 'adamw'
        # self.momentum = 0.9
        # self.weight_decay = 0.05
        # self.lr = 1e-4
        # self.epochs = 1
        # self.sched = 'cosine'
        # self.th = 0.5
        # CIFAR10 
        self.opt = 'sgd'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.lr = 0.007
        self.epochs = 10
        self.sched = 'cosine'
        self.th = 0.2

    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders
        forget_trainloader = self.unlearn_dataloaders['forget_train']
        self.mask = self.get_gradient_ratio(forget_loader=forget_trainloader)
        print(f"SalUn threshold: {self.th}")

        # random label dataset
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
        if self.sched == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )

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

                if self.mask:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param.grad *= self.mask[name]

                optimizer.step()

                acc1 = utils.accuracy(outputs.data, labels)[0]
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))

            finish = time.time()
            lr = optimizer.param_groups[0]['lr']
            print(f"Train Epoch {epoch} Loss {losses.avg:.4f} Acc {top1.avg:.4f} LR {lr:6f} Time: {finish-start:.2f}s")
            
            if self.eval:
                # validata on retain valid, forget train, forget valid
                validate(forget_trainloader, self.model, self.loss_function)
                if retain_validloader:
                    validate(retain_validloader, self.model, self.loss_function)
                if forget_validloader:
                    validate(forget_validloader, self.model, self.loss_function)
            scheduler.step(epoch)
        return self.model 

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def get_gradient_ratio(self, forget_loader):
        optimizer = torch.optim.SGD(
            self.model.parameters(),lr=0,
        )
        criterion = self.loss_function
        gradients = self.zerolike_params_dict(self.model)
        self.model.eval()
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = self.model(image)
            loss = - criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data

        with torch.no_grad():
            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])
            
        sorted_dict_positions = {}
        hard_dict = {}
        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * self.th)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        return hard_dict

    def get_params(self) -> dict:
        self.params = {
            'opt' : self.opt,
            'momentum' : self.momentum,
            'weight_decay' : self.weight_decay,
            'lr' : self.lr,
            'epochs' : self.epochs,
            'sched' : self.sched}
        return self.params