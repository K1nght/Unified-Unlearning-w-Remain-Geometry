import os
from datetime import datetime

import torch 
import torch.nn as nn
import torch.optim as optim

import utils
from .unlearn_method import UnlearnMethod
from models import create_model
from trainer import train, validate


class Retrain(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)
        self.model_name = args.model 
        self.dataset_name = args.dataset
        self.num_classes = args.num_classes
        self.seed = args.seed
        self.retrain_checkpoint = None
        # Tinyimg
        # self.opt = 'adamw'
        # self.momentum = 0.9
        # self.weight_decay = 0.05
        # self.lr = 1e-4
        # self.epochs = 10
        # self.sched = 'cosine'
        # CIFAR-10
        self.opt = 'sgd'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.lr = 0.1
        self.epochs = 200
        self.sched = 'cosine'

    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders 
        if self.retrain_checkpoint is None:
            self.model = create_model(model_name=self.model_name, num_classes=self.num_classes)
            self.model.cuda() 
        else:
            self.model.load_state_dict(torch.load(self.retrain_checkpoint))

    def get_unlearned_model(self) -> nn.Module:
        if self.retrain_checkpoint is not None: return self.model

        retain_trainloader = self.unlearn_dataloaders['retain_train']
        forget_trainloader = self.unlearn_dataloaders['forget_train']
        retain_validloader = self.unlearn_dataloaders['retain_valid']
        forget_validloader = self.unlearn_dataloaders['forget_valid'] 

        if self.opt == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.opt == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.sched == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        weights_save_path = os.path.join(self.save_path, "checkpoint", datetime.now().strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(weights_save_path):
            os.makedirs(weights_save_path)
        weights_path = os.path.join(weights_save_path, f"retrain-{self.seed}.pth")

        best_acc = None 
        best_epoch = None 
        for epoch in range(1, self.epochs + 1):
            # train on retain train
            retain_train_metrics = train(epoch, retain_trainloader, self.model, self.loss_function, optimizer)
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
            utils.update_summary(
                f"{epoch}-retain", retain_train_metrics, retain_valid_metrics, os.path.join(weights_save_path, f"summary-retrain-{self.seed}.csv"),
                write_header=best_acc is None,
            )
            utils.update_summary(
                f"{epoch}-forget", forget_train_metrics, forget_valid_metrics, os.path.join(weights_save_path, f"summary-retrain-{self.seed}.csv"),
            )
            if best_acc is None or best_acc < retain_valid_metrics["top1"]:  
                print("saving weights file to {}".format(weights_path))
                torch.save(self.model.state_dict(), weights_path)
                best_acc = retain_valid_metrics["top1"]
                best_epoch = epoch
        print('*** Best metric: {0} (epoch {1})'.format(best_acc, best_epoch))
        return self.model 
        
    def get_params(self) -> dict:
        self.params = {
            'retrain_checkpoint': self.retrain_checkpoint,
            'opt' : self.opt,
            'momentum' : self.momentum,
            'weight_decay' : self.weight_decay,
            'lr' : self.lr,
            'epochs' : self.epochs,
            'sched' : self.sched}
        return self.params