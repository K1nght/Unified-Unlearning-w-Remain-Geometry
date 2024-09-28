import os
import sys 
import time 
from copy import deepcopy
from datetime import datetime

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils
from .unlearn_method import UnlearnMethod
from trainer import train, validate


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist

class SCRUB(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)
        self.num_classes = args.num_classes
        self.seed = args.seed
        self.model_t = None 
        self.model_s = None 
        self.swa_model = None
        self.module_list = nn.ModuleList([])
        self.trainable_list = nn.ModuleList([])
        self.criterion_list = nn.ModuleList([])
        self.eval = False
        # params
        # TinyImageNet
        self.opt = 'adamw'
        self.gamma = 0.99
        self.alpha = 0.001
        self.beta = 0
        self.smoothing = 0.0
        self.msteps = 2
        self.clip = 0.2
        self.sstart = 10
        self.kd_T = 4
        self.distill = 'kd'
        self.sched = 'cosine'
        self.sgda_epochs = 1
        self.sgda_learning_rate = 1e-4
        self.sgda_weight_decay = 0.05
        self.sgda_momentum = 0.9
        self.print_freq = 100
        # CIFAR10 
        # self.opt = 'sgd'
        # self.gamma = 0.99
        # self.alpha = 0.001
        # self.beta = 0
        # self.smoothing = 0.0
        # self.msteps = 2
        # self.clip = 0.2
        # self.sstart = 10
        # self.kd_T = 4
        # self.distill = 'kd'
        # self.sched = 'cosine'
        # self.sgda_epochs = 6
        # self.sgda_learning_rate = 0.00008
        # self.sgda_weight_decay = 5e-4
        # self.sgda_momentum = 0.9
        # self.print_freq = 100

    
    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders
        self.model_t = deepcopy(self.model)
        self.model_s = deepcopy(self.model)
        def avg_fn(averaged_model_parameter, model_parameter, num_averaged): return (
            1 - self.beta) * averaged_model_parameter + self.beta * model_parameter
        self.swa_model = torch.optim.swa_utils.AveragedModel(
            self.model_s, avg_fn=avg_fn)

        self.module_list.append(self.model_s)
        self.module_list.append(self.model_t)
        self.trainable_list.append(self.model_s)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(self.kd_T)
        criterion_kd = DistillKL(self.kd_T)

        self.criterion_list.append(criterion_cls)    # classification loss
        self.criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
        self.criterion_list.append(criterion_kd)     # other knowledge distillation loss

    def get_unlearned_model(self) -> nn.Module:
        retain_trainloader = self.unlearn_dataloaders['retain_train']
        forget_trainloader = self.unlearn_dataloaders['forget_train']
        retain_validloader = self.unlearn_dataloaders['retain_valid']
        forget_validloader = self.unlearn_dataloaders['forget_valid']

        # optimizer
        if self.opt == "sgd":
            optimizer = torch.optim.SGD(self.trainable_list.parameters(),
                                lr=self.sgda_learning_rate,
                                momentum=self.sgda_momentum,
                                weight_decay=self.sgda_weight_decay)
        elif self.opt == "adam": 
            optimizer = torch.optim.Adam(self.trainable_list.parameters(),
                                lr=self.sgda_learning_rate,
                                weight_decay=self.sgda_weight_decay)
        elif self.opt == "rmsp":
            optimizer = torch.optim.RMSprop(self.trainable_list.parameters(),
                                lr=self.sgda_learning_rate,
                                momentum=self.sgda_momentum,
                                weight_decay=self.sgda_weight_decay)
        elif self.opt == "adamw":
            optimizer = torch.optim.AdamW(self.trainable_list.parameters(), 
                                lr=self.sgda_learning_rate, 
                                weight_decay=self.sgda_weight_decay)
            

        if self.sched == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.sgda_epochs
            )

        if torch.cuda.is_available():
            self.module_list.cuda()
            self.criterion_list.cuda()
            cudnn.benchmark = True
            self.swa_model.cuda()

        print("==> SCRUB unlearning ...")
        for epoch in range(1, self.sgda_epochs + 1):

            maximize_loss = 0
            if epoch <= self.msteps:
                maximize_loss = self.train_distill(epoch, forget_trainloader, optimizer, "maximize")
            train_acc, train_loss = self.train_distill(epoch, retain_trainloader, optimizer, "minimize")
            if epoch >= self.sstart:
                self.swa_model.update_parameters(self.model_s)
            print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))

            if self.eval:
                # validata on retain valid, forget train, forget valid
                validate(forget_trainloader, self.model_s, self.loss_function)
                if retain_validloader:
                    validate(retain_validloader, self.model_s, self.loss_function)
                if forget_validloader:
                    validate(forget_validloader, self.model_s, self.loss_function)
            scheduler.step(epoch)
        return self.model_s 

    def get_params(self) -> dict:
        self.params = {
            'opt' : self.opt,
            'gamma' : self.gamma,
            'alpha' : self.alpha,
            'beta' : self.beta,
            'smoothing' : self.smoothing,
            'msteps': self.msteps,
            'clip': self.clip,
            'sstart': self.sstart,
            'kd_T': self.kd_T,
            'distill': self.distill,
            'sched': self.sched,
            'sgda_epochs': self.sgda_epochs,
            'sgda_learning_rate': self.sgda_learning_rate,
            'sgda_weight_decay': self.sgda_weight_decay,
            'sgda_momentum': self.sgda_momentum,
            'print_freq': self.print_freq,}
        return self.params

    def train_distill(self, epoch, train_loader, optimizer, split, quiet=False):
        """One epoch distillation"""
        # set modules as train()
        for module in self.module_list:
            module.train()
        # set teacher as eval()
        self.model_t.eval()

        criterion_cls = self.criterion_list[0]
        criterion_div = self.criterion_list[1]
        criterion_kd = self.criterion_list[2]

        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        kd_losses = utils.AverageMeter()
        top1 = utils.AverageMeter()


        end = time.time()
        for idx, data in enumerate(train_loader):
            input, target = data
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # ===================forward=====================
            logit_s = self.model_s(input)
            with torch.no_grad():
                logit_t = self.model_t(input)

            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)

            # other kd beyond KL divergence
            if self.distill == 'kd':
                loss_kd = 0
            else:
                raise NotImplementedError(self.distill)

            if split == "minimize":
                loss = self.gamma * loss_cls + self.alpha * loss_div + self.beta * loss_kd
            elif split == "maximize":
                loss = -loss_div

            loss = loss + param_dist(self.model_s, self.swa_model, self.smoothing)

            if split == "minimize" and not quiet:
                acc1 = utils.accuracy(logit_s, target, topk=(1,))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
            elif split == "maximize" and not quiet:
                kd_losses.update(loss.item(), input.size(0))
            elif split == "linear" and not quiet:
                acc1 = utils.accuracy(logit_s, target, topk=(1,))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                kd_losses.update(loss.item(), input.size(0))


            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_value_(model_s.parameters(), clip)
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            if not quiet:
                if idx % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, idx, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1))
                    sys.stdout.flush()

        
        if split == "minimize":
            if not quiet:
                print(' * Acc@1 {top1.avg:.3f} '
                    .format(top1=top1))

            return top1.avg, losses.avg
        else:
            return kd_losses.avg