import time 
from collections import OrderedDict
import torch 
import utils


def train(epoch, loader, model, loss_function, optimizer, scheduler=None, log_freq=10000):
    start = time.time()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    for i, (images, labels) in enumerate(loader):
        model.train()
        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        acc1 = utils.accuracy(outputs.data, labels)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        if scheduler is not None:
            scheduler.step()
        if i % log_freq == 0:
            print(f"Train {i}/{len(loader)} Loss {losses.val:.4f}({losses.avg:.4f}) Acc {top1.val:.4f}({top1.avg:.4f})")
    finish = time.time()
    lr = optimizer.param_groups[0]['lr']
    print(f"Train Epoch {epoch} Loss {losses.avg:.4f} Acc {top1.avg:.4f} LR {lr} Time: {finish-start:.2f}s")

    return OrderedDict([('loss', losses.avg), ('top1', top1.avg), ('lr', lr)])