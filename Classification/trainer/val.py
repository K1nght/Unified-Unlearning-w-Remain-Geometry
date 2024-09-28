import time 
from collections import OrderedDict
import torch 
import utils


@torch.no_grad()
def validate(loader, model, loss_function, desc=""):
    start = time.time()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    for i, (images, labels) in enumerate(loader):
        model.eval()
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        acc1 = utils.accuracy(outputs.data, labels)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

    finish = time.time()
    print(f"{desc}Test Loss {losses.avg:.4f} Acc {top1.avg:.4f} Time: {finish-start:.2f}s")

    return OrderedDict([('loss', losses.avg), ('top1', top1.avg)])