import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from trainer import train, validate
from models import create_model
from datasets import create_dataset


def main():
    parser = argparse.ArgumentParser(description='Pretraining Config')
    parser.add_argument('--data_dir', metavar='DIR', default='./data',
                        help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')

    parser.add_argument('--model', default='resnet18', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet50"')
    parser.add_argument('--num_classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--input_size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input_size 3 224 224), uses model default if empty')
    parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')

    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate, overrides lr-base if set (default: None)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
    args = parser.parse_args()


    utils.random_seed(args.seed)

    # dataloaders
    dataset = create_dataset(
        dataset_name=args.dataset, setting="Pretrain", root=args.data_dir, img_size=args.input_size[-1]
    )
    trainset, testset = dataset.get_datasets()

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # get network
    model = create_model(
        model_name=args.model, num_classes=args.num_classes
    )
    model.cuda() 

    loss_function = nn.CrossEntropyLoss()
    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_path = os.path.join("./checkpoint", args.model, args.dataset, datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"Save result at {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    weights_path = os.path.join(save_path, f"{args.model}-{args.dataset}.pth")

    best_acc = None 
    best_epoch = None 
    for epoch in range(1, args.epochs + 1):
        train_metrics = train(epoch, trainloader, model, loss_function, optimizer, log_freq=50)
        eval_metrics = validate(testloader, model, loss_function)
        scheduler.step()
        utils.update_summary(
            epoch, train_metrics, eval_metrics, os.path.join(save_path, "summary.csv"),
            write_header=best_acc is None,
        )
        if best_acc is None or best_acc < eval_metrics["top1"]:  
            print("saving weights file to {}".format(weights_path))
            torch.save(model.state_dict(), weights_path)
            best_acc = eval_metrics["top1"]
            best_epoch = epoch
    print('*** Best metric: {0} (epoch {1})'.format(best_acc, best_epoch))


if __name__ == '__main__':
    main()