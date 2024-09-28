import os
import sys
import random 
import argparse
import time
import numpy as np
import pandas as pd 
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from models import create_model
from dataset import create_dataset, dataset_convert_to_valid
import unlearn 
from trainer import validate
from evaluation import get_membership_attack_prob, get_js_divergence, get_SVC_MIA


def main():
    parser = argparse.ArgumentParser(description='Random Subset Unlearn Config')
    parser.add_argument('--data_dir', metavar='DIR', default='./data',
                        help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')

    parser.add_argument('--model', default='resnet18', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet50"')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--num_classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--input_size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input_size 3 224 224), uses model default if empty')
    parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

    parser.add_argument("--unlearn", type=str, required=True, nargs="?",
                    help="select unlearning method from choice set")
    parser.add_argument("--forget_perc", type=float, required=True, nargs="?",
                    help="forget random subset percentage")
    parser.add_argument('--retrain_checkpoint', default=None, type=str, metavar='PATH',
                        help='Retrained model checkpoint for evaluation (default: none)')
    parser.add_argument("--record_result", action="store_true", default=False, 
                    help="not record result")
    args = parser.parse_args()


    utils.random_seed(args.seed)

    # dataloaders
    dataset = create_dataset(
        dataset_name=args.dataset, setting="RandomUnlearn", root=args.data_dir, img_size=args.input_size[-1]
    )

    save_path = f'results/{dataset.setting}/{args.model}_{args.dataset}_f-{args.forget_perc}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainset, testset = dataset.get_datasets()
    transform_valid = testset.transform
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    (
        forget_trainset,
        retain_trainset,
    ) = dataset.random_split(args.forget_perc, save_path)
    print(len(forget_trainset), len(retain_trainset))
    # print(np.unique(forget_trainset.targets), np.unique(retain_trainset.targets))
    forget_trainloader = DataLoader(forget_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    retain_trainloader = DataLoader(retain_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    unlearn_dataloaders = OrderedDict(
        forget_train=forget_trainloader, 
        retain_train=retain_trainloader, 
        forget_valid=None, 
        retain_valid=testloader
    )

    # get network
    model = create_model(
        model_name=args.model, num_classes=args.num_classes
    )
    model.load_state_dict(torch.load(args.checkpoint))
    model.cuda() 
    if args.retrain_checkpoint is not None and os.path.exists(args.retrain_checkpoint):
        retrained_model = create_model(
            model_name=args.model, num_classes=args.num_classes
        )
        retrained_model.load_state_dict(torch.load(args.retrain_checkpoint))
        retrained_model.cuda()
    else:
        retrained_model = None 

    loss_function = nn.CrossEntropyLoss()

    # start unlearning
    start = time.time()
    unlearn_method = unlearn.create_unlearn_method(args.unlearn)(model, loss_function, save_path, args)
    unlearn_method.prepare_unlearn(unlearn_dataloaders)
    unlearned_model = unlearn_method.get_unlearned_model()
    end = time.time()
    time_elapsed = end - start 

    # evaluate
    evaluation_result = {'method': args.unlearn, 'seed': args.seed}
    for name, loader in unlearn_dataloaders.items():
        if loader is None: continue 
        dataset_convert_to_valid(loader, transform_valid)
        eval_metrics = validate(loader, unlearned_model, loss_function, name)
        for metr, v in eval_metrics.items():
            evaluation_result[f"{name} {metr}"] = v 
    for mia_metric in ["entropy"]:
        evaluation_result[f"{mia_metric} mia"] = get_membership_attack_prob(retain_trainloader, forget_trainloader, testloader, unlearned_model, mia_metric)
    # SVC MIA 
    dataset_convert_to_valid(retain_trainset, transform_valid)
    retain_len = len(retain_trainset)
    test_len = len(testset)

    # shadow_train = torch.utils.data.Subset(retain_trainset, random.sample(range(retain_len), test_len))
    # shadow_train_loader = torch.utils.data.DataLoader(
    #     shadow_train, batch_size=args.batch_size, shuffle=False, num_workers=4
    # )
    # svc_mia_forget_efficacy = get_SVC_MIA(
    #     shadow_train=shadow_train_loader, 
    #     shadow_test=testloader, 
    #     target_train=None, 
    #     target_test=forget_trainloader,
    #     model=unlearned_model
    # )
    # for metr, v in svc_mia_forget_efficacy.items():
    #     evaluation_result[f"svc_mia_{metr}"] = v 

    # JS divergence and KL divergence
    if retrained_model:
        evaluation_result['js_div'], evaluation_result['kl_div'] = get_js_divergence(forget_trainloader, unlearned_model, retrained_model)
    else:
        evaluation_result['js_div'] = None 
        evaluation_result['kl_div'] = None
    evaluation_result["time"] = time_elapsed
    evaluation_result["params"] = unlearn_method.get_params()
    print("|\t".join([str(k) for k in evaluation_result.keys()]))
    print("|\t".join([str(v) for v in evaluation_result.values()]))

    if args.record_result:
        file_path = os.path.join(save_path, f'results.csv')
    else:
        file_path = os.path.join(save_path, f'tmp.csv')
    new_row = pd.DataFrame({k:[v] for k, v in evaluation_result.items()})
    new_row.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))


if __name__ == '__main__':
    main()