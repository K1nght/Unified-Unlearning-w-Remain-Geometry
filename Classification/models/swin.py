import torch.nn as nn 
from torchvision.models import swin_t, swin_s, swin_b

def Swin_T(num_classes):
    model = swin_t(weights='DEFAULT')
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
    return model 

def Swin_S(num_classes):
    model = swin_s(weights='DEFAULT')
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
    return model 

def Swin_B(num_classes):
    model = swin_b(weights='DEFAULT')
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
    return model 