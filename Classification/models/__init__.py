from .resnet import *
from .swin import * 
from .vit import * 

def create_model(model_name, num_classes):
    return eval(model_name)(num_classes=num_classes)