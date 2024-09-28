import torch.nn as nn 
from torchvision.models import vit_b_16

def ViT_B(num_classes, pretrained=True):
    model = vit_b_16(pretrained=pretrained)
    hidden_dim = model.hidden_dim
    model.head = nn.Linear(hidden_dim, num_classes) if num_classes > 0 else nn.Identity()
    return model