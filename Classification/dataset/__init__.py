from .cifar10 import PretrainCIFAR10, FullClassUnlearnCIFAR10, RandomUnlearnCIFAR10
from .cifar100 import PretrainCIFAR100, FullClassUnlearnCIFAR100, RandomUnlearnCIFAR100
from .SVHN import PretrainSVHN, FullClassUnlearnSVHN, RandomUnlearnSVHN
from .tinyimagenet import PretrainTinyImageNet, RandomUnlearnTinyImageNet


def create_dataset(dataset_name, setting, root, img_size=32):
    dataset = eval(f"{setting}{dataset_name}")
    return dataset(root, img_size)

def dataset_convert_to_valid(dataset, transform_valid):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = transform_valid
    dataset.train = False