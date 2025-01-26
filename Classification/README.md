# **S**aliency **F**orgetting in the **R**emain-preserving manifold **on**line for **Classification**
This is the official repository for SFR-on on Classification.

## Requirements
```
conda create -n sfr-cls python=3.8
conda activate sfr-cls
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tqdm
```

## Scripts
An example for ResNet-18 on CIFAR-10 to unlearn 10% data.
1. Get the pre-trained model.

```
sh scripts/pretrain.sh
```

2. Get the retrained model. 

```
sh scripts/retrain.sh
```

3. Unlearn with SFR-on.

```
sh scripts/unlearn.sh
```
