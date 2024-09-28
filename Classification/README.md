# **S**aliency **F**orgetting in the **R**emain-preserving manifold **on**line for **Classification**
This is the official repository for SFR-on on Classification.

## Requirements
```
pip install -r requirements.txt
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
