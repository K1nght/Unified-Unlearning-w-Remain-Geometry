py=/your/path/to/main_pretrain.py

model=ResNet18
dataset=CIFAR10
num_classes=10
batch_size=128
input_size="3 32 32"
epochs=200 
lr=0.1

python ${py} --dataset ${dataset} --model ${model} --num_classes ${num_classes} \
    --batch_size ${batch_size} --input_size ${input_size} --epochs ${epochs} --lr ${lr}

