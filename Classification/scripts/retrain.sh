py=/your/path/to/main_random.py

model=ResNet18
dataset=CIFAR10
num_classes=10
batch_size=128
input_size="3 32 32"

unlearn=Retrain
forget_perc=0.1
ckpt=/your/path/to/pre-trained/ResNet18-CIFAR10.pth
retrain_ckpt=/your/path/to/retrained/retrain-seed.pth
seed=0

python ${py} --dataset ${dataset} --model ${model} --num_classes ${num_classes} \
    --batch_size ${batch_size} --input_size ${input_size} \
    --unlearn ${unlearn} --forget_perc ${forget_perc} --checkpoint ${ckpt} --retrain_checkpoint ${retrain_ckpt} \
    --record_result --seed ${seed}
