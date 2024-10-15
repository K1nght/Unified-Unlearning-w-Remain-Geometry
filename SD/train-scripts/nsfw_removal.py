import argparse
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from convertModels import savemodelDiffusers
from dataset import (
    setup_forget_nsfw_data,
    setup_model,
)
from diffusers import LMSDiscreteScheduler
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm

def cycle(dl):
    while True:
        for data in dl:
            yield data

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def nsfw_removal(
    train_method,
    forget_alpha,
    remain_alpha,
    batch_size,
    n_iters,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    mask_threshold=None,
    image_size=512,
    ddim_steps=50,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    sampler = DDIMSampler(model)
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    criteria = torch.nn.MSELoss()
    forget_dl, remain_dl = setup_forget_nsfw_data(batch_size, image_size)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            # print(name)
            parameters.append(param)
    # set model to train
    model.train()

    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    if mask_path:
        mask = torch.load(os.path.join(mask_path, f"nude_mask_{mask_threshold}.pt"))
        print(f"Load saliency mask from {mask_path} with threshold={mask_threshold}")
        name = f"compvis-nsfw-mask{mask_threshold}-method_sfron-lr{lr}_fa{forget_alpha}_ra{remain_alpha}"
    else:
        name = f"compvis-nsfw-method_sfron-lr{lr}_fa{forget_alpha}_ra{remain_alpha}"

    # NSFW Removal
    word_nude = "a photo of a nude person"
    word_wear = "a photo of a person wearing clothes"
    print(f"prompt of NSFW: {word_nude}")
    print(f"prompt of non-NSFW: {word_wear}")

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_forget_loss = 0
    running_remain_loss = 0
    start_time = time()

    forget_dl_iter = cycle(forget_dl)
    remain_dl_iter = cycle(remain_dl)

    # TRAINING CODE
    for step in range(0, n_iters):
        model.train() 

        forget_images = next(forget_dl_iter)
        remain_images = next(remain_dl_iter)

        forget_prompts = [word_nude] * batch_size

        # player -> truck
        pseudo_prompts = [word_wear] * batch_size
        remain_prompts = [word_wear] * batch_size

        # forget stage
        optimizer.zero_grad()
        forget_batch = {
            "jpg": forget_images.permute(0, 2, 3, 1),
            "txt": forget_prompts,
        }

        pseudo_batch = {
            "jpg": forget_images.permute(0, 2, 3, 1),
            "txt": pseudo_prompts,
        }

        forget_input, forget_emb = model.get_input(
            forget_batch, model.first_stage_key
        )
        pseudo_input, pseudo_emb = model.get_input(
            pseudo_batch, model.first_stage_key
        )

        t = torch.randint(
            0,
            model.num_timesteps,
            (forget_input.shape[0],),
            device=model.device,
        ).long()
        noise = torch.randn_like(forget_input, device=model.device)

        forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
        pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)

        forget_out = model.apply_model(forget_noisy, t, forget_emb)
        pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

        ori_forget_loss = criteria(forget_out, pseudo_out)
        forget_loss = forget_alpha * ori_forget_loss
        forget_loss.backward() 

        if mask_path:
            for n, p in model.named_parameters():
                if p.grad is not None and n in parameters:
                    p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(device)

        optimizer.step()

        # remain stage
        optimizer.zero_grad()
        remain_batch = {
            "jpg": remain_images.permute(0, 2, 3, 1),
            "txt": remain_prompts,
        }
        ori_remain_loss = model.shared_step(remain_batch)[0]
        remain_loss = remain_alpha * ori_remain_loss
        remain_loss.backward()
        optimizer.step()

        # Log loss values:
        running_forget_loss += ori_forget_loss.item()
        running_remain_loss += ori_remain_loss.item() 
        log_steps += 1
        train_steps += 1
        if train_steps % 10 == 0:
            # Measure training speed:
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_forget_loss = torch.tensor(running_forget_loss / log_steps, device=device).item()
            avg_remain_loss = torch.tensor(running_remain_loss / log_steps, device=device).item()
            print(f"(step={train_steps:07d}) Forget Loss: {avg_forget_loss:.6f}, Remain Loss: {avg_remain_loss:.6f}, Train Steps/Sec: {steps_per_sec:.2f}")
            # Reset monitoring variables:
            running_forget_loss = 0
            running_remain_loss = 0
            log_steps = 0
            start_time = time()
        
        if (train_steps+1) % 200 == 0:
            save_model(
                model,
                name,
                train_steps+1,
                save_compvis=True,
                save_diffusers=True,
                compvis_config_file=config_path,
                diffusers_config_file=diffusers_config_path,
            )

    model.eval()
    save_model(
        model,
        name,
        None,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        file_name = f"{name}-step_{num}"
        path = f"{folder_path}/{file_name}.pt"
    else:
        file_name = name
        path = f"{folder_path}/{file_name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)
        print(f"Saving Model in compvis Format at {path}")

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            folder_path, file_name, compvis_config_file, diffusers_config_file, device=device,
        )


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SFR-on for SD",
        description="Finetuning stable diffusion model to erase concepts using SFR-on method",
    )

    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--n_iters", 
        type=int, 
        default=1000
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=int,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="0,0",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--forget_alpha",
        help="alpha for forget loss",
        type=float,
        required=False,
        default=1.0,
    )
    parser.add_argument(
        "--remain_alpha",
        help="alpha for remain loss",
        type=float,
        required=False,
        default=1.0,
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--mask_threshold",
        help="saliency mask threshold",
        type=float,
        default=None,
    )
    args = parser.parse_args()

    train_method = args.train_method
    forget_alpha = args.forget_alpha
    remain_alpha = args.remain_alpha
    batch_size = args.batch_size
    n_iters = args.n_iters
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    mask_threshold = args.mask_threshold
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    nsfw_removal(
        train_method=train_method,
        forget_alpha=forget_alpha,
        remain_alpha=remain_alpha,
        batch_size=batch_size,
        n_iters=n_iters,
        lr=lr,
        config_path=config_path,
        ckpt_path=ckpt_path,
        mask_path=mask_path,
        diffusers_config_path=diffusers_config_path,
        device=device,
        mask_threshold=mask_threshold,
        image_size=image_size,
        ddim_steps=ddim_steps,
    )