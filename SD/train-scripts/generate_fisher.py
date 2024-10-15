import argparse
import os

import torch
from dataset import setup_forget_data, setup_forget_nsfw_data, setup_model
from tqdm import tqdm

def generate_nsfw_fisher(
    c_guidance,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    image_size=512,
    num_timesteps=1000,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    forget_dl, remain_dl = setup_forget_nsfw_data(batch_size, image_size)
    print(len(forget_dl), len(remain_dl))

    model.eval()
    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.diffusion_model.parameters(), lr=lr)
    word_nude = "a photo of a nude person"
    word_nonnude = "a photo of a person wearing clothes"

    forget_gradients = {}
    for name, param in model.model.diffusion_model.named_parameters():
        forget_gradients[name] = 0

    # get forget fisher diag
    for i, images in tqdm(enumerate(forget_dl), total=len(forget_dl)):
        optimizer.zero_grad()

        images = images.to(device)
        t = torch.randint(
            0, num_timesteps, (images.shape[0],), device=device
        ).long()

        null_prompts = [""] * batch_size
        prompts = [word_nude] * batch_size
        print(prompts)

        forget_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": prompts}

        null_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": null_prompts}

        forget_input, forget_emb = model.get_input(
            forget_batch, model.first_stage_key
        )
        null_input, null_emb = model.get_input(null_batch, model.first_stage_key)

        t = torch.randint(
            0, model.num_timesteps, (forget_input.shape[0],), device=device
        ).long()
        noise = torch.randn_like(forget_input, device=device)

        forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)

        forget_out = model.apply_model(forget_noisy, t, forget_emb)
        null_out = model.apply_model(forget_noisy, t, null_emb)

        preds = (1 + c_guidance) * forget_out - c_guidance * null_out

        # print(images.shape, noise.shape, preds.shape)
        loss = -criteria(noise, preds)
        loss.backward()

        with torch.no_grad():
            for name, param in model.model.diffusion_model.named_parameters():
                if param.grad is not None:
                    forget_gradients[name] += (param.grad.data.cpu()**2) / len(forget_dl)
                    # gradients[name] += param.grad.data

    torch.save(forget_gradients, os.path.join("fisher/nude_forget.pt"))
####################
    remain_gradients = {}
    for name, param in model.model.diffusion_model.named_parameters():
        remain_gradients[name] = 0

    # get remain fisher diag
    for i, images in tqdm(enumerate(remain_dl), total=len(remain_dl)):
        optimizer.zero_grad()

        images = images.to(device)
        t = torch.randint(
            0, num_timesteps, (images.shape[0],), device=device
        ).long()

        null_prompts = [""] * batch_size
        prompts = [word_nonnude] * batch_size
        print(prompts)

        remain_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": prompts}

        null_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": null_prompts}

        remain_input, remain_emb = model.get_input(
            remain_batch, model.first_stage_key
        )
        null_input, null_emb = model.get_input(null_batch, model.first_stage_key)

        t = torch.randint(
            0, model.num_timesteps, (remain_input.shape[0],), device=device
        ).long()
        noise = torch.randn_like(remain_input, device=device)

        remain_noisy = model.q_sample(x_start=remain_input, t=t, noise=noise)

        remain_out = model.apply_model(remain_noisy, t, remain_emb)
        null_out = model.apply_model(remain_noisy, t, null_emb)

        preds = (1 + c_guidance) * remain_out - c_guidance * null_out

        # print(images.shape, noise.shape, preds.shape)
        loss = -criteria(noise, preds)
        loss.backward()

        with torch.no_grad():
            for name, param in model.model.diffusion_model.named_parameters():
                if param.grad is not None:
                    remain_gradients[name] += (param.grad.data.cpu()**2) / len(remain_dl)
                    # gradients[name] += param.grad.data

    torch.save(remain_gradients, os.path.join("fisher/nude_remain.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Generate fisher", description="generate fisher for NSFW-concept"
    )

    parser.add_argument(
        "--c_guidance",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=7.5,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=1
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
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
        default="4",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--num_timesteps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=1000,
    )
    args = parser.parse_args()


    c_guidance = args.c_guidance
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    num_timesteps = args.num_timesteps

    generate_nsfw_fisher(
        c_guidance,
        batch_size,
        epochs,
        lr,
        config_path,
        ckpt_path,
        diffusers_config_path,
        device,
        image_size,
        num_timesteps,
    )
