# **S**aliency **F**orgetting in the **R**emain-preserving manifold **on**line for NSFW-concept removal in SD
This is the official repository for SFR-on for NSFW-concept removal in stable diffusion. The code structure of this project is adapted from the [ESD](https://github.com/rohitgandikota/erasing/tree/main) and [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency/tree/master/SD) codebase.

<table align="center">
  <tr>
    <td align="center"> 
      <img src="NSFW-concept removal.png" alt="Teaser" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> Overview of NSFW-concept removal in SD by our SFR-on.</em>
    </td>
  </tr>
</table>


# Installation Guide
* To get started, clone the following repository of Original Stable Diffusion [Link](https://github.com/CompVis/stable-diffusion)
* Install the requirements from the following link for the Original Stable Diffusion
* Then download the files from our repository to stable-diffusion main directory of stable diffusion. This would replace the ldm folder of the original repo with our custom ldm directory
* Download the weights from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt) and move them to `SD/models/ldm/`

# Folder Structure

```
SD/
├── config/               # Directory for storing configuration files
├── ldm/                  # Directory for model and data files
├── data/                 # Directory for storing generated images
│   ├── nsfw/             # NSFW images generated from SD
│   └── not-nsfw/         # Non-NSFW images generated from SD
├── fisher/               # Directory for storing fisher and saliency masks
├── models/               # Directory for model checkpoints
│   └── ldm/              # Location for SD-v1-4 weights
├── prompts/              # Directory containing prompt files
│   ├── nsfw.csv
│   ├── not-nsfw.csv
├── eval-scripts/         # Main evaluation scripts location
│   ├── generate-images.py
│   └── nudenet-classes.py
└── train-scripts/        # Main training scripts location
    ├── generate_fisher.py
    ├── generate_fisher_mask.py
    └── nsfw_removal.py
```

# NSFW-concept removal with Saliency-Unlearning
1. To remove NSFW-concept, we initially utilize SD V1.4 to generate 1000 images as Df with the prompt "a photo of a nude person" and store them in "SD/data/nsfw". Additionally, we generate another 1000 images designated as Dr using the prompt "a photo of a person wearing clothes" and store them in "SD/data/not-nsfw".
   ```
   python generate-images.py --prompts_path 'prompts/nsfw.csv' --save_path 'SD/data/nsfw' --model_name SD-v1-4 --device 'cuda:0' --num_samples 5
   ```
   ```
   python generate-images.py --prompts_path 'prompts/not-nsfw.csv' --save_path 'SD/data/not-nsfw' --model_name SD-v1-4 --device 'cuda:0' --num_samples 5
   ```

2. Next, we need to generate forget and remain fisher and Forget-Remain Balanced Weight Saliency mask for NSFW-concept.

   ```
   python generate_fisher.py --ckpt_path 'models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt' --batch_size 2 --device '0'
   ```
   ```
   python generate_fisher_mask.py --ckpt_folder fisher --threshold ${gamma}
   ```
   * `gamma` is the threshold for the saliency map.

   This will save fisher and saliency map in `SD/fisher`.

3. Forgetting training with SFR-on

   ```
   python nsfw_removal.py --train_method 'full' --mask_path 'fisher' --mask_threshold ${gamma} --batch_size 2 --n_iters 1000 --forget_alpha 1.0 --remain_alpha 1.0 --device '0'
   ```
   * `gamma` is the threshold for the saliency map.

   This will save the model in `SD/models`.

# Evaluation

1. To evaluate the NSFW-concept removal in diffusion model, we need to generate images from unsafe [I2P](https://github.com/ml-research/safe-latent-diffusion) prompts.

   * original SD-v1-4
   ```
   python generate-images.py --prompts_path 'prompts/unsafe-prompts4703.csv' --save_path 'evaluation_folder/unsafe/SD-v1-4' --model_name SD-v1-4 --device 'cuda:0' --num_samples 2 --num_iters 1
   ```
   * SD-v1-4 unlearned by SFR-on
   ```
   python generate-images.py --prompts_path 'prompts/unsafe-prompts4703.csv' --save_path 'evaluation_folder/unsafe/SD-v1-4-sfron' --model_name ${save_path} --device 'cuda:0' --num_samples 2 --num_iters 1
   ```

2. Now we can detect NSFW-concept in the generated images by running NudeNet.

   We need to install NudeNet by following the instructions [here](https://pypi.org/project/nudenet/).

   ```
   python nudenet-classes.py --folder ${folder} --save_path ${save}
   ```
   * `folder` is the folder containing the generated images.
   * `save` is the path to save the results.


