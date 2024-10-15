# <div align="center"> Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancement </div>

> This repository contains code for the paper [Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancement](https://arxiv.org/pdf/2409.19732v1) by Zhehao Huang, Xinwen Cheng, JingHao Zheng, Haoran Wang, Zhengbao He, Tao Li, Xiaolin Huang.

<table align="center">
  <tr>
    <td align="center"> 
      <img src="overview.png" alt="Teaser" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> Overview of our <strong style="font-size: 18px;">S</strong>aliency <strong style="font-size: 18px;">F</strong>orgetting in the <strong style="font-size: 18px;">R</strong>emain-preserving manifold <strong style="font-size: 18px;">on</strong>line (SFR-on).</em>
    </td>
  </tr>
</table>

# Abstract

Machine unlearning (MU) has emerged to enhance the privacy and trustworthiness of deep neural networks. Approximate MU is a practical method for large-scale models. Our investigation into approximate MU starts with identifying the steepest descent direction, minimizing the output Kullback-Leibler divergence to exact MU inside a parameters' neighborhood. This probed direction decomposes into three components: weighted forgetting gradient ascent, fine-tuning retaining gradient descent, and a weight saliency matrix. Such decomposition derived from Euclidean metric encompasses most existing gradient-based MU methods. Nevertheless, adhering to Euclidean space may result in sub-optimal iterative trajectories due to the overlooked geometric structure of the output probability space. We suggest embedding the unlearning update into a manifold rendered by the remaining geometry, incorporating second-order Hessian from the remaining data. It helps prevent effective unlearning from interfering with the retained performance. However, computing the second-order Hessian for large-scale models is intractable. To efficiently leverage the benefits of Hessian modulation, we propose a fast-slow parameter update strategy to implicitly approximate the up-to-date salient unlearning direction.
Free from specific modal constraints, our approach is adaptable across computer vision unlearning tasks, including classification and generation. Extensive experiments validate our efficacy and efficiency. Notably, our method successfully performs class-forgetting on ImageNet using DiT and forgets a class on CIFAR-10 using DDPM in just 50 steps, compared to thousands of steps required by previous methods.

# 👉 Setup
First, download the repo:
```
git clone https://github.com/K1nght/Unified-Unlearning-w-Remain-Geometry.git
```

# 🔥 Getting Started 

* [SFR-on for Image Classification](Classification) 

* SFR-on for Image Generation 

    * [CIFAR-10 using DDPM](DDPM)

    * [ImageNet using DiT](DiT)

    * [Stable Diffusion](SD)

# 📚 Citation

```
@article{huang2024unified,
  title={Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancement},
  author={Huang, Zhehao and Cheng, Xinwen and Zheng, JingHao and Wang, Haoran and He, Zhengbao and Li, Tao and Huang, Xiaolin},
  journal={arXiv preprint arXiv:2409.19732},
  year={2024}
}
```