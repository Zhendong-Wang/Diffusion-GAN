## Diffusion-GAN &mdash; Official PyTorch implementation

![Illustration](./docs/diffusion-gan.png)

**Diffusion-GAN: Training GANs with Diffusion**<br>
Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen and Mingyuan Zhou <br>
https://arxiv.org/pdf/2206.02262.pdf <br>

Abstract: *For stable training of generative adversarial networks (GANs), injecting instance
noise into the input of the discriminator is considered as a theoretically sound
solution, which, however, has not yet delivered on its promise in practice. This
paper introduces Diffusion-GAN that employs a Gaussian mixture distribution,
defined over all the diffusion steps of a forward diffusion chain, to inject instance
noise. A random sample from the mixture, which is diffused from an observed
or generated data, is fed as the input to the discriminator. The generator is
updated by backpropagating its gradient through the forward diffusion chain,
whose length is adaptively adjusted to control the maximum noise-to-data ratio
allowed at each training step. Theoretical analysis verifies the soundness of the
proposed Diffusion-GAN, which provides model- and domain-agnostic differentiable
augmentation. A rich set of experiments on diverse datasets show that DiffusionGAN can 
provide stable and data-efficient GAN training, bringing consistent
performance improvement over strong GAN baselines for synthesizing photorealistic images.*

## ToDos
- [x] Initial code release
- [x] Providing pretrained models

## Build your Diffusion-GAN
Here, we explain how to train general GANs with diffusion. We provide two ways: 
a. plug-in as simple as a data augmentation methods; 
b. training the GANs on diffusion chains with a timestep-dependent discriminator. 
Currently, we didn't find significant empirical differences of the two approaches, 
while the second approach has stronger theoretical guarantees. We suspect when advanced timestep-dependent structure is applied in the discriminator,
the second approach could become better, and we left that for future study. 

**Simple Plug-in**
* Design a proper diffusion process based on the ```diffusion.py``` file
* Apply diffusion on the inputs of discriminators, 
```logits = Discriminator(Diffusion(gen/real_images))```
* Add adaptiveness of diffusion into your training iterations
``` 
if update_diffusion:  # batch_idx % ada_interval == 0
    adjust = np.sign(sign(Discriminator(real_images)) - ada_target) * C  # C = (batch_size * ada_interval) / (ada_kimg * 1000)
    diffusion.p = (diffusion.p + adjust).clip(min=0., max=1.)
    diffusion.update_T()
```

**Full Version**
* Add diffusion timestep `t` as an input for discriminators `logits = Discriminator(images, t)`. 
You may need some modifications in your discriminator architecture. 
* The other steps are the same as Simple Plug-in. Note that since discriminator depends on timesteps, 
you need to collect `t`.
```
diffused_images, t = Diffusion(images)
logits = Discrimnator(diffused_images, t)
```

## Train our Diffusion-GAN

**Requirements**
* 64-bit Python 3.7 and PyTorch 1.7.1/1.8.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later. 
* Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.

**Data Preparation**

In our paper, we trained our model on [CIFAR-10 (32 x 32)](https://www.cs.toronto.edu/~kriz/cifar.html), [STL-10 (64 x 64)](https://cs.stanford.edu/~acoates/stl10/),
[LSUN (256 x 256)](https://github.com/fyu/lsun), [AFHQ (512 x 512)](https://github.com/clovaai/stargan-v2) and [FFHQ (1024 x 1024)](https://github.com/NVlabs/ffhq-dataset).
You can download the datasets we used in our paper at their respective websites. 
To prepare the dataset at the respective resolution, run for example
```.bash
python dataset_tool.py --source=~/downloads/lsun/raw/bedroom_lmdb --dest=~/datasets/lsun_bedroom200k.zip \
    --transform=center-crop --width=256 --height=256 --max_images=200000

python dataset_tool.py --source=~/downloads/lsun/raw/church_lmdb --dest=~/datasets/lsun_church200k.zip \
    --transform=center-crop-wide --width=256 --height=256 --max_images=200000
```

**Training**

We show the training commands that we used below. In most cases, the training commands are similar, so below we use CIFAR-10 dataset
as an example: 

For Diffusion-GAN,
```.bash
python train.py --outdir=training-runs --data="~/cifar10.zip" --gpus=4 --cfg cifar --kimg 50000 --aug no --target 0.6 --noise_sd 0.05 --ts_dist priority
```
For Diffusion-GAN + DIFF, 
```.bash
python train.py --outdir=training-runs --data="~/cifar10.zip" --gpus=4 --cfg cifar --kimg 50000 --aug diff --target 0.6 --noise_sd 0.05 --ts_dist priority
```
For Diffusion-GAN + ADA, 
```.bash
python train.py --outdir=training-runs --data="~/cifar10.zip" --gpus=4 --cfg cifar --kimg 50000 --aug ada --ada_maxp 0.25 --target 0.6 --noise_sd 0.05 --ts_dist priority
```
For Diffusion-ProjectedGAN
```.bash
python train.py --outdir=training-runs --data="~/cifar10.zip" --gpus=4 --cfg cifar --kimg 50000 --target 0.45 --d_pos first --noise_sd 0.5
```
We follows the `config` setting from [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorchhttps://github.com/NVlabs/stylegan2-ada-pytorch) 
and refer to them for more details. The other major hyperparameters are listed below. 

* `--target` the discriminator target, which balances the level of diffusion intensity.
* `--aug` domain-specific image augmentation, such as ada and differentiable augmentation, which is used for evaluate complementariness with diffusion. 
* `--noise_sd` diffusion noise standard deviation.
* ` --ts_dist` t sampling distribution, $\pi(t)$ in paper.

## Checkpoints
Will be uploaded soon. 

## Citation

```
@InProceedings{wang2022diffusiongan,
  author    = {Wang, Zhendong and Zheng, Huangjie and He, Pengcheng and Chen, Weizhu and Zhou, Mingyuan},
  title     = {Diffusion-GAN: Training GANs with Diffusion},
  journal   = {arXiv.org},
  volume    = {abs/2206.02262},
  year      = {2022},
  url       = {https://arxiv.org/abs/2206.02262},
}
```

## Acknowledgements

Our code builds upon the awesome [StyleGAN2-ADA repo](https://github.com/NVlabs/stylegan2-ada-pytorch) and [ProjectedGAN repo](https://github.com/autonomousvision/projected_gan), respectively by Karras et al and Axel Sauer et al.
