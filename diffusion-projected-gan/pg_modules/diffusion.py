# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import scipy.signal
import torch
from torch_utils import persistence
from torch_utils import misc
from torch_utils.ops import upfirdn2d
from torch_utils.ops import grid_sample_gradfix
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------
# Helpers for doing diffusion process.


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    def continuous_t_beta(t, T):
        b_max = 5.
        b_min = 0.1
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        return 1 - alpha

    if beta_schedule == "continuous_t":
        betas = continuous_t_beta(np.arange(1, num_diffusion_timesteps+1), num_diffusion_timesteps)
    elif beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def q_sample(x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise_type='gauss', noise_std=1.0):
    batch_size, num_channels, _, _ = x_0.shape
    if noise_type == 'gauss':
        noise = torch.randn_like(x_0, device=x_0.device) * noise_std
    elif noise_type == 'bernoulli':
        noise = (torch.bernoulli(torch.ones_like(x_0) * 0.5) * 2 - 1.) * noise_std
    else:
        raise NotImplementedError(noise_type)
    alphas_t_sqrt = alphas_bar_sqrt[t].view(batch_size, num_channels, 1, 1)
    one_minus_alphas_bar_t_sqrt = one_minus_alphas_bar_sqrt[t].view(batch_size, num_channels, 1, 1)
    x_t = alphas_t_sqrt * x_0 + one_minus_alphas_bar_t_sqrt * noise
    return x_t


@persistence.persistent_class
class Diffusion(torch.nn.Module):
    def __init__(self,
        beta_schedule='linear', beta_start=1e-4, beta_end=1e-2,
        t_min=5, t_max=500, noise_std=0.5,
    ):
        super().__init__()
        self.p = 0.0       # Overall multiplier for augmentation probability.
        self.noise_type = self.base_noise_type = 'gauss'
        self.base_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.t_min = t_min
        self.t_max = t_max
        self.t_add = t_max - t_min
        self.update_T()

        # Image-space corruptions.
        self.noise_std = float(noise_std)        # Standard deviation of additive RGB noise.

    def set_diffusion_process(self, t, beta_schedule):

        betas = get_beta_schedule(
            beta_schedule=beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            num_diffusion_timesteps=t,
        )

        betas = self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = betas.shape[0]

        alphas = self.alphas = 1.0 - betas
        alphas_cumprod = torch.cat([torch.tensor([1.]), alphas.cumprod(dim=0)])
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

    def update_T(self):
        t_adjust = round(self.p * self.t_add)
        t = np.clip(int(self.t_min + t_adjust), a_min=self.t_min, a_max=self.t_max)
        self.set_diffusion_process(t, "linear")

        # sampling t
        self.t_epl = np.zeros(64, dtype=np.int)
        diffusion_ind = min(round(self.p * 64), 48)  # 48
        prob_t = np.arange(t) / np.arange(t).sum()
        t_diffusion = np.random.choice(np.arange(1, t+1), size=diffusion_ind, p=prob_t)
        self.t_epl[:diffusion_ind] = t_diffusion

    def forward(self, x_0, noise_std=1.0):
        assert isinstance(x_0, torch.Tensor) and x_0.ndim == 4
        batch_size, num_channels, height, width = x_0.shape
        device = x_0.device

        alphas_bar_sqrt = self.alphas_bar_sqrt.to(device)
        one_minus_alphas_bar_sqrt = self.one_minus_alphas_bar_sqrt.to(device)

        t = torch.from_numpy(np.random.choice(self.t_epl, size=batch_size * num_channels, replace=True)).to(device)

        x_t = q_sample(x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t,
                       noise_type=self.noise_type,
                       noise_std=noise_std)
        return x_t

#----------------------------------------------------------------------------