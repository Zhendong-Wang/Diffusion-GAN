# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

from training.adaaug import AdaAugment

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2LossCL(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.image_disturb = AdaAugment(p=0.2).to(device)

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img, t = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c, t)
        return logits

    def run_cl(self, img, c, sync, contrastive_head, D_ema, loss_name='', loss_only=False, img1=None, update_q=False):
        # contrastive loss fwd 

        # augmentation first via ada-aug
        # assert(self.augment_pipe is not None)
        img0 = self.image_disturb(img)
        img1 = self.image_disturb(img) if img1 is None else self.image_disturb(img1)
        batch_size, device = img.shape[0], img.device
        # img0 = img
        # img1 = img.clone() + torch.randn_like(img) * 0.02 if img1 is None else img1

        # extract features for two views via D and momentum D
        _, logits0 = self.D(img0, c, torch.zeros((batch_size, 1)).long().to(device), return_feats=True)
        with torch.no_grad():
            _, logits1 = D_ema(img1, c, torch.zeros((batch_size, 1)).long().to(device), return_feats=True)

        # project features into the unit sphere and calculate contrastive loss
        loss = contrastive_head(logits0, logits1, loss_only=loss_only, update_q=update_q)
        training_stats.report('Loss/'+loss_name, loss)
        return loss

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, cl_phases=None, D_ema=None, lw_real_cl=1.0, lw_fake_cl=1.0, lw_fake_cl_on_g=1.0, g_fake_cl=False):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)

                # Diversity generation loss from fake instance discrimination
                if cl_phases.get('GHeadmain', None) is not None and g_fake_cl:
                    # when fake cl on g, no params in D encoder and head would be updated, including feature queue.
                    Gphase = cl_phases['GHeadmain']
                    Gphase.module.requires_grad_(False)
                    # fake_cl on g: gradients bp to generator
                    loss_Gmain = loss_Gmain + lw_fake_cl_on_g * self.run_cl(gen_img, gen_c, False, Gphase.module, D_ema, loss_name='G_cl_on_g', loss_only=True)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

            if cl_phases.get('GHeadmain', None) is not None and g_fake_cl:
                Gphase = cl_phases['GHeadmain']
                Gphase.module.requires_grad_(True)

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                    # Contrastive loss would be added to the normal binary cls loss of D  
                    # real instance discrimination
                    if cl_phases.get('DHeadmain', None) is not None:
                        Dphase = cl_phases['DHeadmain']
                        Dphase.opt.zero_grad(set_to_none=True)
                        loss_Dreal = loss_Dreal + lw_real_cl * self.run_cl(real_img_tmp, real_c, sync, Dphase.module, D_ema, loss_name='D_cl')

                    # fake instance discrimination
                    if cl_phases.get('GHeadmain', None) is not None:
                        Gphase = cl_phases['GHeadmain'] 
                        Gphase.opt.zero_grad(set_to_none=True) 
                        # noisy perturbation
                        with torch.no_grad():
                            delta_z = torch.randn(gen_z.shape, device=gen_z.device) * 0.15
                            noisy_gen_img, _ = self.run_G(gen_z + delta_z, gen_c, sync=False)
                        loss_Dreal = loss_Dreal + lw_fake_cl * self.run_cl(gen_img, gen_c, False, Gphase.module, D_ema, loss_name='G_cl', img1=noisy_gen_img, update_q=True)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
            
            # after backward of contrastive loss together with the original loss of D, 
            # manually call optim.step() to update the parameters of contrative head
            if cl_phases.get('DHeadmain', None) is not None and do_Dmain:
                Dphase.opt.step()

            if cl_phases.get('GHeadmain', None) is not None and do_Dmain:
                Gphase.opt.step()

#----------------------------------------------------------------------------
