# Code are mainly borrowed from the official implementation of MoCo (https://github.com/facebookresearch/moco)

import numpy as np
import torch
import torch.nn as nn
from torch_utils import misc
from torch_utils import persistence

#----------------------------------------------------------------------------

# Contrastive head
@persistence.persistent_class
class CLHead(torch.nn.Module):
    def __init__(self,
        inplanes     = 256,   # Number of input features
        temperature  = 0.2,   # Temperature of logits
        queue_size   = 3500,  # Number of stored negative samples
        momentum     = 0.999, # Momentum for updating network
    ):
        super().__init__()
        self.inplanes = inplanes
        self.temperature = temperature
        self.queue_size = queue_size
        self.m = momentum

        self.mlp = nn.Sequential(nn.Linear(inplanes, inplanes), nn.ReLU(), nn.Linear(inplanes, 128))
        self.momentum_mlp = nn.Sequential(nn.Linear(inplanes, inplanes), nn.ReLU(), nn.Linear(inplanes, 128))
        self.momentum_mlp.requires_grad_(False)

        for param_q, param_k in zip(self.mlp.parameters(), self.momentum_mlp.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(128, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.mlp.parameters(), self.momentum_mlp.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        keys = keys.T
        ptr = int(self.queue_ptr) 
        if batch_size > self.queue_size:
            self.queue[:, 0:] = keys[:, :self.queue_size]

        elif ptr + batch_size > self.queue_size:
            self.queue[:, ptr:] = keys[:, :self.queue_size - ptr]
            self.queue[:, :batch_size - (self.queue_size - ptr)] = keys[:, self.queue_size-ptr:]
            self.queue_ptr[0] = batch_size - (self.queue_size - ptr)
        else:
            self.queue[:, ptr:ptr + batch_size] = keys
            self.queue_ptr[0] = ptr + batch_size

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """

        # If non-distributed now, return raw input directly.
        # We have no idea the effect of disabling shuffle BN to MoCo. 
        # Thus, we recommand train InsGen with more than 1 GPU always.
        if not torch.distributed.is_initialized():
            return x, torch.arange(x.shape[0])

        # gather from all gpus
        device = x.device
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda(device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        # If non-distributed now, return raw input directly.
        # We have no idea the effect of disabling shuffle BN to MoCo. 
        # Thus, we recommand train InsGen with more than 1 GPU always.
        if not torch.distributed.is_initialized():
            return x

        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def forward(self, im_q, im_k, loss_only=False, update_q=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        device = im_q.device
        im_q = im_q.to(torch.float32)
        im_k = im_k.to(torch.float32)
        # compute query features
        if im_q.ndim > 2:
            im_q = im_q.mean([2,3])
        q = self.mlp(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            if im_k.ndim > 2:
                im_k = im_k.mean([2,3])
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.momentum_mlp(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(device)

        # dequeue and enqueue
        if not loss_only:
            if update_q:
                with torch.no_grad():
                    temp_im_q, idx_unshuffle = self._batch_shuffle_ddp(im_q)
                    temp_q = self.momentum_mlp(temp_im_q)
                    temp_q = nn.functional.normalize(temp_q, dim=1)
                    temp_q = self._batch_unshuffle_ddp(temp_q, idx_unshuffle)
                    self._dequeue_and_enqueue(temp_q)
            else:
                self._dequeue_and_enqueue(k)

        # calculate loss
        loss = nn.functional.cross_entropy(logits, labels)

        return loss 

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

#----------------------------------------------------------------------------
