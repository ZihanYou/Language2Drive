"""
VQ quantization modules.
Adapted from Emu3.5 (BAAI) vqvae/quantize.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, reduce


def compute_entropy_loss(logits, temperature=0.01,
                         sample_minimization_weight=1.0,
                         batch_maximization_weight=1.0, eps=1e-5):
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)
    avg_probs = reduce(probs, "... D -> D", "mean")
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))
    sample_entropy = -torch.sum(probs * log_probs, -1)
    sample_entropy = torch.mean(sample_entropy)
    loss = sample_minimization_weight * sample_entropy - batch_maximization_weight * avg_entropy
    return sample_entropy, avg_entropy, loss


class IndexPropagationQuantize(nn.Module):

    def __init__(self, n_e, e_dim, beta=0.25, use_entropy_loss=False,
                 remap=None, unknown_index="random", cosine_similarity=False,
                 entropy_temperature=0.01, sample_minimization_weight=1.0,
                 batch_maximization_weight=1.0, l2_normalize=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.embed_dim = e_dim
        self.use_entropy_loss = use_entropy_loss
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
        else:
            self.re_embed = n_e
        self.cosine_similarity = cosine_similarity
        self.entropy_temperature = entropy_temperature
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight
        self.l2_normalize = l2_normalize
        if self.l2_normalize:
            self.init_embedding()

    def init_embedding(self):
        embedding = torch.randn(self.n_e, self.e_dim)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.embedding.weight.data = embedding

    def forward(self, z, temp=None, return_logits=False):
        if self.l2_normalize:
            z = F.normalize(z, dim=1)
            embedding = F.normalize(self.embedding.weight, dim=1)
        else:
            embedding = self.embedding.weight
        logits = einsum('b d h w, n d -> b n h w', z, embedding)
        if self.remap is not None:
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, ...]
        soft_one_hot = F.softmax(logits, dim=1)
        if self.remap is not None:
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        dim = 1
        ind = soft_one_hot.max(dim, keepdim=True)[1]
        hard_one_hot = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, ind, 1.0)
        if self.training:
            one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot
            z_q = einsum('b n h w, n d -> b d h w', one_hot, embedding)
            z_q_2 = einsum('b n h w, n d -> b d h w', hard_one_hot, embedding)
            quant_loss = (torch.mean((z_q - z) ** 2) +
                          torch.mean((z_q_2.detach() - z) ** 2) +
                          torch.mean((z_q_2 - z.detach()) ** 2) * self.beta)
            diff = quant_loss
        else:
            diff = 0.0
            z_q = einsum('b n h w, n d -> b d h w', hard_one_hot, embedding)
        if self.use_entropy_loss:
            sample_entropy, avg_entropy, entropy_loss = compute_entropy_loss(
                logits=logits.permute(0, 2, 3, 1).reshape(-1, self.n_e),
                temperature=self.entropy_temperature,
                sample_minimization_weight=self.sample_minimization_weight,
                batch_maximization_weight=self.batch_maximization_weight,
            )
            diff = (quant_loss, sample_entropy, avg_entropy, entropy_loss)
        ind = torch.flatten(ind)
        if self.remap is not None:
            ind = ind.reshape(z.shape[0], -1)
            ind = ind.reshape(-1, 1)
        return z_q, diff, (None, None, ind)

    def get_codebook_entry(self, indices, shape=None):
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = indices.reshape(-1)
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
