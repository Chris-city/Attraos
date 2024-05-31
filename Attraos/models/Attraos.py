import math
from dataclasses import dataclass
from typing import Union
import numpy as np
from layers import unroll

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from utils.pscan import pscan
from einops import rearrange

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_len = config.patch_len
        self.stride = config.patch_len

        self.PSR_dim = config.PSR_dim
        self.PSR_type = config.PSR_type
        self.PSR_delay = config.PSR_delay
        self.PSR_enc_len = (config.seq_len - (config.PSR_dim - 1) * config.PSR_delay) # 60
        self.seq_len = config.seq_len

        # patch_num = self.PSR_enc_len // config.patch_len
        patch_num = self.seq_len // config.patch_len
        config.d_inner = config.PSR_dim * config.patch_len
        self.pad_len = (config.PSR_dim - 1) * config.PSR_delay

        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.e_layers)]
        )

        # self.K = sparseKernelFT1d(k=self.order, alpha=self.modes, c=self.PSR_dim*config.patch_len)
        # self.out_type = config.out_type
        self.out_layer = nn.Linear(config.d_inner * patch_num, config.pred_len)

    def forward(self, x, x_mark, y, y_mark):
        # x : (B, L, C)

        # y : (B, L, C)
        B, T, C = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        enc_PSR = PSR(x, self.PSR_dim, self.PSR_delay, self.PSR_type)  # [BC, T, D] [32*7, 96, 1]

        pad_tuple = (0, 0, self.pad_len, 0, 0, 0)
        enc_PSR = F.pad(enc_PSR, pad_tuple, "constant", 0)

        enc_PSR = enc_PSR.unfold(dimension=-2, size=self.patch_len, step=self.patch_len) # [BC, L, D, len]
        enc_PSR = enc_PSR.reshape(B*C, -1, self.PSR_dim*self.patch_len) # [BC, L, D*len]

        for layer in self.layers:
            enc_PSR = layer(enc_PSR)  # [BC, L, D*len]

        enc_PSR = self.out_layer(enc_PSR.reshape(B*C, -1))
        y = rearrange(enc_PSR, '(b m) l -> b l m', b=B)

        y = y * stdev
        y = y + means
        return y


class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.get_memory = MDMU(config)
        self.norm = nn.LayerNorm(config.d_inner)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)
        output = self.get_memory(self.norm(x)) + x
        return output


class MDMU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            
        A = torch.ones(config.d_inner, config.d_state, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.K = sparseKernelFT1d(k=config.d_state, alpha=config.modes)
        # self.H_proj = nn.Linear(
        #     config.d_inner, config.d_state, bias=False)

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)
        y = self.ssm(x) # TODO
        return y

    def ssm(self, x):
        D = self.D.float()
        A = -torch.exp(self.A_log.float())  # (ED, N)
        deltaBC = self.x_proj(x)  # (B, L, dt_rank+N)
        # H = self.H_proj(x) # (B, L, N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, ED)
        y = self.Piecewise_scan(x, delta, A, B, C, D)
        return y

    def Piecewise_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        # deltaH = delta.unsqueeze(-1) * H.unsqueeze(2)  # (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)
        hs = pscan(deltaA, BX)

        hs = self.K(hs)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = torch.real(y) + D * x
        return y


def compl_mul1d(x, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", x, weights)


class sparseKernelFT1d(nn.Module):
    def __init__(self, k, modes):
        super(sparseKernelFT1d, self).__init__()

        self.modes1 = modes
        self.scale = 1 / (k * k)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(k, k, self.modes1, dtype=torch.cfloat)
        )
        self.weights1.requires_grad = True

    def forward(self, x):
        B, D, N, k = x.shape  # (B, D, N, k)
        x = x.permute(0, 1, 3, 2) # (B, D, k, N)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, D, k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :, :l] = torch.einsum("bdix,iox->bdox", x_fft[:, :, :, :l], self.weights1[:, :, :, :l])
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=N) # (B, D, k, N)
        x = x.permute(0, 1, 3, 2)
        return x


def PSR(input_data, embedding_dim, delay, mode="indep"):
    batch_size, seq_length, input_channels = input_data.shape

    device = input_data.device
    len_embedded = seq_length - (embedding_dim - 1) * delay
    embedded_data = torch.zeros(
        batch_size, len_embedded, embedding_dim, input_channels, device=device
    )

    for i in range(embedding_dim):
        start_idx = i * delay
        end_idx = start_idx + len_embedded
        embedded_data[:, :, i, :] = input_data[:, start_idx:end_idx, :]

    if mode == "merged_seq":
        embedded_data = embedded_data.permute(0, 1, 3, 2).reshape(
            batch_size, len_embedded, -1
        )
    elif mode == "merged":
        embedded_data = embedded_data.reshape(batch_size, len_embedded, -1)
    else:  # independent
        embedded_data = embedded_data.permute(0, 3, 1, 2).reshape(
            batch_size * input_channels, len_embedded, embedding_dim
        )  # [BC, T, D]
    return embedded_data
