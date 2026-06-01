#!/usr/bin/env python3
"""ResNet-based conditional diffusion backbone (alternative to UNet2DConditionModel)."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.activation = nn.SiLU()

    def forward(self, x, time_emb=None):
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        if time_emb is not None and self.time_emb_dim is not None:
            out = out + self.time_proj(time_emb)[:, :, None, None]
        out = self.activation(self.norm2(self.conv2(out)))
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        return self.activation(out + residual)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        b, n, _ = x.shape
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q = q.view(b, n, h, -1).transpose(1, 2)
        k = k.view(b, -1, h, k.size(-1) // h).transpose(1, 2)
        v = v.view(b, -1, h, v.size(-1) // h).transpose(1, 2)
        sim = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("bhqk,bhvd->bhqd", attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class ResNetDiffusionModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 256,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (32, 16, 8),
        dropout: float = 0.0,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_heads: int = 8,
        cross_attention_dim: int = 768,
        only_cross_attention: bool = True,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_blocks = nn.ModuleList([nn.Conv2d(in_channels, model_channels, 3, padding=1)])
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResNetBlock(ch, mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        CrossAttention(ch, cross_attention_dim, heads=num_heads, dim_head=ch // num_heads)
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                ds *= 2
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                ds *= 2
        self.middle_block = nn.Sequential(
            ResNetBlock(ch, ch, time_embed_dim),
            CrossAttention(ch, cross_attention_dim, heads=num_heads, dim_head=ch // num_heads),
            ResNetBlock(ch, ch, time_embed_dim),
        )
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResNetBlock(ch + mult * model_channels, mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        CrossAttention(ch, cross_attention_dim, heads=num_heads, dim_head=ch // num_heads)
                    )
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
        self.out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, encoder_hidden_states=None, return_dict=True):
        t_emb = self.time_embed(timesteps)
        if encoder_hidden_states is not None:
            if encoder_hidden_states.ndim == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            elif encoder_hidden_states.ndim == 4:
                encoder_hidden_states = encoder_hidden_states.view(
                    encoder_hidden_states.size(0), -1, encoder_hidden_states.size(-1)
                )
        h = x
        hs = []
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResNetBlock):
                        h = layer(h, t_emb)
                    elif isinstance(layer, CrossAttention) and encoder_hidden_states is not None:
                        b, c, h_dim, w_dim = h.shape
                        h_seq = h.view(b, c, -1).transpose(1, 2)
                        h_seq = layer(h_seq, encoder_hidden_states)
                        h = h_seq.transpose(1, 2).view(b, c, h_dim, w_dim)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        for layer in self.middle_block:
            if isinstance(layer, ResNetBlock):
                h = layer(h, t_emb)
            elif isinstance(layer, CrossAttention) and encoder_hidden_states is not None:
                b, c, h_dim, w_dim = h.shape
                h_seq = h.view(b, c, -1).transpose(1, 2)
                h_seq = layer(h_seq, encoder_hidden_states)
                h = h_seq.transpose(1, 2).view(b, c, h_dim, w_dim)
            else:
                h = layer(h)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResNetBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, CrossAttention) and encoder_hidden_states is not None:
                    b, c, h_dim, w_dim = h.shape
                    h_seq = h.view(b, c, -1).transpose(1, 2)
                    h_seq = layer(h_seq, encoder_hidden_states)
                    h = h_seq.transpose(1, 2).view(b, c, h_dim, w_dim)
                else:
                    h = layer(h)
        h = self.out(h)
        return {"sample": h} if return_dict else (h,)


def create_resnet_diffusion_model(**kwargs):
    return ResNetDiffusionModel(**kwargs)
