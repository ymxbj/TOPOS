"""TOPOS-DiT: image-conditioned flow-matching transformer over TOPOS-VAE latents.

Takes the noisy VAE latent ``x_t`` of shape [B, N, C], a scalar timestep, and an
image conditioning sequence (DINOv2 patch tokens), and predicts the flow velocity.

Reference:
    Xiong et al., "TOPOS: High-Fidelity and Efficient Industry-Grade 3D Head
    Generation".
"""
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.utils import convert_module_to_f16, convert_module_to_f32


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class TOPOSDiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_latents: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        pe_mode: Literal['ape', 'rope'] = 'ape',
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_latents = num_latents
        self.share_mod = share_mod
        self.num_heads = num_heads or model_channels // num_head_channels
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        if pe_mode == 'ape':
            pos_embedder = AbsolutePositionEmbedder(model_channels, 1)
            coords = torch.arange(num_latents)[:, None]
            self.register_buffer('pos_emb', pos_embedder(coords))

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                channels=model_channels,
                ctx_channels=cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                use_checkpoint=use_checkpoint,
                use_rope=(pe_mode == 'rope'),
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])
        self.out_layer = nn.Linear(model_channels, out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self):
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero out adaLN gates and output projection so the model starts as an identity.
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x) + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        h = h.type(self.dtype)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        return self.out_layer(h)
