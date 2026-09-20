from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ..attention import MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet


class ModulatedTransformerCrossBlock(nn.Module):
    """Self-attn + cross-attn + FFN, all gated by adaptive layer-norm modulation."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.cross_norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)

        self.self_attn = MultiHeadAttention(
            channels, num_heads=num_heads, type='self',
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels, num_heads=num_heads, ctx_channels=ctx_channels, type='cross',
            qkv_bias=qkv_bias, qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        mod_chunks = (mod if self.share_mod else self.adaLN_modulation(mod)).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_chunks

        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + self.self_attn(h) * gate_msa.unsqueeze(1)

        x = x + self.cross_attn(self.cross_norm(x), context)

        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + self.mlp(h) * gate_mlp.unsqueeze(1)
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        return self._forward(x, mod, context)
