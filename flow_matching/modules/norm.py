import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):
    """LayerNorm that always runs in float32."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
