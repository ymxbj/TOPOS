"""TOPOS-VAE: Perceiver-Resampler encoder + hierarchical GNN decoder.

The encoder maps an unstructured surface point cloud to a latent vec-set.
The decoder maps the latent back to vertex coordinates and normals on a
fixed-topology MetaHuman head mesh, via a hierarchical-graph upsampling
pipeline.

Reference:
    Xiong et al., "TOPOS: High-Fidelity and Efficient Industry-Grade 3D Head
    Generation".
"""
from typing import List

import torch
import torch.nn as nn

from ..hgraph import HGraph
from ..modules.graph import (
    Conv1x1,
    Conv1x1GnSilu,
    FourierEmbedder,
    UnpoolingGraph,
)
from ..modules.perceiver import PerceiverResampler
from ..modules.resblocks import GraphResBlocks


class DiagonalGaussianDistribution(nn.Module):
    def __init__(self, params: torch.Tensor):
        super().__init__()
        self.mean, self.logvar = torch.chunk(params, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.device = params.device

    def sample(self):
        return self.mean + self.std * torch.randn(self.mean.shape, device=self.device)

    def kl(self):
        return 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)


class TOPOSVAE(nn.Module):
    """TOPOS-VAE: Perceiver-Resampler encoder + hierarchical GNN decoder."""

    def __init__(
        self,
        out_channels: int,
        code_channels: int,
        decoder_blocks: List[int],
        decoder_channel: List[int],
        num_latents: int,
        width: int,
        num_encoder_layers: int,
        point_feats: int,
        num_freqs: int,
        include_pi: bool,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.code_channels = code_channels
        self.decoder_stages = len(decoder_blocks)

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.data_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width)

        self.encoder = PerceiverResampler(
            dim=width,
            depth=num_encoder_layers,
            num_learns=num_latents,
        )

        self.pre_kl = Conv1x1(width, code_channels * 2, use_bias=True)
        self.post_kl = Conv1x1(code_channels, decoder_channel[0], use_bias=True)

        self.upsample = nn.ModuleList([UnpoolingGraph() for _ in range(self.decoder_stages)])
        self.decoder = nn.ModuleList([
            GraphResBlocks(decoder_channel[i], decoder_channel[i + 1],
                           resblk_num=decoder_blocks[i], bottleneck=1)
            for i in range(self.decoder_stages)
        ])

        self.header = nn.Sequential(
            Conv1x1GnSilu(decoder_channel[-1], decoder_channel[-1]),
            Conv1x1(decoder_channel[-1], out_channels, use_bias=True),
        )

    # ---- encoder ----
    def encoder_forward(self, pointclouds: torch.Tensor) -> torch.Tensor:
        pc, feats = pointclouds[:, :, :3], pointclouds[:, :, 3:]
        x = self.fourier_embedder(pc)
        if feats.shape[-1] > 0:
            x = torch.cat([x, feats], dim=-1)
        x = self.data_proj(x)
        return self.encoder(x)

    def decoder_forward(self, code: torch.Tensor, hgraph: HGraph, depth: int) -> torch.Tensor:
        h = code
        for i in range(self.decoder_stages):
            h = self.upsample[i](h, hgraph, self.decoder_stages - i)
            h = self.decoder[i](h, hgraph, depth + i + 1)
        return h

    # ---- public API ----
    def forward(self, pointclouds: torch.Tensor, hgraph: HGraph, depth: int):
        latent = self.encoder_forward(pointclouds)
        mean_var = self.pre_kl(latent)
        posterior = DiagonalGaussianDistribution(mean_var)
        z = posterior.sample()

        flat = z.view(-1, z.shape[-1])
        h = self.post_kl(flat)
        h = self.decoder_forward(h, hgraph, depth - self.decoder_stages)
        out = self.header(h)

        kl_loss = posterior.kl().mean()
        return out, kl_loss, z.max(), z.min()

    def extract_code(self, pointclouds: torch.Tensor) -> torch.Tensor:
        latent = self.encoder_forward(pointclouds)
        mean_var = self.pre_kl(latent)
        z = DiagonalGaussianDistribution(mean_var).sample()
        return z.view(-1, z.shape[-1])

    def decode_code(self, latent_code: torch.Tensor, hgraph: HGraph, depth: int) -> torch.Tensor:
        h = self.post_kl(latent_code)
        h = self.decoder_forward(h, hgraph, depth - self.decoder_stages)
        return self.header(h)
