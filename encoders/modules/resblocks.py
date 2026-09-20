import torch
import torch.nn as nn

from ..hgraph import HGraph
from .graph import Conv1x1, GraphConvGn, GraphConvGnSilu


class GraphResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bottleneck: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        bottleneck_channels = out_channels // bottleneck

        self.conv_a = GraphConvGnSilu(in_channels, bottleneck_channels)
        self.conv_b = GraphConvGn(bottleneck_channels, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                Conv1x1(in_channels, out_channels, use_bias=False),
                nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=1e-3),
            )
        self.silu = nn.SiLU(inplace=True)

    def forward(self, data: torch.Tensor, hgraph: HGraph, depth: int):
        h = self.conv_b(self.conv_a(data, hgraph, depth), hgraph, depth)
        if self.in_channels != self.out_channels:
            data = self.skip(data)
        return self.silu(h + data)


class GraphResBlocks(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, resblk_num: int, bottleneck: int = 4):
        super().__init__()
        channels = [in_channels] + [out_channels] * resblk_num
        self.resblks = nn.ModuleList([
            GraphResBlock(channels[i], channels[i + 1], bottleneck=bottleneck)
            for i in range(resblk_num)
        ])

    def forward(self, data: torch.Tensor, hgraph: HGraph, depth: int):
        for blk in self.resblks:
            data = blk(data, hgraph, depth)
        return data
