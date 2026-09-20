"""Graph-convolution building blocks used by the GNN decoder."""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, HeteroLinear

from ..hgraph import Data, HGraph, avg_pool

bn_eps = 1e-3


class MyConvOp(MessagePassing):
    """Edge-typed message passing: type-0 = neighbor, type-1 = self-loop."""

    NUM_TYPES = 2

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='sum')
        self.lin = HeteroLinear(in_channels, out_channels, num_types=self.NUM_TYPES)

    def forward(self, x, edge_index, edge_attr):
        assert edge_attr is not None and edge_attr.min() >= 0 and edge_attr.max() <= self.NUM_TYPES - 1
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.lin(x_j, edge_attr)


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = MyConvOp(in_channels, out_channels)

    def forward(self, feature: torch.Tensor, hgraph: HGraph, depth: int):
        graph = hgraph.treedict[hgraph.depth - depth]
        return self.conv(feature, graph.edge_index, graph.edge_attr)


def _norm(channels):
    return nn.GroupNorm(num_groups=4, num_channels=channels, eps=bn_eps)


class GraphConvGn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.gn = _norm(out_channels)

    def forward(self, data, hgraph, depth):
        return self.gn(self.conv(data, hgraph, depth))


class GraphConvGnSilu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.gn = _norm(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, data, hgraph, depth):
        return self.silu(self.gn(self.conv(data, hgraph, depth)))


class UnpoolingGraph(nn.Module):
    """Broadcast features from a coarser level to the next finer level."""

    def forward(self, x: torch.Tensor, hgraph: HGraph, depth: int):
        assert depth != 0
        cluster = hgraph.cluster[depth][..., None].repeat(1, x.shape[1]).long()
        return torch.gather(input=x, dim=0, index=cluster)


class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, use_bias)

    def forward(self, data):
        return self.linear(data)


class Conv1x1GnSilu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
        self.gn = _norm(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, data):
        return self.silu(self.gn(self.conv(data)))


class FourierEmbedder(nn.Module):
    """Sin/cos positional embedding with log-spaced frequencies."""

    def __init__(self, num_freqs: int = 8, input_dim: int = 3,
                 include_input: bool = True, include_pi: bool = False):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        if include_pi:
            freqs *= torch.pi
        self.register_buffer('frequencies', freqs, persistent=False)
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.out_dim = input_dim * (num_freqs * 2 + (1 if include_input or num_freqs == 0 else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs == 0:
            return x
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)
