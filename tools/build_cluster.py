"""Build the multi-level cluster file (``data/cluster.npz``) for a new topology.

The cluster file is consumed by ``encoders.hgraph.HGraph`` to construct the
coarse-to-fine pyramid during VAE encoding / decoding. Run this once whenever
you change the base mesh.

Usage:
    python tools/build_cluster.py \
        --vertices ./data/template_vertices.npz \
        --edges ./data/edges.npy \
        --output ./data/cluster.npz \
        --depth 8
"""
import argparse
from typing import Callable, Optional

import numpy as np
import torch
import torch_geometric

from encoders.modules.edge_type import decide_edge_type


class _Data:
    def __init__(self, x=None, edge_index=None, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def to(self, target):
        self.x = self.x.to(target)
        self.edge_index = self.edge_index.to(target)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(target)
        return self


def _add_self_loop(data: _Data) -> _Data:
    n = data.x.shape[0]
    loops = torch.arange(n)[None].repeat(2, 1)
    new_edges = torch.zeros([2, data.edge_index.shape[1] + n], dtype=torch.int64)
    new_edges[:, :data.edge_index.shape[1]] = data.edge_index
    new_edges[:, data.edge_index.shape[1]:] = loops
    return _Data(x=data.x, edge_index=new_edges).to(data.x.device)


def _pooling(data: _Data, size: float) -> torch.Tensor:
    cluster = torch_geometric.nn.voxel_grid(
        pos=data.x[..., :3], size=size, batch=None,
        start=[-1, -1, -1.0], end=[1, 1, 1.0],
    )
    mapping = cluster.unique()
    mapping += mapping.shape[0]
    cluster += mapping.shape[0]
    for i in range(int(mapping.shape[0])):
        cluster[cluster == mapping[i]] = i
    return cluster


def _avg_pool(cluster: torch.Tensor, data: _Data, transform: Optional[Callable] = None) -> _Data:
    pg_data = torch_geometric.data.Data(x=data.x, edge_index=data.edge_index)
    new_data = torch_geometric.nn.pool.avg_pool(cluster, pg_data, transform=transform)
    return _Data(x=new_data.x, edge_index=new_data.edge_index)


def build(vertices_path: str, edges_path: str, output: str, depth: int = 8) -> None:
    base = np.load(vertices_path)
    vertices = torch.from_numpy(base['vertices'].astype(np.float32))
    normals = torch.from_numpy(base['normals'].astype(np.float32))
    edges = torch.from_numpy(np.load(edges_path)).t().contiguous().long()
    original = _Data(x=torch.cat([vertices, normals], dim=1), edge_index=edges)

    smallest = 2.0 / (2 ** depth)

    clusters = {}
    graphtree = {}
    for i in range(depth + 1):
        if i == 0:
            g = _add_self_loop(original)
            if g.edge_attr is None:
                ev = g.x[g.edge_index[0]] - g.x[g.edge_index[1]]
                g.edge_attr = decide_edge_type(ev)
            graphtree[0] = g
            continue

        clst = _pooling(graphtree[i - 1], smallest * (2 ** (i - 1)))
        g = _avg_pool(cluster=clst, data=graphtree[i - 1])
        g = _add_self_loop(g)
        ev = g.x[g.edge_index[0]] - g.x[g.edge_index[1]]
        g.edge_attr = decide_edge_type(ev)
        graphtree[i] = g
        clusters[i] = clst

    np.savez(output, **{f'depth{i}': clusters[i].numpy() for i in range(1, depth + 1)})
    print(f'Wrote {output} (depth={depth})')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--vertices', required=True, help='Path to template_vertices.npz')
    parser.add_argument('--edges', required=True, help='Path to edges.npy ([E, 2])')
    parser.add_argument('--output', required=True, help='Output cluster.npz')
    parser.add_argument('--depth', type=int, default=8)
    args = parser.parse_args()
    build(args.vertices, args.edges, args.output, depth=args.depth)


if __name__ == '__main__':
    main()
