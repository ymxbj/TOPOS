"""Hierarchical graph data structure used by the GNN decoder.

Each ``HGraph`` instance keeps a coarse-to-fine pyramid of graphs built from a
single base mesh. Coordinates must lie in the unit cube ``[-1, 1]^3``.
"""
from typing import Callable, List, Optional

import numpy as np
import torch
import torch_geometric

from .modules.edge_type import decide_edge_type


class Data:
    """Lightweight stand-in for ``torch_geometric.data.Data`` without auto re-indexing."""

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

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')


@torch._dynamo.disable
def avg_pool(cluster: torch.Tensor, data: Data, transform: Optional[Callable] = None) -> Data:
    pg_data = torch_geometric.data.Data(x=data.x, edge_index=data.edge_index)
    new_data = torch_geometric.nn.pool.avg_pool(cluster, pg_data, transform=transform)
    return Data(x=new_data.x, edge_index=new_data.edge_index)


def add_self_loop(data: Data) -> Data:
    device = data.x.device
    n = data.x.shape[0]
    loops = torch.arange(n)[None].repeat(2, 1)
    new_edges = torch.zeros([2, data.edge_index.shape[1] + n], dtype=torch.int64)
    new_edges[:, :data.edge_index.shape[1]] = data.edge_index
    new_edges[:, data.edge_index.shape[1]:] = loops
    return Data(x=data.x, edge_index=new_edges).to(device)


class HGraph:
    """Hierarchical graph storing a tree of pooled graphs of ``depth + 1`` levels."""

    def __init__(self, depth: int = 8, batch_size: int = 1, cluster_file: str = ''):
        self.device = 'cuda'
        self.depth = depth
        self.batch_size = batch_size
        self.cluster_file = cluster_file

        self.vertices_sizes = {}
        self.edges_sizes = {}
        self.treedict = {}
        self.cluster = {}

    def build_single_hgraph(self, original_graph: Data):
        """Build a tree from one graph using the precomputed cluster file."""
        precomputed_cluster = np.load(self.cluster_file)

        graphtree, cluster = {}, {}
        vertices_size, edges_size = {}, {}

        for i in range(self.depth + 1):
            if i == 0:
                g = add_self_loop(original_graph)
                if g.edge_attr is None:
                    edges = g.x[g.edge_index[0]] - g.x[g.edge_index[1]]
                    g.edge_attr = decide_edge_type(edges)
                graphtree[0] = g
                cluster[0] = None
            else:
                clst = torch.tensor(precomputed_cluster[f'depth{i}'])
                g = avg_pool(cluster=clst, data=graphtree[i - 1])
                g = add_self_loop(g)
                edges = g.x[g.edge_index[0]] - g.x[g.edge_index[1]]
                g.edge_attr = decide_edge_type(edges)
                graphtree[i] = g
                cluster[i] = clst

            edges_size[i] = graphtree[i].edge_index.shape[1]
            vertices_size[i] = graphtree[i].x.shape[0]

        self.treedict = graphtree
        self.cluster = cluster
        self.vertices_sizes = vertices_size
        self.edges_sizes = edges_size

    def merge_hgraph(self, graphs: List['HGraph']):
        """Merge a batch of single-sample HGraphs into one batched HGraph."""
        assert not self.cluster and not self.treedict, 'merge_hgraph must be called on a fresh instance'
        assert len(graphs) == self.batch_size

        # Re-index edges and clusters per depth level so different samples don't overlap.
        for d in range(self.depth + 1):
            num_vertices = [0] + [g.vertices_sizes[d] for g in graphs]
            cum_sum = torch.cumsum(torch.tensor(num_vertices), dim=0)
            for i, g in enumerate(graphs):
                g.treedict[d].edge_index += cum_sum[i]
                if d != 0:
                    g.cluster[d] += cum_sum[i]

        for d in range(self.depth + 1):
            xs = [g.treedict[d].x for g in graphs]
            es = [g.treedict[d].edge_index for g in graphs]
            ats = [g.treedict[d].edge_attr for g in graphs]
            clusters = [g.cluster[d] for g in graphs]

            self.treedict[d] = Data(
                x=torch.cat(xs, dim=0),
                edge_index=torch.cat(es, dim=1),
                edge_attr=torch.cat(ats, dim=0),
            )
            self.cluster[d] = torch.cat(clusters, dim=0) if d != 0 else None
            self.edges_sizes = self.treedict[d].edge_index.shape[1]
            self.vertices_sizes = len(self.treedict[d].x)

    def cuda(self):
        for k in self.treedict:
            self.treedict[k] = self.treedict[k].cuda()
        for k in self.cluster:
            if self.cluster[k] is not None:
                self.cluster[k] = self.cluster[k].cuda()
        return self
