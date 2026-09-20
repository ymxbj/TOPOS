"""Per-sample mesh transform and collate function used by the VAE / DiT training loops."""
import numpy as np
import torch

from encoders.hgraph import Data, HGraph
from thsolver import Dataset


class Transform:
    """Build the hierarchical graph for one mesh sample."""

    def __init__(self, depth: int, edge_file: str, cluster_file: str):
        self.depth = depth
        self.edges = torch.from_numpy(np.load(edge_file)).t().contiguous().long()
        self.cluster_file = cluster_file

    def __call__(self, sample: dict) -> dict:
        vertices = torch.from_numpy(sample['vertices'].astype(np.float32))
        normals = torch.from_numpy(sample['normals'].astype(np.float32))

        hgraph = HGraph(depth=self.depth, batch_size=1, cluster_file=self.cluster_file)
        hgraph.build_single_hgraph(Data(x=torch.cat([vertices, normals], dim=1), edge_index=self.edges))
        return {'hgraph': hgraph, 'vertices': vertices, 'normals': normals}


def collate_batch(batch: list) -> dict:
    """Merge per-sample dicts into a single batched dict (with a merged HGraph)."""
    outputs = {key: [b[key] for b in batch] for key in batch[0].keys()}

    vertices = torch.cat(outputs.pop('vertices'), dim=0)
    normals = torch.cat(outputs.pop('normals'), dim=0)
    outputs['feature'] = torch.cat([vertices, normals], dim=1)

    if 'pointclouds' in outputs:
        outputs['pointclouds'] = torch.stack(outputs['pointclouds'], dim=0)

    if 'image' in outputs:
        outputs['image'] = torch.stack(outputs['image'], dim=0)

    depth = outputs['hgraph'][0].depth
    merged = HGraph(depth=depth, batch_size=len(batch))
    merged.merge_hgraph(outputs['hgraph'])
    outputs['hgraph'] = merged
    return outputs


def get_dataset(flags):
    transform = Transform(flags.depth, flags.edge_file, flags.cluster_file)
    dataset = Dataset(
        root_dir=flags.root_dir,
        use_surface=flags.use_surface,
        use_images=flags.use_images,
        filelist=flags.filelist,
        transform=transform,
        max_rotation=flags.max_rotation,
        max_translate=flags.max_translate,
        scale_range=flags.scale_range,
        image_nums=flags.image_nums,
    )
    return dataset, collate_batch
