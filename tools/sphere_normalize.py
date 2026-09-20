"""Sphere-normalize meshes or point clouds into the unit sphere ``[-1, 1]^3``.

TOPOS-VAE and TOPOS-DiT require ALL inputs to live in this normalized frame:
  - the surface point clouds fed to TOPOS-VAE (`*_surface_npz/*.npz`)
  - the target meshes fed to TOPOS-VAE training (`*_vertices_npz/*.npz`)
  - the template mesh / vertices (`data/template.obj`, `data/template_vertices.npz`)
  - any standalone `.npz` point cloud used with `scripts/infer_vae.sh`

Normalization recipe (matches the one used to train the released checkpoints):

  1. translate so the **axis-aligned bounding-box centroid** is at the origin
  2. divide all coordinates by ``max ||vertex||`` so the farthest point lies
     exactly on the unit sphere

Usage:
    # Normalize a .obj in place (the rewritten file overwrites the input)
    python tools/sphere_normalize.py --obj ./data/your_template.obj

    # Normalize a single point-cloud .npz (in place)
    python tools/sphere_normalize.py --npz ./examples/face.npz

    # Normalize every .obj or .npz under a directory (recursive)
    python tools/sphere_normalize.py --dir ./my_meshes
"""
import argparse
import os
from typing import Iterable

import numpy as np


def _sphere_normalize(points: np.ndarray) -> np.ndarray:
    """Center on AABB centroid, scale so max radius = 1."""
    centroid = 0.5 * (points.max(axis=0) + points.min(axis=0))
    points = points - centroid
    scale = float(np.linalg.norm(points, axis=1).max())
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


# ---- .obj ----

def _normalize_obj(path: str) -> None:
    import trimesh
    mesh = trimesh.load(path, force='mesh', merge_primitives=True)
    mesh.vertices = _sphere_normalize(np.asarray(mesh.vertices, dtype=np.float64))
    mesh.export(path, file_type='obj')
    print(f'Normalized {path} ({len(mesh.vertices)} vertices)')


# ---- .npz ----

def _normalize_npz(path: str) -> None:
    """Normalize an .npz that stores either ``points`` (surface point cloud) or
    ``vertices`` (mesh). Normals are rotation/scale-invariant in this frame
    (centering + uniform scale only), so they're kept untouched.
    """
    data = dict(np.load(path))
    if 'points' in data:
        key = 'points'
    elif 'vertices' in data:
        key = 'vertices'
    else:
        raise KeyError(f'{path}: expected "points" or "vertices" key, got {list(data)}')
    pts = data[key].astype(np.float32)
    data[key] = _sphere_normalize(pts)
    np.savez(path, **data)
    print(f'Normalized {path} ({pts.shape[0]} {key})')


# ---- directory walk ----

def _walk(root: str) -> Iterable[str]:
    for dirpath, _, files in os.walk(root):
        for name in files:
            yield os.path.join(dirpath, name)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--obj', help='Single .obj to normalize in place')
    g.add_argument('--npz', help='Single .npz to normalize in place')
    g.add_argument('--dir', help='Recursively normalize every .obj / .npz under this dir')
    args = p.parse_args()

    if args.obj:
        _normalize_obj(args.obj)
    elif args.npz:
        _normalize_npz(args.npz)
    else:
        for path in _walk(args.dir):
            if path.endswith('.obj'):
                _normalize_obj(path)
            elif path.endswith('.npz'):
                _normalize_npz(path)


if __name__ == '__main__':
    main()
