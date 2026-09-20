"""Extract ``edges.npy`` and ``faces.npy`` (and ``template_vertices.npz``) from an .obj file.

Usage:
    python tools/extract_topology.py \
        --obj ./data/template.obj \
        --edges-out ./data/edges.npy \
        --faces-out ./data/faces.npy \
        --vertices-out ./data/template_vertices.npz
"""
import argparse

import numpy as np


def _load_obj(mesh_path: str):
    vertices = []
    edges = set()
    faces = []
    with open(mesh_path) as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == 'v':
                vertices.append(list(map(float, tokens[1:4])))
            elif tokens[0] == 'f':
                idx = [int(tok.split('/')[0]) - 1 for tok in tokens[1:]]
                # triangulate fan
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
                # bidirectional edges around the face polygon
                for i in range(len(idx)):
                    a, b = idx[i], idx[(i + 1) % len(idx)]
                    edges.add((a, b))
                    edges.add((b, a))
    return (
        np.array(vertices, dtype=np.float32),
        np.array(sorted(edges), dtype=np.int64),
        np.array(faces, dtype=np.int64),
    )


def _compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8

    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--obj', required=True)
    parser.add_argument('--edges-out', required=True)
    parser.add_argument('--faces-out', required=True)
    parser.add_argument('--vertices-out', required=True,
                        help='Output .npz with arrays "vertices" and "normals"')
    args = parser.parse_args()

    vertices, edges, faces = _load_obj(args.obj)
    normals = _compute_normals(vertices, faces)

    np.save(args.edges_out, edges)
    np.save(args.faces_out, faces)
    np.savez(args.vertices_out, vertices=vertices, normals=normals)
    print(f'Wrote {args.edges_out} ({edges.shape}), '
          f'{args.faces_out} ({faces.shape}), '
          f'{args.vertices_out}')


if __name__ == '__main__':
    main()
