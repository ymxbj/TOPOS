"""Geometric losses used to train the VAE.

Combines MSE/L1 on vertex coordinates with three mesh-aware terms:
- face normal alignment
- face-angle cosine consistency
- per-edge Gaussian curvature consistency (sign-aware dihedral angle)
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussCurvature:
    """Per-edge Gaussian curvature: sign(direction-projected face-normal cross) * dihedral / pi."""

    def __init__(self, faces_path: str, data_dir: str, device: str):
        self.faces = np.load(faces_path)
        self.a2bIdx = torch.load(os.path.join(data_dir, 'a2bIdx.pt')).long().to(device)
        self.sidebyside = torch.load(os.path.join(data_dir, 'sidebyside.pt')).long().to(device)

    def get_edges_gauss(self, verts: torch.Tensor, face_normals: torch.Tensor) -> torch.Tensor:
        a2b = verts[:, self.a2bIdx[:, 1]] - verts[:, self.a2bIdx[:, 0]]
        a2b = F.normalize(a2b, p=2, dim=-1, eps=1e-8)

        adj = face_normals[:, self.sidebyside]
        n1, n2 = adj[:, :, 0], adj[:, :, 1]

        cross_n = torch.cross(n1, n2, dim=-1)
        dot_n = torch.einsum('bni,bni->bn', n1, n2)
        angle = torch.atan2(torch.norm(cross_n, dim=-1), dot_n)

        signs_raw = torch.sum(cross_n * a2b, dim=-1)
        signs = torch.sign(signs_raw)
        signs[signs == 0] = 1.0

        return signs.detach() * angle / torch.pi


class FaceLoss(nn.Module):
    def __init__(self, template_dir: str, faces_path: str, batch_size: int,
                 lambdas_path: str = 'lambdas.json', device: str = 'cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.gaussian = GaussCurvature(faces_path, template_dir, device=device)
        self.faces = torch.from_numpy(np.load(faces_path)).long().to(device)

        with open(lambdas_path) as f:
            self.lambdas = json.load(f)

    @staticmethod
    def _face_normal(n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
        normal = torch.cross(n1, n2)
        return F.normalize(normal.float(), p=2, dim=-1, eps=1e-8).to(n1.dtype)

    def load_gt(self, gt: torch.Tensor):
        """Cache ground-truth-derived quantities for the current batch."""
        self.gt = gt
        verts = gt[:, :3].view(self.batch_size, -1, 3)
        self.verts = verts

        Vf = verts[:, self.faces, :]
        self.n1 = Vf[:, :, 1, :] - Vf[:, :, 0, :]
        self.n2 = Vf[:, :, 2, :] - Vf[:, :, 0, :]
        self.n3 = Vf[:, :, 2, :] - Vf[:, :, 1, :]

        self.costheta1 = F.cosine_similarity(self.n1, self.n2, dim=-1)
        self.costheta2 = F.cosine_similarity(self.n1, self.n3, dim=-1)

        self.face_normal = self._face_normal(self.n1, self.n2)
        self.edges_gauss = self.gaussian.get_edges_gauss(self.verts, self.face_normal)

    def forward(self, pred: torch.Tensor):
        losses = {
            'mse': F.mse_loss(pred, self.gt),
            'l1': F.l1_loss(pred, self.gt),
        }

        verts = pred[:, :3].view(self.batch_size, -1, 3)
        Vf = verts[:, self.faces, :]
        n1 = Vf[:, :, 1, :] - Vf[:, :, 0, :]
        n2 = Vf[:, :, 2, :] - Vf[:, :, 0, :]
        n3 = Vf[:, :, 2, :] - Vf[:, :, 1, :]

        face_normal = self._face_normal(n1, n2)
        losses['normal'] = (1 - F.cosine_similarity(face_normal, self.face_normal, dim=-1)).mean()

        costheta1 = F.cosine_similarity(n1, n2, dim=-1)
        costheta2 = F.cosine_similarity(n1, n3, dim=-1)
        losses['theta'] = F.l1_loss(costheta1, self.costheta1) + F.l1_loss(costheta2, self.costheta2)

        edges_gauss = self.gaussian.get_edges_gauss(verts, face_normal)
        losses['gauss_cur'] = F.l1_loss(edges_gauss, self.edges_gauss)

        losses['total_loss'] = (
            self.lambdas['mse'] * losses['mse']
            + self.lambdas['l1'] * losses['l1']
            + self.lambdas['normal'] * losses['normal']
            + self.lambdas['theta'] * losses['theta']
            + self.lambdas['gauss_cur'] * losses['gauss_cur']
        )
        return losses
