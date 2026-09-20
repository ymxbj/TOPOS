"""TOPOS-VAE entry: training, plus pointcloud-driven inference.

Run modes (``SOLVER.run``):
  - ``train``                : train on the filelist defined in ``DATA.train``.
  - ``test_pointcloud``      : reconstruct a single point cloud
                               (``DATA.test.test_pointcloud_path``).
  - ``test_pointcloud_dir``  : reconstruct every ``.npz`` under
                               ``DATA.test.test_pointcloud_dir``.

The inference inputs are ``.npz`` files with two arrays:
  - ``points``  : float32 ``[N, 3]`` surface points
  - ``normals`` : float32 ``[N, 3]`` surface normals
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import thsolver
from dataset_face import Transform, get_dataset
from encoders.hgraph import HGraph
from encoders.losses import FaceLoss
from encoders.models.vae import TOPOSVAE


def count_parameters(model: torch.nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total / 1e6:.2f}M, trainable: {trainable / 1e6:.2f}M')


class TOPOSVAESolver(thsolver.Solver):

    # ---- model / loss / dataset ----
    def get_model(self, flags):
        model = TOPOSVAE(
            out_channels=flags.vae_out_channel,
            code_channels=flags.code_channel,
            decoder_blocks=flags.decoder_blocks,
            decoder_channel=flags.decoder_channel,
            num_latents=flags.num_latents,
            width=flags.width,
            num_encoder_layers=flags.num_encoder_layers,
            point_feats=flags.point_feats,
            num_freqs=flags.num_freqs,
            include_pi=flags.include_pi,
        )
        if flags.compile and self.FLAGS.SOLVER.run == 'train':
            model = torch.compile(model)
        count_parameters(model)
        return model

    def get_loss_function(self, flags):
        self.kl_weight = flags.kl_weight
        self.train_loss = FaceLoss(flags.template_dir, flags.faces_path,
                                   flags.train_batch_size, device=self.device)

    def get_dataset(self, flags):
        return get_dataset(flags)

    # ---- training ----
    def train_step(self, batch):
        pred, kl_loss, code_max, code_min = self.model(
            batch['pointclouds'].cuda(),
            batch['hgraph'].cuda(),
            batch['hgraph'].depth,
        )

        gt = batch['feature'].cuda()
        self.train_loss.load_gt(gt)
        losses = self.train_loss(pred)
        losses['kl'] = kl_loss
        losses['total_loss'] = losses['total_loss'] + kl_loss * self.kl_weight
        losses['code_max'] = code_max
        losses['code_min'] = code_min
        return losses

    # ---- inference helpers ----
    def _build_template_hgraph(self) -> HGraph:
        flag = self.FLAGS.DATA.test
        transform = Transform(flag.depth, flag.edge_file, flag.cluster_file)
        output = transform(np.load(flag.template_vertices_path))

        merged = HGraph(depth=flag.depth, batch_size=1)
        merged.merge_hgraph([output['hgraph']])
        return merged.cuda()

    @staticmethod
    def _load_pointcloud(path: str) -> torch.Tensor:
        data = np.load(path)
        points = torch.from_numpy(data['points'].astype(np.float32))
        normals = torch.from_numpy(data['normals'].astype(np.float32))
        return torch.cat([points, normals], dim=1).unsqueeze(0).cuda()  # [1, N, 6]

    @torch.no_grad()
    def _reconstruct(self, pointclouds: torch.Tensor, hgraph: HGraph) -> torch.Tensor:
        pred, *_ = self.model(pointclouds, hgraph, hgraph.depth)
        return pred  # [V, 6]

    def _save_pred(self, name: str, pred: torch.Tensor) -> None:
        verts = pred.detach()[:, :3].cpu()
        out = os.path.join(self.result_dir, 'pred_mesh', name + '_pred.obj')
        os.makedirs(os.path.dirname(out), exist_ok=True)

        new_lines = []
        idx = 0
        with open(self.FLAGS.MODEL.base_mesh) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v' and idx < len(verts):
                    v = verts[idx]
                    new_lines.append(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
                    idx += 1
                else:
                    new_lines.append(line)
        with open(out, 'w') as f:
            f.writelines(new_lines)

    # ---- inference hooks (called by Solver.test_pointcloud / test_pointcloud_dir) ----
    def test_single_pointcloud(self):
        flag = self.FLAGS.DATA.test
        hgraph = self._build_template_hgraph()

        pc = self._load_pointcloud(flag.test_pointcloud_path)
        pred = self._reconstruct(pc, hgraph)

        name = os.path.splitext(os.path.basename(flag.test_pointcloud_path))[0]
        self._save_pred(name, pred)
        print(f'Saved reconstruction: {name}')

    def test_dir_pointclouds(self):
        flag = self.FLAGS.DATA.test
        hgraph = self._build_template_hgraph()

        files = sorted(f for f in os.listdir(flag.test_pointcloud_dir) if f.lower().endswith('.npz'))
        for name in files:
            pc = self._load_pointcloud(os.path.join(flag.test_pointcloud_dir, name))
            pred = self._reconstruct(pc, hgraph)
            self._save_pred(os.path.splitext(name)[0], pred)
            print(f'Saved reconstruction: {name}')


if __name__ == '__main__':
    TOPOSVAESolver.main()
