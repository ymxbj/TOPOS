"""Train TOPOS-DiT: image-conditioned flow-matching transformer on frozen TOPOS-VAE latents."""
import os
import sys
import tarfile
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import thsolver
from dataset_face import Transform, get_dataset
from encoders.hgraph import HGraph
from encoders.models.vae import TOPOSVAE
from flow_matching.models.flow import TOPOSDiT
from flow_matching.samplers import FlowEulerCfgSampler


def count_parameters(model: torch.nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total / 1e6:.2f}M, trainable: {trainable / 1e6:.2f}M')


class TOPOSDiTSolver(thsolver.Solver):

    SIGMA_MIN = 1e-5

    def get_model(self, flags):
        self.code_channel = flags.code_channel
        self.num_latents = flags.num_latents
        self.p_uncond = flags.p_uncond

        self.vae = TOPOSVAE(
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
        self._load_vae(flags.vae_ckpt)
        self._init_image_encoder(flags)

        model = TOPOSDiT(
            in_channels=flags.df_in_channels,
            model_channels=flags.model_channels,
            num_latents=flags.num_latents,
            cond_channels=flags.cond_channels,
            out_channels=flags.df_out_channels,
            num_blocks=flags.num_blocks,
            num_heads=flags.num_heads,
            num_head_channels=flags.num_head_channels,
            mlp_ratio=flags.mlp_ratio,
            pe_mode=flags.pe_mode,
            use_fp16=flags.use_fp16,
            use_checkpoint=flags.use_checkpoint,
            qk_rms_norm=flags.qk_rms_norm,
            qk_rms_norm_cross=flags.qk_rms_norm_cross,
        )
        if flags.compile and self.FLAGS.SOLVER.run == 'train':
            model = torch.compile(model)
        count_parameters(model)
        return model

    # ---- helpers ----
    def _load_vae(self, vae_ckpt: str) -> None:
        state = torch.load(vae_ckpt, map_location='cuda')
        prefix = '_orig_mod.'
        if list(state.keys())[0].startswith(prefix):
            state = {k[len(prefix):]: v for k, v in state.items()}
        self.vae.load_state_dict(state)
        self.vae.requires_grad_(False)
        self.vae.to(self.device).eval()

    def _init_image_encoder(self, flags) -> None:
        dinov2 = torch.hub.load(
            repo_or_dir=flags.image_cond_model_dir,
            model=flags.image_cond_model_name,
            pretrained=False,
            source='local',
        )
        dinov2.load_state_dict(torch.load(flags.image_cond_model_path))
        dinov2.eval().to(self.device)
        dinov2.requires_grad_(False)
        self.image_cond_model = {
            'model': dinov2,
            'transform': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        }

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self.image_cond_model['transform'](image).to(self.device)
        with torch.no_grad():
            feats = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        return F.layer_norm(feats, feats.shape[-1:])

    def get_train_cond(self, image: torch.Tensor) -> torch.Tensor:
        """Encoded image tokens, with per-sample CFG dropout."""
        cond = self.encode_image(image)
        if self.p_uncond > 0:
            mask = list(np.random.rand(cond.shape[0]) < self.p_uncond)
            mask = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
            cond = torch.where(mask, torch.zeros_like(cond), cond)
        return cond

    def get_inference_cond(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = self.encode_image(image)
        return cond, torch.zeros_like(cond)

    # ---- flow matching ----
    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        t = t.view(-1, *[1] * (x_0.ndim - 1))
        return (1 - t) * x_0 + (self.SIGMA_MIN + (1 - self.SIGMA_MIN) * t) * noise

    def velocity(self, x_0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (1 - self.SIGMA_MIN) * noise - x_0

    @staticmethod
    def sample_t(batch_size: int) -> torch.Tensor:
        return torch.sigmoid(torch.randn(batch_size))

    def training_losses(self, x_0: torch.Tensor, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_train_cond(image)
        pred = self.model(x_t, t * 1000, cond)
        target = self.velocity(x_0, noise)
        return {'total_loss': F.mse_loss(pred, target).mean()}

    # ---- solver hooks ----
    def get_loss_function(self, flags):
        pass

    def get_dataset(self, flags):
        return get_dataset(flags)

    def train_step(self, batch):
        pointclouds = batch['pointclouds'].cuda()
        hgraph = batch['hgraph'].cuda()
        image = batch['image'].cuda()

        with torch.no_grad():
            latent = self.vae.extract_code(pointclouds)
        x_0 = latent.reshape(hgraph.batch_size, -1, latent.shape[-1])
        return self.training_losses(x_0, image)

    # ---- inference ----
    def image_cond_sample(self, image: torch.Tensor) -> torch.Tensor:
        sampler = FlowEulerCfgSampler(self.SIGMA_MIN)
        noise = torch.randn(1, self.num_latents, self.code_channel, device=self.device)
        cond, neg_cond = self.get_inference_cond(image)
        res = sampler.sample(
            self.model, noise=noise, cond=cond, neg_cond=neg_cond,
            steps=self.FLAGS.MODEL.sample_steps,
            cfg_strength=self.FLAGS.MODEL.cfg_strength,
        )
        return res.samples.squeeze(0)

    def _build_template_hgraph(self) -> HGraph:
        flag = self.FLAGS.DATA.test
        transform = Transform(flag.depth, flag.edge_file, flag.cluster_file)
        output = transform(np.load(flag.template_vertices_path))

        merged = HGraph(depth=flag.depth, batch_size=1)
        merged.merge_hgraph([output['hgraph']])
        return merged.cuda()

    @staticmethod
    def _load_image(image_path: str) -> torch.Tensor:
        raw = Image.open(image_path)
        if raw.mode == 'RGBA':
            img = transforms.ToTensor()(raw.resize((518, 518), Image.BILINEAR))
            alpha = img[3:4]
            img = img[:3] * alpha
        else:
            img = raw.convert('RGB').resize((518, 518), Image.LANCZOS)
            img = torch.from_numpy(np.array(img).astype(np.float32) / 255).permute(2, 0, 1)
        return img.unsqueeze(0).cuda()

    def test_single_image(self):
        flag = self.FLAGS.DATA.test
        image_path = flag.test_image_path
        image = self._load_image(image_path)
        hgraph = self._build_template_hgraph()

        base = os.path.splitext(os.path.basename(image_path))[0]
        seeds = list(flag.seeds) if len(flag.seeds) > 0 else [None]
        for seed in seeds:
            if seed is not None:
                torch.manual_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))
                np.random.seed(int(seed))
            latent = self.image_cond_sample(image)
            mesh = self.vae.decode_code(latent, hgraph, hgraph.depth)
            name = base if seed is None else f'{base}_seed{int(seed)}'
            self._save_pred(name, mesh)
            print(f'Saved generation: {name}')

    def test_dir_images(self):
        flag = self.FLAGS.DATA.test
        hgraph = self._build_template_hgraph()

        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        files = sorted(f for f in os.listdir(flag.test_image_dir) if f.lower().endswith(exts))
        for name in files:
            image = self._load_image(os.path.join(flag.test_image_dir, name))
            latent = self.image_cond_sample(image)
            mesh = self.vae.decode_code(latent, hgraph, hgraph.depth)
            self._save_pred(os.path.splitext(name)[0], mesh)

        pred_dir = os.path.join(self.result_dir, 'pred_mesh')
        tar_name = os.path.basename(flag.test_image_dir.rstrip('/')) + '.tar.gz'
        tar_path = os.path.join(self.result_dir, tar_name)
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(pred_dir, arcname=os.path.basename(pred_dir))
        print(f'Packed predictions to {tar_path}')

    # ---- writing helpers ----
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


if __name__ == '__main__':
    TOPOSDiTSolver.main()
