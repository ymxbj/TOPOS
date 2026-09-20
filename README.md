<h1 align="center">TOPOS: High-Fidelity and Efficient Industry-Grade 3D Head Generation</h1>

<p align="center">
  <b>Bojun Xiong</b><sup>*</sup>,
  <b>Zoubin Bi</b><sup>*</sup>,
  <b>Xinghui Peng</b>,
  <b>Yunmu Wang</b><sup>†</sup>,
  <b>Junchen Deng</b>,
  <b>Jun Liang</b>,
  <b>Jing Li</b>,
  <b>Bowen Cai</b><sup>‡</sup>,
  <b>Huan Fu</b><sup>‡</sup>
</p>

<p align="center">
  HUJING Digital Media &amp; Entertainment Group, Beijing, China
</p>

<p align="center">
  <sup>*</sup> Equal contribution &nbsp;·&nbsp;
  <sup>†</sup> Project lead &nbsp;·&nbsp;
  <sup>‡</sup> Corresponding author
</p>

<p align="center">
  <a href="https://youku-aigc.github.io/TOPOS/"><b>🌐 Project Page</b></a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2605.14594"><b>📄 arXiv</b></a>
</p>

---

TOPOS generates a high-fidelity, industry-grade 3D head with fixed MetaHuman
topology from a single image. Our framework contains three modules:

1. **TOPOS-VAE** — A Perceiver-Resampler encoder consumes an unstructured
   surface point cloud; a hierarchical GNN decoder reconstructs vertex
   coordinates and normals on the MetaHuman topology. The VAE doubles as a
   topology-conversion bridge: any mesh sampled as a point cloud can be
   re-expressed on the MetaHuman base mesh.
2. **TOPOS-DiT** — A single image conditioned flow-matching transformer trained on
   the TOPOS-VAE latents. Conditioning comes from DINOv2 ViT-L/14 (reg4) patch
   tokens of a single RGB(A) input.
3. **TOPOS-Texture** — A single image conditioned texture map generative model that produces a UV-unwrapped texture map for the generated head mesh.

Our repository contains the training and inference code. The training data will not be released; see [`filelists/README.md`](filelists/README.md) for the on-disk
layout the data loader expects.

## Installation
The code has been tested on Python 3.12, CUDA 12.8, PyTorch 2.11 with NVIDIA V100 GPUs.

```bash
git clone https://github.com/youku-aigc/TOPOS.git
cd TOPOS

conda create -n topos python=3.12 -y
conda activate topos

pip3 install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Project deps
pip3 install -r requirements.txt

pip3 install flash-attn --no-build-isolation
```

`flash-attn` is the default attention backend. It requires a GPU with compute
capability >= 8.0 (Ampere or newer, e.g. A100 / RTX 30-series / H100). If your
GPU is older or you cannot install the package, switch to PyTorch SDPA:
`export ATTN_BACKEND=sdpa`. (The code also auto-falls back to SDPA in this
case and prints a warning.)

## Pretrained Checkpoints

TOPOS-DiT uses **DINOv2 ViT-L/14 (reg4)** as the image encoder. Download once
into `./pretrained_ckpt/`:

```bash
mkdir -p pretrained_ckpt
# weights
wget -O pretrained_ckpt/dinov2_vitl14_reg4_pretrain.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
    
git clone --depth 1 https://github.com/facebookresearch/dinov2.git \
    pretrained_ckpt/facebookresearch_dinov2_main
```

Paths are configurable in [`configs/dit.yaml`](configs/dit.yaml)
(`MODEL.image_cond_model_dir`, `MODEL.image_cond_model_path`).

TOPOS-Texture finetunes the pre-trained [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511). Download the base model into `./pretrained_ckpt/`:

```bash
hf download Qwen/Qwen-Image-Edit-2511 \
    --local-dir pretrained_ckpt/Qwen-Image-Edit-2511
```

The released TOPOS-VAE, TOPOS-DiT and TOPOS-Texture weights live on Google Drive. Download
both into `./pretrained_ckpt/` so the inference scripts work out of the box:

```bash
pip3 install gdown
mkdir -p pretrained_ckpt

# TOPOS-VAE
gdown 1yEQIHQAXomgNVJ8chkuNc34RgF5rotoJ -O pretrained_ckpt/TOPOS-VAE.model.pth

# TOPOS-DiT
gdown 1VwiVdPYXJM4gnPbwo8SGCI4og2G1zeul -O pretrained_ckpt/TOPOS-DiT.model.pth

# TOPOS-Texture LoRA
gdown 16JhWYqLddYLLN95mfsJGMhw4DN6_n7Hv -O pretrained_ckpt/TOPOS-Texture.zip
unzip pretrained_ckpt/TOPOS-Texture.zip -d pretrained_ckpt/
```

Direct links: [TOPOS-VAE.model.pth](https://drive.google.com/file/d/1yEQIHQAXomgNVJ8chkuNc34RgF5rotoJ/view?usp=sharing),
[TOPOS-DiT.model.pth](https://drive.google.com/file/d/1VwiVdPYXJM4gnPbwo8SGCI4og2G1zeul/view?usp=sharing),
[TOPOS-Texture.zip](https://drive.google.com/file/d/16JhWYqLddYLLN95mfsJGMhw4DN6_n7Hv/view?usp=sharing).


## Inference

Make sure you've followed [Pretrained Checkpoints](#pretrained-checkpoints) first; after that the
scripts below run end-to-end with no extra config.

The repo ships two demo inputs under [`examples/`](examples/):
- `examples/face.npz` — a sphere-normalized surface point cloud (32k points, with normals)
- `examples/face.png` — a sample portrait for image-conditioned generation

### TOPOS-VAE (pointclouds → mesh)

The input is a `.npz` with two float32 arrays: `points [N, 3]` and
`normals [N, 3]`. The VAE re-expresses it on the MetaHuman topology. Outputs
are saved to `logs/vae/results/pred_mesh/<name>_pred.obj`.

Single point cloud (defaults to `examples/face.npz`):

```bash
bash scripts/infer_vae.sh
# or specify your own input pointclouds
bash scripts/infer_vae.sh DATA.test.test_pointcloud_path ./examples/face.npz
```

Directory of point clouds (every `.npz` under the directory):

```bash
bash scripts/infer_vae_dir.sh DATA.test.test_pointcloud_dir ./my_pointclouds
```

### TOPOS-DiT (image → mesh)

Outputs are saved to `logs/dit/results/pred_mesh/<name>_pred.obj`.

Single image (defaults to `examples/face.png`):

```bash
bash scripts/infer_image.sh
# multi-seed
bash scripts/infer_image.sh DATA.test.seeds "[0,1,2]"
# or specify your own input image
bash scripts/infer_image.sh DATA.test.test_image_path ./examples/face.png
```

Directory of images (every `.png` / `.jpg` / `.jpeg` / `.bmp` / `.webp` under
the directory; results are also packed into a `.tar.gz`):

```bash
bash scripts/infer_image_dir.sh DATA.test.test_image_dir ./examples
```

### TOPOS-Texture (Image → UV texture map)

Generate a UV-unwrapped texture map from a portrait image. The output is saved
as `logs/Texture/<input_image_name>`.

```bash
python3 TOPOS-Texture.py --image_path examples/face.png
```

## TOPOS-VAE training

```bash
# Single GPU
python3 train_vae.py --config configs/vae_train.yaml

# Multi-GPU on a single node
bash scripts/train_vae.sh
```

Override config values on the command line:

```bash
python3 train_vae.py --config configs/vae_train.yaml \
    DATA.train.batch_size 4 SOLVER.lr 5e-4
```

## TOPOS-DiT training

TOPOS-DiT is trained on a frozen TOPOS-VAE — point `MODEL.vae_ckpt` in
[`configs/dit.yaml`](configs/dit.yaml) at the `.model.pth` produced by VAE
training.

```bash
# Single GPU
python3 train_dit.py --config configs/dit_train.yaml

# Multi-GPU
bash scripts/train_dit.sh
```

## Training data layout

Training uses a single filelist (`filelists/train.txt`). Inference takes raw
images directly — there is no inference filelist.

Each line in `train.txt` is a single sample with **two whitespace-separated
columns**:

```
<category>  <relative_filename>
```

The data loader resolves these two columns into file paths under
`DATA.train.root_dir` as follows:

| File needed                                                                     | Used by                       |
|---------------------------------------------------------------------------------|-------------------------------|
| `<root_dir>/<category>_vertices_npz/<relative_filename>.npz`                    | ground truth mesh (TOPOS-VAE)       |
| `<root_dir>/<category>_surface_npz/<relative_filename>.npz`                     | input pointclouds (TOPOS-VAE and TOPOS-VAE) |
| `<root_dir>/<category>_images/<relative_filename>/<k>.png`, k ∈ [0, image_nums) | input image (TOPOS-DiT only)  |

Note the `_vertices_npz`, `_surface_npz`, `_images` suffixes that the loader
appends to the first column — they are NOT in the filelist.

**Concrete example.** The first line of the shipped `filelists/train.txt` is:

```
RandomFace 0/0_00000
```

With `DATA.train.root_dir: ./`, this single line drives the loader to read:

```
./RandomFace_vertices_npz/0/0_00000.npz
./RandomFace_surface_npz/0/0_00000.npz
./RandomFace_images/0/0_00000/0.png      # ... up to 7.png with image_nums=8
```

See [`filelists/README.md`](filelists/README.md) for array shapes and dtype.
The shipped `train.txt` is a tiny placeholder showing the format only —
replace it with your own.

## Topology assets

`data/` ships the MetaHuman head topology used in the paper:

| File                          | Description                                          |
|-------------------------------|------------------------------------------------------|
| `template.obj`                | Base MetaHuman head (24049 vertices, 24002 faces)    |
| `template_vertices.npz`       | `vertices`, `normals` of the template                |
| `edges.npy`                   | Bidirectional edge list `[E, 2]`                     |
| `faces.npy`                   | Triangle index list `[F, 3]`                         |
| `cluster.npz`                 | Pre-built hierarchical voxel-grid clusters           |
| `a2bIdx.pt`, `sidebyside.pt`  | Per-edge adjacency helpers for Gaussian curvature    |

To use a different topology, regenerate these with the
[tools below](#working-with-a-new-topology).

## Working with a new topology

To retarget TOPOS to a different head mesh, regenerate the topology assets:

```bash
# 1. edges.npy, faces.npy, template_vertices.npz
python3 tools/extract_topology.py \
    --obj ./data/your_template.obj \
    --edges-out ./data/edges.npy \
    --faces-out ./data/faces.npy \
    --vertices-out ./data/template_vertices.npz

# 2. cluster.npz (hierarchical pooling)
python3 tools/build_cluster.py \
    --vertices ./data/template_vertices.npz \
    --edges ./data/edges.npy \
    --output ./data/cluster.npz \
    --depth 8
```

`a2bIdx.pt` / `sidebyside.pt` (used by the Gaussian-curvature loss) are
topology-specific and need to be re-derived for any new mesh.

## Repository layout

```
TOPOS/
├── configs/           # YAML configs (base + train / test overrides)
├── data/              # Bundled MetaHuman topology assets (sphere-normalized)
├── examples/          # Demo inputs (face.npz, face.png) used by infer_*.sh
├── filelists/         # Training filelist + dataset README
├── pretrained_ckpt/   # gitignored — populate via the README download commands
├── scripts/           # Shell wrappers for training and inference
├── tools/             # Offline data-prep utilities (sphere normalize, topology, clusters)
├── encoders/          # TOPOS-VAE (Perceiver-Resampler encoder + hierarchical GNN decoder)
├── flow_matching/     # TOPOS-DiT (flow-matching transformer with image cross-attention)
├── thsolver/          # Tiny model-agnostic training framework
├── dataset_face.py    # Per-sample HGraph transform + batched collate
├── train_vae.py       # TOPOS-VAE entry (training + point-cloud inference)
├── train_dit.py       # TOPOS-DiT entry (training + image inference)
├── TOPOS-Texture.py   # TOPOS-Texture entry (single image to UV texture inference)
├── lambdas.json       # TOPOS-VAE loss weights
├── requirements.txt
├── LICENSE
└── README.md
```

## Open-source Progress

- [x] TOPOS-VAE inference code
- [x] TOPOS-DiT inference code
- [x] TOPOS-Texture inference code
- [x] TOPOS-VAE training code
- [x] TOPOS-DiT training code
- [ ] TOPOS-Texture training code
- [ ] Training dataset

## Contact

If you have any questions, please contact xiongbojun@pku.edu.cn or bzb@zju.edu.cn

## Citation

```bibtex
@article{xiong2026topos,
  title={TOPOS: High-Fidelity and Efficient Industry-Grade 3D Head Generation},
  author={Xiong, Bojun and Bi, Zoubin and Peng, Xinghui and Wang, Yunmu and Deng, Junchen and Liang, Jun and Li, Jing and Cai, Bowen and Fu, Huan},
  journal={arXiv preprint arXiv:2605.14594},
  year={2026}
}
```


## License

TOPOS is released under the [MIT License](LICENSE). The license covers:

- All source code in this repository
- The released TOPOS-VAE, TOPOS-DiT and TOPOS-Texture checkpoints (on Google Drive)
- The MetaHuman topology assets shipped under [`data/`](data/)
  (`template.obj`, `template_vertices.npz`, `edges.npy`, `faces.npy`,
  `cluster.npz`, `a2bIdx.pt`, `sidebyside.pt`)
- The demo inputs shipped under [`examples/`](examples/) (`face.npz`, `face.png`)

You are free to use, modify, redistribute, and build commercial products on
top of TOPOS, provided you retain the copyright notice and disclaimer from
[`LICENSE`](LICENSE).

> **Third-party components** that this repository depends on but does **not**
> redistribute (DINOv2 weights from Meta, PyTorch, torch_geometric, flash-attn,
> etc.) remain under their respective upstream licenses — users are
> responsible for complying with those terms when downloading or installing
> them. See [`LICENSE`](LICENSE) for the full list.

If you use TOPOS in academic work, please cite the paper above.
