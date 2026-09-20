import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    """Training dataset for TOPOS-VAE / TOPOS-DiT.

    Each entry in ``filelist`` is one line ``<category> <filename>``. The
    dataset expects the following files per entry:

      <root_dir>/<category>_vertices_npz/<filename>.npz   # required (target mesh)
      <root_dir>/<category>_surface_npz/<filename>.npz    # required iff use_surface
      <root_dir>/<category>_images/<filename>/<k>.png     # required iff use_images
                                                          # (k in [0, image_nums))

    Inference does not use this class — see ``test_single_image`` /
    ``test_dir_images`` in ``train_dit.py`` for the image-driven path.
    """

    def __init__(
        self,
        root_dir: str,
        use_surface: bool,
        use_images: bool,
        filelist: str,
        transform,
        max_rotation: float,
        max_translate: float,
        scale_range,
        image_nums: int = 8,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.filelist = filelist
        self.transform = transform

        self.use_surface = use_surface
        self.use_images = use_images
        self.image_nums = image_nums

        self.max_rotation = max_rotation
        self.max_translate = max_translate
        self.scale_range = scale_range
        self.to_tensor = transforms.ToTensor()

        self.categories, self.filenames = self.load_filenames()

    def __len__(self):
        return len(self.filenames)

    # ---- IO ----
    @staticmethod
    def _load_npz(path: str):
        return np.load(path)

    def load_image(self, idx: int) -> torch.Tensor:
        image_dir = os.path.join(self.root_dir, self.categories[idx] + '_images', self.filenames[idx])
        view_idx = random.randint(0, self.image_nums - 1)
        image_path = os.path.join(image_dir, f'{view_idx}.png')

        image = Image.open(image_path).convert('RGBA').resize((518, 518), Image.BILINEAR)
        image = self.to_tensor(image)
        alpha = image[3:4]
        return image[:3] * alpha  # premultiplied black background

    # ---- augmentation ----
    def sample_augmentation_params(self) -> Tuple[float, Tuple[float, float], float]:
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        translate_frac = (
            random.uniform(-self.max_translate, self.max_translate),
            random.uniform(-self.max_translate, self.max_translate),
        )
        scale = random.uniform(*self.scale_range)
        return angle, translate_frac, scale

    @staticmethod
    def apply_augmentation(image: torch.Tensor, params) -> torch.Tensor:
        angle, translate_frac, scale = params
        _, h, w = image.shape
        translate = (int(translate_frac[0] * w), int(translate_frac[1] * h))
        return TF.affine(
            image, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
            interpolation=transforms.InterpolationMode.BILINEAR, fill=0.0,
        )

    # ---- main ----
    def __getitem__(self, idx):
        mesh_sample = self._load_npz(os.path.join(
            self.root_dir, self.categories[idx] + '_vertices_npz', self.filenames[idx] + '.npz'))
        output = self.transform(mesh_sample)

        if self.use_surface:
            pc_sample = self._load_npz(os.path.join(
                self.root_dir, self.categories[idx] + '_surface_npz', self.filenames[idx] + '.npz'))
            points = torch.from_numpy(pc_sample['points'].astype(np.float32))
            normals = torch.from_numpy(pc_sample['normals'].astype(np.float32))
            output['pointclouds'] = torch.cat([points, normals], dim=1)

        if self.use_images:
            image = self.load_image(idx)
            image = self.apply_augmentation(image, self.sample_augmentation_params())
            output['image'] = image

        output['filename'] = self.filenames[idx]
        return output

    def load_filenames(self):
        categories, filenames = [], []
        with open(self.filelist) as f:
            for line in f:
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                categories.append(tokens[0])
                filenames.append(tokens[1])
        return categories, filenames
