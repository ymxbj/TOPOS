# Training filelist & dataset layout

There is only a training filelist (`train.txt`). Inference does not use a
filelist — see `configs/dit_test.yaml` and `configs/dit_test_dir.yaml`
for the image-based inference paths.

## How a filelist line maps to files

Each line in `train.txt` has **two whitespace-separated columns**:

```
<category>  <relative_filename>
```

- The first column (`<category>`) names a logical group. For each category,
  three sibling directories must exist under `DATA.train.root_dir`, with the
  fixed suffixes `_vertices_npz`, `_surface_npz`, `_images` appended:

      <root_dir>/<category>_vertices_npz/
      <root_dir>/<category>_surface_npz/
      <root_dir>/<category>_images/

- The second column (`<relative_filename>`) is a path **relative to those
  three directories** (slashes are allowed — they become subdirectories).

For a line `CAT REL/sub`, the data loader reads:

```
<root_dir>/CAT_vertices_npz/REL/sub.npz                   # required (target mesh)
<root_dir>/CAT_surface_npz/REL/sub.npz                    # required (input point cloud)
<root_dir>/CAT_images/REL/sub/<k>.png, k=0..image_nums-1  # only for TOPOS-DiT training
```

### Concrete example

The first line of the shipped `train.txt` is:

```
RandomFace 0/0_00000
```

With `DATA.train.root_dir: ./`, the loader pulls in:

```
./RandomFace_vertices_npz/0/0_00000.npz
./RandomFace_surface_npz/0/0_00000.npz
./RandomFace_images/0/0_00000/0.png   # ... through 7.png when image_nums=8
```

You can pack multiple datasets into one `train.txt` by mixing different first
columns; the `_vertices_npz` / `_surface_npz` / `_images` suffixes keep each
group's files isolated on disk.

> **⚠ Sphere normalization (required).** Both `vertices` and `points` must be
> sphere-normalized into the unit cube `[-1, 1]^3` using the recipe in
> `tools/sphere_normalize.py` (center on AABB centroid, divide by
> `max ||vertex||`). The hierarchical-graph pooling baked into TOPOS-VAE
> assumes inputs live in this frame; unnormalized data will silently produce
> empty/garbled HGraph levels. The shipped `data/template.obj` and
> `data/template_vertices.npz` are already normalized — make sure your training
> meshes and point clouds use the **same** normalization, sample-by-sample.

## File contents

- `*_vertices_npz/*.npz` — numpy `.npz` with arrays `vertices` and `normals`,
  both shaped `[V, 3]`, in the same vertex order as `data/template.obj`. `V`
  is the fixed-topology vertex count of that template.

- `*_surface_npz/*.npz` — numpy `.npz` with arrays `points` and `normals`, both
  shaped `[N, 3]`, sampled uniformly on the mesh surface. Any value of `N`
  works (the encoder is set-based); we use a few thousand points per sample.

- `*_images/<sample>/<k>.png` — `k` rendered RGB(A) views of the head at
  `518x518`. If alpha is provided, the dataset uses it to composite onto a
  black background. `image_nums` in the config sets how many views per sample
  exist on disk (default 8); one view is randomly chosen each training step.

The shipped `train.txt` is a tiny placeholder showing the format only — replace
it with your own.
