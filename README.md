# Black-Box Embedding Inversion Attack on Vector Databases

This repository implements **LIBRA**, a black-box image embedding inversion framework for analyzing privacy leakage in vector databases.

Given only embedding queries (no access to target model parameters), LIBRA trains a conditional diffusion model in latent space and reconstructs semantically consistent images from target embeddings.

> For research use only. Evaluate privacy risks only on authorized data/models.

## Method Overview

Modern vector databases store embeddings instead of raw images. LIBRA studies whether those embeddings can still leak sensitive visual information.

![LIBRA framework](docs/images/franework.png)

### Stage 1: Training
- Query the target encoder on an auxiliary dataset to obtain image embeddings.
- Train an embedding-conditioned `UNet` denoiser.
- Perform diffusion in VQ latent space for better efficiency.

### Stage 2: Recovery
- Input a target embedding from the vector database.
- Run reverse diffusion conditioned on the embedding.
- Decode latent output with VQ-VAE to obtain reconstructed images.

## Key Features

- Black-box attack setting (no target model internals required).
- Multiple encoders: `clip`, `dinov2`, `resnet`.
- Unified evaluation script with `MSE`, `PSNR`, `SSIM`, `LPIPS`, and optional semantic cosine (`CLIP` / `DINO`).
- Config-driven workflow for quick dataset switching.

## Project Layout

```text
.
├── configs/
│   ├── config.example.yaml   # copy -> config.yaml
│   └── config.yaml           # local paths (gitignored)
├── src/rar_diffusion/
│   ├── extract.py            # embedding extraction core
│   ├── train.py              # training loop core
│   ├── infer.py              # recovery pipeline core
│   ├── metrics/              # folder-based metrics
│   └── paths.py              # config/path resolver
├── scripts/
│   ├── encode/               # extraction entrypoints
│   ├── train/train.py        # training entrypoint
│   ├── recover/recover.py    # recovery entrypoint
│   └── eval/evaluate.py      # evaluation entrypoint
├── docs/images/
│   └── franework.png         # framework figure
└── outputs/                  # checkpoints / reconstructions / metrics
```

## Quick Start (CIFAR-100)

### 1) Install

```bash
conda create -n rar-diffusion python=3.10 -y
conda activate rar-diffusion
pip install -e .
```

### 2) Prepare config

```bash
cp configs/config.example.yaml configs/config.yaml
```

Edit `configs/config.yaml`:
- `dataset` and `datasets.<name>.*` paths
- `encoder.type` / `encoder.dim` / `encoder.model_dir`
- `models.vqvae` / `models.vqvae_subfolder`

Minimal CIFAR-100 example:

```yaml
dataset: cifar100

datasets:
  cifar100:
    layout: cifar100_class
    images_train: "/path/to/cifar100/images_train"
    embeddings_train: "/path/to/cifar100/embeddings_train"
    test_embedding: "/path/to/cifar100/test_embedding.npy"

encoder:
  type: clip
  dim: 512
  model_dir: "/path/to/clip-or-dino-model"

models:
  vqvae: "/path/to/ldm-super"
  vqvae_subfolder: "vqvae"
```

### 3) Extract embeddings (once)

```bash
# uses encoder.type from config.yaml
python scripts/encode/extract_embeddings.py

# or explicit entrypoints
python scripts/encode/extract_embeddings_clip.py
python scripts/encode/extract_embeddings_dinov2.py
python scripts/encode/extract_embeddings_resnet.py
```

Useful flags:
- `--force` overwrite existing `.npy`
- `--dry-run` check resolved config/counts only
- `--encoder {clip,dinov2,resnet}` override `encoder.type`

Expected CIFAR-100 naming:
- input image: `images_train/<class>/<stem>.jpg`
- output embedding: `embeddings_train/<class>/<class>_<stem>.npy`

### 4) Train denoiser

```bash
python scripts/train/train.py
# optional: python scripts/train/train.py --config /path/to/config.yaml
```

`train.py` validates one embedding sample against `encoder.dim` before training to prevent dimension mismatch.

### 5) Recover images from embeddings

`scripts/recover/recover.py` requires all core inputs explicitly:

```bash
python scripts/recover/recover.py \
  --checkpoint outputs/checkpoints/cifar100/final \
  --vqvae-path /path/to/ldm-super/vqvae \
  --embeddings-dir /path/to/embeddings_root \
  --image-root /path/to/original_images_root \
  --output-dir outputs/recovered/cifar100 \
  --num-steps 1000 \
  --device cuda:0
```

Notes:
- `embeddings-dir` should be class folders for CIFAR-style recovery.
- The script recovers one random embedding per class and also saves comparison grids.

### 6) Evaluate reconstruction quality

```bash
python scripts/eval/evaluate.py \
  --reference-dir /path/to/original_images \
  --generated-dir outputs/recovered/cifar100 \
  --output outputs/metrics/cifar100.json
```

Optional semantic metric:

```bash
python scripts/eval/evaluate.py \
  --reference-dir /path/to/original_images \
  --generated-dir outputs/recovered/cifar100 \
  --semantic-metric clip \
  --semantic-model-dir /path/to/clip-model \
  --output outputs/metrics/cifar100_clip.json
```

Supported semantic metrics:
- `clip`
- `dino`
- `dino-vits16`

## Switching Datasets

1. Change `dataset:` in `configs/config.yaml` (e.g., `celeba`, `cub200`).
2. Fill `datasets.<name>` paths.
3. Set layout:
   - `cifar100_class` for class-subfolder trees
   - `flat` for flattened image/embedding trees
4. Re-run extraction (if needed), training, and recovery with updated paths.

## Config Cheat Sheet

- `dataset`: active dataset key.
- `datasets.<name>.layout`: data organization mode.
- `encoder.type`: `clip` / `dinov2` / `resnet`.
- `encoder.dim`: must match actual embedding output.
- `train.*`: optimization and output settings.
- `recover.*`: default recovery output/checkpoint fields.
- `eval.*`: evaluation defaults.

Encoder dimension reference:
- `clip`: `512`
- `dinov2-base`: `768`
- `resnet101`: `2048`

Environment overrides:
- `RAR_CONFIG`: custom config path
- `RAR_ROOT`: custom repository root

## Common Issues

- `Embedding dim mismatch`: `encoder.dim` and actual `.npy` size differ; regenerate embeddings or fix config.
- `diffusion_pytorch_model.safetensors not found`: checkpoint format/path mismatch; verify model directory and files.
- Training interrupted with `SIGHUP`: run with `tmux`/`screen` or detached shell to avoid terminal hangup.

## What Is Not Committed

Datasets, checkpoints, logs, and local pretrained weights are excluded by `.gitignore`. Keep your local paths in `configs/config.yaml`.

## Citation

If you use this codebase or any third-party pretrained components, please cite the corresponding original papers and repositories.
