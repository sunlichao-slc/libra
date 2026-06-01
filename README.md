# Black-Box Embedding Inversion Attack on Vector Databases

This repository provides the implementation of **LIBRA**, a black-box image embedding inversion framework for evaluating privacy leakage in vector databases.

Given query access to an embedding model (without model parameters), LIBRA trains an embedding-conditioned latent diffusion model and reconstructs semantically consistent images from target embeddings.

> For research use only. Evaluate privacy risks only on authorized data and models.

## Method Overview

Vector databases typically store dense embeddings rather than raw images. LIBRA investigates whether these embeddings still leak semantic information about the original samples.

![LIBRA framework](docs/images/franework.png)

### Stage 1: Training
- Query the target encoder on an auxiliary dataset to collect image embeddings.
- Train an embedding-conditioned `UNet` denoiser in VQ latent space.
- Learn to predict noise under diffusion timesteps with embedding guidance.

### Stage 2: Recovery
- Input a target embedding from the vector database.
- Run reverse diffusion conditioned on that embedding.
- Decode the recovered latent through VQ-VAE to obtain reconstructed images.

## Key Features

- Black-box threat model (no access to target model internals).
- Multiple supported encoders: `clip`, `dinov2`, `resnet`.
- Unified evaluation script for `MSE`, `PSNR`, `SSIM`, `LPIPS`, and optional semantic cosine metrics.
- Config-driven workflow for reproducible cross-dataset experiments.

## Repository Structure

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

## Required Model Checkpoints

- CLIP ViT-B/32: <https://huggingface.co/openai/clip-vit-base-patch32>
- DINOv2 Base: <https://huggingface.co/facebook/dinov2-base>

## Quick Start (CIFAR-100)

### 1) Environment Setup

```bash
conda create -n libra python=3.10 -y
conda activate libra
pip install -e .
```

### 2) Prepare Configuration

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

### 3) Extract Embeddings (Run Once)

```bash
# uses encoder.type from config.yaml
python scripts/encode/extract_embeddings.py

# explicit entrypoints
python scripts/encode/extract_embeddings_clip.py
python scripts/encode/extract_embeddings_dinov2.py
python scripts/encode/extract_embeddings_resnet.py
```

Useful flags:
- `--force` overwrite existing `.npy`
- `--dry-run` print resolved config and counts without writing files
- `--encoder {clip,dinov2,resnet}` override `encoder.type`

Expected CIFAR-100 naming:
- input image: `images_train/<class>/<stem>.jpg`
- output embedding: `embeddings_train/<class>/<class>_<stem>.npy`

### 4) Train the Denoiser

```bash
python scripts/train/train.py
# optional: python scripts/train/train.py --config /path/to/config.yaml
```

Before training starts, `train.py` validates `encoder.dim` against a sample embedding to prevent shape mismatch errors.

### 5) Recover Images from Embeddings

`scripts/recover/recover.py` requires explicit runtime paths:

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
- `embeddings-dir` should use class-wise folders for CIFAR-style recovery.
- The script reconstructs one random embedding per class and saves comparison grids.

### 6) Evaluate Reconstruction Quality

```bash
python scripts/eval/evaluate.py \
  --reference-dir /path/to/original_images \
  --generated-dir outputs/recovered/cifar100 \
  --output outputs/metrics/cifar100.json
```

Optional semantic metric (example: CLIP cosine):

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

## Switching to Other Datasets

1. Update `dataset:` in `configs/config.yaml` (e.g., `celeba`, `cub200`).
2. Fill corresponding `datasets.<name>` paths.
3. Set `layout`:
   - `cifar100_class` for class-subfolder trees
   - `flat` for flattened image/embedding trees
4. Re-run extraction (if needed), training, and recovery.

## Config Cheat Sheet

- `dataset`: active dataset key.
- `datasets.<name>.layout`: data organization mode.
- `encoder.type`: `clip` / `dinov2` / `resnet`.
- `encoder.dim`: must match the actual embedding dimension.
- `train.*`: optimization and training outputs.
- `recover.*`: default recovery fields.
- `eval.*`: default evaluation fields.

Encoder dimensions:
- `clip`: `512`
- `dinov2-base`: `768`
- `resnet101`: `2048`

Environment overrides:
- `RAR_CONFIG`: custom config path
- `RAR_ROOT`: custom repository root

## Baselines

The implementations of `rMLE`, `LM`, `GLASS`, and `DRRAG` are based on:
<https://github.com/ntuaislab/DRAG>

## Common Issues

- `Embedding dim mismatch`: `encoder.dim` differs from `.npy` feature size; regenerate embeddings or update config.
- `diffusion_pytorch_model.safetensors not found`: checkpoint path/format mismatch; verify the model directory layout.
- Training interrupted by `SIGHUP`: run with `tmux` / `screen` or a detached shell session.

## What Is Not Committed

Datasets, checkpoints, logs, and local pretrained weights are ignored by `.gitignore`. Keep machine-specific paths in `configs/config.yaml`.

## Citation

If you use this codebase or third-party pretrained components, please cite the corresponding original papers and repositories.
