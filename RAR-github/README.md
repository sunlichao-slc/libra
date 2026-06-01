# Black-Box Embedding Inversion Attack on Vector Databases



This repository contains the official implementation of LIBRA, a black-box image embedding inversion framework for studying privacy leakage from outsourced vector databases.

Given only query access to an embedding model or API, LIBRA trains a conditional diffusion model on an auxiliary dataset and reconstructs semantically consistent images from target embeddings. The framework uses embedding-guided cross-attention and performs diffusion in a VQGAN latent space to improve reconstruction quality and efficiency.

Note: This repository is intended for academic research on embedding privacy and defense evaluation. Please use it responsibly and only on data and models for which you have proper authorization.

# Overview

Modern vector databases store dense embeddings for efficient similarity search. Although raw data may not be directly outsourced, embeddings can still preserve semantic and structural information about the original inputs. LIBRA investigates this privacy risk by reconstructing plausible visual content from image embeddings under a black-box setting.

# LIBRA consists of two stages:

## Training stage
The attacker queries the target embedding model on an auxiliary dataset and trains a conditional diffusion model in the VQGAN latent space.
Recovery stage
Given a target embedding stored in a vector database, the trained diffusion model generates a latent reconstruction conditioned on the embedding, which is then decoded into image space.
Features
Black-box image embedding inversion without access to target model parameters or architecture.
Conditional diffusion model guided by image embeddings.
Embedding-guided cross-attention inside the U-Net denoising network.
VQGAN latent-space diffusion for efficient training and recovery.
Support for multiple image domains and embedding backbones.
Evaluation scripts for MSE, PSNR, SSIM, LPIPS, and optional semantic cosine similarity (CLIP-cos / DINO-cos).
Defense evaluation with embedding perturbation strategies.


## Project layout

```
.
├── configs/
│   ├── config.example.yaml   # copy → config.yaml
│   └── config.yaml           # your paths (gitignored)
├── src/rar_diffusion/        # dataset, training, recover, metrics, config
├── scripts/
│   ├── encode/               # CLIP / DINOv2 / ResNet embedding extraction
│   ├── train/train.py        # single training entry (reads config)
│   ├── recover/recover.py    # checkpoint + embedding -> reconstructed images
│   ├── eval/evaluate.py      # unified metrics entry
│   └── legacy/               # deprecated compatibility scripts
├── third_party/              # optional local placeholders (no required git submodule)
├── data/                     # CIFAR-100 images & .npy (gitignored)
└── outputs/                  # checkpoints & samples (gitignored)
```

## Quick start (CIFAR-100)

### 1. Environment

```bash
conda create -n rar-diffusion python=3.10 -y
conda activate rar-diffusion
pip install -e .
```

### 2. Prepare local model paths

```bash
cp configs/config.example.yaml configs/config.yaml
```

Configure your own local model directories in `configs/config.yaml`:

- `encoder.model_dir` for CLIP / DINOv2
- `models.vqvae` (+ optional `models.vqvae_subfolder`) for VQ-VAE

Edit `configs/config.yaml` and set your CIFAR-100 paths:

```yaml
dataset: cifar100

datasets:
  cifar100:
    images_train: "data/cifar100/images_train"      # class subfolders
    embeddings_train: "data/cifar100/embeddings_dino"
    test_embedding: "data/cifar100/embeddings_test/apple/apple_image_114.npy"

models:
  vqvae: "${rar_config_dir}/ldm-super"
  backbone: resnet   # or unet
```

### 3. Extract embeddings (once)

```bash
# auto-dispatch by encoder.type in config.yaml
python scripts/encode/extract_embeddings.py

# explicit encoder scripts
python scripts/encode/extract_embeddings_clip.py
python scripts/encode/extract_embeddings_dinov2.py
python scripts/encode/extract_embeddings_resnet.py
```

Expects images under `images_train/<class>/*.jpg` and writes  
`embeddings_train/<class>/<class>_<stem>.npy`.

Useful flags:

- `--force` overwrite existing `.npy`
- `--dry-run` print resolved config + counts, no writes
- `--encoder {clip,dinov2,resnet}` override `encoder.type`

### 4. Train

```bash
python scripts/train/train.py
# or: python scripts/train/train.py --config /path/to/config.yaml
```

### 5. Recover from embeddings

```bash
python scripts/recover/recover.py \
  --checkpoint outputs/checkpoints/cifar100/final \
  --embeddings-dir /path/to/embeddings \
  --output-dir outputs/recovered/cifar100
```

### 6. Evaluate recovered images

```bash
python scripts/eval/evaluate.py \
  --reference-dir /path/to/original_images \
  --generated-dir outputs/recovered/cifar100 \
  --output outputs/metrics/cifar100.json

# optional semantic metric
python scripts/eval/evaluate.py \
  --reference-dir /path/to/original_images \
  --generated-dir outputs/recovered/cifar100 \
  --semantic-metric clip \
  --semantic-model-dir /path/to/clip_model

# DINO ViT-S/16 cosine similarity
python scripts/eval/evaluate.py \
  --reference-dir /path/to/original_images \
  --generated-dir outputs/recovered/cifar100 \
  --semantic-metric dino-vits16 \
  --semantic-model-dir /home/wuyuncheng/RAR/RAR-main/config/dino-vits16
```

## Switching to another dataset

1. Set `dataset: celeba` (or `cub200`, …) in `config.yaml`.
2. Fill the matching block under `datasets:` (see commented examples in `config.example.yaml`).
3. Use `layout: flat` for flat image/embedding trees, or `layout: cifar100_class` for per-class folders.
4. Run the same `python scripts/train/train.py`.

No code changes required—only config.

## Config reference

| Section | Purpose |
|---------|---------|
| `dataset` | Active dataset key (`cifar100`, `celeba`, …) |
| `datasets.<name>` | Paths and `layout` for that dataset |
| `encoder` | Encoder type/dim/model dir (`clip` / `dinov2` / `resnet`) |
| `models` | VQ-VAE path, `backbone` (`resnet` / `unet`) |
| `train` | Batch size, epochs, output dirs |
| `recover` | Optional defaults for recovery output/checkpoint |
| `eval` | Optional defaults for evaluation output and semantic metric |

Encoder dim quick reference:

- `clip`: `dim=512`
- `dinov2` (base): `dim=768`
- `resnet18`: `dim=512`
- `resnet101`: `dim=2048`

`scripts/train/train.py` now validates `encoder.dim` against one embedding sample before training starts.

Environment overrides:

- `RAR_CONFIG` — path to yaml (default: `configs/config.yaml`)
- `RAR_ROOT` — repository root if not inferred from install path

## What is not in git

Datasets, checkpoints, logs, and local model weights are excluded via `.gitignore`. Point `config.yaml` at your existing local directories.


## Citation

If you use third-party pretrained configs, please cite the corresponding original papers/repos.
