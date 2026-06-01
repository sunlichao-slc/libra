#!/usr/bin/env python3
"""Train conditional DDPM from configs/config.yaml (default: CIFAR-100)."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torchvision import transforms

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from rar_diffusion.config import TrainingConfig
from rar_diffusion.checkpoints import save_denoiser
from rar_diffusion.dataset import build_dataset_from_config
from rar_diffusion.embeddings import ensure_3d_embedding
from rar_diffusion.models.resnet_unet import create_resnet_diffusion_model
from rar_diffusion.paths import (
    _resolve_string,
    active_dataset_name,
    dataset_path,
    get_encoder_cfg,
    get_models_cfg,
    get_train_cfg,
    load_config,
)
from rar_diffusion.train import train_loop


def _setup_path():
    config_path = os.path.join(_REPO_ROOT, "configs", "config.yaml")
    if not os.path.isfile(config_path):
        example = os.path.join(_REPO_ROOT, "configs", "config.example.yaml")
        raise FileNotFoundError(
            f"Missing configs/config.yaml. Copy the example first:\n"
            f"  cp {example} configs/config.yaml"
        )


def build_unet(cross_attention_dim: int) -> UNet2DConditionModel:
    unet = UNet2DConditionModel(
        act_fn="silu",
        attention_head_dim=32,
        block_out_channels=[224, 448, 672, 896],
        center_input_sample=False,
        down_block_types=[
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ],
        downsample_padding=1,
        flip_sin_to_cos=True,
        freq_shift=0,
        in_channels=3,
        layers_per_block=2,
        mid_block_scale_factor=1,
        norm_eps=1e-5,
        norm_num_groups=32,
        out_channels=3,
        sample_size=64,
        cross_attention_dim=cross_attention_dim,
        only_cross_attention=True,
        up_block_types=[
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ],
    )
    unet.enable_gradient_checkpointing()
    return unet


def build_denoiser(backbone: str, cross_attention_dim: int):
    if backbone == "resnet":
        model = create_resnet_diffusion_model(
            in_channels=3,
            out_channels=3,
            model_channels=256,
            num_res_blocks=2,
            attention_resolutions=(32, 16, 8),
            channel_mult=(1, 2, 4, 8),
            cross_attention_dim=cross_attention_dim,
            only_cross_attention=True,
        )
        return model
    if backbone == "unet":
        return build_unet(cross_attention_dim)
    raise ValueError(f"Unknown backbone '{backbone}', use resnet or unet")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: configs/config.yaml or $RAR_CONFIG)",
    )
    return p.parse_args()


def _validate_embedding_dim(dataset, expected_dim: int) -> None:
    sample_path = None
    if hasattr(dataset, "embedding_files") and dataset.embedding_files:
        sample_path = dataset.embedding_files[0]
    if sample_path is None:
        return
    arr = np.load(sample_path)
    actual_dim = int(arr.shape[-1]) if arr.ndim > 0 else int(arr.size)
    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dim mismatch: encoder.dim={expected_dim}, "
            f"but sample '{sample_path}' has dim={actual_dim}. "
            "Please regenerate embeddings or update encoder.dim."
        )


def main():
    args = parse_args()
    if args.config:
        os.environ["RAR_CONFIG"] = args.config
    _setup_path()

    tcfg = get_train_cfg()
    encfg = get_encoder_cfg()
    mcfg = get_models_cfg()
    cross_dim = int(encfg.get("dim", 768))
    backbone = str(mcfg.get("backbone", "resnet"))

    config = TrainingConfig(
        image_size=int(tcfg.get("image_size", 256)),
        train_batch_size=int(tcfg.get("batch_size", 32)),
        num_epochs=int(tcfg.get("num_epochs", 1001)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(tcfg.get("learning_rate", 1e-5)),
        lr_warmup_steps=int(tcfg.get("lr_warmup_steps", 500)),
        mixed_precision=str(tcfg.get("mixed_precision", "fp16")),
        save_image_epochs=int(tcfg.get("save_image_epochs", 1)),
        save_model_epochs=int(tcfg.get("save_model_epochs", 5)),
        output_dir=str(tcfg.get("output_dir", "outputs/checkpoints/cifar100")),
        eval_images_dir=str(tcfg.get("eval_images_dir", "outputs/eval_images/cifar100")),
        seed=int(tcfg.get("seed", 0)),
    )

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5]),
        transforms.Lambda(lambda t: t.clamp(0, 1)),
    ])

    print(f"Dataset: {active_dataset_name()} | Backbone: {backbone} | Encoder dim: {cross_dim}")
    dataset = build_dataset_from_config(transform)
    _validate_embedding_dim(dataset, cross_dim)
    num_workers = int(tcfg.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    images, embeddings = next(iter(loader))
    eval_image = reverse_transform(images[0]).cpu().permute(1, 2, 0).numpy()
    eval_embedding = ensure_3d_embedding(embeddings[0])

    test_path = dataset_path("test_embedding")
    if test_path.is_file():
        test_embedding = ensure_3d_embedding(
            torch.tensor(np.load(test_path), dtype=torch.float32)
        )
    else:
        test_embedding = eval_embedding

    os.makedirs(config.eval_images_dir, exist_ok=True)
    plt.imsave(os.path.join(config.eval_images_dir, "base_image.jpg"), eval_image)

    cfg = load_config()
    vqvae_root = str(_resolve_string(str(mcfg.get("vqvae", "${rar_config_dir}/ldm-super")), cfg))
    subfolder = str(mcfg.get("vqvae_subfolder", "vqvae"))
    vqvae = VQModel.from_pretrained(vqvae_root, subfolder=subfolder)

    scheduler = DDPMScheduler(
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon",
        num_train_timesteps=1000,
    )

    denoiser = build_denoiser(backbone, cross_dim)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(loader) * config.num_epochs,
    )

    train_loop(
        config,
        denoiser,
        vqvae,
        scheduler,
        optimizer,
        loader,
        lr_scheduler,
        eval_embedding,
        test_embedding,
    )

    save_root = os.path.join(config.output_dir, "final")
    os.makedirs(save_root, exist_ok=True)
    save_denoiser(denoiser, save_root)


if __name__ == "__main__":
    main()
