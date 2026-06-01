"""Shared embedding extraction loop for configured datasets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from rar_diffusion.encoders import build_encoder
from rar_diffusion.paths import (
    active_dataset_cfg,
    active_dataset_name,
    dataset_path,
    get_encoder_cfg,
    load_config,
)


def parse_args(encoder_type: Optional[str] = None):
    p = argparse.ArgumentParser(
        description="Extract image embeddings for the active dataset layout."
    )
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda:1, cuda:0, cpu, ...")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embedding .npy files instead of skipping.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings and image counts, then exit without writing files.",
    )
    if encoder_type is None:
        p.add_argument(
            "--encoder",
            choices=["clip", "dinov2", "dino", "resnet"],
            default=None,
            help="Override encoder.type in config.yaml",
        )
    return p.parse_args()


def run_extract(
    *,
    encoder_type: Optional[str] = None,
    config: Optional[str] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    if config:
        os.environ["RAR_CONFIG"] = config

    ds_name = active_dataset_name()
    ds_cfg = active_dataset_cfg()
    layout = str(ds_cfg.get("layout", "flat"))

    encfg = get_encoder_cfg()
    encoder_type = (encoder_type or encfg.get("type", "dinov2")).lower()
    if encoder_type == "dino":
        encoder_type = "dinov2"

    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(encoder_type, encfg, device)
    configured_dim = int(encfg.get("dim", encoder.embedding_dim))
    if configured_dim != int(encoder.embedding_dim):
        raise ValueError(
            f"encoder.dim mismatch: config={configured_dim}, "
            f"{encoder_type} model={encoder.embedding_dim}. "
            "Update config.yaml encoder.dim to match the selected encoder."
        )

    if layout == "cifar100_class":
        image_root = dataset_path("images_train")
        output_root = dataset_path("embeddings_train")
    elif layout == "flat":
        images = ds_cfg.get("images") or ds_cfg.get("images_train")
        embeddings = ds_cfg.get("embeddings") or ds_cfg.get("embeddings_train")
        if not images or not embeddings:
            raise KeyError(
                f"Dataset '{ds_name}' (layout=flat) needs 'images' and 'embeddings' in config."
            )
        cfg = load_config()
        from rar_diffusion.paths import _resolve_string

        image_root = _resolve_string(str(images), cfg)
        output_root = _resolve_string(str(embeddings), cfg)
    else:
        raise ValueError(f"Unknown dataset layout '{layout}' for '{ds_name}'")

    output_root.mkdir(parents=True, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png")
    print(
        f"Dataset: {ds_name} ({layout}) | Encoder: {encoder_type} | "
        f"dim: {encoder.embedding_dim} | output: {output_root}"
    )

    if layout == "cifar100_class":
        class_folders = [d for d in sorted(image_root.iterdir()) if d.is_dir()]
        total = 0
        for class_folder in class_folders:
            total += len([f for f in os.listdir(class_folder) if f.lower().endswith(valid_ext)])
        print(f"Classes: {len(class_folders)} | images: {total}")
        if dry_run:
            print("Dry run complete (no files written).")
            return

        for class_folder in class_folders:
            class_name = class_folder.name
            class_out = output_root / class_name
            class_out.mkdir(parents=True, exist_ok=True)
            image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(valid_ext)]
            print(f"\n{class_name}: {len(image_files)} images")

            for i in tqdm(range(0, len(image_files), batch_size), desc=class_name):
                batch_files = image_files[i : i + batch_size]
                images, names = [], []
                for fname in batch_files:
                    path = class_folder / fname
                    try:
                        images.append(Image.open(path).convert("RGB"))
                        names.append(fname)
                    except OSError as exc:
                        print(f"[skip] {fname}: {exc}")

                if not images:
                    continue

                feats = encoder.encode_batch(images).cpu().numpy()
                for feat, fname in zip(feats, names):
                    stem = os.path.splitext(fname)[0]
                    save_path = class_out / f"{class_name}_{stem}.npy"
                    if save_path.is_file() and not force:
                        continue
                    np.save(save_path, np.atleast_2d(feat))
    else:
        image_paths = [
            p for p in sorted(image_root.rglob("*")) if p.is_file() and p.suffix.lower() in valid_ext
        ]
        print(f"Flat images: {len(image_paths)}")
        if dry_run:
            print("Dry run complete (no files written).")
            return

        for i in tqdm(range(0, len(image_paths), batch_size), desc="flat"):
            batch_paths = image_paths[i : i + batch_size]
            images, stems = [], []
            for image_path in batch_paths:
                try:
                    images.append(Image.open(image_path).convert("RGB"))
                    stems.append(image_path.stem)
                except OSError as exc:
                    print(f"[skip] {image_path.name}: {exc}")

            if not images:
                continue

            feats = encoder.encode_batch(images).cpu().numpy()
            for feat, stem in zip(feats, stems):
                save_path = output_root / f"{stem}.npy"
                if save_path.is_file() and not force:
                    continue
                np.save(save_path, np.atleast_2d(feat))

    print("Done.")


def main(encoder_type: Optional[str] = None) -> None:
    args = parse_args(encoder_type)
    chosen = encoder_type or getattr(args, "encoder", None)
    run_extract(
        encoder_type=chosen,
        config=args.config,
        batch_size=args.batch_size,
        device=args.device,
        force=args.force,
        dry_run=args.dry_run,
    )
