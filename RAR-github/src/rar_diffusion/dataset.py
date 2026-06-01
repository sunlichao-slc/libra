from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedImageEmbeddingDataset(Dataset):
    """Dataset pairing images with precomputed embedding .npy files."""

    def __init__(
        self,
        image_folder: str | os.PathLike,
        embedding_folder: str | os.PathLike,
        transform: Optional[Callable] = None,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        self.image_folder = os.fspath(image_folder)
        self.embedding_folder = os.fspath(embedding_folder)
        self.transform = transform
        self.image_files: list[str] = []
        self.embedding_files: list[str] = []

        if not os.path.isdir(self.image_folder):
            raise ValueError(f"Image folder not found: {self.image_folder}")
        if not os.path.isdir(self.embedding_folder):
            raise ValueError(f"Embedding folder not found: {self.embedding_folder}")

        for root, _, files in os.walk(self.image_folder):
            for image_file in files:
                if not image_file.lower().endswith(image_extensions):
                    continue
                embedding_file = os.path.splitext(image_file)[0] + ".npy"
                embedding_path = os.path.join(self.embedding_folder, embedding_file)
                if os.path.isfile(embedding_path):
                    self.image_files.append(os.path.join(root, image_file))
                    self.embedding_files.append(embedding_path)

        if not self.image_files:
            raise ValueError(
                f"No valid image-embedding pairs under {self.image_folder} "
                f"and {self.embedding_folder}"
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        embedding = np.load(self.embedding_files[idx])
        return image, torch.tensor(embedding, dtype=torch.float32)


class Cifar100Dataset(Dataset):
    """
    CIFAR-100 layout: images in class subfolders; embeddings named {class}_{stem}.npy
    in parallel class subfolders under embeddings_train.
    """

    def __init__(
        self,
        image_folder: str | os.PathLike,
        embedding_folder: str | os.PathLike,
        transform: Optional[Callable] = None,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        self.transform = transform
        self.image_files: list[str] = []
        self.embedding_files: list[str] = []
        image_folder = os.fspath(image_folder)
        embedding_folder = os.fspath(embedding_folder)

        if not os.path.isdir(image_folder):
            raise ValueError(f"Image folder not found: {image_folder}")
        if not os.path.isdir(embedding_folder):
            raise ValueError(f"Embedding folder not found: {embedding_folder}")

        for class_name in sorted(os.listdir(image_folder)):
            class_dir = os.path.join(image_folder, class_name)
            embedding_class_dir = os.path.join(embedding_folder, class_name)
            if not os.path.isdir(class_dir):
                continue
            for image_name in os.listdir(class_dir):
                if not image_name.lower().endswith(image_extensions):
                    continue
                image_path = os.path.join(class_dir, image_name)
                stem = os.path.splitext(image_name)[0]
                embedding_path = os.path.join(embedding_class_dir, f"{class_name}_{stem}.npy")
                if os.path.isfile(embedding_path):
                    self.image_files.append(image_path)
                    self.embedding_files.append(embedding_path)

        if not self.image_files:
            raise ValueError(
                f"No CIFAR-100 image-embedding pairs under {image_folder} / {embedding_folder}"
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        embedding = np.load(self.embedding_files[idx])
        return image, torch.tensor(embedding, dtype=torch.float32)


def build_dataset_from_config(transform):
    """Build train dataset for the active entry in config.yaml."""
    from rar_diffusion.paths import active_dataset_cfg, active_dataset_name, dataset_path

    ds_cfg = active_dataset_cfg()
    layout = ds_cfg.get("layout", "flat")
    name = active_dataset_name()

    if layout == "cifar100_class":
        return Cifar100Dataset(
            dataset_path("images_train"),
            dataset_path("embeddings_train"),
            transform=transform,
        )
    if layout == "flat":
        images = ds_cfg.get("images") or ds_cfg.get("images_train")
        embeddings = ds_cfg.get("embeddings") or ds_cfg.get("embeddings_train")
        if not images or not embeddings:
            raise KeyError(
                f"Dataset '{name}' (layout=flat) needs 'images' and 'embeddings' in config."
            )
        from rar_diffusion.paths import load_config, _resolve_string

        cfg = load_config()
        return PairedImageEmbeddingDataset(
            _resolve_string(images, cfg),
            _resolve_string(embeddings, cfg),
            transform=transform,
        )
    raise ValueError(f"Unknown dataset layout '{layout}' for '{name}'")
