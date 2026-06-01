"""Shared utilities for embedding-conditioned latent diffusion."""

from rar_diffusion.config import TrainingConfig
from rar_diffusion.dataset import Cifar100Dataset, PairedImageEmbeddingDataset, build_dataset_from_config
from rar_diffusion.embeddings import ensure_3d_embedding
from rar_diffusion.paths import active_dataset_name, get_project_root, load_config, resolve_path

__all__ = [
    "TrainingConfig",
    "PairedImageEmbeddingDataset",
    "Cifar100Dataset",
    "build_dataset_from_config",
    "ensure_3d_embedding",
    "get_project_root",
    "load_config",
    "resolve_path",
    "active_dataset_name",
]
