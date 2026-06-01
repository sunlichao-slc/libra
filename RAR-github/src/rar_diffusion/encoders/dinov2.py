"""DINOv2 image embeddings (see reverse-face-latent-diffusion cifar100_embedding_dino.py)."""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from rar_diffusion.paths import _resolve_string, load_config


class Dinov2Encoder:
    """Extract DINO/DINOv2 CLS-token features."""

    def __init__(self, encfg: dict, device: str):
        cfg = load_config()
        model_dir = str(
            _resolve_string(str(encfg.get("model_dir", "${rar_config_dir}/DINO")), cfg)
        )
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(device).eval()

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def encode_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(
            images=list(images), return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]
