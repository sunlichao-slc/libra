"""CLIP image embeddings (see reverse-face-latent-diffusion cub200_clip_encode.py)."""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from transformers import AutoImageProcessor, CLIPModel

from rar_diffusion.paths import _resolve_string, load_config


class ClipEncoder:
    """Extract CLIP image features with get_image_features()."""

    def __init__(self, encfg: dict, device: str):
        cfg = load_config()
        model_dir = str(
            _resolve_string(str(encfg.get("model_dir", "${rar_config_dir}/clip")), cfg)
        )
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = CLIPModel.from_pretrained(model_dir).to(device).eval()

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.projection_dim)

    def encode_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.get_image_features(**inputs)
