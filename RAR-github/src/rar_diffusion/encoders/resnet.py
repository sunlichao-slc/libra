"""ResNet image embeddings (see reverse-face-latent-diffusion cub200_resnet.py)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

_RESNET_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048}


def _load_resnet(arch: str) -> nn.Module:
    arch = arch.lower()
    factory = getattr(models, arch, None)
    if factory is None:
        raise ValueError(f"Unsupported ResNet arch '{arch}'. Choose from: {sorted(_RESNET_DIMS)}")

    try:
        weights_enum = getattr(models, f"{arch.replace('resnet', 'ResNet')}_Weights", None)
        if weights_enum is not None:
            model = factory(weights=weights_enum.IMAGENET1K_V1)
        else:
            model = factory(pretrained=True)
    except TypeError:
        model = factory(pretrained=True)

    # Drop the final FC layer; keep global pooled features only.
    return nn.Sequential(*list(model.children())[:-1])


class ResNetEncoder:
    """Extract ImageNet ResNet features from class-folder datasets."""

    def __init__(self, encfg: dict, device: str):
        arch = str(encfg.get("resnet_arch", "resnet101")).lower()
        self.arch = arch
        self.device = device
        self.model = _load_resnet(arch).to(device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @property
    def embedding_dim(self) -> int:
        return int(_RESNET_DIMS.get(self.arch, 2048))

    def encode_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
        return feats.view(feats.size(0), -1)
