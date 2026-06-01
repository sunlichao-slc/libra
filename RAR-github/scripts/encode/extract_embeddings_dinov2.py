#!/usr/bin/env python3
"""Extract DINOv2 embeddings for CIFAR-100 (class-folder layout).

Reference: reverse-face-latent-diffusion-main/encode_embedding/cifar100_embedding_dino.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from rar_diffusion.extract import main

if __name__ == "__main__":
    main(encoder_type="dinov2")
