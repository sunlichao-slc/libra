#!/usr/bin/env python3
"""Extract encoder embeddings for CIFAR-100 (class-folder layout).

Dispatches to clip / dinov2 / resnet based on encoder.type in config.yaml,
or override with --encoder.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from rar_diffusion.extract import main

if __name__ == "__main__":
    main()
