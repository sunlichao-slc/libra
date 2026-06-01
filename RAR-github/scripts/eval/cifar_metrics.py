#!/usr/bin/env python3
"""Backward-compatible wrapper around scripts/eval/evaluate.py."""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from rar_diffusion.metrics import evaluate_folders


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dir", required=True, help="Generated/eval images folder")
    p.add_argument("--original_dir", required=True, help="Original/reference images folder")
    p.add_argument("--output", default="outputs/metrics/cifar_metrics.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--semantic_metric", choices=["clip", "dino"], default=None)
    p.add_argument("--semantic_model_dir", default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    results = evaluate_folders(
        reference_dir=args.original_dir,
        generated_dir=args.eval_dir,
        device=args.device,
        semantic_metric=args.semantic_metric,
        semantic_model_dir=args.semantic_model_dir,
        limit=args.limit,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    import json

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
