#!/usr/bin/env python3
"""Unified evaluation entrypoint for reconstructed image folders."""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from rar_diffusion.metrics import evaluate_folders


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", required=True, help="Ground-truth image folder")
    p.add_argument("--generated-dir", required=True, help="Recovered/generated image folder")
    p.add_argument("--output", default="outputs/metrics/evaluate.json", help="Output JSON path")
    p.add_argument("--device", default="auto", help="auto/cuda/cpu")
    p.add_argument(
        "--semantic-metric",
        choices=["clip", "dino", "dino-vits16"],
        default=None,
        help="Optional semantic metric backend",
    )
    p.add_argument(
        "--semantic-model-dir",
        default=None,
        help="Model directory for semantic metric (required when semantic metric is enabled)",
    )
    p.add_argument("--limit", type=int, default=0, help="Evaluate at most N pairs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results = evaluate_folders(
        reference_dir=args.reference_dir,
        generated_dir=args.generated_dir,
        device=args.device,
        semantic_metric=args.semantic_metric,
        semantic_model_dir=args.semantic_model_dir,
        limit=args.limit,
    )

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Pairs: {results['pairs_evaluated']}")
    for metric_name, stats in results["summary"].items():
        print(
            f"{metric_name}: "
            f"{stats['mean']:.6f} ± {stats['std']:.6f} "
            f"[{stats['min']:.6f}, {stats['max']:.6f}]"
        )
    print(f"Saved metrics JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

