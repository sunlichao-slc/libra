"""Folder-based metrics for reconstruction quality."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.regression import MeanSquaredError
from transformers import AutoImageProcessor, AutoModel, CLIPModel, ViTImageProcessor, ViTModel

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            rel_no_ext = str(p.relative_to(root).with_suffix(""))
            images[rel_no_ext] = p
    return images


def _match_pairs(reference_dir: Path, generated_dir: Path) -> list[tuple[Path, Path, str]]:
    ref = _collect_images(reference_dir)
    gen = _collect_images(generated_dir)

    pairs: list[tuple[Path, Path, str]] = []
    for key, ref_path in ref.items():
        # Primary rule: exact relative-path+stem match.
        candidates = [key]
        # Fallback for flat outputs named like "<class>_<stem>.png".
        if "/" in key:
            cls, stem = key.split("/", 1)
            candidates.append(f"{cls}_{stem}")
        match_key = next((c for c in candidates if c in gen), None)
        if match_key is not None:
            pairs.append((ref_path, gen[match_key], key))
    return pairs


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor.clamp(0, 1)


def _build_semantic_fn(
    metric: str | None,
    model_dir: str | None,
    device: torch.device,
):
    if metric is None:
        return None, None
    key = metric.lower()
    if key == "clip":
        if not model_dir:
            raise ValueError("semantic metric 'clip' requires --semantic-model-dir")
        processor = AutoImageProcessor.from_pretrained(model_dir)
        model = CLIPModel.from_pretrained(model_dir).to(device).eval()

        def clip_similarity(ref_img: Image.Image, gen_img: Image.Image) -> float:
            inputs = processor(images=[ref_img, gen_img], return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            score = F.cosine_similarity(feats[0:1], feats[1:2]).item()
            return float(score)

        return "CLIP-cos", clip_similarity

    if key == "dino":
        if not model_dir:
            raise ValueError("semantic metric 'dino' requires --semantic-model-dir")
        processor = AutoImageProcessor.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir).to(device).eval()

        def dino_similarity(ref_img: Image.Image, gen_img: Image.Image) -> float:
            inputs = processor(images=[ref_img, gen_img], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs).last_hidden_state[:, 0, :]
            score = F.cosine_similarity(outputs[0:1], outputs[1:2]).item()
            return float(score)

        return "DINO-cos", dino_similarity

    if key in {"dino-vits16", "dino_vits16", "dino-vit-s16"}:
        default_local = "/home/wuyuncheng/RAR/RAR-main/config/dino-vits16"
        local_dir = model_dir or default_local
        processor = None
        model = None
        if local_dir and Path(local_dir).is_dir():
            try:
                print(f"[INFO] Loading DINO ViT-S/16 from local dir: {local_dir}")
                processor = ViTImageProcessor.from_pretrained(local_dir, local_files_only=True)
                model = ViTModel.from_pretrained(local_dir, local_files_only=True).to(device).eval()
            except Exception as e:
                print(f"[WARN] Local DINO ViT-S/16 load failed, fallback to online model: {e}")
        if processor is None or model is None:
            processor = ViTImageProcessor.from_pretrained("facebook/dino-vits16")
            model = ViTModel.from_pretrained("facebook/dino-vits16").to(device).eval()

        def dino_vits16_similarity(ref_img: Image.Image, gen_img: Image.Image) -> float:
            with torch.no_grad():
                xa = processor(images=ref_img, return_tensors="pt")["pixel_values"].to(device)
                xb = processor(images=gen_img, return_tensors="pt")["pixel_values"].to(device)
                fa = model(pixel_values=xa).last_hidden_state[:, 0]
                fb = model(pixel_values=xb).last_hidden_state[:, 0]
                fa = F.normalize(fa, dim=1)
                fb = F.normalize(fb, dim=1)
                cos = (fa * fb).sum(dim=1).item()
            return float(cos)

        return "DINO-cos", dino_vits16_similarity

    raise ValueError(
        "Unknown semantic metric "
        f"'{metric}', choose from: clip, dino, dino-vits16"
    )


def _summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.array(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def evaluate_folders(
    *,
    reference_dir: str,
    generated_dir: str,
    device: str = "auto",
    semantic_metric: str | None = None,
    semantic_model_dir: str | None = None,
    limit: int = 0,
) -> dict:
    device_name = "cuda" if device == "auto" and torch.cuda.is_available() else device
    torch_device = torch.device(device_name)

    reference_path = Path(reference_dir).expanduser().resolve()
    generated_path = Path(generated_dir).expanduser().resolve()
    if not reference_path.is_dir():
        raise FileNotFoundError(f"Reference folder not found: {reference_path}")
    if not generated_path.is_dir():
        raise FileNotFoundError(f"Generated folder not found: {generated_path}")

    pairs = _match_pairs(reference_path, generated_path)
    if limit > 0:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError("No matched image pairs found by relative path + stem.")

    mse = MeanSquaredError().to(torch_device).eval()
    psnr = PSNR(data_range=1.0).to(torch_device).eval()
    ssim = SSIM(data_range=1.0).to(torch_device).eval()
    lpips = LPIPS(net_type="alex").to(torch_device).eval()
    semantic_name, semantic_fn = _build_semantic_fn(semantic_metric, semantic_model_dir, torch_device)

    records: list[dict] = []
    metric_values: dict[str, list[float]] = {
        "MSE": [],
        "PSNR": [],
        "SSIM": [],
        "LPIPS": [],
    }
    if semantic_fn and semantic_name:
        metric_values[semantic_name] = []

    for ref_path, gen_path, pair_key in pairs:
        ref = _load_image(ref_path, torch_device)
        gen = _load_image(gen_path, torch_device)
        if ref.shape != gen.shape:
            gen = F.interpolate(gen, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        with torch.no_grad():
            result = {
                "MSE": float(mse(gen, ref).cpu()),
                "PSNR": float(psnr(gen, ref).cpu()),
                "SSIM": float(ssim(gen, ref).cpu()),
                "LPIPS": float(lpips(gen, ref).cpu()),
            }

        if semantic_fn:
            ref_img = Image.open(ref_path).convert("RGB")
            gen_img = Image.open(gen_path).convert("RGB")
            result[semantic_name] = semantic_fn(ref_img, gen_img)

        for key, val in result.items():
            metric_values[key].append(val)
        records.append(
            {
                "pair": pair_key,
                "reference": str(ref_path),
                "generated": str(gen_path),
                "metrics": result,
            }
        )

    summary = {name: _summarize(vals) for name, vals in metric_values.items()}
    return {
        "reference_dir": str(reference_path),
        "generated_dir": str(generated_path),
        "device": device_name,
        "pairs_evaluated": len(records),
        "summary": summary,
        "results": records,
    }

