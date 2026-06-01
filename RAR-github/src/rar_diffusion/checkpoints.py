"""Save/load helpers for diffusion denoisers."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel

from rar_diffusion.models.resnet_unet import create_resnet_diffusion_model


def save_denoiser(model, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(model, UNet2DConditionModel):
        denoiser_dir = output_path / "denoiser"
        model.save_pretrained(str(denoiser_dir))
        return denoiser_dir

    state_path = output_path / "denoiser.pt"
    torch.save(model.state_dict(), state_path)
    meta_path = output_path / "denoiser_meta.json"
    meta_path.write_text(
        json.dumps({"format": "state_dict", "backbone": "resnet"}, indent=2),
        encoding="utf-8",
    )
    return state_path


def load_denoiser(
    *,
    backbone: str,
    checkpoint_dir: str | Path,
    cross_attention_dim: int,
    device: str,
):
    checkpoint_dir = Path(checkpoint_dir)
    backbone = backbone.lower()

    if backbone == "unet":
        if (checkpoint_dir / "config.json").is_file():
            source = checkpoint_dir
        elif (checkpoint_dir / "denoiser").is_dir():
            source = checkpoint_dir / "denoiser"
        else:
            raise FileNotFoundError(
                f"UNet checkpoint missing under {checkpoint_dir}; "
                "expected config.json or denoiser/."
            )
        return UNet2DConditionModel.from_pretrained(str(source)).to(device).eval()

    if backbone == "resnet":
        model = create_resnet_diffusion_model(
            in_channels=3,
            out_channels=3,
            model_channels=256,
            num_res_blocks=2,
            attention_resolutions=(32, 16, 8),
            channel_mult=(1, 2, 4, 8),
            cross_attention_dim=cross_attention_dim,
            only_cross_attention=True,
        )
        state_path = checkpoint_dir / "denoiser.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"ResNet checkpoint not found: {state_path}")
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
        return model.to(device).eval()

    raise ValueError(f"Unsupported backbone '{backbone}'")

