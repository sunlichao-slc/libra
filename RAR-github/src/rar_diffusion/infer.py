"""Image recovery pipeline (class-wise random embedding sampling)."""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel

from rar_diffusion.embeddings import ensure_3d_embedding


@dataclass
class RecoverConfig:
    image_size: int
    device: str
    unet_path: str
    vqvae_path: str
    embedding_root: str
    image_root: str
    output_dir: str
    num_steps: int
    limit: int
    seed: int
    grid_nrow: int


def parse_args():
    p = argparse.ArgumentParser(description="Recover images from embeddings and save grids.")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint", required=True, help="UNet checkpoint directory")
    p.add_argument("--vqvae-path", required=True, help="VQ-VAE directory")
    p.add_argument("--embeddings-dir", required=True, help="Class-folder embedding root")
    p.add_argument("--image-root", required=True, help="Class-folder original image root")
    p.add_argument("--output-dir", required=True, help="Output directory for recovered images/grids")
    p.add_argument("--num-steps", type=int, default=1000, help="Reverse diffusion steps")
    p.add_argument("--limit", type=int, default=0, help="Recover at most N classes (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-nrow", type=int, default=10)
    return p.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(path, image_size: int):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    return transform(img)


def load_unet_model(config: RecoverConfig):
    """
    Load UNet with two compatible formats:
    1) save_pretrained directory
    2) torch.save pickle state_dict stored as diffusion_pytorch_model.safetensors
    """
    model_path = config.unet_path
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
    try:
        print(f"Trying from_pretrained: {model_path}")
        unet = UNet2DConditionModel.from_pretrained(model_path).to(config.device).eval()
        print("Loaded UNet via from_pretrained")
        return unet
    except Exception as e:
        print(f"from_pretrained failed: {e}")
        print("Trying torch.load fallback...")
        try:
            unet = UNet2DConditionModel.from_config(model_path)
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            unet.load_state_dict(state_dict, strict=False)
            unet = unet.to(config.device).eval()
            print("Loaded UNet via torch.load fallback")
            return unet
        except Exception as e2:
            raise RuntimeError(
                "Failed to load UNet checkpoint.\n"
                f"from_pretrained error: {e}\n"
                f"torch.load error: {e2}"
            ) from e2


def _embedding_to_image_name(embedding_file: str) -> str:
    stem = embedding_file.replace(".npy", "")
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{'_'.join(parts[:-1])}_{parts[-1]}.png"
    return embedding_file.replace(".npy", ".png")


def generate_images_and_compare(config: RecoverConfig):
    unet = load_unet_model(config)
    vqvae = VQModel.from_pretrained(config.vqvae_path).to(config.device).eval()

    noise_scheduler = DDPMScheduler(
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        num_train_timesteps=1000,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_root = Path(config.embedding_root)
    image_root = Path(config.image_root)
    if not embedding_root.is_dir():
        raise FileNotFoundError(f"Embedding root directory not found: {embedding_root}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root directory not found: {image_root}")

    all_recons, all_orig = [], []
    class_folders = sorted([d for d in embedding_root.iterdir() if d.is_dir()])
    if config.limit > 0:
        class_folders = class_folders[: config.limit]
    print(f"Found {len(class_folders)} classes.")

    with torch.no_grad():
        dummy = torch.randn(1, 3, config.image_size, config.image_size, device=config.device)
        latent_shape = vqvae.encode(dummy).latents.shape

    for class_dir in class_folders:
        class_name = class_dir.name
        image_class_path = image_root / class_name
        npy_files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
        if not npy_files:
            continue

        selected = random.choice(npy_files)
        npy_path = class_dir / selected
        jpg_path = image_class_path / _embedding_to_image_name(selected)

        if not jpg_path.exists():
            print(f"Missing original image: {jpg_path}")
            continue

        embedding = torch.from_numpy(np.load(npy_path)).float()
        embedding = ensure_3d_embedding(embedding).to(config.device)

        img_start_time = time.time()
        with torch.no_grad():
            latents = torch.randn(latent_shape, device=config.device)
            total_steps = int(noise_scheduler.config.num_train_timesteps)
            steps = max(1, int(config.num_steps))
            stride = max(1, total_steps // steps)
            schedule = list(range(total_steps - 1, -1, -stride))

            for t in schedule:
                t_tensor = torch.tensor([t], device=config.device)
                model_output = unet(
                    sample=latents,
                    timestep=t_tensor,
                    encoder_hidden_states=embedding,
                ).sample
                latents = noise_scheduler.step(model_output, t, latents, return_dict=True).prev_sample

            recon = vqvae.decode(latents).sample.squeeze(0).cpu()
            recon_img = (recon * 0.5 + 0.5).clamp(0, 1)

        img_end_time = time.time()
        print(f"Time for {class_name}/{selected}: {img_end_time - img_start_time:.2f}s")

        orig_img = load_image(str(jpg_path), config.image_size)
        if orig_img.shape != recon_img.shape:
            recon_img = T.Resize(orig_img.shape[1:])(recon_img)

        output_image_name = f"{class_name}_{selected.replace('.npy', '.png')}"
        output_image_path = output_dir / output_image_name
        vutils.save_image(recon_img, str(output_image_path))
        print(f"Saved image: {output_image_path}")

        all_recons.append(recon_img)
        all_orig.append(orig_img)

    if len(all_orig) == 0:
        print("No images were successfully processed. Check paths and naming.")
        return

    grid_orig = vutils.make_grid(torch.stack(all_orig), nrow=config.grid_nrow, padding=2)
    orig_grid_path = output_dir / "grid_original.jpg"
    vutils.save_image(grid_orig, str(orig_grid_path))
    print(f"Saved original grid: {orig_grid_path}")

    grid_recon = vutils.make_grid(torch.stack(all_recons), nrow=config.grid_nrow, padding=2)
    recon_grid_path = output_dir / "grid_reconstructed.jpg"
    vutils.save_image(grid_recon, str(recon_grid_path))
    print(f"Saved reconstructed grid: {recon_grid_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = RecoverConfig(
        image_size=args.image_size,
        device=args.device,
        unet_path=args.checkpoint,
        vqvae_path=args.vqvae_path,
        embedding_root=args.embeddings_dir,
        image_root=args.image_root,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        limit=args.limit,
        seed=args.seed,
        grid_nrow=args.grid_nrow,
    )
    generate_images_and_compare(cfg)

