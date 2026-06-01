#!/usr/bin/env bash
# Optional one-time physical move of legacy scripts into scripts/{train,encode,eval}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p scripts/train scripts/encode scripts/eval src/rar_diffusion/models

move() {
  local src="$1" dest="$2"
  if [[ -f "$src" && ! -f "$dest" ]]; then
    git mv "$src" "$dest" 2>/dev/null || mv "$src" "$dest"
    echo "moved $src -> $dest"
  fi
}

move diffuse_cifar100_clip.py scripts/train/cifar100_clip_impl.py
move diffuse_cifar100_resnet18.py scripts/train/cifar100_resnet_impl.py
move diffusers_celeba_clip.py scripts/train/celeba_clip_impl.py
move diffusers_cub200_clip.py scripts/train/cub200_clip_impl.py
move diffusers_cub200_resnet.py scripts/train/cub200_resnet_impl.py
move diffuser_cub200_clip_gan.py scripts/train/cub200_clip_gan_impl.py
move diffusers_clip_cifar100_pick.py scripts/train/cifar100_clip_pick_impl.py
move duffusers_cifar100.py scripts/train/cifar100_dino_impl.py

move cifar_metrics.py scripts/eval/cifar_metrics_impl.py
move compare_folders.py scripts/eval/compare_folders_impl.py
move predict_face.py scripts/eval/predict_face_impl.py
move resnet_diffusion_model.py src/rar_diffusion/models/resnet_unet.py

if [[ -d reverse-face-latent-diffusion-main/encode_embedding ]]; then
  cp -n reverse-face-latent-diffusion-main/encode_embedding/*.py scripts/encode/ || true
fi

echo "Done. Update scripts/train/* wrappers to point at *_impl.py if you ran this."
