#!/usr/bin/env bash
# Export GitHub-ready copy to ../RAR-github
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$(dirname "$ROOT")/RAR-github}"

rm -rf "$DEST"
mkdir -p "$DEST"

rsync -a \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.vscode/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude 'RAR-main/' \
  --exclude 'DRAG-main/' \
  --exclude 'reverse-face-latent-diffusion-main/' \
  --exclude 'Imagenet/' \
  --exclude 'Imagenet_npy/' \
  --exclude 'cifar/' \
  --exclude 'cifar100/' \
  --exclude 'lfw_gene/' \
  --exclude 'embedding_clip/' \
  --exclude 'embedding_resnet/' \
  --exclude 'testembedding/' \
  --exclude 'data/' \
  --exclude 'ddpm-retrain*/' \
  --exclude '/models/' \
  --exclude 'output/' \
  --exclude 'eval_images*/' \
  --exclude 'tensorboard/' \
  --exclude 'wandb/' \
  --exclude '*.log' \
  --exclude '*.npy' \
  --exclude '*.npz' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.ckpt' \
  --exclude '*.safetensors' \
  --exclude '*.bin' \
  --exclude '*.jpg' \
  --exclude '*.jpeg' \
  --exclude '*.png' \
  --exclude '*.gif' \
  --exclude '=2.6' \
  --exclude 'optimized_image.jpg' \
  --exclude 'configs/config.yaml' \
  --exclude 'configs/paths.yaml' \
  --exclude '.env' \
  --exclude 'venv/' \
  --exclude '.venv/' \
  --exclude 'test.ipynb' \
  --exclude 'readme.txt' \
  --exclude 'test_resnet_model.py' \
  --exclude 'RAR-github/' \
  "$ROOT/" "$DEST/"

if [[ -f "$ROOT/docs/GITHUB_EXPORT.md" ]]; then
  cp "$ROOT/docs/GITHUB_EXPORT.md" "$DEST/GITHUB_EXPORT.md"
fi
chmod +x "$DEST/scripts/export_for_github.sh" 2>/dev/null || true

echo "Exported to: $DEST"
du -sh "$DEST"
echo "Files: $(find "$DEST" -type f | wc -l)"
