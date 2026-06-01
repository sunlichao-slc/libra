# Third-party dependencies

This folder is optional. Keep your local model assets here only if convenient.

No external repository clone is required by default. You can point paths directly in
`configs/config.yaml`:

- `encoder.model_dir` for CLIP / DINOv2
- `models.vqvae` for VQ-VAE

Optional baselines (not required for core training scripts):

- **DRAG-main** — defense / reconstruction baselines
- **reverse-face-latent-diffusion-main** — legacy scripts (equivalent tools live under `scripts/`)
