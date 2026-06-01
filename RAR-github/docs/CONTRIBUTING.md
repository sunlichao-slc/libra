# Publishing to GitHub

## CIFAR-100 setup

1. `cp configs/config.example.yaml configs/config.yaml`
2. Set `datasets.cifar100.images_train`, `embeddings_train`, `test_embedding`
3. `pip install -e .`
4. `python scripts/encode/extract_embeddings.py`
5. `python scripts/train/train.py`

## Other datasets

Change `dataset:` and fill the corresponding block under `datasets:` in `config.yaml` (see commented templates in `config.example.yaml`). No code changes.

## Git

```bash
git add .gitignore README.md requirements.txt pyproject.toml configs/ src/ scripts/ docs/
git status   # confirm no data/ or outputs/ staged
git commit -m "CIFAR-100 focused release with unified config"
```
