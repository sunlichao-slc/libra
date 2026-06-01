"""Load paths and training options from configs/config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None


def get_project_root() -> Path:
    env_root = os.environ.get("RAR_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _config_path() -> Path:
    env_cfg = os.environ.get("RAR_CONFIG") or os.environ.get("RAR_PATHS_CONFIG")
    if env_cfg:
        return Path(env_cfg).expanduser().resolve()
    root = get_project_root()
    for name in ("config.yaml", "paths.yaml"):
        candidate = root / "configs" / name
        if candidate.is_file():
            return candidate
    return root / "configs" / "config.yaml"


def load_config() -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    cfg_path = _config_path()
    if not cfg_path.is_file():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    root = Path(raw.get("project_root", "."))
    if not root.is_absolute():
        root = (get_project_root() / root).resolve()
    raw["project_root"] = str(root)
    _CONFIG_CACHE = raw
    return _CONFIG_CACHE


def _expand_ctx(cfg: dict[str, Any]) -> dict[str, str]:
    rar = str(cfg.get("rar_config_dir", "third_party/models"))
    if not Path(rar).is_absolute():
        rar = str((Path(cfg["project_root"]) / rar).resolve())
    return {
        "project_root": cfg.get("project_root", str(get_project_root())),
        "rar_config_dir": rar,
    }


def _expand_value(value: str, ctx: dict[str, str]) -> str:
    prev = None
    cur = value
    while prev != cur:
        prev = cur
        for key, val in ctx.items():
            cur = cur.replace(f"${{{key}}}", val)
    return cur


def _resolve_string(value: str, cfg: dict[str, Any]) -> Path:
    ctx = _expand_ctx(cfg)
    value = _expand_value(str(value), ctx)
    path = Path(value)
    if not path.is_absolute():
        path = Path(cfg["project_root"]) / path
    return path.resolve()


def resolve_path(*keys: str, default: str | None = None) -> Path:
    """Resolve nested key, e.g. resolve_path('models', 'vqvae')."""
    cfg = load_config()
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            node = None
            break
        node = node[key]

    if node is None:
        if default is None:
            dotted = ".".join(keys)
            raise KeyError(
                f"Missing config key '{dotted}'. "
                f"Copy configs/config.example.yaml to configs/config.yaml."
            )
        node = default
    return _resolve_string(str(node), cfg)


def active_dataset_name() -> str:
    return load_config().get("dataset", "cifar100")


def active_dataset_cfg() -> dict[str, Any]:
    cfg = load_config()
    name = active_dataset_name()
    datasets = cfg.get("datasets", {})
    if name not in datasets:
        raise KeyError(
            f"dataset '{name}' not in config.datasets. "
            f"Available: {list(datasets.keys())}"
        )
    return datasets[name]


def get_train_cfg() -> dict[str, Any]:
    return load_config().get("train", {})


def get_encoder_cfg() -> dict[str, Any]:
    return load_config().get("encoder", {})


def get_models_cfg() -> dict[str, Any]:
    return load_config().get("models", {})


def dataset_path(key: str, *, required: bool = True) -> Path:
    """Path from the active dataset block, e.g. dataset_path('images_train')."""
    ds = active_dataset_cfg()
    if key not in ds:
        if not required:
            return Path()
        raise KeyError(
            f"Key '{key}' missing for dataset '{active_dataset_name()}'. "
            f"Available: {list(ds.keys())}"
        )
    cfg = load_config()
    return _resolve_string(str(ds[key]), cfg)
