"""Configuration loader for MQL4 to MQL5 converter.

Loads YAML config files and resolves relative paths against project root.
"""

import os
import sys
import torch
import yaml
from types import SimpleNamespace


def load_config(config_path):
    """Load a YAML config file and return as a nested SimpleNamespace.

    Resolves all relative data paths against the project root.
    """
    project_root = _find_project_root()

    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    cfg = _to_namespace(raw)
    cfg._project_root = project_root
    cfg._config_path = os.path.abspath(config_path)

    _resolve_paths(cfg)
    return cfg


def resolve_device(cfg):
    """Return device string from config ('auto' -> cuda/cpu)."""
    device = getattr(cfg, 'device', 'auto')
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def _find_project_root():
    """Walk up from this file to find the directory containing pyproject.toml."""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.exists(os.path.join(current, 'pyproject.toml')):
            return current
        current = os.path.dirname(current)
    raise RuntimeError(
        "Could not find project root (pyproject.toml not found). "
        "Ensure pyproject.toml exists in the project root."
    )


def _to_namespace(d):
    """Recursively convert dict to SimpleNamespace."""
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        d[k] = _to_namespace(v)
    return SimpleNamespace(**d)


def _resolve_paths(cfg):
    """Resolve relative data paths in the config to absolute paths."""
    data = getattr(cfg, 'data', None)
    if data is None:
        return

    root = cfg._project_root

    raw_dir = getattr(data, 'raw_dir', None)
    processed_dir = getattr(data, 'processed_dir', None)
    checkpoint_dir = getattr(data, 'checkpoint_dir', 'checkpoints')

    if raw_dir and not os.path.isabs(raw_dir):
        data.raw_dir = os.path.join(root, raw_dir)
    if processed_dir and not os.path.isabs(processed_dir):
        data.processed_dir = os.path.join(root, processed_dir)
    if not os.path.isabs(checkpoint_dir):
        data.checkpoint_dir = os.path.join(root, checkpoint_dir)

    # Resolve individual file keys
    for key in ('input_jsonl', 'output_pkl', 'train_file', 'val_file'):
        value = getattr(data, key, None)
        if value and not os.path.isabs(value):
            parent = data.processed_dir if key != 'input_jsonl' else data.raw_dir
            setattr(data, key, os.path.join(parent, value))

    # Also resolve output_dir if present (legacy)
    output_dir = getattr(data, 'output_dir', None)
    if output_dir and not os.path.isabs(output_dir):
        data.output_dir = os.path.join(root, output_dir)