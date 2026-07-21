"""config_loader.py -- loads the single config.yaml shared by
curriculum_builder.py, dataset.py, and train.py."""
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'config.yaml'


def load_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault('paths', {})
    cfg.setdefault('data', {})
    cfg.setdefault('val', {})
    cfg.setdefault('curriculum', {})
    cfg.setdefault('dynamic_batch', {})
    cfg.setdefault('train_scale', {})
    return cfg