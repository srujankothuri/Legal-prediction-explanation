"""
YAML configuration loader.
All hyperparameters and paths are defined in configs/ YAML files.
This module loads them into a simple dictionary with dot-access support.

Usage:
    from src.utils.config import load_config
    cfg = load_config("configs/models/xlnet_bigru.yaml")
    print(cfg["encoder"]["pretrained"])   # "xlnet-base-cased"
    print(cfg.encoder.pretrained)         # also works with dot access
"""

import os
import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigDict(dict):
    """Dictionary subclass that supports dot-notation access."""

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


def load_config(config_path: str) -> ConfigDict:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to .yaml config file

    Returns:
        ConfigDict with parsed configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    config = ConfigDict(raw_config)
    logger.info(f"Loaded config from {config_path}")
    return config


def merge_configs(*config_paths: str) -> ConfigDict:
    """
    Load and merge multiple config files. Later files override earlier ones.

    Args:
        config_paths: Paths to .yaml config files

    Returns:
        Merged ConfigDict
    """
    merged = ConfigDict()
    for path in config_paths:
        cfg = load_config(path)
        _deep_update(merged, cfg)
    return merged


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update base dict with values from update dict."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base