"""
Device detection and management.
Auto-detects CUDA, Apple MPS, or falls back to CPU.
Supports override via DEVICE environment variable.

Usage:
    from src.utils.device import get_device
    device = get_device()             # auto-detect
    device = get_device("cuda")       # force CUDA
"""

import os
import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device(override: str = None) -> torch.device:
    """
    Detect the best available compute device.

    Priority: override > env var > CUDA > MPS > CPU

    Args:
        override: Force a specific device ("cuda", "mps", "cpu")

    Returns:
        torch.device instance
    """
    # Check for override
    device_str = override or os.getenv("DEVICE", "auto")

    if device_str != "auto":
        device = torch.device(device_str)
        logger.info(f"Using device (override): {device}")
        return device

    # Auto-detect
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using device: CUDA — {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU (no GPU detected)")

    return device


def log_device_info():
    """Log detailed device information for debugging."""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {props.name} — "
                f"{props.total_memory / 1e9:.1f} GB, "
                f"Compute capability: {props.major}.{props.minor}"
            )

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    logger.info(f"MPS available: {mps_available}")