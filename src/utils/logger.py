"""
Centralized logging configuration.
All modules import their logger from here for consistent formatting.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import os
import sys
import logging
from datetime import datetime


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Create a configured logger instance.

    Args:
        name: Logger name (typically __name__ of calling module)
        log_to_file: Whether to also write logs to logs/ directory

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # ── Console Handler ─────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # ── File Handler ────────────────────────────────────────────────────────
    if log_to_file:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
        os.makedirs(log_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"{date_str}.log")

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger