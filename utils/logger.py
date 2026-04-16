"""
utils/logger.py – Centralised structured logger

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Model trained", algorithm="SVD", rmse=0.82)
"""
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config.settings import LOG_LEVEL, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with console + rotating file handlers.
    Safe to call multiple times with the same name — returns the same instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger   # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ────────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Rotating file handler ──────────────────────────────────────────────────
    try:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass   # if log dir is read-only, just use console

    logger.propagate = False
    return logger
