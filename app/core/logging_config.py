"""Logging setup for DocMind."""

import logging

from app.core.config import settings


def setup_logging() -> logging.Logger:
    """Configure application logging and return the docmind logger."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("docmind")
    logger.info("DocMind logging initialized (level=%s)", settings.LOG_LEVEL)
    return logger
