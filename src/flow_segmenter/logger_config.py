"""
Logger configuration for the FLOW Preprocessing Service with loguru.
"""

import sys

from loguru import logger

# setup_logger() is intended for standalone use only, not when imported as a library.
logger.disable("flow_segmenter")


def setup_logger(level: str = "DEBUG") -> None:
    """
    Configure the Loguru logger for the application.

    Args:
        level: Log level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
    """
    logger.remove()

    diagnose = level == "DEBUG"

    # Console handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=diagnose,
        enqueue=False,  # stderr is fast; no need for async queue
    )

    logger.enable("flow_segmenter")

    logger.debug(f"Logger initialized with level: {level}")
