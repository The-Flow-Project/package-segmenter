"""
Logger configuration for the FLOW Preprocessing Service with loguru.
"""

import contextlib
import sys
from pathlib import Path

from loguru import logger


_VALID_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}


def setup_logger(level: str = "DEBUG", log_files: bool = False) -> None:
    """
    Configure the Loguru logger for the application.

    Args:
        level: Log level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
        log_files: Write logging to files (defaults: False).
    """
    if level.upper() not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid log level: {level!r}. Must be one of {sorted(_VALID_LEVELS)}"
        )
    level = level.upper()

    with contextlib.suppress(ValueError):
        logger.remove(0)

    # Console handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )

    if log_files:
        # File handler for all logs with rotation
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        diagnose = level == "DEBUG"

        logger.add(
            logs_dir / "flow_segmenter.log",
            rotation="5 MB",
            retention="10 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=diagnose,
            enqueue=True,  # Thread-safe logging
        )

        # Separate error log file
        logger.add(
            logs_dir / "flow_segmenter_errors.log",
            rotation="5 MB",
            retention="30 days",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=diagnose,
            enqueue=True,
        )

    logger.debug(f"Logger initialized with level: {level}")
