"""
Logger configuration for the FLOW Preprocessing Service with loguru.
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(level: str = "DEBUG") -> None:
    """
    Configure the Loguru logger for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove default handler
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | " \
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )

    # File handler for all logs with rotation
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    if level == "DEBUG":
        diagnose = True
    else:
        diagnose = False

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


