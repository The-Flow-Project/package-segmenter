"""
Flow Segmenter Package Initialization
"""

from .baseline_utils import BaselineUtils
from .config import SegmenterBaseConfig, SegmenterConfig
from .exceptions import (
    EmptyCollectionError,
    InvalidImageError,
    InvalidXMLError,
    ModelLoadError,
    PageNotFoundError,
    SegmentationError,
)
from .logger_config import setup_logger
from .segmenter import SegmenterKrakenLinemasks, SegmenterYolo
from .xml_utils import XMLUtils

__version__ = "0.3.0"
__license__ = "MIT"
__authors__ = ["jnswidmer"]

__all__ = [
    "SegmenterConfig",
    "SegmenterBaseConfig",
    "SegmenterKrakenLinemasks",
    "SegmenterYolo",
    "SegmentationError",
    "PageNotFoundError",
    "InvalidImageError",
    "ModelLoadError",
    "InvalidXMLError",
    "EmptyCollectionError",
    "XMLUtils",
    "BaselineUtils",
    "setup_logger",
]
