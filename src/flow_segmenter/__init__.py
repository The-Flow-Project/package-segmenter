"""
Flow Segmenter Package Initialization
"""

from .baseline_utils import BaselineUtils
from .config import SegmenterConfig, SegmenterBaseConfig
from .exceptions import (
    EmptyCollectionError,
    InvalidImageError,
    InvalidXMLError,
    ModelLoadError,
    PageNotFoundError,
    SegmentationError,
)
from .segmenter import SegmenterKrakenLinemasks, SegmenterYolo
from .xml_utils import XMLUtils

__version__ = "0.2.0"
__license__ = "MIT"
__authors__ = ["l0rn0r"]

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
]
