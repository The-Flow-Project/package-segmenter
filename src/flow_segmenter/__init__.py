"""
Flow Segmenter Package Initialization
"""

from .baseline_utils import BaselineUtils
from .config import SegmenterConfig
from .exceptions import (
    EmptyCollectionError,
    InvalidImageError,
    InvalidXMLError,
    ModelLoadError,
    PageNotFoundError,
    SegmentationError,
)
from .segmenter import SegmenterYOLO
from .xml_utils import XMLUtils

__version__ = "0.2.0"
__license__ = "MIT"
__authors__ = ["l0rn0r"]

__all__ = [
    "SegmenterConfig",
    "SegmenterYOLO",
    "SegmentationError",
    "PageNotFoundError",
    "InvalidImageError",
    "ModelLoadError",
    "InvalidXMLError",
    "EmptyCollectionError",
    "XMLUtils",
    "BaselineUtils",
]
