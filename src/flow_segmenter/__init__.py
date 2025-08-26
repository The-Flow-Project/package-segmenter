"""
Flow Segmenter Package Initialization
"""
from .config import SegmenterConfig
from .segmenter import SegmenterYOLO

__version__ = "0.1.5"
__license__ = "MIT"
__authors__ = ["l0rn0r"]

__all__ = [
    "SegmenterConfig",
    "SegmenterYOLO"
]