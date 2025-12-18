"""
Custom exceptions for the flow_segmenter package.
"""


class SegmentationError(Exception):
    """Base exception for segmentation-related errors."""

    pass


class PageNotFoundError(SegmentationError):
    """Raised when a <Page> element is not found in XML."""

    pass


class InvalidImageError(SegmentationError):
    """Raised when an image cannot be loaded or is invalid."""

    pass


class ModelLoadError(SegmentationError):
    """Raised when a model cannot be loaded."""

    pass


class InvalidXMLError(SegmentationError):
    """Raised when XML structure is invalid or cannot be parsed."""

    pass


class EmptyCollectionError(SegmentationError):
    """Raised when no pages are found in the collection."""

    pass
