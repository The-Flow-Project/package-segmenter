"""
Tests for flow_segmenter.exceptions module.
"""

import pytest

from flow_segmenter.exceptions import (
    EmptyCollectionError,
    InvalidImageError,
    InvalidXMLError,
    ModelLoadError,
    PageNotFoundError,
    SegmentationError,
)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_segmentation_error_is_exception(self):
        """Test that SegmentationError inherits from Exception."""
        assert issubclass(SegmentationError, Exception)

    def test_page_not_found_error_inherits_segmentation_error(self):
        """Test PageNotFoundError inheritance."""
        assert issubclass(PageNotFoundError, SegmentationError)

    def test_invalid_image_error_inherits_segmentation_error(self):
        """Test InvalidImageError inheritance."""
        assert issubclass(InvalidImageError, SegmentationError)

    def test_invalid_xml_error_inherits_segmentation_error(self):
        """Test InvalidXMLError inheritance."""
        assert issubclass(InvalidXMLError, SegmentationError)

    def test_empty_collection_error_inherits_segmentation_error(self):
        """Test EmptyCollectionError inheritance."""
        assert issubclass(EmptyCollectionError, SegmentationError)

    def test_model_load_error_inherits_segmentation_error(self):
        """Test ModelLoadError inheritance."""
        assert issubclass(ModelLoadError, SegmentationError)


class TestExceptionRaising:
    """Test that exceptions can be raised with messages."""

    def test_segmentation_error_with_message(self):
        """Test raising SegmentationError with message."""
        with pytest.raises(SegmentationError, match="Test error"):
            raise SegmentationError("Test error")

    def test_page_not_found_error_with_message(self):
        """Test raising PageNotFoundError with message."""
        with pytest.raises(PageNotFoundError, match="Page not found"):
            raise PageNotFoundError("Page not found")

    def test_invalid_image_error_with_message(self):
        """Test raising InvalidImageError with message."""
        with pytest.raises(InvalidImageError, match="Invalid image"):
            raise InvalidImageError("Invalid image")

    def test_invalid_xml_error_with_message(self):
        """Test raising InvalidXMLError with message."""
        with pytest.raises(InvalidXMLError, match="Invalid XML"):
            raise InvalidXMLError("Invalid XML")

    def test_empty_collection_error_with_message(self):
        """Test raising EmptyCollectionError with message."""
        with pytest.raises(EmptyCollectionError, match="Empty collection"):
            raise EmptyCollectionError("Empty collection")

    def test_model_load_error_with_message(self):
        """Test raising ModelLoadError with message."""
        with pytest.raises(ModelLoadError, match="Model load failed"):
            raise ModelLoadError("Model load failed")


class TestExceptionCatching:
    """Test that specific exceptions can be caught as base exception."""

    def test_catch_page_not_found_as_segmentation_error(self):
        """Test catching PageNotFoundError as SegmentationError."""
        with pytest.raises(SegmentationError):
            raise PageNotFoundError("Test")

    def test_catch_invalid_image_as_segmentation_error(self):
        """Test catching InvalidImageError as SegmentationError."""
        with pytest.raises(SegmentationError):
            raise InvalidImageError("Test")

    def test_catch_specific_exception_type(self):
        """Test catching specific exception type."""
        try:
            raise InvalidXMLError("XML problem")
        except InvalidXMLError:
            pytest.raises(InvalidXMLError, match="XML problem")
        except SegmentationError:
            pytest.fail("Should catch specific exception first")
