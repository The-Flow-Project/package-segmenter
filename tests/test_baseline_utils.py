"""
Tests for flow_segmenter.baseline_utils module.
"""

from unittest.mock import Mock, patch

import lxml.etree as et
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from flow_segmenter.baseline_utils import BaselineUtils
from flow_segmenter.exceptions import InvalidImageError
from tests.fixtures.mock_data import (
    create_mock_xml_with_baselines,
)


class TestLoadImageGrayscale:
    """Test BaselineUtils.load_image_grayscale()."""

    def test_load_valid_image(self, mock_image_file):
        """Test loading a valid image file."""
        img = BaselineUtils.load_image_grayscale(mock_image_file)

        assert img is not None
        assert img.mode == "L"  # Grayscale

    def test_load_nonexistent_image_raises_error(self):
        """Test that loading nonexistent image raises InvalidImageError."""
        with pytest.raises(InvalidImageError, match="Cannot open"):
            BaselineUtils.load_image_grayscale("/nonexistent/image.jpg")

    def test_load_invalid_path_raises_error(self):
        """Test that invalid path raises InvalidImageError."""
        with pytest.raises(InvalidImageError):
            BaselineUtils.load_image_grayscale("")


class TestExtractMasksFromXML:
    """Test BaselineUtils.extract_masks_from_xml()."""

    def test_extract_masks_from_xml_with_textlines(
        self, mock_xml_with_textlines, mock_namespace
    ):
        """Test extracting masks from XML with TextLine elements."""
        masks = BaselineUtils.extract_masks_from_xml(
            mock_xml_with_textlines, mock_namespace
        )

        assert len(masks) > 0
        assert all(isinstance(mask, Polygon) for mask in masks)

    def test_extract_masks_returns_list(self, mock_xml_with_textlines, mock_namespace):
        """Test that extract_masks returns a list."""
        masks = BaselineUtils.extract_masks_from_xml(
            mock_xml_with_textlines, mock_namespace
        )
        assert isinstance(masks, list)

    def test_extract_masks_from_xml_without_textlines(self, mock_namespace):
        """Test extracting masks from XML without TextLines."""
        xml = et.fromstring(
            b'<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
            b'<Page><TextRegion id="r1"/></Page></PcGts>'
        )

        masks = BaselineUtils.extract_masks_from_xml(xml, mock_namespace)
        assert len(masks) == 0

    def test_extract_masks_handles_invalid_coordinates(self, mock_namespace):
        """Test handling of TextLines with invalid coordinates."""
        xml = et.fromstring(
            b'<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
            b'<Page><TextRegion><TextLine id="l1">'
            b'<Coords points="invalid,data 100,200"/>'
            b"</TextLine></TextRegion></Page></PcGts>"
        )

        # Should not crash, just skip invalid lines
        masks = BaselineUtils.extract_masks_from_xml(xml, mock_namespace)
        assert isinstance(masks, list)


class TestExtractBaselinesFromSegmentation:
    """Test BaselineUtils.extract_baselines_from_segmentation()."""

    def test_extract_baselines_from_segmentation(self):
        """Test extracting baselines from segmentation result."""
        # Create mock segmentation
        mock_line1 = Mock()
        mock_line1.baseline = [(100, 100), (200, 100)]

        mock_line2 = Mock()
        mock_line2.baseline = [(100, 150), (200, 150)]

        mock_seg = Mock()
        mock_seg.lines = [mock_line1, mock_line2]

        baselines = BaselineUtils.extract_baselines_from_segmentation(mock_seg)

        assert len(baselines) == 2
        assert all(isinstance(bl, LineString) for bl in baselines)

    def test_extract_baselines_filters_none(self):
        """Test that None baselines are filtered out."""
        mock_line1 = Mock()
        mock_line1.baseline = [(100, 100), (200, 100)]

        mock_line2 = Mock()
        mock_line2.baseline = None  # No baseline

        mock_seg = Mock()
        mock_seg.lines = [mock_line1, mock_line2]

        baselines = BaselineUtils.extract_baselines_from_segmentation(mock_seg)

        assert len(baselines) == 1

    def test_extract_baselines_filters_short_baselines(self):
        """Test that baselines with <2 points are filtered."""
        mock_line1 = Mock()
        mock_line1.baseline = [(100, 100), (200, 100)]

        mock_line2 = Mock()
        mock_line2.baseline = [(100, 150)]  # Only 1 point

        mock_seg = Mock()
        mock_seg.lines = [mock_line1, mock_line2]

        baselines = BaselineUtils.extract_baselines_from_segmentation(mock_seg)

        assert len(baselines) == 1


class TestComputeOverlapMatrixOptimized:
    """Test BaselineUtils.compute_overlap_matrix_optimized()."""

    def test_compute_overlap_matrix_basic(self):
        """Test basic overlap matrix computation."""
        baselines = [LineString([(0, 0), (10, 0)]), LineString([(0, 5), (10, 5)])]
        masks = [
            Polygon([(0, -2), (10, -2), (10, 2), (0, 2)]),
            Polygon([(0, 3), (10, 3), (10, 7), (0, 7)]),
        ]

        overlap_matrix = BaselineUtils.compute_overlap_matrix_optimized(
            baselines, masks
        )

        assert overlap_matrix.shape == (2, 2)
        assert isinstance(overlap_matrix, np.ndarray)

        # First baseline should overlap more with first mask
        assert overlap_matrix[0, 0] > overlap_matrix[0, 1]
        # Second baseline should overlap more with second mask
        assert overlap_matrix[1, 1] > overlap_matrix[1, 0]

    def test_compute_overlap_matrix_empty_baselines(self):
        """Test with empty baselines list."""
        baselines = []
        masks = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]

        overlap_matrix = BaselineUtils.compute_overlap_matrix_optimized(
            baselines, masks
        )

        assert overlap_matrix.shape == (0, 1)

    def test_compute_overlap_matrix_empty_masks(self):
        """Test with empty masks list."""
        baselines = [LineString([(0, 0), (10, 0)])]
        masks = []

        overlap_matrix = BaselineUtils.compute_overlap_matrix_optimized(
            baselines, masks
        )

        assert overlap_matrix.shape == (1, 0)

    def test_compute_overlap_matrix_no_overlap(self):
        """Test with baselines and masks that don't overlap."""
        baselines = [LineString([(0, 0), (10, 0)])]
        masks = [Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])]

        overlap_matrix = BaselineUtils.compute_overlap_matrix_optimized(
            baselines, masks
        )

        # No overlap should result in 0
        assert overlap_matrix[0, 0] == 0.0

    def test_compute_overlap_matrix_performance(self):
        """Test performance with larger dataset."""
        import time

        # Create 100 baselines and 100 masks
        baselines = [LineString([(i, 0), (i + 10, 0)]) for i in range(0, 1000, 10)]
        masks = [
            Polygon([(i, -2), (i + 10, -2), (i + 10, 2), (i, 2)])
            for i in range(0, 1000, 10)
        ]

        start = time.time()
        overlap_matrix = BaselineUtils.compute_overlap_matrix_optimized(
            baselines, masks
        )
        duration = time.time() - start

        assert overlap_matrix.shape == (100, 100)
        # Should complete in reasonable time (< 1 second for 100x100)
        assert duration < 1.0


class TestPredictKrakenBaselines:
    """Test BaselineUtils.predict_kraken_baselines()."""

    @patch("flow_segmenter.baseline_utils.blla.segment")
    def test_predict_kraken_baselines_integration(
        self,
        mock_blla_segment,
        mock_image_file,
        mock_xml_with_textlines,
        mock_namespace,
    ):
        """Test baseline prediction integration."""
        # Mock Kraken segmentation
        mock_line = Mock()
        mock_line.baseline = [(110, 130), (490, 130)]

        mock_seg = Mock()
        mock_seg.lines = [mock_line]
        mock_blla_segment.return_value = mock_seg

        result = BaselineUtils.predict_kraken_baselines(
            mock_image_file, mock_xml_with_textlines, mock_namespace
        )

        assert result is not None
        # Check that blla.segment was called
        mock_blla_segment.assert_called_once()

    @patch("flow_segmenter.baseline_utils.blla.segment")
    def test_predict_kraken_baselines_with_no_baselines(
        self,
        mock_blla_segment,
        mock_image_file,
        mock_xml_with_textlines,
        mock_namespace,
    ):
        """Test handling when no baselines found."""
        mock_seg = Mock()
        mock_seg.lines = []  # No lines
        mock_blla_segment.return_value = mock_seg

        result = BaselineUtils.predict_kraken_baselines(
            mock_image_file, mock_xml_with_textlines, mock_namespace
        )

        # Should return XML unchanged
        assert result is not None


class TestAddLineMasksToTextlines:
    """Test BaselineUtils.add_linemasks_to_textlines()."""

    @patch("flow_segmenter.baseline_utils.calculate_polygonal_environment")
    def test_add_linemasks_to_textlines(
        self, mock_calc_env, mock_image_file, mock_namespace
    ):
        """Test adding linemasks to textlines."""
        xml_string = create_mock_xml_with_baselines()
        xml = et.fromstring(xml_string.encode())

        # Mock mask calculation
        mock_calc_env.return_value = [[(100, 100), (200, 100), (200, 150), (100, 150)]]

        result = BaselineUtils.calc_and_add_linemasks_to_textlines(
            mock_image_file, xml, mock_namespace
        )

        assert result is not None

        # Check that calculate_polygonal_environment was called
        mock_calc_env.assert_called()

        # Check that Coords were updated
        textline = result.find(".//ns:TextLine", namespaces=mock_namespace)
        coords = textline.find(".//ns:Coords", namespaces=mock_namespace)
        assert coords is not None
        assert "points" in coords.attrib

    @patch("flow_segmenter.baseline_utils.calculate_polygonal_environment")
    def test_add_linemasks_handles_calculation_error(
        self, mock_calc_env, mock_image_file, mock_namespace
    ):
        """Test handling when mask calculation fails."""
        xml_string = create_mock_xml_with_baselines()
        xml = et.fromstring(xml_string.encode())

        # Mock error in calculation
        mock_calc_env.side_effect = Exception("Calculation error")

        # Should not crash, just skip the line
        result = BaselineUtils.calc_and_add_linemasks_to_textlines(
            mock_image_file, xml, mock_namespace
        )

        assert result is not None
