"""
Tests for flow_segmenter.segmenter module.
"""

from unittest.mock import MagicMock, patch

import pytest

from flow_segmenter import SegmenterConfig, SegmenterYolo
from flow_segmenter.exceptions import (
    EmptyCollectionError,
    InvalidImageError,
    InvalidXMLError,
)
from tests.fixtures.mock_data import (
    create_mock_dataset_example,
    create_mock_image_array,
    create_mock_xml_with_namespace,
    create_mock_xml_with_textlines,
)


class TestSegmenterYOLOInit:
    """Test SegmenterYolo initialization."""

    def test_init_with_single_model(self):
        """Test initialization with single model name."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        assert segmenter.model_names == ["model.pt"]
        assert len(segmenter.batch_sizes) == 1

    def test_init_with_multiple_models(self):
        """Test initialization with multiple model names."""
        config = SegmenterConfig(
            model_names=["model1.pt", "model2.pt"], batch_sizes=[2, 4]
        )
        segmenter = SegmenterYolo(config)

        assert segmenter.model_names == ["model1.pt", "model2.pt"]
        assert segmenter.batch_sizes == [2, 4]

    def test_init_sets_device(self):
        """Test that device is set correctly."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        assert segmenter.device is not None
        assert segmenter.devicename in ["cuda", "mps", "cpu"]

    def test_init_creates_pipeline(self):
        """Test that pipeline is created."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        assert segmenter.pipeline is not None

    def test_init_with_baselines_enabled(self):
        """Test initialization with baselines enabled."""
        config = SegmenterConfig(
            model_names="model.pt", baselines=True, kraken_linemasks=True
        )
        segmenter = SegmenterYolo(config)

        assert segmenter.baselines is True
        assert segmenter.kraken_linemasks is True


class TestCreateAndValidateCollection:
    """Test SegmenterYolo._create_and_validate_collection()."""

    @patch("flow_segmenter.segmenter.Collection")
    def test_create_collection_success(self, mock_collection_class):
        """Test successful collection creation."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        # Mock collection with pages
        mock_collection = MagicMock()
        mock_collection.pages = [MagicMock()]
        mock_collection_class.return_value = mock_collection

        collection, temp_paths = segmenter._create_and_validate_collection(
            create_mock_image_array(), None
        )

        assert collection is not None
        assert isinstance(temp_paths, list)
        mock_collection_class.assert_called_once()

    @patch("flow_segmenter.segmenter.Collection")
    def test_create_collection_empty_raises_error(self, mock_collection_class):
        """Test that empty collection raises EmptyCollectionError."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        # Mock empty collection
        mock_collection = MagicMock()
        mock_collection.pages = []
        mock_collection_class.return_value = mock_collection

        with pytest.raises(EmptyCollectionError, match="No pages found"):
            segmenter._create_and_validate_collection(create_mock_image_array(), None)

    @patch("flow_segmenter.segmenter.Collection")
    def test_create_collection_io_error_raises_invalid_image(
        self, mock_collection_class
    ):
        """Test that IOError raises InvalidImageError."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        mock_collection_class.side_effect = OSError("File not found")

        with pytest.raises(InvalidImageError, match="Cannot create collection"):
            segmenter._create_and_validate_collection(create_mock_image_array(), None)


class TestRunPipelineAndSerialize:
    """Test SegmenterYolo._run_pipeline_and_serialize()."""

    @patch("flow_segmenter.segmenter.XMLUtils.safe_parse_xml")
    @patch("htrflow.serialization.serialization.PageXML")
    def test_run_pipeline_and_serialize_success(
        self, mock_pagexml_class, mock_safe_parse, mock_collection
    ):
        """Test successful pipeline run and serialization."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        # Prevent the real pipeline from calling HuggingFace
        segmenter.pipeline = MagicMock()
        segmenter.pipeline.run.return_value = mock_collection

        # Mock serializer
        mock_serializer = MagicMock()
        mock_serializer.serialize_collection.return_value = [
            [create_mock_xml_with_namespace()]
        ]
        mock_pagexml_class.return_value = mock_serializer

        # Mock XML parsing
        mock_xml = MagicMock()
        mock_safe_parse.return_value = mock_xml

        result = segmenter._run_pipeline_and_serialize(mock_collection)

        assert result is not None
        segmenter.pipeline.run.assert_called_once_with(mock_collection)
        mock_safe_parse.assert_called_once()


class TestApplyPostprocessing:
    """Test SegmenterYolo._apply_postprocessing()."""

    @patch("flow_segmenter.segmenter.XMLUtils.convert_textregions_to_textlines")
    def test_apply_postprocessing_with_textline_check(self, mock_convert):
        """Test postprocessing with textline check enabled."""
        config = SegmenterConfig(model_names="model.pt", textline_check=True)
        segmenter = SegmenterYolo(config)

        mock_xml = MagicMock()
        mock_convert.return_value = mock_xml

        segmenter._apply_postprocessing(mock_xml, "./tests/fixtures/test.jpg")

        mock_convert.assert_called_once()

    @patch("flow_segmenter.segmenter.BaselineUtils.predict_kraken_baselines")
    @patch("flow_segmenter.segmenter.XMLUtils.get_xml_namespace")
    def test_apply_postprocessing_with_baselines(
        self, mock_get_ns, mock_predict_baselines
    ):
        """Test postprocessing with baselines enabled."""
        config = SegmenterConfig(
            model_names="model.pt", baselines=True, textline_check=False
        )
        segmenter = SegmenterYolo(config)

        mock_xml = MagicMock()
        mock_get_ns.return_value = {"ns": "test"}
        mock_predict_baselines.return_value = mock_xml

        segmenter._apply_postprocessing(mock_xml, "./tests/fixtures/test.jpg")

        mock_predict_baselines.assert_called_once()

    @patch("flow_segmenter.segmenter.BaselineUtils.calc_and_add_linemasks_to_textlines")
    @patch("flow_segmenter.segmenter.BaselineUtils.predict_kraken_baselines")
    @patch("flow_segmenter.segmenter.XMLUtils.get_xml_namespace")
    def test_apply_postprocessing_with_linemasks(
        self, mock_get_ns, mock_predict_baselines, mock_add_linemasks
    ):
        """Test postprocessing with linemasks enabled."""
        config = SegmenterConfig(
            model_names="model.pt",
            baselines=True,
            kraken_linemasks=True,
            textline_check=False,
        )
        segmenter = SegmenterYolo(config)

        mock_xml = MagicMock()
        mock_get_ns.return_value = {"ns": "test"}
        mock_predict_baselines.return_value = mock_xml
        mock_add_linemasks.return_value = mock_xml

        segmenter._apply_postprocessing(mock_xml, "./tests/fixtures/test.jpg")

        mock_add_linemasks.assert_called_once()


class TestMergeOrFinalizeXML:
    """Test SegmenterYolo._merge_or_finalize_xml()."""

    @patch("flow_segmenter.segmenter.XMLUtils.merge_xml_pages")
    @patch("flow_segmenter.segmenter.XMLUtils.get_xml_namespace")
    def test_merge_with_existing_xml(self, mock_get_ns, mock_merge):
        """Test merging with existing XML."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        mock_new_xml = MagicMock()
        mock_existing_xml = MagicMock()
        mock_get_ns.return_value = {"ns": "test"}
        mock_merge.return_value = mock_existing_xml

        result = segmenter._merge_or_finalize_xml(mock_new_xml, mock_existing_xml)

        mock_merge.assert_called_once()
        assert result == mock_existing_xml

    @patch("flow_segmenter.segmenter.XMLUtils.add_creator_metadata")
    def test_finalize_without_existing_xml(self, mock_add_creator):
        """Test finalizing without existing XML."""
        config = SegmenterConfig(model_names="model.pt", creator="TestApp")
        segmenter = SegmenterYolo(config)

        mock_xml = MagicMock()
        mock_add_creator.return_value = mock_xml

        segmenter._merge_or_finalize_xml(mock_xml, None)

        mock_add_creator.assert_called_once_with(mock_xml, "TestApp")


class TestSegmentMethod:
    """Test SegmenterYolo.segment() method."""

    @patch.object(SegmenterYolo, "_merge_or_finalize_xml")
    @patch.object(SegmenterYolo, "_apply_postprocessing")
    @patch.object(SegmenterYolo, "_run_pipeline_and_serialize")
    @patch.object(SegmenterYolo, "_create_and_validate_collection")
    def test_segment_orchestrates_workflow(
        self, mock_create, mock_run, mock_postprocess, mock_finalize
    ):
        """Test that segment() orchestrates the workflow correctly."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        # Setup mocks
        mock_collection = MagicMock()
        mock_xml = MagicMock()

        mock_create.return_value = (mock_collection, [])
        mock_run.return_value = mock_xml
        mock_postprocess.return_value = mock_xml
        mock_finalize.return_value = mock_xml

        result = segmenter.segment("./tests/fixtures/test.jpg")

        # Verify workflow steps
        mock_create.assert_called_once_with("./tests/fixtures/test.jpg", None)
        mock_run.assert_called_once_with(mock_collection)
        mock_postprocess.assert_called_once_with(mock_xml, "./tests/fixtures/test.jpg")
        mock_finalize.assert_called_once_with(mock_xml, None)

        assert result == mock_xml

    @patch.object(SegmenterYolo, "_create_and_validate_collection")
    def test_segment_propagates_exceptions(self, mock_create):
        """Test that segment() propagates exceptions."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        mock_create.side_effect = InvalidImageError("Test error")

        with pytest.raises(InvalidImageError, match="Test error"):
            segmenter.segment("./tests/fixtures/test.jpg")


class TestProcessSingleDatasetExample:
    """Test SegmenterYolo._process_single_dataset_example()."""

    @patch("flow_segmenter.segmenter.XMLUtils.safe_parse_xml")
    @patch("flow_segmenter.segmenter.XMLUtils.serialize_xml")
    @patch.object(SegmenterYolo, "segment")
    def test_process_dataset_example_success(
        self, mock_segment, mock_serialize, mock_parse
    ):
        """Test successful dataset example processing."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        mock_xml = MagicMock()
        mock_parse.return_value = mock_xml
        mock_segment.return_value = mock_xml
        mock_serialize.return_value = "<xml>result</xml>"

        example = create_mock_dataset_example()
        result = segmenter._process_single_dataset_example(example, "xml_out")

        assert result["xml_out"] == "<xml>result</xml>"
        mock_segment.assert_called_once()

    @patch("flow_segmenter.segmenter.XMLUtils.safe_parse_xml")
    def test_process_dataset_example_invalid_xml(self, mock_parse):
        """Test handling invalid XML in dataset example."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        mock_parse.side_effect = InvalidXMLError("Invalid XML")

        example = create_mock_dataset_example()
        result = segmenter._process_single_dataset_example(example, "xml_out")

        assert result["xml_out"] == create_mock_xml_with_textlines()

    @patch("flow_segmenter.segmenter.XMLUtils.safe_parse_xml")
    @patch.object(SegmenterYolo, "segment")
    def test_process_dataset_example_segmentation_error(
        self, mock_segment, mock_parse
    ):
        """Test handling segmentation errors."""
        config = SegmenterConfig(model_names="model.pt")
        segmenter = SegmenterYolo(config)

        mock_xml = MagicMock()
        mock_parse.return_value = mock_xml
        mock_segment.side_effect = InvalidImageError("Segmentation failed")

        example = create_mock_dataset_example()
        result = segmenter._process_single_dataset_example(example, "xml_out")

        assert result["xml_out"] == create_mock_xml_with_textlines()


class TestGetBatchsize:
    """Test SegmenterYolo.get_batchsize()."""

    def test_get_batchsize_single_int(self):
        """Test get_batchsize with single integer."""
        config = SegmenterConfig(model_names=["model1.pt", "model2.pt"], batch_sizes=4)
        segmenter = SegmenterYolo(config)

        # Should return list with same value for each model
        assert len(segmenter.batch_sizes) == 2
        assert all(b == 4 for b in segmenter.batch_sizes)

    def test_get_batchsize_list(self):
        """Test get_batchsize with list."""
        config = SegmenterConfig(
            model_names=["model1.pt", "model2.pt"], batch_sizes=[2, 8]
        )
        segmenter = SegmenterYolo(config)

        assert segmenter.batch_sizes == [2, 8]

    def test_get_batchsize_enforces_minimum(self):
        """Test that get_batchsize enforces minimum of 1."""
        config = SegmenterConfig(
            model_names="model.pt", batch_sizes=1  # Minimum allowed
        )
        segmenter = SegmenterYolo(config)

        assert segmenter.batch_sizes[0] >= 1
