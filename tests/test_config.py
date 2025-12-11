"""
Tests for flow_segmenter.config module.
"""

import pytest
from pydantic import ValidationError

from flow_segmenter.config import SegmenterConfig


class TestSegmenterConfigBasic:
    """Test basic SegmenterConfig functionality."""

    def test_config_with_single_model_name(self):
        """Test config with single model name string."""
        config = SegmenterConfig(model_names="model.pt")
        assert config.model_names == "model.pt"

    def test_config_with_multiple_model_names(self):
        """Test config with list of model names."""
        config = SegmenterConfig(model_names=["model1.pt", "model2.pt"])
        assert config.model_names == ["model1.pt", "model2.pt"]

    def test_config_with_single_batch_size(self):
        """Test config with single batch size."""
        config = SegmenterConfig(model_names="model.pt", batch_sizes=4)
        assert config.batch_sizes == 4

    def test_config_with_multiple_batch_sizes(self):
        """Test config with list of batch sizes."""
        config = SegmenterConfig(
            model_names=["model1.pt", "model2.pt"], batch_sizes=[2, 4]
        )
        assert config.batch_sizes == [2, 4]

    def test_config_default_values(self):
        """Test config default values."""
        config = SegmenterConfig(model_names="model.pt")
        assert config.batch_sizes == 2
        assert config.order_lines is False
        assert config.export is False
        assert config.baselines is False
        assert config.kraken_linemasks is False
        assert config.textline_check is True
        assert config.creator == "The-Flow-Project"

    def test_config_custom_creator(self):
        """Test config with custom creator."""
        config = SegmenterConfig(model_names="model.pt", creator="CustomApp")
        assert config.creator == "CustomApp"

    def test_config_yolo_args(self):
        """Test config with YOLO arguments."""
        yolo_args = {"conf": 0.5, "iou": 0.7}
        config = SegmenterConfig(model_names="model.pt", yolo_args=yolo_args)
        assert config.yolo_args == yolo_args


class TestSegmenterConfigValidation:
    """Test SegmenterConfig validation."""

    def test_kraken_linemasks_requires_baselines(self):
        """Test that kraken_linemasks=True requires baselines=True."""
        with pytest.raises(ValidationError, match="requires baselines=True"):
            SegmenterConfig(
                model_names="model.pt", kraken_linemasks=True, baselines=False
            )

    def test_kraken_linemasks_allowed_with_baselines(self):
        """Test that kraken_linemasks=True works with baselines=True."""
        config = SegmenterConfig(
            model_names="model.pt", kraken_linemasks=True, baselines=True
        )
        assert config.kraken_linemasks is True
        assert config.baselines is True

    def test_batch_sizes_length_must_match_model_names(self):
        """Test that batch_sizes list length must match model_names length."""
        with pytest.raises(ValidationError, match="must match"):
            SegmenterConfig(
                model_names=["model1.pt", "model2.pt"],
                batch_sizes=[2, 4, 6],  # Wrong length
            )

    def test_batch_sizes_matching_length_is_valid(self):
        """Test that matching batch_sizes length is valid."""
        config = SegmenterConfig(
            model_names=["model1.pt", "model2.pt"], batch_sizes=[2, 4]
        )
        assert len(config.batch_sizes) == len(config.model_names)

    def test_negative_batch_size_raises_error(self):
        """Test that negative batch size raises error."""
        with pytest.raises(ValidationError, match="must be positive"):
            SegmenterConfig(model_names="model.pt", batch_sizes=-1)

    def test_zero_batch_size_raises_error(self):
        """Test that zero batch size raises error."""
        with pytest.raises(ValidationError, match="must be positive"):
            SegmenterConfig(model_names="model.pt", batch_sizes=0)

    def test_negative_batch_size_in_list_raises_error(self):
        """Test that negative batch size in list raises error."""
        with pytest.raises(ValidationError, match="must be positive"):
            SegmenterConfig(model_names=["model1.pt", "model2.pt"], batch_sizes=[2, -1])

    def test_positive_batch_sizes_are_valid(self):
        """Test that positive batch sizes are valid."""
        config = SegmenterConfig(
            model_names=["model1.pt", "model2.pt"], batch_sizes=[1, 10]
        )
        assert config.batch_sizes == [1, 10]


class TestSegmenterConfigEdgeCases:
    """Test edge cases for SegmenterConfig."""

    def test_empty_yolo_args(self):
        """Test config with empty YOLO args dict."""
        config = SegmenterConfig(model_names="model.pt", yolo_args={})
        assert config.yolo_args == {}

    def test_none_yolo_args(self):
        """Test config with None YOLO args."""
        config = SegmenterConfig(model_names="model.pt", yolo_args=None)
        assert config.yolo_args is None

    def test_all_boolean_flags_enabled(self):
        """Test config with all boolean flags enabled."""
        config = SegmenterConfig(
            model_names="model.pt",
            order_lines=True,
            export=True,
            baselines=True,
            kraken_linemasks=True,
            textline_check=True,
        )
        assert config.order_lines is True
        assert config.export is True
        assert config.baselines is True
        assert config.kraken_linemasks is True
        assert config.textline_check is True

    def test_large_batch_size(self):
        """Test config with large batch size."""
        config = SegmenterConfig(model_names="model.pt", batch_sizes=1000)
        assert config.batch_sizes == 1000

    def test_single_model_with_list_batch_size(self):
        """Test single model name (string) with list of batch sizes fails."""
        with pytest.raises(ValidationError):
            SegmenterConfig(
                model_names="model.pt", batch_sizes=[2, 4]  # Single string  # List
            )
