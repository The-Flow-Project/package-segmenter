"""
Pydantic Model for SegmenterConfig.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class SegmenterBaseConfig(BaseModel):
    """
    Base configuration model for the segmentation process.
    """
    order_lines: bool = Field(False, description="Whether to order text lines")
    export: bool = Field(False, description="Export results to PageXML files or not")
    baselines: bool = Field(
        False, description="Generate baselines by the kraken default-blla-model "
                           "(expected if not existing in the XML file)"
    )
    kraken_linemasks: bool = Field(
        False,
        description="Recalculate line masks using kraken's default blla model (baselines needed)",
    )
    creator: str = Field(
        "The-Flow-Project",
        description="Creator name for metadata, default is 'The-Flow-Project'.",
    )


class SegmenterConfig(SegmenterBaseConfig):
    """
    Configuration model for the segmentation process.
    """
    model_names: str | list[str] = Field(
        ...,
        description="Huggingface model name(s) and/or local path(s) as string or list of strings.",
    )
    batch_sizes: int | list[int] = Field(
        2,
        description="Batch size(s) per model as integer or list of integers (as long as the model_names list).",
    )
    textline_check: bool = Field(
        True,
        description="Check textline IDs and convert TextRegions to TextLines if ID contains 'textline'",
    )
    load_existing_segmentation: bool = Field(
        False,
        description="Whether to load the existing segmentation from the XML file before using the segmenter."
                    "Makes sense, if you use a line recognition model and you want to keep the regions (default False)."
    )
    yolo_args: dict[str, Any] | None = Field(
        None,
        description="Additional YOLO pipeline arguments. "
                    "See https://docs.ultralytics.com/modes/predict/#inference-arguments for details.",
    )

    @model_validator(mode="after")
    def validate_batch_sizes_length(self):
        """Ensure batch_sizes list matches model_names list length."""
        # Convert to lists for validation
        model_names_list = (
            [self.model_names]
            if isinstance(self.model_names, str)
            else self.model_names
        )

        if isinstance(self.batch_sizes, list):
            if len(self.batch_sizes) != len(model_names_list):
                raise ValueError(
                    f"Length of batch_sizes ({len(self.batch_sizes)}) must match "
                    f"length of model_names ({len(model_names_list)})"
                )
        return self

    @field_validator("batch_sizes")
    @classmethod
    def validate_batch_sizes_positive(cls, v):
        """Ensure all batch sizes are positive integers."""
        if isinstance(v, int):
            if v < 1:
                raise ValueError(f"Batch size must be positive, got {v}")
        elif isinstance(v, list):
            for i, size in enumerate(v):
                if size < 1:
                    raise ValueError(
                        f"Batch size at index {i} must be positive, got {size}"
                    )
        return v
