"""
Pydantic Model for SegmenterConfig.
"""
from typing import Union, List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SegmenterConfig(BaseModel):
    model_names: Union[str, List[str]] = Field(
        ...,
        description="Huggingface model name(s) and/or local path(s) as string or list of strings."
    )
    batch_sizes: Union[int, List[int]] = Field(
        2,
        description="Batch size(s) per model as integer or list of integers (as long as the model_names list)."
    )
    order_lines: bool = Field(
        False,
        description="Whether to order text lines"
    )
    export: bool = Field(
        False,
        description="Export results to PageXML files or not"
    )
    baselines: bool = Field(
        False,
        description="Include kraken default-blla-model baselines in the output"
    )
    kraken_linemasks: bool = Field(
        False,
        description="Recalculate line masks using kraken's default blla model (only if baselines is True)"
    )
    textline_check: bool = Field(
        True,
        description="Check textline IDs and convert TextRegions to TextLines if ID contains 'textline'"
    )
    creator: str = Field(
        "The-Flow-Project",
        description="Creator name for metadata, default is 'The-Flow-Project'."
    )
    yolo_args: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional YOLO pipeline arguments. "
                    "See https://docs.ultralytics.com/modes/predict/#inference-arguments for details."
    )