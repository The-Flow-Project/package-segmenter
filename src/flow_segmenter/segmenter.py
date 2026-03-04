"""
Package to recognize text segmentation
"""

# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================

import logging
import copy
import tempfile
from io import BytesIO

from abc import ABC, abstractmethod
from typing import Any

import datasets
import lxml.etree as ET
import numpy as np
import torch
import yaml

from htrflow.volume.volume import Collection
from PIL import Image

from .baseline_utils import BaselineUtils
from .config import SegmenterBaseConfig, SegmenterConfig
from .exceptions import (
    EmptyCollectionError,
    InvalidImageError,
    InvalidXMLError,
    SegmentationError,
)
from .xml_utils import XMLUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants
DEFAULT_YOLO_ARGS = {
    "conf": 0.25,  # Confidence threshold
    "iou": 0.45,  # IoU threshold
    "max_det": 100,  # Maximum detections per image
    "device": (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),  # Device to run the model on
}

DEFAULT_BATCH_SIZE = 2
MIN_BATCH_SIZE = 1
BASELINE_INSERT_POSITION = 0
COORDS_INSERT_POSITION = 1
DEFAULT_JPEG_QUALITY = 95
TEMP_IMAGE_PREFIX = "flow_segmenter_temp_"


# ===============================================================================
# CLASS
# ===============================================================================
class Segmenter(ABC):
    """
    Abstract Base Class for segmenter classes
    to recognize text segmentation in images based on XML-files
    """

    def __init__(self):
        self.devicename = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.devicename)

        if torch.cuda.is_available():
            # Allow matrix multiplication with TensorFloat-32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_names = None
        self.batch_size = None
        self.text_direction = None

    @abstractmethod
    def segment(
            self, image: str | bytes, xml_etree: ET.Element | None = None
    ) -> ET.Element | None:
        """
        Method to segment the image with the loaded model
        :param image: Path to the image or numpy array
        :param xml_etree: Optional XML tree of the unsegmented XML file
        :return: XML tree with segmentation, or None if segmentation fails
        """
        pass

    def get_batchsize(
            self, batch_sizes: list[int] | int
    ) -> int | list[int]:
        """
        Method to get the batch size of the model
        :param batch_sizes: List of batch sizes or a single batch size int
        :return: Batch size of the model
        """
        if self.model_names:
            batch_sizes = (
                [max(batch_sizes, MIN_BATCH_SIZE)] * len(self.model_names)
                if isinstance(batch_sizes, int)
                else [max(b, MIN_BATCH_SIZE) for b in batch_sizes]
            )
            if len(batch_sizes) != len(self.model_names):
                # If the batch sizes are not equal to the number of models, set them to default
                batch_sizes = [DEFAULT_BATCH_SIZE] * len(self.model_names)
        else:
            raise ValueError(
                "No model names provided. Please provide a list of model names."
            )

        return batch_sizes

    @staticmethod
    def _load_pil_image_from_example(image_example: dict) -> Image.Image:
        """
        Load a PIL image from a dataset image example.

        Supports raw bytes, file paths, or array-like images.
        """
        if "bytes" in image_example and image_example["bytes"]:
            try:
                return Image.open(BytesIO(image_example["bytes"])).convert("RGB")
            except (OSError, ValueError) as e:
                raise InvalidImageError(f"Cannot decode image bytes: {e}")
        if "path" in image_example and image_example["path"]:
            try:
                return Image.open(image_example["path"]).convert("RGB")
            except (OSError, ValueError) as e:
                raise InvalidImageError(f"Cannot open image path '{image_example['path']}': {e}")
        try:
            image_array = np.array(image_example)
            if image_array.ndim == 2:
                return Image.fromarray(image_array).convert("RGB")
            if image_array.ndim == 3:
                return Image.fromarray(image_array).convert("RGB")
        except (TypeError, ValueError) as e:
            raise InvalidImageError(f"Cannot convert image to array: {e}")

        raise InvalidImageError("Unsupported image example format")

    def _process_single_dataset_example(
            self, example: dict, new_column_name: str
    ) -> dict:
        """
        Process a single example from the dataset.

        :param example: Dataset example with 'image' and 'xml' fields
        :param new_column_name: Name of column to store result
        :return: Modified example with segmented XML
        """
        logger.debug(f"Processing single dataset example")
        image_example = example["image"]["bytes"] if "bytes" in example["image"] else None
        if image_example is None:
            raise InvalidImageError("No image bytes found in dataset example")
        logger.debug(f"Type of image example: {type(image_example)}")
        if "xml" not in example and "xml_content" in example:
            xml_content = example["xml_content"]
        elif "xml" in example:
            xml_content = example["xml"]
        else:
            logger.error("No XML content found in dataset example")
            raise ValueError("Dataset example must contain 'xml' or 'xml_content' field")
        xml_bytes = xml_content.encode("utf-8")
        logger.debug(f"Processing XML bytes: {len(xml_bytes)}")

        # Parse XML
        logger.debug(f"Parse example")
        try:
            xml_etree = XMLUtils.safe_parse_xml(xml_bytes)
        except InvalidXMLError as e:
            logger.error(f"Invalid XML in dataset example: {e}")
            example[new_column_name] = xml_content
            return example

        # Segment the image
        try:
            logger.debug("Running segmentation on image")
            segmented_etree = self.segment(image=image_example, xml_etree=xml_etree)
            # Serialize result
            logger.debug("Segmentation succeeded, serializing result")
            if segmented_etree is not None:
                example[new_column_name] = XMLUtils.serialize_xml(segmented_etree)
            else:
                logger.warning("Segmentation returned None; falling back to original XML")
                example[new_column_name] = xml_content

        except InvalidImageError as e:
            logger.error(f"Invalid image in dataset example: {e}")
            example[new_column_name] = xml_content
        except InvalidXMLError as e:
            logger.error(f"Invalid XML in dataset example: {e}")
            example[new_column_name] = xml_content
        except SegmentationError as e:
            logger.error(f"Segmentation error in dataset example: {e}")
            example[new_column_name] = xml_content
        except Exception as e:
            logger.error(f"Unexpected error segmenting image: {e}")
            example[new_column_name] = xml_content

        return example

    def segment_dataset(
            self, dataset: datasets.Dataset, new_column_name: str | None = None
    ) -> datasets.Dataset:
        """
        Segment a HuggingFace dataset with the loaded model.

        Processes each example in the dataset, applying segmentation to images
        and updating the XML annotations.

        :param dataset: HuggingFace dataset with 'image' and 'xml' columns
        :param new_column_name: Column name for segmented XML (default: 'xml')
        :return: Dataset with segmented XML
        :raises ValueError: If required columns are missing
        """
        if not isinstance(dataset, datasets.Dataset):
            raise ValueError("Input must be a HuggingFace Dataset")

        logger.debug(f"Segmenting dataset with columns: {dataset.column_names}")

        if "image" not in dataset.column_names:
            raise ValueError("Dataset must contain 'image' column")
        elif "xml" not in dataset.column_names and "xml_content" not in dataset.column_names:
            raise ValueError("Dataset must contain 'xml' or 'xml_content' column")

        new_column_name = new_column_name if new_column_name else "xml"

        # Map the processing function to all examples
        logger.debug("Call dataset.map with processing function")
        segmented_dataset = dataset.map(
            lambda example: self._process_single_dataset_example(
                example, new_column_name
            )
        )
        return segmented_dataset


class SegmenterKrakenLinemasks(Segmenter):
    """
    Kraken Linemask Segmenter.
    Use kraken's default blla model to predict baselines and line masks for existing text lines in the XML.

    :param config: SegmenterBaseConfig instance with options for baselines and linemasks.
    """

    def __init__(self, config: SegmenterBaseConfig) -> None:
        super().__init__()
        self.baselines = config.baselines
        self.kraken_linemasks = config.kraken_linemasks

    def segment(
            self, image: bytes, xml_etree: ET.Element | None = None
    ) -> ET.Element | None:
        """
        Segment an image by adding baselines and linemasks using kraken's default blla model.

        :param image: Path to image file or numpy array
        :param xml_etree: XML element tree to process
        :return: XML element tree with added baselines and linemasks, or None on failure
        :raises InvalidImageError: If image cannot be processed
        :raises InvalidXMLError: If XML parsing fails
        """
        logger.info(f"Processing image for baselines and linemasks")

        if xml_etree is None:
            logger.error("No XML provided for baseline/linemask processing")
            return None

        namespace = XMLUtils.get_xml_namespace(xml_etree)
        if xml_etree.findall(".//ns:Baseline", namespaces=namespace) is None:
            logger.warning("No TextLines with Baselines found in XML; predicting baselines")
            self.baselines = True

        # Add baselines if configured
        if self.baselines:
            xml_etree = BaselineUtils.predict_kraken_baselines(
                image, xml_etree, namespace
            )

        # Add line masks if configured
        if self.kraken_linemasks:
            if xml_etree.findall(".//ns:TextLine", namespaces=namespace) is None:
                logger.warning("No TextLines found in XML; cannot add linemasks")
            else:
                xml_etree = BaselineUtils.calc_and_add_linemasks_to_textlines(
                    image, xml_etree, namespace
                )

        return xml_etree


class SegmenterYolo(Segmenter):
    """
    YOLO-based Segmenter.

    :param config: SegmenterConfig instance with model names, options, and YOLO-specific arguments.
    """

    def __init__(self, config: SegmenterConfig) -> None:
        from htrflow.pipeline.pipeline import Pipeline

        super().__init__()
        self.model_names = (
            [config.model_names]
            if isinstance(config.model_names, str)
            else config.model_names
        )
        self.batch_sizes = self.get_batchsize(config.batch_sizes)
        self.export = config.export
        self.creator = config.creator
        self.yolo_args = {**DEFAULT_YOLO_ARGS, **(config.yolo_args or {})}

        self.baselines = config.baselines
        self.kraken_linemasks = [config.kraken_linemasks if self.baselines else False]
        self.textline_check = config.textline_check
        self.order_lines = config.order_lines

        # Initiate htrflow pipeline htrflowConfig
        self.htrflowConfig: dict[str, Any] = {"steps": []}

        # Add the segmentation steps to the pipeline htrflowConfig
        for model, batchsize in zip(self.model_names, self.batch_sizes):
            settings = {
                "model": "yolo",
                "model_settings": {
                    "model": model,
                    "device": str(self.device),
                },
                "generation_settings": {
                    "batch_size": batchsize,
                },
            }
            if self.yolo_args:
                settings["generation_settings"].update(self.yolo_args)
            self.htrflowConfig["steps"].append(
                {
                    "step": "Segmentation",
                    "settings": settings,
                }
            )
        if self.order_lines:
            self.htrflowConfig["steps"].append({"step": "OrderLines"})
        if self.export:
            settings = {
                "format": "page",
                "dest": ".",
            }
            self.htrflowConfig["steps"].append(
                {
                    "step": "Export",
                    "settings": settings,
                }
            )

        logger.debug(
            yaml.safe_dump(
                self.htrflowConfig, sort_keys=False,  # default_flow_style=False,
            )
        )

        # Create the htrflow pipeline
        config_for_pipeline = copy.deepcopy(self.htrflowConfig)
        self.pipeline = Pipeline.from_config(config_for_pipeline)

    @staticmethod
    def _create_and_validate_collection(image: bytes | np.ndarray) -> Collection:
        """
        Create and validate a collection from an image.

        :param image: Image bytes or numpy array
        :return: Validated Collection object
        :raises InvalidImageError: If collection cannot be created
        :raises EmptyCollectionError: If no pages found in collection
        """
        try:
            if isinstance(image, bytes):
                pil_image = Image.open(BytesIO(image)).convert("RGB")
            else:
                pil_image = Image.fromarray(image)
            with tempfile.NamedTemporaryFile(
                    prefix=TEMP_IMAGE_PREFIX, suffix=".jpg", delete=False
            ) as temp_file:
                pil_image.save(temp_file.name, format="JPEG", quality=DEFAULT_JPEG_QUALITY)
                image_path = temp_file.name
        except (OSError, ValueError) as e:
            raise InvalidImageError(f"Cannot process image: {e}")

        if image_path is None:
            raise InvalidImageError("Failed to create temporary image file")
        image = image_path

        try:
            collection = Collection(paths=[image])
        except (OSError, FileNotFoundError) as e:
            raise InvalidImageError(
                f"Cannot create collection from image '{image}': {e}"
            )

        if len(collection.pages) < 1:
            error_msg = f"No pages found in the collection for image {image}"
            logger.error(error_msg)
            raise EmptyCollectionError(error_msg)

        return collection

    def _run_pipeline_and_serialize(self, collection: Collection) -> ET.Element | None:
        """
        Run the segmentation pipeline and serialize the result to XML.

        :param collection: Collection to process
        :return: XML element tree of the segmented result
        :raises InvalidXMLError: If XML parsing fails
        """
        from htrflow.serialization.serialization import PageXML
        serializer = PageXML()
        collection = self.pipeline.run(collection)
        logger.debug("Serializing collection to XML")
        serialized = serializer.serialize_collection(collection)

        logger.debug(f"Collection: {collection}")
        logger.debug(f"Serialized Collection: {serialized}")
        if serialized is None or len(serialized) == 0 or len(serialized[0]) == 0:
            logger.error("Pipeline did not produce any serialized output")
            return None
        else:
            logger.debug("#" * 20 + " START Serialized PageXML")
            logger.debug(serializer.serialize_collection(collection)[0][0].encode())
            logger.debug("#" * 20 + " END Serialized PageXML")

            xml_content = serializer.serialize_collection(collection)[0][0].encode()
            return XMLUtils.safe_parse_xml(xml_content)

    def _apply_postprocessing(
            self, xml_etree: ET.Element, image: str | bytes | np.ndarray
    ) -> ET.Element:
        """
        Apply post-processing steps to the segmented XML.

        :param xml_etree: XML element tree to process
        :param image: Image path for baseline/linemask calculations
        :return: Processed XML element tree
        """
        # Check and convert TextRegions to TextLines if needed
        if self.textline_check:
            xml_etree = XMLUtils.convert_textregions_to_textlines(xml_etree)

        # Add baselines if configured
        if self.baselines:
            namespace = XMLUtils.get_xml_namespace(xml_etree)
            xml_etree = BaselineUtils.predict_kraken_baselines(
                image, xml_etree, namespace
            )

        # Add line masks if configured
        if self.kraken_linemasks:
            namespace = XMLUtils.get_xml_namespace(xml_etree)
            xml_etree = BaselineUtils.calc_and_add_linemasks_to_textlines(
                image, xml_etree, namespace
            )

        return xml_etree

    def _merge_or_finalize_xml(
            self, new_etree: ET.Element, original_etree: ET.Element | None
    ) -> ET.Element:
        """
        Merge with original XML or finalize the new XML with metadata.

        :param new_etree: Newly segmented XML
        :param original_etree: Optional original XML to merge with
        :return: Final XML element tree
        """
        if original_etree is not None:
            if new_etree is None:
                logger.warning("Segmentation produced no XML; returning original XML")
                return original_etree
            else:
                # Merge with existing XML
                logger.debug("Merging with existing XML")
                xml_namespace_old = XMLUtils.get_xml_namespace(original_etree)
                xml_namespace_new = XMLUtils.get_xml_namespace(new_etree)
                return XMLUtils.merge_xml_pages(
                    existing_etree=original_etree,
                    new_etree=new_etree,
                    namespace_existing=xml_namespace_old,
                    namespace_new=xml_namespace_new,
                )
        else:
            if new_etree is None:
                logger.error("Segmentation produced no XML and no original XML provided")
                raise InvalidXMLError("No XML produced from segmentation and no original XML to fall back on")
            else:
                # Add creator metadata if configured
                if self.creator is not None:
                    new_etree = XMLUtils.add_creator_metadata(new_etree, self.creator)
                return new_etree

    def segment(
            self, image: bytes | np.ndarray, xml_etree: ET.Element | None = None
    ) -> ET.Element | None:
        """
        Segment an image using the loaded YOLO model.

        This method orchestrates the complete segmentation workflow:
        1. Create and validate collection from image
        2. Run the segmentation pipeline
        3. Parse the result to XML
        4. Apply post-processing (baselines, linemasks, textline conversion)
        5. Merge with original XML or add metadata

        :param image: Path to image file or numpy array
        :param xml_etree: Optional existing XML to merge with
        :return: Segmented XML element tree, or None on failure
        :raises InvalidImageError: If image cannot be processed
        :raises EmptyCollectionError: If no pages found
        :raises InvalidXMLError: If XML parsing fails
        """
        logger.info(f"Segmenting image")

        # Step 1: Create and validate collection
        collection = self._create_and_validate_collection(image)

        # Step 2: Run pipeline and serialize to XML
        new_etree = self._run_pipeline_and_serialize(collection)

        # Step 3: Apply post-processing
        if new_etree is not None:
            new_etree = self._apply_postprocessing(new_etree, image)

        # Step 4: Merge or finalize
        new_element = self._merge_or_finalize_xml(new_etree, xml_etree)
        return new_element
