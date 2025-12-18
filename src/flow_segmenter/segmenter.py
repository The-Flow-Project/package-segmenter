"""
Package to recognize text segmentation
"""

import logging
import tempfile

# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from abc import ABC, abstractmethod

import datasets
import lxml.etree as ET
import numpy as np
import torch
import yaml
from htrflow.pipeline.pipeline import Pipeline
from htrflow.serialization.serialization import PageXML

# TODO: Implement htrflow fork (lightweight, e.g. without PyLaia or RTMDet)
from htrflow.volume.volume import Collection
from PIL import Image

from .baseline_utils import BaselineUtils
from .config import SegmenterConfig
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
            self, image: str | np.ndarray, xml_etree: ET.Element | None = None
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


class SegmenterYOLO(Segmenter):
    """
    YOLO-based Segmenter.

    :param config: SegmenterConfig instance with model names, options, and YOLO-specific arguments.
    """

    def __init__(self, config: SegmenterConfig) -> None:
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
        self.htrflowConfig = {"steps": []}

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
            yaml.dump(self.htrflowConfig, default_flow_style=False, sort_keys=False)
        )
        self.htrflowConfig = yaml.safe_load(yaml.dump(self.htrflowConfig))
        # Create the htrflow pipeline
        self.pipeline = Pipeline.from_config(self.htrflowConfig)

    @staticmethod
    def _create_and_validate_collection(image: str | np.ndarray) -> Collection:
        """
        Create and validate a collection from an image.

        :param image: Image path or numpy array
        :return: Validated Collection object
        :raises InvalidImageError: If collection cannot be created
        :raises EmptyCollectionError: If no pages found in collection
        """
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

    def _run_pipeline_and_serialize(self, collection: Collection) -> ET.Element:
        """
        Run the segmentation pipeline and serialize the result to XML.

        :param collection: Collection to process
        :return: XML element tree of the segmented result
        :raises InvalidXMLError: If XML parsing fails
        """
        serializer = PageXML()
        collection = self.pipeline.run(collection)

        logger.debug(f"Collection: {collection}")
        logger.debug("#" * 20 + " START Serialized PageXML")
        logger.debug(serializer.serialize_collection(collection)[0][0].encode())
        logger.debug("#" * 20 + " END Serialized PageXML")

        xml_content = serializer.serialize_collection(collection)[0][0].encode()
        return XMLUtils.safe_parse_xml(xml_content)

    def _apply_postprocessing(
            self, xml_etree: ET.Element, image: str | np.ndarray
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
            xml_etree = BaselineUtils.add_linemasks_to_textlines(
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
            # Add creator metadata if configured
            if self.creator is not None:
                new_etree = XMLUtils.add_creator_metadata(new_etree, self.creator)
            return new_etree

    def segment(
            self, image: str | np.ndarray, xml_etree: ET.Element | None = None
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
        logger.info(f"Segmenting image {image}")

        # Step 1: Create and validate collection
        collection = self._create_and_validate_collection(image)

        # Step 2: Run pipeline and serialize to XML
        new_etree = self._run_pipeline_and_serialize(collection)

        # Step 3: Apply post-processing
        new_etree = self._apply_postprocessing(new_etree, image)

        # Step 4: Merge or finalize
        return self._merge_or_finalize_xml(new_etree, xml_etree)

    def _process_single_dataset_example(
            self, example: dict, new_column_name: str
    ) -> dict:
        """
        Process a single example from the dataset.

        :param example: Dataset example with 'image' and 'xml' fields
        :param new_column_name: Name of column to store result
        :return: Modified example with segmented XML
        """
        image = np.array(example["image"])
        xml_content = example["xml"]
        xml_bytes = xml_content.encode("utf-8")

        # Parse XML
        try:
            xml_etree = XMLUtils.safe_parse_xml(xml_bytes)
        except InvalidXMLError as e:
            logger.error(f"Invalid XML in dataset example: {e}")
            example[new_column_name] = None
            return example

        # Process with temporary file
        with tempfile.NamedTemporaryFile(
                mode="w+b", suffix=".jpg", prefix=TEMP_IMAGE_PREFIX, delete=True
        ) as tmp_file:
            try:
                # Save image to temporary file
                try:
                    Image.fromarray(image).save(
                        tmp_file.name, "JPEG", quality=DEFAULT_JPEG_QUALITY
                    )
                except (OSError, ValueError) as e:
                    raise InvalidImageError(
                        f"Cannot save image array to temporary file: {e}"
                    )

                # Segment the image
                segmented_etree = self.segment(image=tmp_file.name, xml_etree=xml_etree)

                # Serialize result
                if segmented_etree is not None:
                    example[new_column_name] = XMLUtils.serialize_xml(segmented_etree)
                else:
                    example[new_column_name] = None

            except InvalidImageError as e:
                logger.error(f"Invalid image in dataset example: {e}")
                example[new_column_name] = None
            except InvalidXMLError as e:
                logger.error(f"Invalid XML in dataset example: {e}")
                example[new_column_name] = None
            except SegmentationError as e:
                logger.error(f"Segmentation error in dataset example: {e}")
                example[new_column_name] = None
            except Exception as e:
                logger.error(f"Unexpected error segmenting image: {e}")
                example[new_column_name] = None

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
        if "image" not in dataset.column_names or "xml" not in dataset.column_names:
            raise ValueError("Dataset must contain 'image' and 'xml' columns")

        new_column_name = new_column_name if new_column_name else "xml"

        # Map the processing function to all examples
        segmented_dataset = dataset.map(
            lambda example: self._process_single_dataset_example(
                example, new_column_name
            )
        )
        return segmented_dataset
