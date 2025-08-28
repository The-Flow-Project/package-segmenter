"""
Package to recognize text segmentation
"""

# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from abc import ABC, abstractmethod
import copy
import logging
import os
from typing import List, Union, Dict, Optional  # , Literal
import yaml
import torch
import datasets
import numpy as np
import scipy.optimize as opt

# TODO: Implement htrflow fork (lightweight, e.g. without PyLaia or RTMDet)
from htrflow.volume.volume import Collection
from htrflow.pipeline.pipeline import Pipeline
from htrflow.serialization.serialization import PageXML

import lxml.etree as ET

from kraken import blla  # , serialization
from kraken.lib.segmentation import calculate_polygonal_environment
# polygonal_reading_order, extract_polygons,
# from kraken.kraken import SEGMENTATION_DEFAULT_MODEL
# from kraken.lib import vgsl
from PIL import Image
from shapely.geometry import Polygon, LineString

from .config import SegmenterConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_YOLO_ARGS = {
    'conf': 0.25,  # Confidence threshold
    'iou': 0.45,  # IoU threshold
    'max_det': 100,  # Maximum detections per image
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Device to run the model on
}


# ===============================================================================
# CLASS
# ===============================================================================
class Segmenter(ABC):
    """
    Abstract Base Class for segmenter classes
    to recognize text segmentation in images based on XML-files
    """

    def __init__(self):
        self.devicename = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.devicename)

        if torch.cuda.is_available():
            # Allow matrix multiplication with TensorFloat-32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_names = None
        self.batch_size = None
        self.text_direction = None

    @abstractmethod
    def segment(self, xml_etree: ET.Element, image: str) -> ET.Element:
        """
        Method to segment the image with the loaded model
        :param xml_etree: XML tree of the unsegmented XML file
        :param image: Path to the image
        :return: XML tree with segmentation
        """
        pass

    def get_batchsize(self, batch_sizes: Union[List[int], int]) -> Union[int, List[int]]:
        """
        Method to get the batch size of the model
        :param batch_sizes: List of batch sizes or a single batch size int
        :return: Batch size of the model
        """
        if self.model_names:
            batch_sizes = [max(batch_sizes, 1)] * len(self.model_names) \
                if isinstance(batch_sizes, int) \
                else [max(b, 1) for b in batch_sizes]
            if len(batch_sizes) != len(self.model_names):
                # If the batch sizes are not equal to the number of models, set them to 2
                batch_sizes = [2] * len(self.model_names)
        else:
            raise ValueError('No model names provided. Please provide a list of model names.')

        return batch_sizes

    @staticmethod
    def get_xml_namespace(xml_etree: ET.Element) -> Dict[str, str]:
        """
        Method to get the namespace of the XML file
        :param xml_etree: XML tree of the unsegmented XML file
        :return: Dictionary {'ns': 'namespace_uri'} with the namespace URI
        """
        # Get the namespace
        root = copy.deepcopy(xml_etree)
        return root.nsmap if None not in root.nsmap else {'ns': root.nsmap[None]}

    @staticmethod
    def get_new_xml_page(
            existing_etree: ET.Element,
            new_etree: ET.Element,
            namespace_existing: Dict[str, str],
            namespace_new: Dict[str, str]
    ) -> ET.Element:
        """
        Change the existing XML page by replacing the <Page> element
        with the new <Page> element from the new XML file.

        :param namespace_new: New namespace of the XML file with segmentation
        :param namespace_existing: Existing namespace of the XML file without segmentation
        :param existing_etree: Existing XML tree of the unsegmented XML file
        :param new_etree: New XML tree with segmentation
        :return: existing_etree with the new <Page> element
        """
        existing_etree = copy.deepcopy(existing_etree)

        existing_root = existing_etree
        new_root = new_etree

        new_page = new_root.find('.//ns:Page', namespaces=namespace_new)
        existing_page = existing_root.find('.//ns:Page', namespaces=namespace_existing)

        logger.debug(f'Existing page: {existing_page}')
        logger.debug(f'New page: {new_page}')

        if existing_page is None:
            raise ValueError("No <Page> element found in existing XML")
        if new_page is None:
            raise ValueError("No <Page> element found in new XML")

        for child in list(existing_page):
            existing_page.remove(child)
        for element in new_page:
            existing_page.append(copy.deepcopy(element))
        return existing_etree

    @staticmethod
    def _predict_kraken_baselines_for_textlines(
            image_path: str,
            xml_etree: ET.Element,
    ) -> ET.Element:
        """
        Predict baselines for text lines in the XML file using Kraken's blla segmentation.
        :param image_path: Path to the image file
        :param xml_etree: XML tree of the segmented XML file missing baselines
        :return: Element with baselines added to the text lines
        """
        logger.info('Predicting baselines for text lines')
        img = Image.open(image_path).convert("L")
        seg = blla.segment(img)
        ns = Segmenter.get_xml_namespace(xml_etree)
        masks = []
        baselines = [
            LineString(b.baseline) for b in seg.lines if b.baseline is not None and len(b.baseline) > 1
        ]

        for line_el in xml_etree.findall('.//ns:TextLine', namespaces=ns):
            coords = line_el.find('.//ns:Coords', namespaces=ns)
            if coords is None:
                logger.info(f'No coordinates found for line {line_el}')
                continue
            masks.append(Polygon([
                tuple(map(int, point.split(','))) for point in coords.attrib['points'].split()
            ]))

        n, m = len(baselines), len(masks)
        if n == 0 or m == 0:
            logger.warning('No baselines or masks found. Skipping baseline prediction.')
            return xml_etree
        logger.debug(f'Found {n} baselines and {m} masks')
        overlap_matrix = np.zeros((n, m), dtype=float)

        for i, baseline in enumerate(baselines):
            total_length = baseline.length
            for j, poly in enumerate(masks):
                inter = baseline.intersection(poly)
                if inter.is_empty:
                    overlap = 0.0
                elif inter.geom_type == 'LineString':
                    overlap = inter.length
                elif inter.geom_type == 'MultiLineString':
                    overlap = sum(line.length for line in inter.geoms)
                else:
                    overlap = 0.0
                overlap_matrix[i, j] = (overlap / total_length)

        logger.debug('Overlap matrix (%):')
        logger.debug(overlap_matrix)

        row_ind, col_ind = opt.linear_sum_assignment(-overlap_matrix)  # Maximize overlap

        textlines_el = xml_etree.findall('.//ns:TextLine', namespaces=ns)
        for i, j in zip(row_ind, col_ind):
            baseline = baselines[i]
            line_el = textlines_el[j]
            baseline_el = line_el.find('.//ns:Baseline', namespaces=ns)
            if baseline_el is not None:
                line_el.remove(baseline_el)
            baseline_el = ET.Element(f'Baseline', nsmap={'ns': ns['ns']})
            line_el.insert(0, baseline_el)
            baseline_points = ' '.join(f'{int(x)},{int(y)}' for x, y in baseline.coords)
            baseline_el.attrib['points'] = baseline_points
            logger.debug(f'Added baseline to line {line_el.attrib.get("id", "unknown")}: {baseline_points}')
        logger.debug('Finished adding baselines to text lines')
        return xml_etree

    @staticmethod
    def _add_linemasks_to_pagexml(
            image_path: str,
            xml_etree: ET.Element,
    ) -> ET.Element:
        """
        Add kraken linemasks to the text lines in the XML file using the baseline points.
        :param image_path: Path to the image file
        :param xml_etree: XML tree of the segmented XML file to (re)calculate the linemasks
        :return: Element with (new) linemasks added to the text lines
        """
        img = Image.open(image_path).convert("L")
        ns = Segmenter.get_xml_namespace(xml_etree)

        for line_el in xml_etree.findall('.//ns:TextLine', namespaces=ns):
            baseline_el = line_el.find('.//ns:Baseline', namespaces=ns)
            if baseline_el is None:
                logger.info(f'No baseline found for line {line_el}')
                continue
            points = [(int(x), int(y)) for x, y in [p.split(',') for p in baseline_el.attrib['points'].split()]]

            try:
                mask = calculate_polygonal_environment(img, baselines=[points])[0]
            except Exception as e:
                logger.error(f'Error calculating mask for line {line_el}: {e}')
                continue

            if not mask:
                logger.info(f'No mask found for line {line_el}')
                continue

            mask_str = ' '.join(f'{int(x)},{int(y)}' for x, y in mask)
            coords_el = line_el.find('.//ns:Coords', namespaces=ns)
            if coords_el is not None:
                coords_el.attrib['points'] = mask_str
            else:
                coords_el = ET.Element(f'Coords', points=mask_str, nsmap=ns)
                line_el.insert(1, coords_el)
        return xml_etree


class SegmenterYOLO(Segmenter):
    """
    YOLO-based Segmenter.

    :param config: SegmenterConfig instance with model names, options, and YOLO-specific arguments.
    """

    def __init__(self, config: SegmenterConfig) -> None:
        super().__init__()
        self.model_names = [config.model_names] if isinstance(config.model_names, str) else config.model_names
        self.batch_sizes = self.get_batchsize(config.batch_sizes)
        self.export = config.export
        self.creator = config.creator
        self.yolo_args = {**DEFAULT_YOLO_ARGS, **(config.yolo_args or {})}

        self.baselines = config.baselines
        self.kraken_linemasks = [config.kraken_linemasks if self.baselines else False]
        self.textline_check = config.textline_check
        self.order_lines = config.order_lines

        # Initiate htrflow pipeline htrflowConfig
        self.htrflowConfig = {'steps': []}

        # Add the segmentation steps to the pipeline htrflowConfig
        for model, batchsize in zip(self.model_names, self.batch_sizes):
            settings = {
                'model': 'yolo',
                'model_settings': {
                    'model': model,
                    'device': str(self.device),
                },
                'generation_settings': {
                    'batch_size': batchsize,
                }
            }
            if self.yolo_args:
                settings['generation_settings'].update(self.yolo_args)
            self.htrflowConfig['steps'].append({
                'step': 'Segmentation',
                'settings': settings,
            })
        if self.order_lines:
            self.htrflowConfig['steps'].append({'step': 'OrderLines'})
        if self.export:
            settings = {
                'format': 'page',
                'dest': '.',
            }
            self.htrflowConfig['steps'].append({
                'step': 'Export',
                'settings': settings,
            })

        logger.debug(yaml.dump(self.htrflowConfig, default_flow_style=False, sort_keys=False))
        self.htrflowConfig = yaml.safe_load(yaml.dump(self.htrflowConfig))
        # Create the htrflow pipeline
        self.pipeline = Pipeline.from_config(self.htrflowConfig)

    def segment(
            self,
            image: Union[str, np.ndarray],
            xml_etree: Optional[ET.Element] = None
    ) -> Union[ET.Element, None]:
        """
        Method to segment the image with the loaded model
        """
        # Use htrflow to run the pipeline
        serializer = PageXML()
        logger.info(f'Segmenting image {image}')
        collection = Collection(paths=[image])
        if len(collection.pages) < 1:
            logger.error(f'No pages found in the collection for image {image}')
            return None
        collection = self.pipeline.run(collection)
        logger.debug(f'Collection: {collection}')
        logger.debug('#' * 20 + ' START Serialized PageXML')
        logger.debug(serializer.serialize_collection(collection)[0][0].encode())
        logger.debug('#' * 20 + ' END Serialized PageXML')
        new_etree = ET.fromstring(
            serializer.serialize_collection(collection)[0][0].encode(),
            parser=ET.XMLParser(
                encoding='utf-8',
                ns_clean=True,
                compact=False,
            )
        )
        logger.debug(type(new_etree))
        # Check, if textregions ids contain textlines ids
        if self.textline_check:
            logger.info('Checking TextRegion ids for "textline"')
            ns = self.get_xml_namespace(new_etree)
            textregions = new_etree.findall('.//ns:TextRegion', namespaces=ns)

            logger.debug(f"Root tag: {new_etree.tag}")
            logger.debug(ns)

            # for elem in new_etree.iter():
            #     logger.debug(f"Element: {elem.tag}, ID: {elem.attrib.get('id', '-')}")

            if textregions:
                for i, tregion in enumerate(textregions):
                    id_tregion = tregion.attrib['id']
                    # logger.info(f'Checking TextRegion id "{i}" for "textline": {id_tregion}')
                    if id_tregion and 'textline' in id_tregion.lower():
                        # logger.debug(f'TextRegion {id_tregion} contains "textline" in its id.')
                        tregion.tag = f'{{{ns["ns"]}}}TextLine'

        # Shared baseline/linemask post-processing
        if self.baselines:
            new_etree = self._predict_kraken_baselines_for_textlines(image, new_etree)
        if self.kraken_linemasks:
            new_etree = self._add_linemasks_to_pagexml(image, new_etree)
        if xml_etree is not None:
            logger.debug(xml_etree)
            logger.debug(type(xml_etree))
            logger.debug(ET.tostring(xml_etree, pretty_print=True).decode())
            xml_namespace_old = self.get_xml_namespace(xml_etree)
            xml_namespace = self.get_xml_namespace(new_etree)
            existing_etree = self.get_new_xml_page(
                existing_etree=xml_etree,
                new_etree=new_etree,
                namespace_existing=xml_namespace_old,
                namespace_new=xml_namespace
            )
            return existing_etree
        else:
            if self.creator is not None:
                logger.info(f'Adding creator "{self.creator}" to the metadata of the XML file')
                # Add creator to the metadata of the XML file
                xml_namespace = self.get_xml_namespace(new_etree)
                metadata = new_etree.find('.//ns:Metadata', namespaces=xml_namespace)
                if metadata is None:
                    metadata = ET.Element('Metadata', nsmap=xml_namespace)
                    new_etree.insert(0, metadata)
                creator_el = ET.SubElement(metadata, 'Creator', nsmap=xml_namespace)
                creator_el.text = self.creator
            return new_etree

    def segment_dataset(
            self,
            dataset: datasets.Dataset,
            new_column_name: Optional[str] = None
    ) -> datasets.Dataset:
        """
        Method to segment a HuggingFace dataset with the loaded model
        :param dataset: HuggingFace dataset with 'image' and 'xml' (XML content string) columns
        :param new_column_name: Name of the new column to store the segmented XML, default is None (xml is replaced)
        :return: HuggingFace dataset with segmented XML in 'xml_segmented' column
        """
        if 'image' not in dataset.column_names or 'xml' not in dataset.column_names:
            raise ValueError("Dataset must contain 'image' and 'xml' columns")

        new_column_name = new_column_name if new_column_name else 'xml'

        def segment_example(example):
            """
            Generator for mapping the segmentation function to each example in the dataset
            :param example:
            :return:
            """
            image = np.array(example['image'])
            temp_image_path = 'temp_image.jpg'
            xml_content = example['xml']
            xml_bytes = xml_content.encode('utf-8')
            xml_etree = ET.fromstring(xml_bytes)
            try:
                Image.fromarray(image).save(temp_image_path, 'JPEG', quality=95)
                segmented_etree = self.segment(image=temp_image_path, xml_etree=xml_etree)
            except Exception as e:
                logger.error(f'Error segmenting image {example["image"]}: {e}')
                segmented_etree = None
            finally:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

            if segmented_etree is not None:
                segmented_xml = ET.tostring(segmented_etree, encoding='utf-8').decode('utf-8')
                example[new_column_name] = segmented_xml
            else:
                example[new_column_name] = None
            return example

        segmented_dataset = dataset.map(segment_example)
        return segmented_dataset


# TODO: Add linemask_only functionality to SegmenterKraken
"""
class SegmenterKraken(Segmenter):
    "."."
    Class to recognize text segmentation in images based on XML-files
    with Kraken model

    :param models: Singel model or List of loaded vsgl.TorchVGSL models
    :param text_direction: Direction of the text in the image \
    ('horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'), default is 'horizontal-lr'
    :param polygon_length_threshold: Maximum length of the polygon before it is simplified, default is 50
    "."."

    def __init__(
            self,
            models: Union[List[vgsl.TorchVGSLModel], vgsl.TorchVGSLModel] = None,
            text_direction: Literal[
                "horizontal-lr", "horizontal-rl", "vertical-lr", "vertical-rl"
            ] = 'horizontal-lr',
            polygon_length_threshold: int = 50,
            **kwargs
    ) -> None:
        super().__init__()
        # Check if models are provided
        if models:
            self.models = [models] if isinstance(models, vgsl.TorchVGSLModel) else models
        elif models is None:
            # load the blla default model
            self.models = [vgsl.TorchVGSLModel.load_model(SEGMENTATION_DEFAULT_MODEL)]
        self.text_direction = text_direction
        self.polygon_length_threshold = polygon_length_threshold
        self.linemasks_only = kwargs.get('linemasks_only', False)

    # noinspection PyTypeChecker
    def segment(
            self,
            image: str,
            xml_etree: Optional[Element] = None,
            image_save: bool = False,
    ) -> Union[Element, None]:
        # Load the image
        img = Image.open(image)

        # Load the segmentation model
        xml_page_seg = blla.segment(
            img,
            model=self.models,
            device=self.devicename,
            text_direction=self.text_direction,
        )

        # Calculate the mask of each line
        lines = []
        for baseline in xml_page_seg.lines:
            baseline_coords = baseline.baseline
            mask = calculate_polygonal_environment(img, baselines=[baseline_coords])
            logger.debug(mask)
            if mask:
                if len(mask[0]) > self.polygon_length_threshold:
                    baseline.boundary = Polygon(mask[0]).simplify(2).exterior.coords[:]
                else:
                    baseline.boundary = mask[0]
            if baseline.boundary is None:
                continue
            lines.append(baseline)

        xml_page_seg.lines = lines
        polygonal_list = [
            {"tags": bl.tags, "baseline": bl.baseline, "boundary": list(bl.boundary)}
            for bl in xml_page_seg.lines
        ]

        # Create reading order of the text lines
        regions_list = [Polygon(r.boundary) for r in xml_page_seg.regions['text']]
        reading_order = polygonal_reading_order(polygonal_list, regions=regions_list)
        xml_page_seg.reading_order = reading_order

        # Assign new IDs to the lines based on the reading order and region index
        regions_ids = [r.id for r in xml_page_seg.regions['text']]
        for i, bl in enumerate(xml_page_seg.lines):
            if bl.regions and bl.regions[0] in regions_ids:
                region_index = regions_ids.index(bl.regions[0]) + 1
                new_id = f'tr_{region_index}_tl_{reading_order[i] + 1}'
                xml_page_seg.lines[i].id = new_id

        if image_save:
            generator = extract_polygons(img, xml_page_seg)
            for mask_image in generator:
                mask_image[0].save(f'{mask_image[1].id}.jpg', 'JPEG', quality=95)

        xml_page = serialization.serialize(
            xml_page_seg,
            image_size=img.size,
            template="pagexml",
        )
        new_etree = etree.fromstring(
            xml_page.encode(),
            parser=etree.XMLParser(
                encoding='utf-8',
                ns_clean=True,
                # remove_blank_text=False,
                compact=False,
            )
        )

        if self.linemasks_only:
            new_etree = self._add_linemasks_to_pagexml(image, new_etree)
        if xml_etree:
            xml_namespace = self.get_xml_namespace(new_etree)
            existing_etree = self.get_new_xml_page(
                existing_etree=new_etree,
                new_etree=xml_etree,
                namespace=xml_namespace
            )
            return existing_etree
        else:
            return new_etree
"""
