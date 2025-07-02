"""
Package to recognize text segmentation
"""

# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from abc import ABC, abstractmethod
import copy
import logging
from typing import List, Union, Dict, Optional  # , Literal
import yaml
import torch
import numpy as np
import scipy.optimize as opt

# TODO: Implement htrflow fork (lightweight, e.g. without PyLaia or RTMDet)
from htrflow.volume.volume import Collection
from htrflow.pipeline.pipeline import Pipeline
from htrflow.serialization.serialization import PageXML

from lxml import etree
from lxml.etree import _ElementTree

from kraken import blla  # , serialization
# from kraken.lib import vgsl
from kraken.lib.segmentation import (
    calculate_polygonal_environment,
    # polygonal_reading_order,
    # extract_polygons,
)
# from kraken.kraken import SEGMENTATION_DEFAULT_MODEL
from PIL import Image
from shapely.geometry import Polygon, LineString

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    def segment(self, xml_etree: _ElementTree, image: str) -> _ElementTree:
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
    def get_xml_namespace(xml_etree: _ElementTree) -> Dict[str, str]:
        """
        Method to get the namespace of the XML file
        :param xml_etree: XML tree of the unsegmented XML file
        :return: Dictionary {'ns': 'namespace_uri'} with the namespace URI
        """
        # Get the namespace
        existing_etree = copy.deepcopy(xml_etree)
        root = existing_etree.getroot()
        namespace_uri = root.tag.split('}')[0][1:]
        xmlns = {'ns': namespace_uri}
        return xmlns

    @staticmethod
    def get_new_xml_page(
            existing_etree: _ElementTree,
            new_etree: Union[_ElementTree, str],
            namespace: Dict[str, str]
    ) -> _ElementTree:
        """
        Change the existing XML page by replacing the <Page> element
        with the new <Page> element from the new XML file.

        :param existing_etree: Existing XML tree of the unsegmented XML file
        :param new_etree: New XML tree with segmentation
        :param namespace: Namespace dictionary of the XML file
        :return: existing_etree with the new <Page> element
        """
        existing_etree = copy.deepcopy(existing_etree)
        root = existing_etree.getroot()

        new_page = new_etree.find('.//ns:Page', namespaces=namespace)
        existing_page = root.find('.//ns:Page', namespaces=namespace)

        # Remove all the elements inside <Page> from the existing page
        # since it has no segmentation
        for child in list(existing_page):
            existing_page.remove(child)

        for element in new_page:
            existing_page.append(copy.deepcopy(element))

        return new_etree

    @staticmethod
    def _predict_kraken_baselines_for_textlines(
            image_path: str,
            xml_etree: _ElementTree,
    ) -> _ElementTree:
        """
        Predict baselines for text lines in the XML file using Kraken's blla segmentation.
        :param image_path: Path to the image file
        :param xml_etree: XML tree of the segmented XML file missing baselines
        :return: ElementTree with baselines added to the text lines
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
            baseline_el = etree.Element(f'{{{ns["ns"]}}}Baseline')
            line_el.insert(0, baseline_el)
            baseline_points = ' '.join(f'{int(x)},{int(y)}' for x, y in baseline.coords)
            baseline_el.attrib['points'] = baseline_points
            logger.debug(f'Added baseline to line {line_el.attrib.get("id", "unknown")}: {baseline_points}')
        logger.debug('Finished adding baselines to text lines')
        return xml_etree

    @staticmethod
    def _add_linemasks_to_pagexml(
            image_path: str,
            xml_etree: _ElementTree,
    ) -> _ElementTree:
        """
        Add kraken linemasks to the text lines in the XML file using the baseline points.
        :param image_path: Path to the image file
        :param xml_etree: XML tree of the segmented XML file to (re)calculate the linemasks
        :return: ElementTree with (new) linemasks added to the text lines
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
                coords_el = etree.Element(f'{{{ns["ns"]}}}Coords', points=mask_str)
                line_el.insert(1, coords_el)
        return xml_etree


class SegmenterYOLO(Segmenter):
    """
    Class to recognize text segmentation in images based on XML-files
    with YOLO model

    :param model_names: String or List of Huggignface models names
    :param batch_sizes: Int or List of ints specifying batch sizes
    :param order_lines: Boolean if the recognized lines should be ordered
    :param export: Boolean if the recognized lines should be exported to a file
    :param baselines: Boolean if the recognized lines should include baselines
    :param kraken_linemasks: Boolean if the recognized lines should get a kraken linemask (vs. Yolo)
    :param textline_check: Boolean if the recognized regions should be checked by id
    :param creator: Creator of the XML-File (in Metadata), default is 'The-Flow-Project'
    :param kwargs: Additional keyword arguments for the htrflow pipeline, accepts the same parameters as YOLO.predict()
    """

    def __init__(
            self,
            model_names: Union[List[str], str],
            batch_sizes: Union[List[int], int] = 2,
            order_lines: bool = False,
            export: bool = False,
            baselines: bool = False,
            kraken_linemasks: bool = False,
            textline_check: bool = True,
            creator: str = 'The-Flow-Project',
            **kwargs: Union[str, int, float, bool],
    ) -> None:
        super().__init__()
        self.model_names = [model_names] if isinstance(model_names, str) else model_names
        self.batch_sizes = self.get_batchsize(batch_sizes)
        self.export = export
        self.creator = creator
        self.kwargs = kwargs

        self.baselines = baselines
        self.kraken_linemasks = kraken_linemasks
        self.textline_check = textline_check

        # Initiate htrflow pipeline config
        self.config = {'steps': []}

        # Add the segmentation steps to the pipeline config
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
            if self.kwargs:
                settings['generation_settings'].update(self.kwargs)
            self.config['steps'].append({
                'step': 'Segmentation',
                'settings': settings,
            })
        if order_lines:
            self.config['steps'].append({'step': 'OrderLines'})
        if export:
            settings = {
                'format': 'page',
                'dest': '.',
            }
            self.config['steps'].append({
                'step': 'Export',
                'settings': settings,
            })

        # logger.debug(self.config)
        logger.debug(yaml.dump(self.config, default_flow_style=False, sort_keys=False))
        self.config = yaml.safe_load(yaml.dump(self.config))
        # logger.debug(self.config)
        # Create the htrflow pipeline
        self.pipeline = Pipeline.from_config(self.config)

    def segment(self, image: str, xml_etree: Optional[_ElementTree] = None) -> _ElementTree:
        # Use htrflow to run the pipeline
        serializer = PageXML()
        collection = Collection(paths=[image])
        collection = self.pipeline.run(collection)
        # logger.debug('#' * 20 + ' START Serialized PageXML')
        # logger.debug(serializer.serialize_collection(collection)[0][0].encode())
        # logger.debug('#' * 20 + ' END Serialized PageXML')
        new_etree = etree.ElementTree(
            etree.fromstring(
                serializer.serialize_collection(collection)[0][0].encode(),
                parser=etree.XMLParser(
                    encoding='utf-8',
                    ns_clean=True,
                    compact=False,
                )
            )
        )
        logger.debug(type(new_etree))
        # Check, if textregions ids contain textlines ids
        if self.textline_check:
            logger.info('Checking TextRegion ids for "textline"')
            ns = self.get_xml_namespace(new_etree)
            textregions = new_etree.findall('.//ns:TextRegion', namespaces=ns)

            logger.debug(f"Root tag: {new_etree.getroot().tag}")
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
        if xml_etree:
            xml_namespace = self.get_xml_namespace(new_etree)
            existing_etree = self.get_new_xml_page(
                existing_etree=xml_etree,
                new_etree=new_etree,
                namespace=xml_namespace
            )
            return existing_etree
        else:
            if self.creator:
                # Add creator to the metadata of the XML file
                xml_namespace = self.get_xml_namespace(new_etree)
                metadata = new_etree.find('.//ns:Metadata', namespaces=xml_namespace)
                if metadata is None:
                    metadata = etree.Element(f'{{{xml_namespace["ns"]}}}Metadata')
                    new_etree.getroot().insert(0, metadata)
                creator_el = etree.SubElement(metadata, f'{{{xml_namespace["ns"]}}}Creator')
                creator_el.text = self.creator
            return new_etree


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
            xml_etree: Optional[_ElementTree] = None,
            image_save: bool = False,
    ) -> Union[_ElementTree, None]:
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
