"""
Baseline utilities with performance optimizations.
"""

import logging

import lxml.etree as ET
import numpy as np
import scipy.optimize as opt
from kraken import blla
from kraken.lib.segmentation import calculate_polygonal_environment
from PIL import Image
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree

from .exceptions import InvalidImageError

logger = logging.getLogger(__name__)

# Constants
BASELINE_INSERT_POSITION = 0
COORDS_INSERT_POSITION = 1


class BaselineUtils:
    """Utility class for baseline prediction and line mask operations."""

    @staticmethod
    def load_image_grayscale(image_path: str) -> Image.Image:
        """
        Load an image and convert to grayscale.

        :param image_path: Path to the image file
        :return: PIL Image in grayscale mode
        :raises InvalidImageError: If image cannot be loaded
        """
        try:
            return Image.open(image_path).convert("L")
        except OSError as e:
            raise InvalidImageError(f"Cannot open or process image '{image_path}': {e}")

    @staticmethod
    def extract_masks_from_xml(
            xml_etree: ET.Element, namespace: dict[str, str]
    ) -> list[Polygon]:
        """
        Extract polygon masks from TextLine elements in XML.

        :param xml_etree: XML element tree
        :param namespace: XML namespace dictionary
        :return: List of Shapely Polygon objects representing line masks
        """
        masks = []
        for line_el in xml_etree.findall(".//ns:TextLine", namespaces=namespace):
            coords = line_el.find(".//ns:Coords", namespaces=namespace)
            if coords is None:
                logger.debug(
                    f'No coordinates found for line {line_el.attrib.get("id", "unknown")}'
                )
                continue

            try:
                points = [
                    tuple(map(int, point.split(",")))
                    for point in coords.attrib["points"].split()
                ]
                masks.append(Polygon(points))
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid coordinates for line: {e}")
                continue

        return masks

    @staticmethod
    def extract_baselines_from_segmentation(seg) -> list[LineString]:
        """
        Extract baselines from Kraken segmentation result.

        :param seg: Kraken segmentation result
        :return: List of Shapely LineString objects representing baselines
        """
        baselines = []
        for b in seg.lines:
            if b.baseline is not None and len(b.baseline) > 1:
                baselines.append(LineString(b.baseline))
        return baselines

    @staticmethod
    def compute_overlap_matrix_optimized(
            baselines: list[LineString], masks: list[Polygon]
    ) -> np.ndarray:
        """
        Compute overlap matrix between baselines and masks using spatial indexing.

        This uses an STRtree (Sort-Tile-Recursive tree) for efficient spatial queries,
        which significantly improves performance for large numbers of baselines/masks.

        :param baselines: List of baseline LineStrings
        :param masks: List of mask Polygons
        :return: Matrix of overlap ratios (n_baselines x n_masks)
        """
        n, m = len(baselines), len(masks)
        overlap_matrix = np.zeros((n, m), dtype=float)

        if n == 0 or m == 0:
            return overlap_matrix

        # Build spatial index for masks (O(m log m))
        tree = STRtree(masks)

        # For each baseline, query nearby masks and compute overlaps
        for i, baseline in enumerate(baselines):
            total_length = baseline.length
            if total_length == 0:
                continue

            # Query spatial index for candidate masks (O(log m + k) where k = results)
            candidate_indices = tree.query(baseline)

            # Compute overlap only with candidates
            for j in candidate_indices:
                poly = masks[j]
                inter = baseline.intersection(poly)

                if inter.is_empty:
                    overlap = 0.0
                elif inter.geom_type == "LineString":
                    overlap = inter.length
                elif inter.geom_type == "MultiLineString":
                    overlap = sum(line.length for line in inter.geoms)
                else:
                    overlap = 0.0

                overlap_matrix[i, j] = overlap / total_length

        return overlap_matrix

    @staticmethod
    def assign_baselines_to_textlines(
            baselines: list[LineString],
            textlines: list[ET.Element],
            overlap_matrix: np.ndarray,
            namespace: dict[str, str],
    ) -> None:
        """
        Assign baselines to text lines using optimal matching.

        Uses the Hungarian algorithm (linear_sum_assignment) to find the optimal
        one-to-one assignment between baselines and text lines.

        :param baselines: List of baseline LineStrings
        :param textlines: List of TextLine XML elements
        :param overlap_matrix: Precomputed overlap matrix
        :param namespace: XML namespace dictionary
        """
        # Use Hungarian algorithm to maximize overlap
        row_ind, col_ind = opt.linear_sum_assignment(-overlap_matrix)

        for i, j in zip(row_ind, col_ind):
            baseline = baselines[i]
            line_el = textlines[j]

            # Remove existing baseline if present
            baseline_el = line_el.find(".//ns:Baseline", namespaces=namespace)
            if baseline_el is not None:
                line_el.remove(baseline_el)

            # Create and insert new baseline element
            baseline_el = ET.Element("Baseline", nsmap={"ns": namespace["ns"]})
            line_el.insert(BASELINE_INSERT_POSITION, baseline_el)

            # Set baseline points
            baseline_points = " ".join(f"{int(x)},{int(y)}" for x, y in baseline.coords)
            baseline_el.attrib["points"] = baseline_points

            logger.debug(
                f'Added baseline to line {line_el.attrib.get("id", "unknown")}: '
                f"{baseline_points}"
            )

    @staticmethod
    def predict_kraken_baselines(
            image_path: str, xml_etree: ET.Element, namespace: dict[str, str]
    ) -> ET.Element:
        """
        Predict baselines for text lines using Kraken with optimized matching.

        :param image_path: Path to the image file
        :param xml_etree: XML tree with text lines
        :param namespace: XML namespace dictionary
        :return: XML tree with added baselines
        """
        logger.info("Predicting baselines for text lines")

        # Load image
        img = BaselineUtils.load_image_grayscale(image_path)

        # Run Kraken segmentation
        seg = blla.segment(img)

        # Extract masks and baselines
        masks = BaselineUtils.extract_masks_from_xml(xml_etree, namespace)
        baselines = BaselineUtils.extract_baselines_from_segmentation(seg)

        n, m = len(baselines), len(masks)
        if n == 0 or m == 0:
            logger.warning("No baselines or masks found. Skipping baseline prediction.")
            return xml_etree

        logger.debug(f"Found {n} baselines and {m} masks")

        # Compute overlap matrix using spatial indexing (optimized)
        overlap_matrix = BaselineUtils.compute_overlap_matrix_optimized(
            baselines, masks
        )

        logger.debug("Overlap matrix (%):")
        logger.debug(overlap_matrix * 100)

        # Assign baselines to text lines
        textlines = xml_etree.findall(".//ns:TextLine", namespaces=namespace)
        BaselineUtils.assign_baselines_to_textlines(
            baselines, textlines, overlap_matrix, namespace
        )

        logger.info("Finished adding baselines to text lines")
        return xml_etree

    @staticmethod
    def add_linemasks_to_textlines(
            image_path: str, xml_etree: ET.Element, namespace: dict[str, str]
    ) -> ET.Element:
        """
        Calculate and add line masks to text lines based on their baselines.

        :param image_path: Path to the image file
        :param xml_etree: XML tree with baselines
        :param namespace: XML namespace dictionary
        :return: XML tree with updated line masks
        """
        logger.info("Adding line masks to text lines")

        # Load image
        img = BaselineUtils.load_image_grayscale(image_path)

        # Process each text line
        for line_el in xml_etree.findall(".//ns:TextLine", namespaces=namespace):
            baseline_el = line_el.find(".//ns:Baseline", namespaces=namespace)
            if baseline_el is None:
                logger.debug(
                    f'No baseline found for line {line_el.attrib.get("id", "unknown")}'
                )
                continue

            # Parse baseline points
            try:
                points = [
                    (int(x), int(y))
                    for x, y in [
                        p.split(",") for p in baseline_el.attrib["points"].split()
                    ]
                ]
            except (ValueError, KeyError) as e:
                logger.error(
                    f'Invalid baseline points for line {line_el.attrib.get("id", "unknown")}: {e}'
                )
                continue

            # Calculate mask using Kraken
            try:
                mask = calculate_polygonal_environment(img, baselines=[points])[0]
            except Exception as e:
                logger.error(
                    f'Error calculating mask for line {line_el.attrib.get("id", "unknown")}: {e}'
                )
                continue

            if not mask:
                logger.debug(
                    f'No mask calculated for line {line_el.attrib.get("id", "unknown")}'
                )
                continue

            # Update or create Coords element
            mask_str = " ".join(f"{int(x)},{int(y)}" for x, y in mask)
            coords_el = line_el.find(".//ns:Coords", namespaces=namespace)

            if coords_el is not None:
                coords_el.attrib["points"] = mask_str
            else:
                coords_el = ET.Element("Coords", points=mask_str, nsmap=namespace)
                line_el.insert(COORDS_INSERT_POSITION, coords_el)

        logger.info("Finished adding line masks")
        return xml_etree
