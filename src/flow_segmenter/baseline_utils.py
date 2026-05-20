"""
Baseline utilities with performance optimizations.
"""

from io import BytesIO
from typing import cast

import lxml.etree as et
import numpy as np
import scipy.optimize as opt
from kraken import blla
from kraken.lib.segmentation import calculate_polygonal_environment
from loguru import logger
from PIL import Image
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree

from .exceptions import InvalidImageError

# Constants
BASELINE_INSERT_POSITION = 0
COORDS_INSERT_POSITION = 1


class BaselineUtils:
    """Utility class for baseline prediction and line mask operations."""

    @staticmethod
    def _open_image_grayscale(image: str | bytes | np.ndarray) -> Image.Image:
        """
        Load any supported image format and return a grayscale PIL Image.

        :param image: File path, raw bytes, or numpy array
        :return: PIL Image in grayscale ('L') mode
        :raises InvalidImageError: If the image cannot be opened
        """
        if isinstance(image, str):
            return BaselineUtils.load_image_grayscale(image)
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("L")
        return Image.open(BytesIO(image)).convert("L")

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
            raise InvalidImageError(
                f"Cannot open or process image '{image_path}': {e}"
            ) from e

    @staticmethod
    def extract_masks_from_xml(
        xml_etree: et._Element, namespace: dict[str, str]
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
                    for point in cast(str, coords.attrib["points"]).split()
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
        textlines: list[et._Element],
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
            if overlap_matrix[i, j] == 0.0:
                continue  # skip pairs with no spatial overlap

            baseline = baselines[i]
            line_el = textlines[j]

            # Remove existing baseline if present
            baseline_el = line_el.find(".//ns:Baseline", namespaces=namespace)
            if baseline_el is not None:
                line_el.remove(baseline_el)

            # Create and insert new baseline element
            baseline_el = et.Element(
                f"{{{namespace['ns']}}}Baseline", nsmap={"ns": namespace["ns"]}
            )
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
        image: str | bytes | np.ndarray,
        xml_etree: et._Element,
        namespace: dict[str, str],
    ) -> et._Element:
        """
        Predict baselines for text lines using Kraken with optimized matching.

        :param image: Image path or bytes or numpy array
        :param xml_etree: XML tree with text lines
        :param namespace: XML namespace dictionary
        :return: XML tree with added baselines
        """
        logger.info("Predicting baselines for text lines")

        img = BaselineUtils._open_image_grayscale(image)

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
    def calc_and_add_linemasks_to_textlines(
        image: str | bytes | np.ndarray,
        xml_etree: et._Element,
        namespace: dict[str, str],
    ) -> et._Element:
        """
        Calculate and add line masks to text lines based on their baselines.

        :param image: Image path or bytes or numpy array
        :param xml_etree: XML tree with baselines
        :param namespace: XML namespace dictionary
        :return: XML tree with updated line masks
        """
        logger.info("Adding line masks to text lines")

        img = BaselineUtils._open_image_grayscale(image)

        # Process text lines with baselines
        baseline_points = []
        lines = []
        for line_el in xml_etree.findall(".//ns:TextLine", namespaces=namespace):
            line_nr = line_el.attrib.get("id", "unknown")
            baseline_el = line_el.find("ns:Baseline", namespaces=namespace)
            if baseline_el is None:
                logger.debug(f"No baseline found for line {line_nr}")
                continue

            # Parse baseline points
            try:
                points = [
                    (int(x), int(y))
                    for x, y in [
                        p.split(",") for p in cast(str, baseline_el.attrib["points"]).split()
                    ]
                ]
                if len(points) < 2:
                    logger.warning(
                        f"Skipping line {line_nr}: baseline has only {len(points)} point(s)"
                    )
                    continue
                baseline_points.append(points)
                lines.append(line_el)
            except (ValueError, KeyError) as e:
                logger.error(f"Invalid baseline points for line {line_nr}: {e}")
                continue

        # Calculate mask using Kraken
        masks = None
        try:
            masks = calculate_polygonal_environment(img, baselines=baseline_points)
        except Exception as e:
            logger.error(f"Error calculating mask: {e}")

        if masks is None:
            logger.debug("No mask calculated")
            return xml_etree
        logger.debug(f"Calculated {len(masks)} line masks")
        logger.debug(
            f"There are {len([m for m in masks if m is not None])} valid masks"
        )

        # Update or create Coords element
        for line_el, mask in zip(lines, masks):
            coords_el = line_el.find(".//ns:Coords", namespaces=namespace)
            logger.debug(f"Mask for line {line_el.attrib.get('id', 'unknown')}: {mask}")
            if mask is None:
                mask_str = coords_el.attrib["points"] if coords_el is not None else ""
            else:
                mask_str = " ".join(f"{int(x)},{int(y)}" for x, y in mask)

            if coords_el is not None:
                coords_el.attrib["points"] = mask_str
            else:
                ns_uri = namespace["ns"]
                coords_el = et.Element(f"{{{ns_uri}}}Coords", nsmap={"ns": ns_uri})
                coords_el.attrib["points"] = mask_str
                line_el.insert(COORDS_INSERT_POSITION, coords_el)

        logger.info("Finished adding line masks")
        return xml_etree
