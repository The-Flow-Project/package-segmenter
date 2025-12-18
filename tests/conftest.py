"""
PyTest configuration and shared fixtures.
"""

import tempfile
from unittest.mock import MagicMock

import lxml.etree as ET
import pytest

from tests.fixtures.mock_data import (
    create_mock_image,
    create_mock_xml_with_namespace,
    create_mock_xml_with_textlines,
)


@pytest.fixture
def mock_image_file(tmp_path):
    """
    Create a temporary image file for testing.

    :param tmp_path: PyTest temporary directory
    :return: Path to temporary image file
    """
    image = create_mock_image(width=200, height=300)
    image_path = tmp_path / "test_image.jpg"
    image.save(image_path, "JPEG")
    return str(image_path)


@pytest.fixture
def mock_xml_etree():
    """
    Create a mock XML element tree.

    :return: lxml Element tree
    """
    xml_string = create_mock_xml_with_namespace()
    return ET.fromstring(xml_string.encode("utf-8"))


@pytest.fixture
def mock_xml_with_textlines():
    """
    Create a mock XML with TextLine elements.

    :return: lxml Element tree
    """
    xml_string = create_mock_xml_with_textlines()
    return ET.fromstring(xml_string.encode("utf-8"))


@pytest.fixture
def mock_namespace():
    """
    Create a mock namespace dictionary.

    :return: Namespace dict
    """
    return {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


@pytest.fixture
def mock_config():
    """
    Create a mock SegmenterConfig.

    :return: Mock config object
    """
    from flow_segmenter import SegmenterConfig

    return SegmenterConfig(
        model_names="test_model.pt",
        batch_sizes=2,
        baselines=False,
        kraken_linemasks=False,
        textline_check=True,
        order_lines=False,
        export=False,
        creator="TestCreator",
    )


@pytest.fixture
def mock_pipeline():
    """
    Create a mock htrflow Pipeline.

    :return: MagicMock Pipeline
    """
    mock = MagicMock()
    mock.run.return_value = MagicMock(pages=[MagicMock()])
    return mock


@pytest.fixture
def mock_collection():
    """
    Create a mock htrflow Collection.

    :return: MagicMock Collection
    """
    mock = MagicMock()
    mock.pages = [MagicMock(), MagicMock()]
    return mock


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory that is cleaned up after test.

    :return: Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def reset_logging():
    """
    Reset logging configuration after each test.
    """
    import logging

    yield
    # Reset logging
    logging.getLogger("flow_segmenter").handlers = []
