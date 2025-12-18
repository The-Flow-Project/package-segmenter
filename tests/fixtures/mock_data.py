"""
Mock data generators for testing flow_segmenter.
"""


import numpy as np
from PIL import Image


def create_mock_image(width: int = 100, height: int = 100) -> Image.Image:
    """
    Create a mock grayscale image for testing.

    :param width: Image width
    :param height: Image height
    :return: PIL Image
    """
    # Create a simple gradient image
    array = np.linspace(0, 255, width * height, dtype=np.uint8).reshape(height, width)
    return Image.fromarray(array, mode="L")


def create_mock_image_array(width: int = 100, height: int = 100) -> np.ndarray:
    """
    Create a mock image as numpy array.

    :param width: Image width
    :param height: Image height
    :return: numpy array
    """
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def create_mock_xml_with_namespace() -> str:
    """
    Create a mock PageXML document with namespace.

    :return: XML string with namespace
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15" 
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
       xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd">
    <Metadata>
        <Creator>TestCreator</Creator>
    </Metadata>
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion id="region_1">
            <Coords points="100,100 200,100 200,200 100,200"/>
            <TextLine id="line_1">
                <Coords points="110,110 190,110 190,130 110,130"/>
                <Baseline points="110,125 190,125"/>
                <TextEquiv>
                    <Unicode>Test text</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"""


def create_mock_xml_without_page() -> str:
    """
    Create a mock XML without <Page> element.

    :return: XML string without Page
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Metadata>
        <Creator>TestCreator</Creator>
    </Metadata>
</PcGts>"""


def create_mock_xml_with_textline_in_id() -> str:
    """
    Create a mock XML with 'textline' in TextRegion ID.

    :return: XML string
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion id="textline_region_1">
            <Coords points="100,100 200,100 200,200 100,200"/>
        </TextRegion>
        <TextRegion id="normal_region_2">
            <Coords points="300,300 400,300 400,400 300,400"/>
        </TextRegion>
    </Page>
</PcGts>"""


def create_mock_xml_with_textlines() -> str:
    """
    Create a mock XML with TextLine elements.

    :return: XML string
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion id="region_1">
            <Coords points="100,100 500,100 500,300 100,300"/>
            <TextLine id="line_1">
                <Coords points="110,110 490,110 490,150 110,150"/>
            </TextLine>
            <TextLine id="line_2">
                <Coords points="110,160 490,160 490,200 110,200"/>
            </TextLine>
            <TextLine id="line_3">
                <Coords points="110,210 490,210 490,250 110,250"/>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"""


def create_mock_xml_with_baselines() -> str:
    """
    Create a mock XML with Baseline elements.

    :return: XML string
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion id="region_1">
            <TextLine id="line_1">
                <Coords points="110,110 490,110 490,150 110,150"/>
                <Baseline points="110,130 490,130"/>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"""


def create_malicious_xxe_xml() -> str:
    """
    Create a malicious XML with XXE payload for security testing.

    :return: XML string with XXE
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion id="region_1">
            <Coords points="&xxe;"/>
        </TextRegion>
    </Page>
</PcGts>"""


def create_invalid_xml() -> str:
    """
    Create invalid XML for error testing.

    :return: Invalid XML string
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion id="region_1">
            <Coords points="100,100 200,100 200,200 100,200"
        </TextRegion>
    </Page>
<!-- Missing closing tag -->"""


def create_mock_kraken_segmentation_result():
    """
    Create a mock Kraken segmentation result.

    :return: Mock segmentation object
    """
    from collections import namedtuple

    Line = namedtuple("Line", ["baseline", "boundary"])
    Segmentation = namedtuple("Segmentation", ["lines"])

    lines = [
        Line(
            baseline=[(110, 130), (490, 130)],
            boundary=[(110, 110), (490, 110), (490, 150), (110, 150)],
        ),
        Line(
            baseline=[(110, 230), (490, 230)],
            boundary=[(110, 210), (490, 210), (490, 250), (110, 250)],
        ),
    ]

    return Segmentation(lines=lines)


def create_mock_collection_with_pages():
    """
    Create a mock htrflow Collection object.

    :return: Mock Collection
    """
    from unittest.mock import MagicMock

    mock_collection = MagicMock()
    mock_collection.pages = [MagicMock(), MagicMock()]
    return mock_collection


def create_mock_empty_collection():
    """
    Create a mock empty htrflow Collection.

    :return: Mock empty Collection
    """
    from unittest.mock import MagicMock

    mock_collection = MagicMock()
    mock_collection.pages = []
    return mock_collection


def create_mock_dataset_example() -> dict:
    """
    Create a mock HuggingFace dataset example.

    :return: Dictionary with image and xml
    """
    return {"image": create_mock_image_array(), "xml": create_mock_xml_with_textlines()}
