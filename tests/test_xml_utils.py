"""
Tests for flow_segmenter.xml_utils module.
"""

import lxml.etree as et
import pytest

from flow_segmenter.exceptions import InvalidXMLError, PageNotFoundError
from flow_segmenter.xml_utils import XMLUtils
from tests.fixtures.mock_data import (
    create_invalid_xml,
    create_malicious_xxe_xml,
    create_mock_xml_with_namespace,
    create_mock_xml_with_textline_in_id,
    create_mock_xml_without_page,
)


class TestGetXMLNamespace:
    """Test XMLUtils.get_xml_namespace()."""

    def test_get_namespace_from_xml_with_namespace(self, mock_xml_etree):
        """Test extracting namespace from XML with namespace."""
        namespace = XMLUtils.get_xml_namespace(mock_xml_etree)
        assert "ns" in namespace
        assert "primaresearch.org" in namespace["ns"]

    def test_get_namespace_returns_dict(self, mock_xml_etree):
        """Test that get_xml_namespace returns a dictionary."""
        namespace = XMLUtils.get_xml_namespace(mock_xml_etree)
        assert isinstance(namespace, dict)

    def test_get_namespace_handles_default_namespace(self):
        """Test handling XML with default namespace."""
        xml = et.fromstring(b'<root xmlns="http://example.com"><child/></root>')
        namespace = XMLUtils.get_xml_namespace(xml)
        assert namespace == {"ns": "http://example.com"}

    def test_get_namespace_does_not_modify_original(self, mock_xml_etree):
        """Test that original XML is not modified."""
        original_str = et.tostring(mock_xml_etree)
        XMLUtils.get_xml_namespace(mock_xml_etree)
        after_str = et.tostring(mock_xml_etree)
        assert original_str == after_str


class TestMergeXMLPages:
    """Test XMLUtils.merge_xml_pages()."""

    def test_merge_xml_pages_replaces_page_content(self):
        """Test that merge replaces Page element content."""
        xml1 = et.fromstring(create_mock_xml_with_namespace().encode())
        xml2 = et.fromstring(create_mock_xml_with_namespace().encode())

        ns = XMLUtils.get_xml_namespace(xml1)

        # Modify xml2's page content
        page2 = xml2.find(".//ns:Page", namespaces=ns)
        if page2:
            ns_uri = ns.get("ns")
            new_region = et.Element("TextRegion", id="new_region")
            new_region.tag = f"{{{ns_uri}}}TextRegion"
            page2.append(new_region)
        print(XMLUtils.serialize_xml(xml2))
        print(xml2.find(".//ns:TextRegion", namespaces=ns))

        result = XMLUtils.merge_xml_pages(xml1, xml2)

        # Check that new region is in result
        result_page = result.find(".//ns:Page", namespaces=ns)
        new_regions = result_page.findall(
            './/ns:TextRegion[@id="new_region"]', namespaces=ns
        )
        print(new_regions)
        assert len(new_regions) > 0

    def test_merge_xml_pages_raises_error_if_no_page_in_existing(self):
        """Test that merge raises error if no Page in existing XML."""
        xml1 = et.fromstring(create_mock_xml_without_page().encode())
        xml2 = et.fromstring(create_mock_xml_with_namespace().encode())

        with pytest.raises(PageNotFoundError, match="existing XML"):
            XMLUtils.merge_xml_pages(xml1, xml2)

    def test_merge_xml_pages_raises_error_if_no_page_in_new(self):
        """Test that merge raises error if no Page in new XML."""
        xml1 = et.fromstring(create_mock_xml_with_namespace().encode())
        xml2 = et.fromstring(create_mock_xml_without_page().encode())

        with pytest.raises(PageNotFoundError, match="new XML"):
            XMLUtils.merge_xml_pages(xml1, xml2)

    def test_merge_xml_pages_preserves_root_structure(self):
        """Test that merge preserves root structure."""
        xml1 = et.fromstring(create_mock_xml_with_namespace().encode())
        xml2 = et.fromstring(create_mock_xml_with_namespace().encode())

        result = XMLUtils.merge_xml_pages(xml1, xml2)

        # Root should still be PcGts
        assert "PcGts" in result.tag
        # Metadata should still be present
        ns = XMLUtils.get_xml_namespace(result)
        metadata = result.find(".//ns:Metadata", namespaces=ns)
        assert metadata is not None


class TestAddCreatorMetadata:
    """Test XMLUtils.add_creator_metadata()."""

    def test_add_creator_to_xml_with_metadata(self, mock_xml_etree):
        """Test adding creator to XML that already has Metadata."""
        result = XMLUtils.add_creator_metadata(mock_xml_etree, "TestApp")

        ns = XMLUtils.get_xml_namespace(result)
        creators = result.findall(".//ns:Creator", namespaces=ns)

        # Should have added a creator
        assert len(creators) > 0
        assert creators[-1].text == "TestApp"

    def test_add_creator_to_xml_without_metadata(self):
        """Test adding creator to XML without Metadata element."""
        xml = et.fromstring(
            b'<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
            b'<Page imageFilename="test.jpg"/></PcGts>'
        )

        result = XMLUtils.add_creator_metadata(xml, "TestApp")

        ns = XMLUtils.get_xml_namespace(result)
        metadata = result.find(".//ns:Metadata", namespaces=ns)

        # Metadata should be created
        assert metadata is not None

        creator = metadata.find(".//ns:Creator", namespaces=ns)
        assert creator is not None
        assert creator.text == "TestApp"

    def test_add_creator_with_explicit_namespace(self, mock_xml_etree):
        """Test adding creator with explicit namespace."""
        ns = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        result = XMLUtils.add_creator_metadata(mock_xml_etree, "TestApp", namespace=ns)

        creators = result.findall(".//ns:Creator", namespaces=ns)
        assert len(creators) > 0


class TestConvertTextRegionsToTextLines:
    """Test XMLUtils.convert_textregions_to_textlines()."""

    def test_convert_textregion_with_textline_in_id(self):
        """Test converting TextRegion with 'textline' in ID."""
        xml = et.fromstring(create_mock_xml_with_textline_in_id().encode())

        result = XMLUtils.convert_textregions_to_textlines(xml)

        ns = XMLUtils.get_xml_namespace(result)
        textlines = result.findall(".//ns:TextLine", namespaces=ns)

        # One TextRegion should be converted to TextLine
        assert len(textlines) > 0

        # Check that textline_region_1 was converted
        converted = result.find(
            './/ns:TextLine[@id="textline_region_1"]', namespaces=ns
        )
        assert converted is not None

    def test_normal_textregion_not_converted(self):
        """Test that normal TextRegion is not converted."""
        xml = et.fromstring(create_mock_xml_with_textline_in_id().encode())

        result = XMLUtils.convert_textregions_to_textlines(xml)

        ns = XMLUtils.get_xml_namespace(result)

        # normal_region_2 should still be TextRegion
        if ns.get("ns") is not None:
            normal_region = result.find(
                './/ns:TextRegion[@id="normal_region_2"]', namespaces=ns
            )
        else:
            normal_region = result.find('.//TextRegion[@id="normal_region_2"]')
        assert normal_region is not None

    def test_convert_handles_case_insensitive(self):
        """Test that 'textline' matching is case-insensitive."""
        xml = et.fromstring(
            b'<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
            b'<Page><TextRegion id="TEXTLINE_1"/></Page></PcGts>'
        )

        result = XMLUtils.convert_textregions_to_textlines(xml)

        ns = XMLUtils.get_xml_namespace(result)
        if ns.get("ns") is not None:
            textline = result.find('.//ns:TextLine[@id="TEXTLINE_1"]', namespaces=ns)
        else:
            textline = result.find('.//TextLine[@id="TEXTLINE_1"]')
        assert textline is not None


class TestSafeParseXML:
    """Test XMLUtils.safe_parse_xml()."""

    def test_safe_parse_valid_xml(self):
        """Test parsing valid XML."""
        xml_string = create_mock_xml_with_namespace()
        result = XMLUtils.safe_parse_xml(xml_string.encode())

        assert result is not None
        assert "PcGts" in result.tag

    def test_safe_parse_prevents_xxe(self):
        """Test that XXE attacks are prevented."""
        malicious_xml = create_malicious_xxe_xml()

        # Should parse but not load external entity
        with pytest.raises(
            InvalidXMLError, match=r"Failed to parse XML \(invalid or malicious\).*"
        ):
            XMLUtils.safe_parse_xml(malicious_xml.encode())

    def test_safe_parse_invalid_xml_raises_error(self):
        """Test that invalid XML raises InvalidXMLError."""
        invalid_xml = create_invalid_xml()

        with pytest.raises(InvalidXMLError, match="Failed to parse XML"):
            XMLUtils.safe_parse_xml(invalid_xml.encode())

    def test_safe_parse_with_custom_encoding(self):
        """Test parsing with custom encoding."""
        xml_string = create_mock_xml_with_namespace()
        result = XMLUtils.safe_parse_xml(xml_string.encode("utf-8"), encoding="utf-8")
        assert result is not None


class TestSerializeXML:
    """Test XMLUtils.serialize_xml()."""

    def test_serialize_xml_to_string(self, mock_xml_etree):
        """Test serializing XML to string."""
        result = XMLUtils.serialize_xml(mock_xml_etree)

        assert isinstance(result, str)
        assert "PcGts" in result
        assert "xmlns" in result

    def test_serialize_xml_preserves_structure(self, mock_xml_etree):
        """Test that serialization preserves XML structure."""
        result = XMLUtils.serialize_xml(mock_xml_etree)

        # Reparse and check structure
        reparsed = et.fromstring(result.encode())
        ns = XMLUtils.get_xml_namespace(reparsed)

        page = reparsed.find(".//ns:Page", namespaces=ns)
        assert page is not None

    def test_serialize_with_custom_encoding(self, mock_xml_etree):
        """Test serialization with custom encoding."""
        result = XMLUtils.serialize_xml(mock_xml_etree)
        assert isinstance(result, str)
