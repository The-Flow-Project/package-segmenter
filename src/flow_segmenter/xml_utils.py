"""
XML Utility functions for PageXML manipulation.
"""

import copy

import lxml.etree as et
from loguru import logger

from .exceptions import InvalidXMLError, PageNotFoundError


class XMLUtils:
    """Utility class for XML operations on PageXML documents."""

    @staticmethod
    def get_xml_namespace(xml_etree: et._Element) -> dict[str, str]:
        """
        Extract the namespace from an XML element.

        :param xml_etree: XML tree element
        :return: Dictionary {'ns': 'namespace_uri'} with the namespace URI
        """
        nsmap = {k or "ns": v for k, v in xml_etree.nsmap.items()}
        return nsmap

    @staticmethod
    def merge_xml_pages(
        existing_etree: et._Element,
        new_etree: et._Element,
    ) -> et._Element:
        """
        Merge two PageXML documents by replacing the <Page> element.

        This replaces the <Page> element in the existing XML with the <Page>
        element from the new XML, preserving the rest of the existing structure.

        :param existing_etree: Existing XML tree (will be modified)
        :param new_etree: New XML tree containing the updated <Page>
        :return: Modified existing_etree with new <Page> element
        :raises PageNotFoundError: If <Page> element is not found in either XML
        """
        existing_etree = copy.deepcopy(existing_etree)
        new_etree = copy.deepcopy(new_etree)

        namespace_existing = XMLUtils.get_xml_namespace(existing_etree)
        namespace_new = XMLUtils.get_xml_namespace(new_etree)

        new_page = new_etree.find(".//ns:Page", namespaces=namespace_new)
        existing_page = existing_etree.find(".//ns:Page", namespaces=namespace_existing)

        if existing_page is None:
            raise PageNotFoundError("No <Page> element found in existing XML")
        if new_page is None:
            raise PageNotFoundError("No <Page> element found in new XML")

        # Remove all children from existing page
        for child in list(existing_page):
            existing_page.remove(child)

        # Copy all elements from new page to existing page
        for element in new_page:
            existing_page.append(copy.deepcopy(element))

        # Update page attributes from new page (e.g. imageFilename, dimensions)
        for attr, val in new_page.attrib.items():
            existing_page.attrib[attr] = val

        return existing_etree

    @staticmethod
    def add_creator_metadata(
        xml_etree: et._Element, creator: str, namespace: dict[str, str] | None = None
    ) -> et._Element:
        """
        Add creator information to the metadata of a PageXML document.

        :param xml_etree: XML tree to modify
        :param creator: Name of the creator to add
        :param namespace: Optional namespace dict (will be extracted if not provided)
        :return: Modified XML tree with creator metadata
        """
        if namespace is None:
            namespace = XMLUtils.get_xml_namespace(xml_etree)

        ns_uri = namespace.get("ns")
        if not ns_uri:
            raise InvalidXMLError("Cannot add metadata: no namespace found in document")

        metadata = xml_etree.find(".//ns:Metadata", namespaces=namespace)

        if metadata is None:
            metadata = et.Element(f"{{{ns_uri}}}Metadata")
            xml_etree.insert(0, metadata)

        creator_el = metadata.find("ns:Creator", namespaces=namespace)
        if creator_el is None:
            creator_el = et.Element(f"{{{ns_uri}}}Creator")
            metadata.insert(0, creator_el)

        creator_el.text = creator

        logger.info(f'Added creator "{creator}" to XML metadata')
        return xml_etree

    @staticmethod
    def convert_textregions_to_textlines(
        xml_etree: et._Element, namespace: dict[str, str] | None = None
    ) -> et._Element:
        """
        Convert TextRegion elements to TextLine if their ID contains 'textline'.

        This is useful for fixing incorrectly classified elements where the
        model output uses 'textline' in TextRegion IDs.

        :param xml_etree: XML tree to modify
        :param namespace: Optional namespace dict (will be extracted if not provided)
        :return: Modified XML tree with converted elements
        """
        if namespace is None:
            namespace = XMLUtils.get_xml_namespace(xml_etree)
        ns_uri = namespace.get("ns")

        textregions = xml_etree.findall(".//ns:TextRegion", namespaces=namespace)

        converted_count = 0
        for tregion in textregions:
            id_tregion = tregion.attrib.get("id", "")
            if id_tregion and "textline" in id_tregion.lower():
                if ns_uri:
                    tregion.tag = f"{{{ns_uri}}}TextLine"
                else:
                    tregion.tag = "TextLine"
                converted_count += 1

        if converted_count > 0:
            logger.info(f"Converted {converted_count} TextRegion(s) to TextLine(s)")

        return xml_etree

    @staticmethod
    def safe_parse_xml(xml_content: bytes, encoding: str = "utf-8") -> et._Element:
        """
        Safely parse XML content with security measures against XXE attacks.

        :param xml_content: XML content as bytes
        :param encoding: Character encoding (default: 'utf-8')
        :return: Parsed XML element tree
        :raises InvalidXMLError: If XML parsing fails
        """
        # AIDEV-NOTE: return type is et._Element; et.fromstring return type is declared as et._Element in lxml stubs
        try:
            return et.fromstring(
                xml_content,
                parser=et.XMLParser(
                    encoding=encoding,
                    ns_clean=True,
                    compact=False,
                    resolve_entities=False,  # Prevent XXE attacks
                ),
            )
        except et.XMLSyntaxError as e:
            raise InvalidXMLError(
                f"Failed to parse XML (invalid or malicious): {e}"
            ) from e

    @staticmethod
    def serialize_xml(xml_etree: et._Element) -> str:
        """
        Serialize XML element tree to string with default encoding (utf-8).

        :param xml_etree: XML element tree to serialize
        :return: XML as string
        :raises InvalidXMLError: If serialization fails
        """
        try:
            result = et.tostring(xml_etree, encoding="unicode")
            return result if isinstance(result, str) else str(result)
        except (et.XMLSyntaxError, TypeError) as e:
            raise InvalidXMLError(f"Cannot serialize XML to string: {e}") from e
