"""
XML Utility functions for PageXML manipulation.
"""

import copy
import logging

import lxml.etree as et

from .exceptions import InvalidXMLError, PageNotFoundError

logger = logging.getLogger(__name__)


class XMLUtils:
    """Utility class for XML operations on PageXML documents."""

    @staticmethod
    def get_xml_namespace(xml_etree: et.Element) -> dict[str, str]:
        """
        Extract the namespace from an XML element.

        :param xml_etree: XML tree element
        :return: Dictionary {'ns': 'namespace_uri'} with the namespace URI
        """
        root = copy.deepcopy(xml_etree)
        return root.nsmap if None not in root.nsmap else {"ns": root.nsmap[None]}

    @staticmethod
    def merge_xml_pages(
            existing_etree: et.Element,
            new_etree: et.Element,
            namespace_existing: dict[str, str],
            namespace_new: dict[str, str],
            remove_namespaces: bool = True,
    ) -> et.Element:
        """
        Merge two PageXML documents by replacing the <Page> element.

        This replaces the <Page> element in the existing XML with the <Page>
        element from the new XML, preserving the rest of the existing structure.

        :param existing_etree: Existing XML tree (will be modified)
        :param new_etree: New XML tree containing the updated <Page>
        :param namespace_existing: Namespace dict of existing XML
        :param namespace_new: Namespace dict of new XML
        :param remove_namespaces: Whether to remove namespaces from the result
        :return: Modified existing_etree with new <Page> element
        :raises PageNotFoundError: If <Page> element is not found in either XML
        """
        existing_etree = copy.deepcopy(existing_etree)
        existing_root = existing_etree
        new_root = new_etree

        new_page = new_root.find(".//ns:Page", namespaces=namespace_new)
        existing_page = existing_root.find(".//ns:Page", namespaces=namespace_existing)

        # logger.debug(f"Existing page: {existing_page}")
        # logger.debug(f"New page: {new_page}")

        if existing_page is None:
            raise PageNotFoundError("No <Page> element found in existing XML")
        if new_page is None:
            raise PageNotFoundError("No <Page> element found in new XML")

        # Remove all children from existing page
        for child in list(existing_page):
            existing_page.remove(child)

        if remove_namespaces:
            new_page = XMLUtils._remove_namespaces(new_page)

        # Copy all elements from new page to existing page
        for element in new_page:
            existing_page.append(copy.deepcopy(element))

        return existing_etree

    @staticmethod
    def add_creator_metadata(
            xml_etree: et.Element, creator: str, namespace: dict[str, str] | None = None
    ) -> et.Element:
        """
        Add creator information to the metadata of a PageXML document.

        :param xml_etree: XML tree to modify
        :param creator: Name of the creator to add
        :param namespace: Optional namespace dict (will be extracted if not provided)
        :return: Modified XML tree with creator metadata
        """
        if namespace is None:
            namespace = XMLUtils.get_xml_namespace(xml_etree)

        metadata = xml_etree.find(".//ns:Metadata", namespaces=namespace)

        if metadata is None:
            metadata = et.Element("Metadata", nsmap=namespace)
            xml_etree.insert(0, metadata)

        creator_el = xml_etree.find(".//ns:Creator", namespaces=namespace)
        if creator_el is None:
            creator_el = et.Element("Creator")
            metadata.insert(0, creator_el)

        creator_el.text = creator

        logger.info(f'Added creator "{creator}" to XML metadata')
        return xml_etree

    @staticmethod
    def convert_textregions_to_textlines(
            xml_etree: et.Element, namespace: dict[str, str] | None = None
    ) -> et.Element:
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

        textregions = xml_etree.findall(".//ns:TextRegion", namespaces=namespace)

        converted_count = 0
        for tregion in textregions:
            id_tregion = tregion.attrib.get("id", "")
            if id_tregion and "textline" in id_tregion.lower():
                tregion.tag = f'TextLine'
                converted_count += 1

        if converted_count > 0:
            logger.info(f"Converted {converted_count} TextRegion(s) to TextLine(s)")

        return xml_etree

    @staticmethod
    def safe_parse_xml(xml_content: bytes, encoding: str = "utf-8") -> et.Element:
        """
        Safely parse XML content with security measures against XXE attacks.

        :param xml_content: XML content as bytes
        :param encoding: Character encoding (default: 'utf-8')
        :return: Parsed XML element tree
        :raises InvalidXMLError: If XML parsing fails
        """
        try:
            return et.fromstring(
                xml_content,
                parser=et.XMLParser(
                    encoding=encoding,
                    ns_clean=True,
                    compact=False,
                    resolve_entities=False,  # Prevent XXE attacks
                    # no_network=True,  # Disable network access
                ),
            )
        except et.XMLSyntaxError as e:
            raise InvalidXMLError(f"Failed to parse XML: {e}")

    @staticmethod
    def serialize_xml(xml_etree: et.Element, encoding: str = "utf-8") -> str:
        """
        Serialize XML element tree to string.

        :param xml_etree: XML element tree to serialize
        :param encoding: Character encoding (default: 'utf-8')
        :return: XML as string
        :raises InvalidXMLError: If serialization fails
        """
        try:
            return et.tostring(xml_etree, encoding=encoding).decode(encoding)
        except (et.XMLSyntaxError, TypeError) as e:
            raise InvalidXMLError(f"Cannot serialize XML to string: {e}")

    @staticmethod
    def _remove_namespaces(xml_etree: et.Element) -> et.Element:
        """
        Remove namespaces from an XML element tree.

        :param xml_etree: XML element tree to modify
        :return: Modified XML element tree without namespaces
        """
        for elem in xml_etree.iter():
            elem.tag = et.QName(elem).localname
            if elem.prefix:
                elem.attrib.pop('xmlns:' + elem.prefix, None)
            else:
                elem.attrib.pop('xmlns:', None)
        et.cleanup_namespaces(xml_etree)
        return xml_etree
