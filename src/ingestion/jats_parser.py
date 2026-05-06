"""Parse PMC full-text articles in JATS/NLM XML format."""

from __future__ import annotations

from dataclasses import dataclass, field

from lxml import etree

# Section type normalization map — maps common title variants to canonical types
_SECTION_TYPE_MAP: dict[str, str] = {
    # Introduction
    "introduction": "introduction",
    "background": "introduction",
    "intro": "introduction",
    # Methods
    "methods": "methods",
    "materials and methods": "methods",
    "material and methods": "methods",
    "patients and methods": "methods",
    "subjects and methods": "methods",
    "study design": "methods",
    "experimental procedures": "methods",
    "experimental design": "methods",
    "methodology": "methods",
    "methods and materials": "methods",
    # Results
    "results": "results",
    "findings": "results",
    "outcomes": "results",
    # Discussion
    "discussion": "discussion",
    "interpretation": "discussion",
    # Conclusion
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "concluding remarks": "conclusion",
    "summary": "conclusion",
    # Abstract (when it appears in body)
    "abstract": "abstract",
}

_KNOWN_SECTION_TYPES = {"introduction", "methods", "results", "discussion", "conclusion", "abstract"}


@dataclass
class JATSSection:
    section_type: str   # normalized canonical type
    section_order: int
    title: str | None
    content: str        # plain text, tags stripped


@dataclass
class ParsedFullText:
    pmcid: str | None
    pmid: str | None
    sections: list[JATSSection] = field(default_factory=list)


def _normalize_section_type(title_text: str) -> str:
    key = title_text.strip().lower().rstrip(".")
    return _SECTION_TYPE_MAP.get(key, "other")


def _get_section_type(sec_el: etree._Element) -> str:
    """Determine section type from sec-type attribute or title text."""
    sec_type = sec_el.get("sec-type", "").lower()
    if sec_type:
        # JATS sec-type often matches canonical names directly
        for canonical in _KNOWN_SECTION_TYPES:
            if canonical in sec_type:
                return canonical

    title_el = sec_el.find("title")
    if title_el is not None:
        title_text = "".join(title_el.itertext()).strip()
        return _normalize_section_type(title_text)

    return "other"


def _extract_text(element: etree._Element) -> str:
    """Extract plain text from a JATS element, skipping figure/table/ref elements."""
    _SKIP_TAGS = {"fig", "table-wrap", "supplementary-material", "ref-list", "ref", "xref"}

    parts: list[str] = []

    def _walk(el: etree._Element) -> None:
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if tag in _SKIP_TAGS:
            return
        if el.text:
            parts.append(el.text)
        for child in el:
            _walk(child)
            if child.tail:
                parts.append(child.tail)

    _walk(element)
    return " ".join(" ".join(parts).split())  # normalize whitespace


def _extract_sections(body: etree._Element, order_start: int = 0) -> list[JATSSection]:
    sections: list[JATSSection] = []
    order = order_start

    for sec in body.findall("sec"):
        sec_type = _get_section_type(sec)
        title_el = sec.find("title")
        title_text = "".join(title_el.itertext()).strip() if title_el is not None else None

        # Get direct paragraph content (not nested sec content)
        paragraphs: list[str] = []
        for child in sec:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag in ("p", "list"):
                text = _extract_text(child)
                if text:
                    paragraphs.append(text)
            elif tag == "sec":
                # Nested section — flatten into parent with its own entry
                pass  # handled below by recursion

        content = " ".join(paragraphs).strip()

        # Only add if there's meaningful content
        if content:
            sections.append(JATSSection(
                section_type=sec_type,
                section_order=order,
                title=title_text,
                content=content,
            ))
            order += 1

        # Recurse into nested sections
        nested = _extract_sections(sec, order_start=order)
        sections.extend(nested)
        order += len(nested)

    return sections


def parse_jats_xml(xml_bytes: bytes) -> ParsedFullText | None:
    """Parse a JATS XML article into sections.

    Returns None if the XML is malformed or does not contain a body.
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return None

    # Handle namespace-prefixed roots
    article = root if (root.tag == "article" or root.tag.endswith("}article")) else root.find(".//article")
    if article is None:
        return None

    # Extract IDs
    pmcid = None
    pmid = None
    for art_id in article.findall(".//front/article-meta/article-id"):
        id_type = art_id.get("pub-id-type", "")
        val = (art_id.text or "").strip()
        if id_type == "pmc":
            pmcid = val
        elif id_type == "pmid":
            pmid = val

    sections: list[JATSSection] = []
    order = 0

    # Abstract from front matter
    abstract_el = article.find(".//front/article-meta/abstract")
    if abstract_el is not None:
        abstract_text = _extract_text(abstract_el)
        if abstract_text:
            sections.append(JATSSection(
                section_type="abstract",
                section_order=order,
                title="Abstract",
                content=abstract_text,
            ))
            order += 1

    # Body sections
    body = article.find("body")
    if body is not None:
        body_sections = _extract_sections(body, order_start=order)
        sections.extend(body_sections)

    return ParsedFullText(pmcid=pmcid, pmid=pmid, sections=sections)
