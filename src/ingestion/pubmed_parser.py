"""Parse PubMed XML records returned by the NCBI EFetch API."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from lxml import etree


@dataclass
class ParsedAuthor:
    last_name: str | None
    fore_name: str | None
    initials: str | None
    orcid: str | None
    affiliations: list[str] = field(default_factory=list)
    is_collective: bool = False


@dataclass
class ParsedMeshTerm:
    descriptor_ui: str
    descriptor_name: str
    qualifier_name: str | None
    is_major_topic: bool


@dataclass
class ParsedPaper:
    pmid: str
    pmcid: str | None
    doi: str | None
    title: str
    abstract: str | None
    abstract_json: dict[str, str]   # {section_label: text}
    pub_year: int | None
    pub_date: str | None            # ISO date string when available
    journal_title: str | None
    journal_abbreviation: str | None
    journal_issn: str | None
    journal_eissn: str | None
    journal_nlm_id: str | None
    publication_types: list[str]
    language: str
    mesh_terms: list[ParsedMeshTerm]
    keywords: list[str]
    authors: list[ParsedAuthor]
    grant_agencies: list[str]
    cited_pmids: list[str]          # reference list
    xml_checksum: str               # SHA-256 of the raw article XML bytes


def _text(element: etree._Element | None, default: str = "") -> str:
    if element is None:
        return default
    parts = [element.text or ""]
    for child in element:
        parts.append(etree.tostring(child, method="text", encoding="unicode"))
    return "".join(parts).strip()


def _find_text(root: etree._Element, xpath: str, default: str | None = None) -> str | None:
    el = root.find(xpath)
    if el is None:
        return default
    return _text(el) or default


def _parse_abstract(article: etree._Element) -> tuple[str | None, dict[str, str]]:
    """Return (plain_abstract, structured_dict)."""
    abstract_el = article.find(".//Abstract")
    if abstract_el is None:
        return None, {}

    sections: dict[str, str] = {}
    texts: list[str] = []

    for at in abstract_el.findall("AbstractText"):
        label = at.get("Label", "").strip()
        content = _text(at)
        if not content:
            continue
        if label:
            sections[label.lower()] = content
            texts.append(f"{label}: {content}")
        else:
            sections["text"] = content
            texts.append(content)

    plain = " ".join(texts) if texts else None
    return plain, sections


def _parse_pub_date(medline_citation: etree._Element) -> tuple[int | None, str | None]:
    """Extract (pub_year, iso_date_str)."""
    # Try PubDate inside JournalIssue first
    pub_date = medline_citation.find(".//JournalIssue/PubDate")
    if pub_date is None:
        pub_date = medline_citation.find(".//PubDate")
    if pub_date is None:
        return None, None

    year_el = pub_date.find("Year")
    month_el = pub_date.find("Month")
    day_el = pub_date.find("Day")
    medline_date_el = pub_date.find("MedlineDate")

    if year_el is not None:
        year = int(_text(year_el))
        month = _text(month_el) if month_el is not None else None
        day = _text(day_el) if day_el is not None else None
        if month and day:
            # Month may be abbreviated name — keep as string
            iso = f"{year}-{month}-{day}"
        else:
            iso = str(year)
        return year, iso

    if medline_date_el is not None:
        raw = _text(medline_date_el)
        # Typically "2021 Jan-Feb" or "2021 Spring"
        try:
            year = int(raw[:4])
            return year, raw
        except (ValueError, IndexError):
            pass

    return None, None


def _parse_authors(article: etree._Element) -> list[ParsedAuthor]:
    authors: list[ParsedAuthor] = []
    for author_el in article.findall(".//AuthorList/Author"):
        collective = author_el.find("CollectiveName")
        if collective is not None:
            authors.append(ParsedAuthor(
                last_name=_text(collective),
                fore_name=None,
                initials=None,
                orcid=None,
                is_collective=True,
            ))
            continue

        affiliations = [
            _text(aff.find("Affiliation")) if aff.find("Affiliation") is not None else _text(aff)
            for aff in author_el.findall(".//AffiliationInfo")
        ]
        orcid_ids = [
            _text(idf)
            for idf in author_el.findall(".//Identifier[@Source='ORCID']")
        ]

        authors.append(ParsedAuthor(
            last_name=_find_text(author_el, "LastName"),
            fore_name=_find_text(author_el, "ForeName"),
            initials=_find_text(author_el, "Initials"),
            orcid=orcid_ids[0] if orcid_ids else None,
            affiliations=[a for a in affiliations if a],
        ))
    return authors


def _parse_mesh(medline_citation: etree._Element) -> list[ParsedMeshTerm]:
    terms: list[ParsedMeshTerm] = []
    for heading in medline_citation.findall(".//MeshHeadingList/MeshHeading"):
        descriptor = heading.find("DescriptorName")
        if descriptor is None:
            continue
        descriptor_ui = descriptor.get("UI", "")
        descriptor_name = _text(descriptor)
        is_major = descriptor.get("MajorTopicYN", "N") == "Y"

        qualifiers = heading.findall("QualifierName")
        if qualifiers:
            for q in qualifiers:
                q_major = q.get("MajorTopicYN", "N") == "Y"
                terms.append(ParsedMeshTerm(
                    descriptor_ui=descriptor_ui,
                    descriptor_name=descriptor_name,
                    qualifier_name=_text(q),
                    is_major_topic=is_major or q_major,
                ))
        else:
            terms.append(ParsedMeshTerm(
                descriptor_ui=descriptor_ui,
                descriptor_name=descriptor_name,
                qualifier_name=None,
                is_major_topic=is_major,
            ))
    return terms


def _parse_elocation(article: etree._Element) -> str | None:
    for el in article.findall(".//ELocationID[@EIdType='doi']"):
        doi = _text(el)
        if doi:
            return doi
    return None


def _parse_grants(article: etree._Element) -> list[str]:
    agencies = []
    for grant in article.findall(".//GrantList/Grant"):
        agency = _find_text(grant, "Agency")
        if agency:
            agencies.append(agency)
    return list(set(agencies))


def _parse_references(pubmed_article: etree._Element) -> list[str]:
    """Extract cited PMIDs from the ReferenceList."""
    pmids: list[str] = []
    for ref in pubmed_article.findall(".//ReferenceList/Reference"):
        for art_id in ref.findall(".//ArticleIdList/ArticleId[@IdType='pubmed']"):
            pmid = _text(art_id).strip()
            if pmid:
                pmids.append(pmid)
    return pmids


def parse_pubmed_article(article_xml_bytes: bytes) -> ParsedPaper | None:
    """Parse a single PubmedArticle XML element (as bytes) into a ParsedPaper.

    The bytes should contain one <PubmedArticle> element.
    Returns None if the record is not a standard article (e.g. book).
    """
    checksum = hashlib.sha256(article_xml_bytes).hexdigest()
    try:
        root = etree.fromstring(article_xml_bytes)
    except etree.XMLSyntaxError:
        return None

    # Handle both bare <PubmedArticle> and wrapped <PubmedArticleSet>
    if root.tag == "PubmedArticleSet":
        article_el = root.find("PubmedArticle")
    elif root.tag == "PubmedArticle":
        article_el = root
    else:
        return None

    if article_el is None:
        return None

    medline = article_el.find("MedlineCitation")
    if medline is None:
        return None
    article = medline.find("Article")
    if article is None:
        return None

    pmid_el = medline.find("PMID")
    if pmid_el is None:
        return None
    pmid = _text(pmid_el).strip()

    # IDs
    pmcid = None
    doi = _parse_elocation(article)
    for art_id in article_el.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = art_id.get("IdType", "")
        val = _text(art_id).strip()
        if id_type == "pmc":
            pmcid = val
        elif id_type == "doi" and doi is None:
            doi = val

    # Title
    title_el = article.find(".//ArticleTitle")
    title = _text(title_el) if title_el is not None else ""

    # Abstract
    abstract, abstract_json = _parse_abstract(article)

    # Dates
    pub_year, pub_date = _parse_pub_date(medline)

    # Journal
    journal = article.find(".//Journal")
    journal_title = _find_text(journal, "Title") if journal is not None else None
    journal_abbrev = _find_text(journal, "ISOAbbreviation") if journal is not None else None
    journal_issn = None
    journal_eissn = None
    if journal is not None:
        for issn_el in journal.findall("ISSN"):
            if issn_el.get("IssnType") == "Print":
                journal_issn = _text(issn_el)
            elif issn_el.get("IssnType") == "Electronic":
                journal_eissn = _text(issn_el)
    journal_nlm_id = _find_text(medline, "MedlineJournalInfo/NlmUniqueID")

    # Publication types
    pub_types = [
        _text(pt)
        for pt in article.findall(".//PublicationTypeList/PublicationType")
        if _text(pt)
    ]

    # Language
    lang_el = article.find("Language")
    language = _text(lang_el)[:3] if lang_el is not None else "eng"

    # MeSH
    mesh_terms = _parse_mesh(medline)

    # Keywords
    keywords = [
        _text(kw)
        for kw in medline.findall(".//KeywordList/Keyword")
        if _text(kw)
    ]

    # Authors
    authors = _parse_authors(article)

    # Grants
    grant_agencies = _parse_grants(article)

    # References
    cited_pmids = _parse_references(article_el)

    return ParsedPaper(
        pmid=pmid,
        pmcid=pmcid,
        doi=doi,
        title=title,
        abstract=abstract,
        abstract_json=abstract_json,
        pub_year=pub_year,
        pub_date=pub_date,
        journal_title=journal_title,
        journal_abbreviation=journal_abbrev,
        journal_issn=journal_issn,
        journal_eissn=journal_eissn,
        journal_nlm_id=journal_nlm_id,
        publication_types=pub_types,
        language=language,
        mesh_terms=mesh_terms,
        keywords=keywords,
        authors=authors,
        grant_agencies=grant_agencies,
        cited_pmids=cited_pmids,
        xml_checksum=checksum,
    )


def parse_pubmed_xml_batch(xml_bytes: bytes) -> list[ParsedPaper]:
    """Parse a full EFetch XML response containing multiple PubmedArticle elements."""
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"Malformed PubMed XML: {exc}") from exc

    articles = root.findall("PubmedArticle") if root.tag == "PubmedArticleSet" else [root]
    results: list[ParsedPaper] = []
    for article_el in articles:
        raw = etree.tostring(article_el)
        paper = parse_pubmed_article(raw)
        if paper is not None:
            results.append(paper)
    return results
