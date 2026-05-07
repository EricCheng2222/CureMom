"""Elasticsearch client — index setup and BM25 search."""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

INDEX_NAME = "papers"

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "similarity": {
            "default": {
                "type": "BM25",
                "b": 0.75,
                "k1": 1.2,
            }
        },
        "analysis": {
            "analyzer": {
                "biomedical": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "biomedical_synonyms"],
                },
            },
            "filter": {
                "biomedical_synonyms": {
                    "type": "synonym_graph",
                    # Synonyms loaded from file in production; inline for now
                    "synonyms": [
                        "SLE, systemic lupus erythematosus, lupus",
                        "HCQ, hydroxychloroquine",
                        "MMF, mycophenolate mofetil",
                        "ANA, antinuclear antibody, antinuclear antibodies",
                        "anti-dsDNA, anti-double-stranded DNA",
                    ],
                }
            },
        },
    },
    "mappings": {
        "properties": {
            "pmid":              {"type": "keyword"},
            "pmcid":             {"type": "keyword"},
            "doi":               {"type": "keyword"},
            "title":             {"type": "text", "analyzer": "english",
                                  "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
            "abstract":          {"type": "text", "analyzer": "english"},
            "full_text":         {"type": "text", "analyzer": "english", "index_options": "offsets"},
            "mesh_terms":        {"type": "keyword"},
            "mesh_major_terms":  {"type": "keyword"},
            "publication_types": {"type": "keyword"},
            "pub_year":          {"type": "integer"},
            "journal_title":     {"type": "keyword"},
            "language":          {"type": "keyword"},
            "authors":           {"type": "text"},
            "grant_agencies":    {"type": "keyword"},
            "has_full_text":     {"type": "boolean"},
        }
    },
}


def get_client(es_host: str) -> Elasticsearch:
    return Elasticsearch(es_host, request_timeout=30)


def ensure_index(es: Elasticsearch, index: str = INDEX_NAME) -> None:
    """Create the papers index if it doesn't exist."""
    if not es.indices.exists(index=index):
        es.indices.create(index=index, body=INDEX_MAPPING)
        logger.info("Created Elasticsearch index: %s", index)
    else:
        logger.debug("Elasticsearch index already exists: %s", index)


def index_paper(es: Elasticsearch, paper_doc: dict[str, Any], index: str = INDEX_NAME) -> None:
    """Index a single paper document. paper_doc must contain 'pmid'."""
    es.index(index=index, id=paper_doc["pmid"], document=paper_doc)


def bulk_index_papers(
    es: Elasticsearch, paper_docs: list[dict[str, Any]], index: str = INDEX_NAME
) -> tuple[int, int]:
    """Bulk index papers. Returns (success_count, error_count)."""
    from elasticsearch.helpers import bulk

    actions = [
        {"_op_type": "index", "_index": index, "_id": doc["pmid"], "_source": doc}
        for doc in paper_docs
    ]
    success, errors = bulk(es, actions, raise_on_error=False, stats_only=False)
    error_count = len(errors) if isinstance(errors, list) else errors
    return success, error_count


_QUERY_FILLER_RE = __import__("re").compile(
    r"^\s*(?:"
    r"please\s+)?"
    r"(?:"
    r"can\s+you\s+(?:tell\s+me|explain|describe)|"
    r"could\s+you\s+(?:tell\s+me|explain|describe)|"
    r"tell\s+me\s+about|"
    r"explain(?:\s+to\s+me)?|"
    r"describe|"
    r"what\s+(?:is|are|does|do|helps?|happens?|causes?)|"
    r"how\s+(?:does|do|is|are|can|to)|"
    r"why\s+(?:does|do|is|are)|"
    r"where\s+(?:does|do|is|are|can)|"
    r"when\s+(?:does|do|is|are)|"
    r"i\s+(?:want|need|would\s+like)\s+to\s+(?:know|understand|learn)|"
    r"i\s+(?:wonder|am\s+wondering)"
    r")\s+",
    __import__("re").IGNORECASE,
)


def _strip_query_filler(query: str) -> str:
    """Strip natural-language preamble so BM25 sees content terms only.

    'tell me about muscle growth' -> 'muscle growth'
    'how do muscles grow'         -> 'muscles grow'
    'what is lupus'               -> 'lupus'

    Falls back to the original query if stripping leaves it too short.
    """
    cleaned = _QUERY_FILLER_RE.sub("", query).strip(" ?.!,")
    if len(cleaned.split()) < 1:
        return query
    return cleaned


def search_bm25(
    es: Elasticsearch,
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 20,
    index: str = INDEX_NAME,
) -> list[dict[str, Any]]:
    """BM25 search across abstract, mesh_terms, full_text, title.

    filters: optional dict with keys like pub_year_from, pub_year_to,
             publication_types (list), mesh_terms (list), language.
    Returns list of {pmid, score, highlight} dicts.
    """
    cleaned_query = _strip_query_filler(query)

    must_clauses: list[dict] = [
        {
            "multi_match": {
                "query": cleaned_query,
                # Field weighting: content fields dominate; title kept low so
                # papers with interrogative titles ("How do X work?") don't
                # match natural-language queries by tone alone.
                #   abstract     2.0 — primary content
                #   mesh_terms   2.0 — curated, normalized concept tags
                #   full_text    1.5 — secondary content
                #   title        0.5 — tiebreaker only
                "fields": ["abstract^2", "mesh_terms^2", "full_text^1.5", "title^0.5"],
                # cross_fields lets multi-word queries match across fields
                # (e.g. "muscle" in title + "growth" in abstract counts).
                "type": "cross_fields",
                # At least half the query terms must match — generous on
                # natural-language phrasing while still requiring real overlap.
                "minimum_should_match": "50%",
            }
        }
    ]

    filter_clauses: list[dict] = []
    if filters:
        if "pub_year_from" in filters or "pub_year_to" in filters:
            year_range: dict[str, int] = {}
            if "pub_year_from" in filters:
                year_range["gte"] = filters["pub_year_from"]
            if "pub_year_to" in filters:
                year_range["lte"] = filters["pub_year_to"]
            filter_clauses.append({"range": {"pub_year": year_range}})

        if filters.get("publication_types"):
            filter_clauses.append({"terms": {"publication_types": filters["publication_types"]}})

        if filters.get("mesh_terms"):
            filter_clauses.append({"terms": {"mesh_terms": filters["mesh_terms"]}})

        if filters.get("language"):
            filter_clauses.append({"term": {"language": filters["language"]}})

    es_query: dict[str, Any] = {
        "bool": {
            "must": must_clauses,
            "filter": filter_clauses,
        }
    }

    response = es.search(
        index=index,
        query=es_query,
        size=top_k,
        highlight={
            "fields": {
                "title": {"number_of_fragments": 0},
                "abstract": {"number_of_fragments": 2, "fragment_size": 200},
            }
        },
        _source=["pmid", "title", "pub_year", "journal_title", "publication_types", "mesh_terms"],
    )

    results: list[dict[str, Any]] = []
    for hit in response["hits"]["hits"]:
        results.append({
            "pmid": hit["_source"]["pmid"],
            "es_score": hit["_score"],
            "title": hit["_source"].get("title", ""),
            "pub_year": hit["_source"].get("pub_year"),
            "journal_title": hit["_source"].get("journal_title"),
            "publication_types": hit["_source"].get("publication_types", []),
            "mesh_terms": hit["_source"].get("mesh_terms", []),
            "highlights": hit.get("highlight", {}),
        })
    return results
