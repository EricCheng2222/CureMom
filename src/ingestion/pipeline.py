"""Resumable PubMed ingestion pipeline.

State machine:
  queued → fetching → fetched → parsing → parsed → indexed → done
                                                           → error (retryable)

On restart, any PMID stuck in a transitional state (fetching/parsing/embedding)
for more than STALE_THRESHOLD_MINUTES is re-queued automatically.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Generator

import httpx
import psycopg
from psycopg.rows import dict_row
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .pubmed_parser import ParsedPaper, parse_pubmed_xml_batch
from .topics import IngestionTopic

try:
    from elasticsearch import Elasticsearch
    from ..search.elasticsearch_client import ensure_index, index_paper as es_index_paper
    _ES_AVAILABLE = True
except ImportError:
    _ES_AVAILABLE = False

logger = logging.getLogger(__name__)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{NCBI_BASE}/esearch.fcgi"
EFETCH_URL = f"{NCBI_BASE}/efetch.fcgi"
STALE_THRESHOLD_MINUTES = 60
BATCH_SIZE = 200   # PMIDs per EFetch request (NCBI allows up to 10,000 but 200 is safe)
MAX_RETRIES = 3


class PipelineConfig:
    def __init__(
        self,
        db_dsn: str,
        ncbi_api_key: str | None = None,
        ncbi_email: str = "curemom@example.com",
        requests_per_second: float | None = None,
    ) -> None:
        self.db_dsn = db_dsn
        self.ncbi_api_key = ncbi_api_key
        self.ncbi_email = ncbi_email
        # Rate limiting: 10 req/s with API key, 3 without
        self.requests_per_second = requests_per_second or (10.0 if ncbi_api_key else 3.0)
        self._last_request_time: float = 0.0

    def ncbi_params(self) -> dict[str, str]:
        params: dict[str, str] = {
            "tool": "curemom",
            "email": self.ncbi_email,
            "retmode": "xml",
        }
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        return params

    def throttle(self) -> None:
        min_interval = 1.0 / self.requests_per_second
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()


@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(MAX_RETRIES),
)
def _ncbi_get(client: httpx.Client, url: str, params: dict[str, str]) -> bytes:
    response = client.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.content


def esearch_pmids(
    client: httpx.Client,
    config: PipelineConfig,
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[str]:
    """Return all PMIDs matching a query via ESearch (handles pagination).

    NCBI caps a single ESearch at 9,999 PMIDs even with retstart pagination.
    When the count exceeds that, we recursively halve the date range until
    each window is under the cap. If the caller didn't supply a date range
    and the count is too big, we default to splitting from 1900 to today.
    """
    from datetime import date

    cap = 9999

    def _do_count(df: str | None, dt: str | None) -> tuple[int, str, str]:
        # Returns (count, web_env, query_key) for the window.
        return _esearch_count(client, config, query, df, dt)

    count, web_env, query_key = _do_count(date_from, date_to)
    logger.info(
        "ESearch found %d results for query (date_from=%s, date_to=%s): %s",
        count, date_from or "*", date_to or "*", query,
    )
    if count == 0:
        return []

    if count > cap:
        # Need to split. Default the window if open-ended so we have edges
        # to halve. PubMed's earliest record is from 1781; 1900 is fine as
        # a lower bound for biomedical lit and keeps year-math simple.
        df = date_from or "1900/01/01"
        dt = date_to or date.today().strftime("%Y/%m/%d")
        return _split_and_collect(client, config, query, df, dt, cap)

    # Single-window pagination path.
    return _esearch_pages(client, config, query, web_env, query_key, count)


def _esearch_count(
    client: httpx.Client,
    config: PipelineConfig,
    query: str,
    date_from: str | None,
    date_to: str | None,
) -> tuple[int, str, str]:
    """Just get count + history-server tokens. Retries on backend error."""
    from lxml import etree

    params = {
        **config.ncbi_params(),
        "db": "pubmed",
        "term": query,
        "retmax": "0",
        "usehistory": "y",
    }
    if date_from:
        params["mindate"] = date_from
    if date_to:
        params["maxdate"] = date_to

    import time
    from lxml import etree

    last_err: str | None = None
    for attempt in range(1, 4):
        config.throttle()
        xml = _ncbi_get(client, ESEARCH_URL, params)
        root = etree.fromstring(xml)
        err = root.findtext("ERROR")
        if err:
            last_err = err
            wait = 5 * attempt
            logger.warning(
                "ESearch backend error (attempt %d/3): %s — retrying in %ds",
                attempt, err, wait,
            )
            time.sleep(wait)
            continue
        count = int((root.findtext("Count") or "0"))
        web_env = root.findtext("WebEnv") or ""
        query_key = root.findtext("QueryKey") or ""
        return count, web_env, query_key
    raise RuntimeError(
        f"NCBI esearch count kept returning <ERROR> after 3 attempts: {last_err}"
    )


def _esearch_pages(
    client: httpx.Client,
    config: PipelineConfig,
    query: str,
    web_env: str,
    query_key: str,
    count: int,
) -> list[str]:
    """Page through a history-server result that's known to be ≤ 9,999."""
    import time
    from lxml import etree

    all_pmids: list[str] = []
    retstart = 0
    page_size = 10000

    while retstart < count:
        fetch_params = {
            **config.ncbi_params(),
            "db": "pubmed",
            "WebEnv": web_env,
            "query_key": query_key,
            "retstart": str(retstart),
            "retmax": str(page_size),
            "rettype": "uilist",
        }
        page_pmids: list[str] | None = None
        last_err: str | None = None
        for attempt in range(1, 4):
            config.throttle()
            response = _ncbi_get(client, ESEARCH_URL, fetch_params)
            page_root = etree.fromstring(response)
            err = page_root.findtext("ERROR")
            if err:
                last_err = err
                wait = 5 * attempt
                logger.warning(
                    "ESearch page-fetch error (retstart=%d, attempt %d/3): %s — retrying in %ds",
                    retstart, attempt, err, wait,
                )
                time.sleep(wait)
                continue
            page_pmids = [el.text.strip() for el in page_root.findall(".//Id") if el.text]
            break
        if page_pmids is None:
            raise RuntimeError(
                f"ESearch page-fetch kept returning <ERROR> after 3 attempts "
                f"at retstart={retstart}: {last_err}"
            )
        all_pmids.extend(page_pmids)
        retstart += page_size
        logger.info("Fetched %d/%d PMIDs", len(all_pmids), count)

    return all_pmids


def _split_and_collect(
    client: httpx.Client,
    config: PipelineConfig,
    query: str,
    date_from: str,
    date_to: str,
    cap: int,
) -> list[str]:
    """Recursively halve [date_from, date_to] until each window is ≤ cap,
    then page-fetch each window and concatenate. PMIDs may repeat across
    windows in edge cases (papers indexed twice with different dates) —
    final dedup is handled at queue-insert (UNIQUE constraint).
    """
    from datetime import date, datetime, timedelta

    def _parse(s: str) -> date:
        return datetime.strptime(s, "%Y/%m/%d").date()

    def _fmt(d: date) -> str:
        return d.strftime("%Y/%m/%d")

    df = _parse(date_from)
    dt = _parse(date_to)
    if df > dt:
        return []

    count, web_env, query_key = _esearch_count(client, config, query, _fmt(df), _fmt(dt))
    logger.info(
        "  window %s..%s → %d PMIDs", _fmt(df), _fmt(dt), count,
    )
    if count == 0:
        return []
    if count <= cap:
        return _esearch_pages(client, config, query, web_env, query_key, count)

    # Single-day window that's still over the cap — can't split further;
    # accept the truncation but log loudly.
    if df == dt:
        logger.warning(
            "Window %s alone has %d PMIDs (> cap of %d); first %d will be "
            "kept, the rest lost. Narrow the query if you need full coverage.",
            _fmt(df), count, cap, cap,
        )
        return _esearch_pages(client, config, query, web_env, query_key, cap)

    # Halve the range. Midpoint is the floor.
    days = (dt - df).days
    mid = df + timedelta(days=days // 2)
    left = _split_and_collect(client, config, query, _fmt(df), _fmt(mid), cap)
    right = _split_and_collect(client, config, query, _fmt(mid + timedelta(days=1)), _fmt(dt), cap)
    return left + right


def efetch_batch(
    client: httpx.Client,
    config: PipelineConfig,
    pmids: list[str],
) -> tuple[list[ParsedPaper], list[str]]:
    """Fetch and parse a batch of papers from EFetch.

    Returns (parsed_articles, skipped_book_pmids). Skipped PMIDs are valid
    PubMed entries that aren't journal articles (NCBI Bookshelf monographs,
    gov-agency reports). The caller marks those with status='skipped' so
    they don't sit in the retry queue or inflate the error counter.
    """
    params = {
        **config.ncbi_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    config.throttle()
    xml_bytes = _ncbi_get(client, EFETCH_URL, params)
    from .pubmed_parser import parse_pubmed_xml_batch_with_skipped
    return parse_pubmed_xml_batch_with_skipped(xml_bytes)


def _queue_new_pmids(conn: psycopg.Connection, pmids: list[str]) -> int:
    """Insert PMIDs not yet in ingestion_log. Returns count of newly queued."""
    if not pmids:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO ingestion_log (pmid, status)
            VALUES (%s, 'queued')
            ON CONFLICT (pmid) DO NOTHING
            """,
            [(pmid,) for pmid in pmids],
        )
    conn.commit()
    return cur.rowcount  # type: ignore[return-value]


def _reset_stale_jobs(conn: psycopg.Connection) -> int:
    """Re-queue any jobs stuck in transitional states."""
    stale_before = datetime.now(timezone.utc) - timedelta(minutes=STALE_THRESHOLD_MINUTES)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingestion_log
            SET status = 'queued', error_message = 'reset from stale state'
            WHERE status IN ('fetching', 'parsing', 'embedding')
              AND updated_at < %s
            """,
            (stale_before,),
        )
        count = cur.rowcount
    conn.commit()
    if count:
        logger.info("Reset %d stale ingestion jobs", count)
    return count


def _iter_queued_pmids(
    conn: psycopg.Connection, batch_size: int = BATCH_SIZE
) -> Generator[list[str], None, None]:
    """Yield batches of queued PMIDs, marking them as 'fetching'."""
    while True:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT pmid FROM ingestion_log
                WHERE status = 'queued'
                ORDER BY queued_at
                LIMIT %s
                FOR UPDATE SKIP LOCKED
                """,
                (batch_size,),
            )
            rows = cur.fetchall()

        if not rows:
            break

        pmids = [r["pmid"] for r in rows]

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ingestion_log SET status = 'fetching' WHERE pmid = ANY(%s)",
                (pmids,),
            )
        conn.commit()
        yield pmids


def _upsert_paper(conn: psycopg.Connection, paper: ParsedPaper) -> int:
    """Upsert a parsed paper into PostgreSQL. Returns the internal paper.id."""
    with conn.cursor() as cur:
        # Journal (upsert)
        journal_id = None
        if paper.journal_title:
            cur.execute(
                """
                INSERT INTO journals (title, abbreviation, issn, eissn, nlm_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (nlm_id) DO UPDATE
                    SET title = EXCLUDED.title,
                        abbreviation = EXCLUDED.abbreviation
                RETURNING id
                """,
                (
                    paper.journal_title,
                    paper.journal_abbreviation,
                    paper.journal_issn,
                    paper.journal_eissn,
                    paper.journal_nlm_id,
                ),
            )
            row = cur.fetchone()
            journal_id = row[0] if row else None

        # Paper upsert
        pub_date = None
        if paper.pub_date:
            try:
                # Attempt to parse ISO date; store NULL if unparseable
                from datetime import date
                parts = paper.pub_date.split("-")
                if len(parts) == 3:
                    pub_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
            except (ValueError, TypeError):
                pass

        cur.execute(
            """
            INSERT INTO papers (
                pmid, pmcid, doi, title, abstract, abstract_json,
                pub_year, pub_date, journal_id, publication_types,
                language, grant_agencies, last_updated
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (pmid) DO UPDATE SET
                pmcid              = EXCLUDED.pmcid,
                doi                = EXCLUDED.doi,
                title              = EXCLUDED.title,
                abstract           = EXCLUDED.abstract,
                abstract_json      = EXCLUDED.abstract_json,
                pub_year           = EXCLUDED.pub_year,
                pub_date           = EXCLUDED.pub_date,
                journal_id         = EXCLUDED.journal_id,
                publication_types  = EXCLUDED.publication_types,
                language           = EXCLUDED.language,
                grant_agencies     = EXCLUDED.grant_agencies,
                last_updated       = NOW()
            RETURNING id
            """,
            (
                paper.pmid,
                paper.pmcid,
                paper.doi,
                paper.title,
                paper.abstract,
                psycopg.types.json.Jsonb(paper.abstract_json) if paper.abstract_json else None,
                paper.pub_year,
                pub_date,
                journal_id,
                paper.publication_types,
                paper.language,
                paper.grant_agencies,
            ),
        )
        paper_db_id = cur.fetchone()[0]  # type: ignore[index]

        # Authors
        for pos, author in enumerate(paper.authors):
            cur.execute(
                """
                INSERT INTO authors (last_name, fore_name, initials, orcid)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (last_name, fore_name, orcid) DO UPDATE SET initials = EXCLUDED.initials
                RETURNING id
                """,
                (author.last_name, author.fore_name, author.initials, author.orcid),
            )
            author_id = cur.fetchone()[0]  # type: ignore[index]
            cur.execute(
                """
                INSERT INTO paper_authors (paper_id, author_id, position, affiliations)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (paper_id, author_id) DO NOTHING
                """,
                (paper_db_id, author_id, pos, author.affiliations or None),
            )

        # MeSH terms
        for term in paper.mesh_terms:
            cur.execute(
                """
                INSERT INTO mesh_terms (descriptor_ui, descriptor_name)
                VALUES (%s, %s)
                ON CONFLICT (descriptor_ui) DO NOTHING
                RETURNING id
                """,
                (term.descriptor_ui, term.descriptor_name),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute("SELECT id FROM mesh_terms WHERE descriptor_ui = %s", (term.descriptor_ui,))
                row = cur.fetchone()
            mesh_id = row[0]  # type: ignore[index]

            cur.execute(
                """
                INSERT INTO paper_mesh (paper_id, mesh_id, qualifier_name, is_major_topic)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (paper_db_id, mesh_id, term.qualifier_name or '', term.is_major_topic),
            )

        # References
        for cited_pmid in paper.cited_pmids:
            cur.execute(
                """
                INSERT INTO citations (citing_paper_id, cited_pmid_raw)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                (paper_db_id, cited_pmid),
            )

    return paper_db_id


def _mark_done(conn: psycopg.Connection, pmid: str, checksum: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingestion_log
            SET status = 'indexed', xml_checksum = %s, indexed_at = NOW()
            WHERE pmid = %s
            """,
            (checksum, pmid),
        )
    # No explicit commit — caller is responsible (either inside conn.transaction()
    # or via explicit conn.commit() for the error path).


def _mark_error(conn: psycopg.Connection, pmid: str, message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingestion_log
            SET status = 'error', error_message = %s, retry_count = retry_count + 1
            WHERE pmid = %s
            """,
            (message[:2000], pmid),
        )
    conn.commit()


def _mark_skipped(conn: psycopg.Connection, pmid: str, reason: str) -> None:
    """Terminal state for PMIDs that aren't journal articles — NCBI Bookshelf
    monographs, gov-agency reports, etc. Distinct from 'error' so they don't
    get retried and don't pollute the error counter."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingestion_log
            SET status = 'skipped', error_message = %s, indexed_at = NOW()
            WHERE pmid = %s
            """,
            (reason[:2000], pmid),
        )
    conn.commit()


def _index_paper_to_es(es: "Elasticsearch", paper: ParsedPaper, journal_title: str | None) -> None:
    """Index a single paper to Elasticsearch. Errors are logged, not raised."""
    try:
        authors_text = "; ".join(
            f"{a.last_name}{' ' + a.initials if a.initials else ''}"
            for a in paper.authors[:10]
        )
        doc = {
            "pmid":             paper.pmid,
            "pmcid":            paper.pmcid,
            "doi":              paper.doi,
            "title":            paper.title or "",
            "abstract":         paper.abstract or "",
            "pub_year":         paper.pub_year,
            "journal_title":    journal_title or paper.journal_title,
            "publication_types": paper.publication_types,
            "mesh_terms":       [t.descriptor_name for t in paper.mesh_terms],
            "mesh_major_terms": [t.descriptor_name for t in paper.mesh_terms if t.is_major_topic],
            "language":         paper.language,
            "has_full_text":    False,
            "grant_agencies":   paper.grant_agencies,
            "authors":          authors_text,
        }
        es_index_paper(es, doc)
    except Exception as exc:
        logger.warning("ES index failed for PMID %s: %s", paper.pmid, exc)


def _upsert_abstract_chunk(conn: psycopg.Connection, paper_db_id: int, paper: ParsedPaper) -> None:
    """Insert one abstract chunk for the paper if none exists yet."""
    text = paper.abstract or paper.title or ""
    if not text:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO paper_chunks
                    (paper_id, chunk_index, chunk_text, source_type,
                     start_char, end_char, paragraph_index, token_count)
                VALUES (%s, 0, %s, 'abstract', 0, %s, 0, %s)
                ON CONFLICT DO NOTHING
                """,
                (paper_db_id, text, len(text), len(text) // 4),
            )
    except Exception as exc:
        logger.warning("Chunk insert failed for paper_id %s: %s", paper_db_id, exc)


def run_ingestion(
    config: PipelineConfig,
    topics: list[IngestionTopic],
    date_from: str | None = None,
    date_to: str | None = None,
    dry_run: bool = False,
    es: "Elasticsearch | None" = None,
) -> None:
    """Main entry point. Discovers and ingests papers for the given topics.

    Args:
        config: Pipeline configuration (DB DSN, NCBI credentials, etc.)
        topics: List of IngestionTopic to query
        date_from: Optional NCBI date filter (YYYY/MM/DD)
        date_to: Optional NCBI date filter (YYYY/MM/DD)
        dry_run: If True, discover PMIDs and queue them but don't fetch full records.
    """
    conn = psycopg.connect(config.db_dsn, autocommit=False)
    _reset_stale_jobs(conn)
    if es is not None and _ES_AVAILABLE:
        ensure_index(es)

    with httpx.Client(follow_redirects=True) as client:
        # Phase 1: Discover PMIDs for each topic and queue them
        for topic in topics:
            logger.info("Discovering PMIDs for topic: %s", topic.name)
            try:
                pmids = esearch_pmids(client, config, topic.mesh_query, date_from, date_to)
                new_count = _queue_new_pmids(conn, pmids)
                logger.info("Topic %s: %d PMIDs found, %d newly queued", topic.name, len(pmids), new_count)
            except Exception as exc:
                logger.error("ESearch failed for topic %s: %s", topic.name, exc)

        if dry_run:
            logger.info("Dry run complete — PMIDs queued, skipping fetch.")
            conn.close()
            return

        # Phase 2: Fetch and parse queued papers
        total_done = 0
        total_error = 0
        total_skipped = 0
        for pmid_batch in _iter_queued_pmids(conn):
            logger.info("Fetching batch of %d PMIDs", len(pmid_batch))
            try:
                papers, skipped_pmids = efetch_batch(client, config, pmid_batch)
            except Exception as exc:
                logger.error("EFetch failed for batch: %s", exc)
                for pmid in pmid_batch:
                    _mark_error(conn, pmid, str(exc))
                total_error += len(pmid_batch)
                continue

            fetched_pmids = {p.pmid for p in papers}
            skipped_pmids_set = set(skipped_pmids)
            for paper in papers:
                try:
                    with conn.transaction():
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE ingestion_log SET status = 'parsing' WHERE pmid = %s",
                                (paper.pmid,),
                            )
                        paper_db_id = _upsert_paper(conn, paper)
                        _upsert_abstract_chunk(conn, paper_db_id, paper)
                        _mark_done(conn, paper.pmid, paper.xml_checksum)
                    # ES indexing outside the DB transaction — failure won't roll back PG
                    if es is not None and _ES_AVAILABLE:
                        _index_paper_to_es(es, paper, paper.journal_title)
                    total_done += 1
                except Exception as exc:
                    logger.error("Failed to upsert paper %s: %s", paper.pmid, exc)
                    _mark_error(conn, paper.pmid, str(exc))
                    total_error += 1

            # Mark book/monograph entries as 'skipped' — not retryable, not
            # counted as errors. PMIDs truly missing from the response (NCBI
            # deleted, transient drop) stay as 'error' so a future re-run picks
            # them up.
            for pmid in skipped_pmids_set:
                _mark_skipped(conn, pmid, "not a journal article (book/monograph)")
                total_skipped += 1

            for pmid in pmid_batch:
                if pmid not in fetched_pmids and pmid not in skipped_pmids_set:
                    _mark_error(conn, pmid, "not returned by EFetch")
                    total_error += 1

        logger.info("Ingestion complete. Done: %d, Skipped: %d, Errors: %d",
                    total_done, total_skipped, total_error)

    conn.close()
