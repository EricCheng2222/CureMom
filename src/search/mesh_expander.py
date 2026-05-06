"""MeSH-based query expansion using tree number hierarchy and synonyms.

Enables hierarchical expansion: querying "diabetes" can automatically expand
to all MeSH child terms (type 1, type 2, gestational, etc.).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class MeSHExpander:
    def __init__(self, db_dsn: str) -> None:
        self._db_dsn = db_dsn
        self._conn: psycopg.Connection | None = None

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._db_dsn, row_factory=dict_row)
        return self._conn

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def search_mesh(self, term: str, limit: int = 10) -> list[dict]:
        """Find MeSH descriptors matching a term (name or synonym)."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT descriptor_ui, descriptor_name, tree_numbers, synonyms
                FROM mesh_terms
                WHERE lower(descriptor_name) = lower(%s)
                   OR lower(%s) = ANY(SELECT lower(s) FROM unnest(synonyms) AS s)
                LIMIT %s
                """,
                (term, term, limit),
            )
            return cur.fetchall()

    def suggest_mesh(self, partial: str, limit: int = 10) -> list[dict]:
        """Autocomplete MeSH descriptor names."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT descriptor_ui, descriptor_name, tree_numbers
                FROM mesh_terms
                WHERE lower(descriptor_name) LIKE lower(%s)
                ORDER BY descriptor_name
                LIMIT %s
                """,
                (f"{partial}%", limit),
            )
            return cur.fetchall()

    def get_descendants(self, descriptor_ui: str) -> list[dict]:
        """Return all MeSH terms that are descendants of the given descriptor.

        Uses tree_numbers prefix matching — e.g. if parent tree is 'C17.800.500',
        all terms with tree numbers starting with 'C17.800.500.' are descendants.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            # Get parent tree numbers first
            cur.execute(
                "SELECT tree_numbers FROM mesh_terms WHERE descriptor_ui = %s",
                (descriptor_ui,),
            )
            row = cur.fetchone()
            if not row or not row["tree_numbers"]:
                return []

            tree_numbers = row["tree_numbers"]

            # Build LIKE patterns for each tree number
            patterns = [f"{tn}.%" for tn in tree_numbers]
            if not patterns:
                return []

            # Match any descendant
            where_clauses = " OR ".join(
                "EXISTS (SELECT 1 FROM unnest(tree_numbers) AS tn WHERE tn LIKE %s)"
                for _ in patterns
            )
            cur.execute(
                f"""
                SELECT descriptor_ui, descriptor_name, tree_numbers
                FROM mesh_terms
                WHERE {where_clauses}
                ORDER BY descriptor_name
                """,
                patterns,
            )
            return cur.fetchall()

    def expand_query_terms(self, terms: list[str], include_descendants: bool = True) -> dict[str, list[str]]:
        """Given a list of query terms, return expanded MeSH descriptor UIs.

        Returns dict mapping input term → list of matching descriptor_ui values
        (including descendants if include_descendants=True).
        """
        expansion: dict[str, list[str]] = {}

        for term in terms:
            matches = self.search_mesh(term)
            uis: list[str] = []

            for match in matches:
                uis.append(match["descriptor_ui"])
                if include_descendants:
                    descendants = self.get_descendants(match["descriptor_ui"])
                    uis.extend(d["descriptor_ui"] for d in descendants)

            if uis:
                expansion[term] = list(set(uis))
                logger.debug("Expanded '%s' to %d MeSH terms", term, len(expansion[term]))

        return expansion
