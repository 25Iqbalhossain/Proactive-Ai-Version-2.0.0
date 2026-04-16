"""
ingestion/sql_query_executor.py – SQL query executor with schema-aware query building

Accepts either a user-provided raw SQL string or constructs an interaction
query automatically from discovered schema metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from ingestion.db_connector import DBConnector
from config.database import SQLConfig
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class QueryResult:
    df: pd.DataFrame
    sql_used: str
    rows: int
    columns: list[str]
    source_dialect: str


class SQLQueryExecutor:
    """
    High-level query executor that sits on top of DBConnector.

    Supports:
      - Raw custom SQL queries
      - Auto-generated interaction queries from schema metadata
      - Query validation before execution
    """

    # SQL injection blocklist
    _FORBIDDEN_KEYWORDS = {"drop", "delete", "truncate", "alter", "insert", "update", "exec", "execute"}

    def __init__(self, config: SQLConfig):
        self.config    = config
        self._connector = DBConnector(config)

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute_custom(self, sql: str, params: Optional[dict] = None) -> QueryResult:
        """
        Execute a user-provided SQL query with safety validation.

        Parameters
        ----------
        sql    : raw SQL string
        params : named parameters dict (recommended for user inputs)
        """
        self._validate_sql(sql)
        log.info("Executing custom SQL query")

        with self._connector as conn:
            df = conn.query(sql, params)

        return QueryResult(
            df=df,
            sql_used=sql,
            rows=len(df),
            columns=list(df.columns),
            source_dialect=self.config.dialect,
        )

    def execute_interaction_query(
        self,
        user_col: str,
        item_col: str,
        rating_col: Optional[str],
        table: str,
        timestamp_col: Optional[str] = None,
        extra_filters: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> QueryResult:
        """
        Auto-build and execute a standard interaction query.

        Produces:
            SELECT user_col, item_col, [rating_col], [timestamp_col]
            FROM table
            [WHERE extra_filters]
            [LIMIT limit]
        """
        select_cols = [user_col, item_col]
        if rating_col:
            select_cols.append(rating_col)
        if timestamp_col:
            select_cols.append(timestamp_col)

        cols_str = ", ".join(select_cols)
        sql = f"SELECT {cols_str} FROM {table}"
        if extra_filters:
            sql += f" WHERE {extra_filters}"
        if limit:
            sql += f" LIMIT {limit}"

        log.info("Auto-built SQL: %s", sql)
        return self.execute_custom(sql)

    def list_tables(self) -> list[str]:
        """Return all table names in the connected database."""
        with self._connector as conn:
            return conn.get_table_names()

    def describe_table(self, table: str) -> list[dict]:
        """Return column metadata for a given table."""
        with self._connector as conn:
            return conn.get_columns(table)

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate_sql(self, sql: str) -> None:
        """
        Basic SQL injection guard.
        Raises ValueError if forbidden keywords are detected.
        """
        lowered = sql.lower()
        # Strip string literals before checking to avoid false positives
        import re
        cleaned = re.sub(r"'[^']*'", "", lowered)
        cleaned = re.sub(r'"[^"]*"', "", cleaned)

        found = [kw for kw in self._FORBIDDEN_KEYWORDS if f" {kw} " in f" {cleaned} "]
        if found:
            raise ValueError(
                f"SQL contains forbidden keyword(s): {found}. "
                f"Only SELECT queries are permitted."
            )

        if not cleaned.strip().startswith("select") and "with" not in cleaned.strip()[:10]:
            raise ValueError("Only SELECT (or WITH ... SELECT) queries are permitted.")

        log.debug("SQL validation passed")
