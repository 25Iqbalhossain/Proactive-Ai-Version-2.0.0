"""
ingestion/db_connector.py – SQL database connector

Supports: PostgreSQL, MySQL, SQL Server, Snowflake, Redshift
Converts query results directly into pandas DataFrames.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

from config.database import SQLConfig
from utils.logger import get_logger

log = get_logger(__name__)


class DBConnector:
    """
    Manages connection lifecycle to relational databases.

    Usage
    -----
    config = SQLConfig(dialect="postgres", host="...", ...)
    with DBConnector(config) as conn:
        df = conn.query("SELECT user_id, item_id, rating FROM interactions")
    """

    def __init__(self, config: SQLConfig):
        self.config = config
        self._engine = None

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ── Connection lifecycle ───────────────────────────────────────────────────

    def connect(self):
        """Open a SQLAlchemy engine connection to the configured database."""
        try:
            from sqlalchemy import create_engine, text
            url = self.config.connection_url()
            self._engine = create_engine(
                url,
                connect_args={"connect_timeout": self.config.connect_timeout},
                pool_pre_ping=True,
            )
            # Verify connectivity
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info(
                "DB connected | dialect=%s host=%s db=%s",
                self.config.dialect, self.config.host, self.config.database,
            )
        except ImportError:
            raise ImportError(
                f"Driver for '{self.config.dialect}' not installed. "
                f"Install the appropriate SQLAlchemy dialect package."
            )
        except Exception as e:
            log.error("DB connection failed: %s", e)
            raise ConnectionError(f"Cannot connect to {self.config.dialect}: {e}") from e

    def disconnect(self):
        """Dispose the connection pool."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            log.info("DB connection closed")

    # ── Query execution ────────────────────────────────────────────────────────

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Parameters
        ----------
        sql    : SQL query string (use :param_name for named params)
        params : optional dict of query parameters (prevents SQL injection)
        """
        if self._engine is None:
            raise RuntimeError("Not connected. Call connect() first.")

        log.info("Executing query (first 120 chars): %s", sql[:120].replace("\n", " "))
        try:
            from sqlalchemy import text
            with self._engine.connect() as conn:
                df = pd.read_sql(text(sql), conn, params=params or {})
            log.info("Query returned %d rows × %d columns", *df.shape)
            return df
        except Exception as e:
            log.error("Query failed: %s", e)
            raise RuntimeError(f"Query execution failed: {e}") from e

    def get_table_names(self) -> list[str]:
        """List all tables visible in the connected database / schema."""
        from sqlalchemy import inspect
        inspector = inspect(self._engine)
        schema = self.config.schema
        return inspector.get_table_names(schema=schema)

    def get_columns(self, table: str) -> list[dict]:
        """
        Return column metadata for a table.
        Each dict has keys: name, type, nullable, primary_key.
        """
        from sqlalchemy import inspect, text
        inspector = inspect(self._engine)
        schema    = self.config.schema
        cols = inspector.get_columns(table, schema=schema)
        pk   = {c["name"] for c in inspector.get_pk_constraint(table, schema=schema).get("constrained_columns", [])}
        return [
            {
                "name":        c["name"],
                "type":        str(c["type"]),
                "nullable":    c.get("nullable", True),
                "primary_key": c["name"] in pk,
            }
            for c in cols
        ]
