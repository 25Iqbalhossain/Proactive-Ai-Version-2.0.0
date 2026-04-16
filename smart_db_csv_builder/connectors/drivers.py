"""
connectors/drivers.py

One connector class per supported DB type.
All connectors expose the same interface:
  - test()            → raises on failure
  - get_schema()      → SchemaResponse
  - execute(sql, lim) → list[dict]    (SQL only)
  - fetch_collection(name, lim) → list[dict]  (MongoDB only)
  - close()
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from smart_db_csv_builder.models.schemas import (
    DBType, SchemaResponse, TableInfo, ColumnInfo, FKRelationship,
    ConnectionCredential,
)

logger = logging.getLogger(__name__)


# ── Base ───────────────────────────────────────────────────────────────────

class BaseConnector:
    def __init__(self, cred: ConnectionCredential):
        self.cred = cred

    def test(self) -> None:
        raise NotImplementedError

    def get_schema(self, conn_id: str) -> SchemaResponse:
        raise NotImplementedError

    def execute(self, sql: str, limit: int = 50_000) -> list[dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ── MySQL ──────────────────────────────────────────────────────────────────

class MySQLConnector(BaseConnector):

    def __init__(self, cred: ConnectionCredential):
        super().__init__(cred)
        import pymysql
        self._pymysql = pymysql
        self._conn = None
        self._connect()

    def _connect(self):
        self._conn = self._pymysql.connect(
            host=self.cred.host,
            port=self.cred.port or 3306,
            user=self.cred.username,
            password=self.cred.password or "",
            database=self.cred.database,
            charset="utf8mb4",
            cursorclass=self._pymysql.cursors.DictCursor,
            ssl={"ssl": {}} if self.cred.ssl else None,
            connect_timeout=10,
        )

    def test(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT 1")

    def execute(self, sql: str, limit: int = 50_000) -> list[dict]:
        self._conn.ping(reconnect=True)
        with self._conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchmany(limit)
        return [dict(r) for r in rows]

    def get_schema(self, conn_id: str) -> SchemaResponse:
        db = self.cred.database
        tables = []

        # Tables + columns
        cols_sql = f"""
            SELECT
                c.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE,
                c.IS_NULLABLE,
                IF(kcu.COLUMN_NAME IS NOT NULL, 1, 0) AS is_pk
            FROM information_schema.COLUMNS c
            LEFT JOIN information_schema.TABLE_CONSTRAINTS tc
                ON tc.TABLE_SCHEMA = c.TABLE_SCHEMA
               AND tc.TABLE_NAME   = c.TABLE_NAME
               AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            LEFT JOIN information_schema.KEY_COLUMN_USAGE kcu
                ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
               AND kcu.TABLE_NAME      = c.TABLE_NAME
               AND kcu.COLUMN_NAME     = c.COLUMN_NAME
            WHERE c.TABLE_SCHEMA = '{db}'
            ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION
        """
        rows = self.execute(cols_sql, limit=5000)

        table_map: dict[str, TableInfo] = {}
        for r in rows:
            tbl = r["TABLE_NAME"]
            if tbl not in table_map:
                table_map[tbl] = TableInfo(table_name=tbl, schema_name=db)
            col = ColumnInfo(
                name=r["COLUMN_NAME"],
                data_type=r["DATA_TYPE"],
                nullable=r["IS_NULLABLE"] == "YES",
                is_pk=bool(r["is_pk"]),
            )
            # Sample values
            try:
                sample = self.execute(
                    f"SELECT `{r['COLUMN_NAME']}` FROM `{tbl}` WHERE `{r['COLUMN_NAME']}` IS NOT NULL LIMIT 3",
                    limit=3,
                )
                col.sample_values = [list(s.values())[0] for s in sample]
            except Exception:
                pass
            table_map[tbl].columns.append(col)

        # Row counts
        for tbl, meta in table_map.items():
            try:
                cnt = self.execute(f"SELECT COUNT(*) AS n FROM `{tbl}`", limit=1)
                meta.row_count = cnt[0]["n"] if cnt else 0
            except Exception:
                pass

        # FK relationships
        fk_sql = f"""
            SELECT
                kcu.TABLE_NAME       AS from_table,
                kcu.COLUMN_NAME      AS from_col,
                kcu.REFERENCED_TABLE_NAME  AS to_table,
                kcu.REFERENCED_COLUMN_NAME AS to_col
            FROM information_schema.KEY_COLUMN_USAGE kcu
            JOIN information_schema.REFERENTIAL_CONSTRAINTS rc
              ON rc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
             AND rc.CONSTRAINT_SCHEMA = kcu.TABLE_SCHEMA
            WHERE kcu.TABLE_SCHEMA = '{db}'
              AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
        """
        fk_rows = self.execute(fk_sql, limit=500)
        rels = [
            FKRelationship(
                from_table=r["from_table"], from_column=r["from_col"],
                to_table=r["to_table"],     to_column=r["to_col"],
            )
            for r in fk_rows
        ]
        # Mark FK columns
        fk_index = {(r.from_table, r.from_column): r for r in rels}
        for tbl, meta in table_map.items():
            for col in meta.columns:
                key = (tbl, col.name)
                if key in fk_index:
                    col.is_fk = True
                    col.fk_ref_table  = fk_index[key].to_table
                    col.fk_ref_column = fk_index[key].to_column

        return SchemaResponse(
            connection_id=conn_id,
            db_type=DBType.MYSQL,
            tables=list(table_map.values()),
            relationships=rels,
        )

    def close(self):
        if self._conn:
            self._conn.close()


# ── PostgreSQL ─────────────────────────────────────────────────────────────

class PostgresConnector(BaseConnector):

    def __init__(self, cred: ConnectionCredential):
        super().__init__(cred)
        import psycopg2
        import psycopg2.extras
        self._psycopg2 = psycopg2
        self._extras = psycopg2.extras
        self._conn = None
        self._connect()

    def _connect(self):
        self._conn = self._psycopg2.connect(
            host=self.cred.host,
            port=self.cred.port or 5432,
            user=self.cred.username,
            password=self.cred.password or "",
            dbname=self.cred.database,
            connect_timeout=10,
            sslmode="require" if self.cred.ssl else "prefer",
        )
        self._conn.autocommit = True

    def test(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT 1")

    def execute(self, sql: str, limit: int = 50_000) -> list[dict]:
        with self._conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchmany(limit)
        return [dict(r) for r in rows]

    def get_schema(self, conn_id: str) -> SchemaResponse:
        db = self.cred.database
        cols_sql = """
            SELECT
                c.table_schema, c.table_name, c.column_name, c.data_type,
                c.is_nullable,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END AS is_pk
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT kcu.table_schema, kcu.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON kcu.constraint_name = tc.constraint_name
                 AND kcu.table_schema    = tc.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
            ) pk ON pk.table_schema = c.table_schema
                 AND pk.table_name   = c.table_name
                 AND pk.column_name  = c.column_name
            WHERE c.table_schema NOT IN ('pg_catalog','information_schema')
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
        """
        rows = self.execute(cols_sql, limit=5000)

        table_map: dict[str, TableInfo] = {}
        for r in rows:
            full = f"{r['table_schema']}.{r['table_name']}"
            if full not in table_map:
                table_map[full] = TableInfo(
                    table_name=r["table_name"],
                    schema_name=r["table_schema"],
                )
            col = ColumnInfo(
                name=r["column_name"],
                data_type=r["data_type"],
                nullable=r["is_nullable"] == "YES",
                is_pk=bool(r["is_pk"]),
            )
            try:
                sample = self.execute(
                    f'SELECT "{r["column_name"]}" FROM "{r["table_schema"]}"."{r["table_name"]}" '
                    f'WHERE "{r["column_name"]}" IS NOT NULL LIMIT 3',
                    limit=3,
                )
                col.sample_values = [list(s.values())[0] for s in sample]
            except Exception:
                pass
            table_map[full].columns.append(col)

        for full, meta in table_map.items():
            try:
                cnt = self.execute(
                    f'SELECT COUNT(*) AS n FROM "{meta.schema_name}"."{meta.table_name}"',
                    limit=1,
                )
                meta.row_count = cnt[0]["n"] if cnt else 0
            except Exception:
                pass

        # FK relationships
        fk_sql = """
            SELECT
                kcu.table_schema  || '.' || kcu.table_name   AS from_table,
                kcu.column_name                              AS from_col,
                ccu.table_schema  || '.' || ccu.table_name   AS to_table,
                ccu.column_name                              AS to_col
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON kcu.constraint_name = tc.constraint_name
             AND kcu.table_schema    = tc.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema NOT IN ('pg_catalog','information_schema')
        """
        fk_rows = self.execute(fk_sql, limit=500)
        rels = [
            FKRelationship(
                from_table=r["from_table"], from_column=r["from_col"],
                to_table=r["to_table"],     to_column=r["to_col"],
            )
            for r in fk_rows
        ]

        return SchemaResponse(
            connection_id=conn_id,
            db_type=DBType.POSTGRES,
            tables=list(table_map.values()),
            relationships=rels,
        )

    def close(self):
        if self._conn:
            self._conn.close()


# ── MSSQL ──────────────────────────────────────────────────────────────────

class MSSQLConnector(BaseConnector):

    def __init__(self, cred: ConnectionCredential):
        super().__init__(cred)
        import pyodbc
        self._pyodbc = pyodbc
        self._conn = None
        self._connect()

    def _connect(self):
        driver = self.cred.options.get("driver", "ODBC Driver 18 for SQL Server")
        cs = (
            f"DRIVER={{{driver}}};"
            f"SERVER={self.cred.host},{self.cred.port or 1433};"
            f"DATABASE={self.cred.database};"
            f"UID={self.cred.username};"
            f"PWD={self.cred.password or ''};"
            f"Encrypt={'yes' if self.cred.ssl else 'no'};"
            f"TrustServerCertificate=yes;"
            f"Connection Timeout=10;"
        )
        self._conn = self._pyodbc.connect(cs)

    def test(self) -> None:
        cur = self._conn.cursor()
        cur.execute("SELECT 1")

    def execute(self, sql: str, limit: int = 50_000) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchmany(limit)
        return [dict(zip(cols, row)) for row in rows]

    def get_schema(self, conn_id: str) -> SchemaResponse:
        cols_sql = """
            SELECT
                t.TABLE_SCHEMA, t.TABLE_NAME,
                c.COLUMN_NAME, c.DATA_TYPE, c.IS_NULLABLE,
                CASE WHEN kcu.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END AS is_pk
            FROM INFORMATION_SCHEMA.TABLES t
            JOIN INFORMATION_SCHEMA.COLUMNS c
              ON c.TABLE_NAME   = t.TABLE_NAME
             AND c.TABLE_SCHEMA = t.TABLE_SCHEMA
            LEFT JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
              ON tc.TABLE_NAME   = t.TABLE_NAME
             AND tc.TABLE_SCHEMA = t.TABLE_SCHEMA
             AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
              ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
             AND kcu.TABLE_NAME      = c.TABLE_NAME
             AND kcu.COLUMN_NAME     = c.COLUMN_NAME
            WHERE t.TABLE_TYPE = 'BASE TABLE'
            ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME, c.ORDINAL_POSITION
        """
        rows = self.execute(cols_sql, limit=5000)
        table_map: dict[str, TableInfo] = {}
        for r in rows:
            full = f"{r['TABLE_SCHEMA']}.{r['TABLE_NAME']}"
            if full not in table_map:
                table_map[full] = TableInfo(
                    table_name=r["TABLE_NAME"],
                    schema_name=r["TABLE_SCHEMA"],
                )
            col = ColumnInfo(
                name=r["COLUMN_NAME"],
                data_type=r["DATA_TYPE"],
                nullable=r["IS_NULLABLE"] == "YES",
                is_pk=bool(r["is_pk"]),
            )
            table_map[full].columns.append(col)

        for full, meta in table_map.items():
            try:
                cnt = self.execute(
                    f"SELECT COUNT(*) AS n FROM [{meta.schema_name}].[{meta.table_name}]",
                    limit=1,
                )
                meta.row_count = cnt[0]["n"] if cnt else 0
            except Exception:
                pass

        fk_sql = """
            SELECT
                tp.TABLE_SCHEMA + '.' + tp.TABLE_NAME AS from_table,
                cp.COLUMN_NAME                        AS from_col,
                tr.TABLE_SCHEMA + '.' + tr.TABLE_NAME AS to_table,
                cr.COLUMN_NAME                        AS to_col
            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
            JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tp
              ON tp.CONSTRAINT_NAME = rc.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tr
              ON tr.CONSTRAINT_NAME = rc.UNIQUE_CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE cp
              ON cp.CONSTRAINT_NAME = rc.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE cr
              ON cr.CONSTRAINT_NAME = rc.UNIQUE_CONSTRAINT_NAME
             AND cr.ORDINAL_POSITION = cp.ORDINAL_POSITION
        """
        fk_rows = self.execute(fk_sql, limit=500)
        rels = [
            FKRelationship(
                from_table=r["from_table"], from_column=r["from_col"],
                to_table=r["to_table"],     to_column=r["to_col"],
            )
            for r in fk_rows
        ]

        return SchemaResponse(
            connection_id=conn_id,
            db_type=DBType.MSSQL,
            tables=list(table_map.values()),
            relationships=rels,
        )

    def close(self):
        if self._conn:
            self._conn.close()


# ── MongoDB ────────────────────────────────────────────────────────────────

class MongoDBConnector(BaseConnector):

    def __init__(self, cred: ConnectionCredential):
        super().__init__(cred)
        from pymongo import MongoClient
        uri = cred.uri or f"mongodb://{cred.host}:{cred.port or 27017}"
        kwargs: dict = {
            "serverSelectionTimeoutMS": 5000,
            "connectTimeoutMS": 5000,
        }
        if cred.username and cred.password:
            kwargs["username"] = cred.username
            kwargs["password"] = cred.password
            kwargs["authSource"] = cred.options.get("auth_source", "admin")
        if cred.ssl:
            kwargs["tls"] = True
        self._client = MongoClient(uri, **kwargs)
        self._db = self._client[cred.database or "admin"]

    def test(self) -> None:
        self._client.admin.command("ping")

    def get_schema(self, conn_id: str) -> SchemaResponse:
        collections = []
        for name in self._db.list_collection_names():
            sample_docs = list(self._db[name].find({}, {"_id": 0}).limit(5))
            # Infer schema from sample
            all_keys: dict[str, set] = {}
            for doc in sample_docs:
                for k, v in doc.items():
                    all_keys.setdefault(k, set()).add(type(v).__name__)
            collections.append({
                "name":        name,
                "doc_count":   self._db[name].estimated_document_count(),
                "fields":      {k: list(v) for k, v in all_keys.items()},
                "sample":      sample_docs[:2],
            })
        return SchemaResponse(
            connection_id=conn_id,
            db_type=DBType.MONGODB,
            tables=[],
            collections=collections,
        )

    def fetch_collection(self, name: str, limit: int = 50_000) -> list[dict]:
        docs = list(self._db[name].find({}, {"_id": 0}).limit(limit))
        return docs

    def close(self):
        if self._client:
            self._client.close()


# ── SQLite ─────────────────────────────────────────────────────────────────

class SQLiteConnector(BaseConnector):

    def __init__(self, cred: ConnectionCredential):
        super().__init__(cred)
        import sqlite3
        self._sqlite3 = sqlite3
        filepath = cred.filepath or cred.database or ":memory:"
        self._conn = sqlite3.connect(filepath, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def test(self) -> None:
        self._conn.execute("SELECT 1")

    def execute(self, sql: str, limit: int = 50_000) -> list[dict]:
        cur = self._conn.execute(sql)
        rows = cur.fetchmany(limit)
        return [dict(r) for r in rows]

    def get_schema(self, conn_id: str) -> SchemaResponse:
        tables_q = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = []
        for t in tables_q:
            tname = t["name"]
            cols_q = self.execute(f'PRAGMA table_info("{tname}")')
            cols = [
                ColumnInfo(
                    name=c["name"],
                    data_type=c["type"] or "TEXT",
                    nullable=not c["notnull"],
                    is_pk=bool(c["pk"]),
                )
                for c in cols_q
            ]
            cnt = self.execute(f'SELECT COUNT(*) AS n FROM "{tname}"', limit=1)
            tables.append(TableInfo(
                table_name=tname,
                row_count=cnt[0]["n"] if cnt else 0,
                columns=cols,
            ))

        fk_rels = []
        for t in tables:
            fks = self.execute(f'PRAGMA foreign_key_list("{t.table_name}")')
            for fk in fks:
                fk_rels.append(FKRelationship(
                    from_table=t.table_name, from_column=fk["from"],
                    to_table=fk["table"],    to_column=fk["to"],
                ))

        return SchemaResponse(
            connection_id=conn_id,
            db_type=DBType.SQLITE,
            tables=tables,
            relationships=fk_rels,
        )

    def close(self):
        if self._conn:
            self._conn.close()


# ── Factory ─────────────────────────────────────────────────────────────────

def build_connector(cred: ConnectionCredential) -> BaseConnector:
    """Instantiate the right connector for a credential."""
    match cred.db_type:
        case DBType.MYSQL:
            return MySQLConnector(cred)
        case DBType.POSTGRES:
            return PostgresConnector(cred)
        case DBType.MSSQL:
            return MSSQLConnector(cred)
        case DBType.MONGODB:
            return MongoDBConnector(cred)
        case DBType.SQLITE:
            return SQLiteConnector(cred)
        case _:
            raise ValueError(f"Unsupported DB type: {cred.db_type}")
