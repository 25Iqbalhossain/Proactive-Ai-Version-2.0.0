from __future__ import annotations

from smart_db_csv_builder.connectors.base import BaseConnector, split_table_reference
from smart_db_csv_builder.models.schemas import (
    ColumnInfo,
    ConnectionCredential,
    DBType,
    FKRelationship,
    SchemaResponse,
    TableInfo,
)


def _quote_identifier(identifier: str) -> str:
    return f"[{identifier}]"


def _quote_table(table: str) -> str:
    parts = split_table_reference(table)
    if not parts:
        return table
    return ".".join(_quote_identifier(part) for part in parts)


def build_select_sql(
    table: str,
    columns: list[str],
    where: str = "",
    limit: int = 50_000,
) -> str:
    cols = ", ".join(_quote_identifier(column) for column in columns) if columns else "*"
    where_clause = f" WHERE {where.strip()}" if where and where.strip() else ""
    return f"SELECT TOP {limit} {cols} FROM {_quote_table(table)}{where_clause}"


class MSSQLConnector(BaseConnector):
    def __init__(self, cred: ConnectionCredential):
        super().__init__(cred)
        import pyodbc

        self._pyodbc = pyodbc
        self._conn = None
        self._connect()

    def _connect(self):
        driver = self.cred.options.get("driver", "ODBC Driver 18 for SQL Server")
        connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER={self.cred.host},{self.cred.port or 1433};"
            f"DATABASE={self.cred.database};"
            f"UID={self.cred.username};"
            f"PWD={self.cred.password or ''};"
            f"Encrypt={'yes' if self.cred.ssl else 'no'};"
            f"TrustServerCertificate=yes;"
            f"Connection Timeout=10;"
        )
        self._conn = self._pyodbc.connect(connection_string)

    def test(self) -> None:
        cur = self._conn.cursor()
        cur.execute("SELECT 1")

    def execute(self, sql: str, limit: int = 50_000) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchmany(limit)
        return [dict(zip(columns, row)) for row in rows]

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
        for row in rows:
            full_name = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"
            if full_name not in table_map:
                table_map[full_name] = TableInfo(
                    table_name=row["TABLE_NAME"],
                    schema_name=row["TABLE_SCHEMA"],
                )

            table_map[full_name].columns.append(
                ColumnInfo(
                    name=row["COLUMN_NAME"],
                    data_type=row["DATA_TYPE"],
                    nullable=row["IS_NULLABLE"] == "YES",
                    is_pk=bool(row["is_pk"]),
                )
            )

        for _, meta in table_map.items():
            try:
                count = self.execute(
                    f"SELECT COUNT(*) AS n FROM [{meta.schema_name}].[{meta.table_name}]",
                    limit=1,
                )
                meta.row_count = count[0]["n"] if count else 0
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
        relationships = [
            FKRelationship(
                from_table=row["from_table"],
                from_column=row["from_col"],
                to_table=row["to_table"],
                to_column=row["to_col"],
            )
            for row in fk_rows
        ]

        return SchemaResponse(
            connection_id=conn_id,
            db_type=DBType.MSSQL,
            tables=list(table_map.values()),
            relationships=relationships,
        )

    def close(self):
        if self._conn:
            self._conn.close()
