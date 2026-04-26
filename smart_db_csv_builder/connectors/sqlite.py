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
    return f'"{identifier}"'


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
    return f"SELECT {cols} FROM {_quote_table(table)}{where_clause} LIMIT {limit}"


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
        return [dict(row) for row in rows]

    def get_schema(self, conn_id: str) -> SchemaResponse:
        table_rows = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )

        tables = []
        for row in table_rows:
            table_name = row["name"]
            column_rows = self.execute(f'PRAGMA table_info("{table_name}")')
            columns = [
                ColumnInfo(
                    name=column["name"],
                    data_type=column["type"] or "TEXT",
                    nullable=not column["notnull"],
                    is_pk=bool(column["pk"]),
                )
                for column in column_rows
            ]
            count = self.execute(f'SELECT COUNT(*) AS n FROM "{table_name}"', limit=1)
            tables.append(
                TableInfo(
                    table_name=table_name,
                    row_count=count[0]["n"] if count else 0,
                    columns=columns,
                )
            )

        relationships = []
        for table in tables:
            fk_rows = self.execute(f'PRAGMA foreign_key_list("{table.table_name}")')
            for row in fk_rows:
                relationships.append(
                    FKRelationship(
                        from_table=table.table_name,
                        from_column=row["from"],
                        to_table=row["table"],
                        to_column=row["to"],
                    )
                )

        return SchemaResponse(
            connection_id=conn_id,
            db_type=DBType.SQLITE,
            tables=tables,
            relationships=relationships,
        )

    def close(self):
        if self._conn:
            self._conn.close()
