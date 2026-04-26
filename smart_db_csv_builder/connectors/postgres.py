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
        return [dict(row) for row in rows]

    def get_schema(self, conn_id: str) -> SchemaResponse:
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
        for row in rows:
            full_name = f"{row['table_schema']}.{row['table_name']}"
            if full_name not in table_map:
                table_map[full_name] = TableInfo(
                    table_name=row["table_name"],
                    schema_name=row["table_schema"],
                )

            column = ColumnInfo(
                name=row["column_name"],
                data_type=row["data_type"],
                nullable=row["is_nullable"] == "YES",
                is_pk=bool(row["is_pk"]),
            )

            try:
                sample = self.execute(
                    f'SELECT "{row["column_name"]}" FROM "{row["table_schema"]}"."{row["table_name"]}" '
                    f'WHERE "{row["column_name"]}" IS NOT NULL LIMIT 3',
                    limit=3,
                )
                column.sample_values = [list(item.values())[0] for item in sample]
            except Exception:
                pass

            table_map[full_name].columns.append(column)

        for _, meta in table_map.items():
            try:
                count = self.execute(
                    f'SELECT COUNT(*) AS n FROM "{meta.schema_name}"."{meta.table_name}"',
                    limit=1,
                )
                meta.row_count = count[0]["n"] if count else 0
            except Exception:
                pass

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
            db_type=DBType.POSTGRES,
            tables=list(table_map.values()),
            relationships=relationships,
        )

    def close(self):
        if self._conn:
            self._conn.close()
