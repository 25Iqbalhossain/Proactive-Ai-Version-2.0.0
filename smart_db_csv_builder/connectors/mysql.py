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
    return f"`{identifier}`"


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
        for row in rows:
            table_name = row["TABLE_NAME"]
            if table_name not in table_map:
                table_map[table_name] = TableInfo(table_name=table_name, schema_name=db)

            column = ColumnInfo(
                name=row["COLUMN_NAME"],
                data_type=row["DATA_TYPE"],
                nullable=row["IS_NULLABLE"] == "YES",
                is_pk=bool(row["is_pk"]),
            )

            try:
                sample = self.execute(
                    f"SELECT `{row['COLUMN_NAME']}` FROM `{table_name}` "
                    f"WHERE `{row['COLUMN_NAME']}` IS NOT NULL LIMIT 3",
                    limit=3,
                )
                column.sample_values = [list(item.values())[0] for item in sample]
            except Exception:
                pass

            table_map[table_name].columns.append(column)

        for table_name, meta in table_map.items():
            try:
                count = self.execute(f"SELECT COUNT(*) AS n FROM `{table_name}`", limit=1)
                meta.row_count = count[0]["n"] if count else 0
            except Exception:
                pass

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
        relationships = [
            FKRelationship(
                from_table=row["from_table"],
                from_column=row["from_col"],
                to_table=row["to_table"],
                to_column=row["to_col"],
            )
            for row in fk_rows
        ]

        fk_index = {(rel.from_table, rel.from_column): rel for rel in relationships}
        for table_name, meta in table_map.items():
            for column in meta.columns:
                key = (table_name, column.name)
                if key in fk_index:
                    column.is_fk = True
                    column.fk_ref_table = fk_index[key].to_table
                    column.fk_ref_column = fk_index[key].to_column

        return SchemaResponse(
            connection_id=conn_id,
            db_type=DBType.MYSQL,
            tables=list(table_map.values()),
            relationships=relationships,
        )

    def close(self):
        if self._conn:
            self._conn.close()
