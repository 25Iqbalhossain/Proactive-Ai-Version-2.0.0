from __future__ import annotations

from collections.abc import Callable

from smart_db_csv_builder.connectors.base import BaseConnector
from smart_db_csv_builder.connectors.mongodb import MongoDBConnector
from smart_db_csv_builder.connectors.mssql import MSSQLConnector, build_select_sql as build_mssql_select_sql
from smart_db_csv_builder.connectors.mysql import MySQLConnector, build_select_sql as build_mysql_select_sql
from smart_db_csv_builder.connectors.postgres import PostgresConnector, build_select_sql as build_postgres_select_sql
from smart_db_csv_builder.connectors.sqlite import SQLiteConnector, build_select_sql as build_sqlite_select_sql
from smart_db_csv_builder.models.schemas import ConnectionCredential, DBType


CONNECTOR_BUILDERS: dict[DBType, type[BaseConnector]] = {
    DBType.MYSQL: MySQLConnector,
    DBType.POSTGRES: PostgresConnector,
    DBType.MSSQL: MSSQLConnector,
    DBType.MONGODB: MongoDBConnector,
    DBType.SQLITE: SQLiteConnector,
}

SQL_SELECT_BUILDERS: dict[DBType, Callable[[str, list[str], str, int], str]] = {
    DBType.MYSQL: build_mysql_select_sql,
    DBType.POSTGRES: build_postgres_select_sql,
    DBType.MSSQL: build_mssql_select_sql,
    DBType.SQLITE: build_sqlite_select_sql,
}


def build_connector(cred: ConnectionCredential) -> BaseConnector:
    builder = CONNECTOR_BUILDERS.get(cred.db_type)
    if not builder:
        raise ValueError(f"Unsupported DB type: {cred.db_type}")
    return builder(cred)


def build_select_sql(
    db_type: DBType | str,
    table: str,
    columns: list[str],
    where: str = "",
    limit: int = 50_000,
) -> str:
    resolved_type = db_type if isinstance(db_type, DBType) else DBType(db_type)
    builder = SQL_SELECT_BUILDERS.get(resolved_type)
    if not builder:
        raise ValueError(f"Unsupported SQL DB type: {resolved_type}")
    return builder(table, columns, where, limit)
