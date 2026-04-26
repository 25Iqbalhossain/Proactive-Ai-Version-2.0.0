"""
Compatibility wrapper for connector imports.
"""

from smart_db_csv_builder.connectors.base import BaseConnector
from smart_db_csv_builder.connectors.factory import build_connector, build_select_sql
from smart_db_csv_builder.connectors.mongodb import MongoDBConnector
from smart_db_csv_builder.connectors.mssql import MSSQLConnector
from smart_db_csv_builder.connectors.mysql import MySQLConnector
from smart_db_csv_builder.connectors.postgres import PostgresConnector
from smart_db_csv_builder.connectors.sqlite import SQLiteConnector

__all__ = [
    "BaseConnector",
    "MySQLConnector",
    "PostgresConnector",
    "MSSQLConnector",
    "MongoDBConnector",
    "SQLiteConnector",
    "build_connector",
    "build_select_sql",
]
