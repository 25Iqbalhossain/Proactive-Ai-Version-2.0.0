"""
config/database.py – Database connection profiles

Credentials are loaded from environment variables or a .env file.
Never hard-code credentials here.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SQLConfig:
    """Configuration for relational database connections."""
    dialect: str          # postgres | mysql | mssql | snowflake | redshift
    host: str
    port: int
    database: str
    username: str
    password: str
    schema: Optional[str] = None
    ssl: bool = True
    connect_timeout: int = 30
    extra_params: dict = field(default_factory=dict)

    def connection_url(self) -> str:
        """Build a SQLAlchemy-compatible connection URL."""
        dialect_map = {
            "postgres":  "postgresql+psycopg2",
            "mysql":     "mysql+pymysql",
            "mssql":     "mssql+pyodbc",
            "snowflake": "snowflake",
            "redshift":  "redshift+redshift_connector",
        }
        driver = dialect_map.get(self.dialect, self.dialect)
        return (
            f"{driver}://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class NoSQLConfig:
    """Configuration for NoSQL database connections."""
    engine: str           # mongodb | cassandra | dynamodb
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    region: Optional[str] = None         # for DynamoDB
    ssl: bool = True
    connect_timeout: int = 30


def load_sql_config_from_env() -> Optional[SQLConfig]:
    """Load SQL config from environment variables."""
    dialect = os.getenv("DB_DIALECT")
    if not dialect:
        return None
    return SQLConfig(
        dialect=dialect,
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", ""),
        username=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASSWORD", ""),
        schema=os.getenv("DB_SCHEMA"),
        ssl=os.getenv("DB_SSL", "true").lower() == "true",
    )


def load_nosql_config_from_env() -> Optional[NoSQLConfig]:
    """Load NoSQL config from environment variables."""
    engine = os.getenv("NOSQL_ENGINE")
    if not engine:
        return None
    return NoSQLConfig(
        engine=engine,
        host=os.getenv("NOSQL_HOST", "localhost"),
        port=int(os.getenv("NOSQL_PORT", "27017")),
        database=os.getenv("NOSQL_DB", ""),
        username=os.getenv("NOSQL_USER"),
        password=os.getenv("NOSQL_PASSWORD"),
        region=os.getenv("AWS_REGION"),
        ssl=os.getenv("NOSQL_SSL", "true").lower() == "true",
    )
