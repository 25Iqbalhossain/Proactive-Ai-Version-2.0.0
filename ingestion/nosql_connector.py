"""
ingestion/nosql_connector.py – NoSQL database connector

Supports: MongoDB, Cassandra, DynamoDB
Each backend is lazily imported so users only need to install the driver
they actually use.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

from config.database import NoSQLConfig
from utils.logger import get_logger

log = get_logger(__name__)


class NoSQLConnector:
    """
    Unified NoSQL connector. Backend is selected from NoSQLConfig.engine.

    Usage
    -----
    config = NoSQLConfig(engine="mongodb", host="localhost", port=27017, database="shop")
    with NoSQLConnector(config) as conn:
        df = conn.fetch_collection("interactions", query={"rating": {"$gte": 1}})
    """

    def __init__(self, config: NoSQLConfig):
        self.config  = config
        self._client = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self):
        engine = self.config.engine.lower()
        if engine == "mongodb":
            self._connect_mongo()
        elif engine == "cassandra":
            self._connect_cassandra()
        elif engine == "dynamodb":
            self._connect_dynamodb()
        else:
            raise ValueError(f"Unsupported NoSQL engine: '{engine}'. Choose: mongodb, cassandra, dynamodb")

    def disconnect(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            log.info("NoSQL connection closed | engine=%s", self.config.engine)

    # ── MongoDB ────────────────────────────────────────────────────────────────

    def _connect_mongo(self):
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("Install pymongo: pip install pymongo")

        uri = self._build_mongo_uri()
        self._client = MongoClient(uri, serverSelectionTimeoutMS=self.config.connect_timeout * 1000)
        self._client.server_info()   # raises if unreachable
        log.info("MongoDB connected | host=%s db=%s", self.config.host, self.config.database)

    def _build_mongo_uri(self) -> str:
        cfg = self.config
        if cfg.username and cfg.password:
            return f"mongodb://{cfg.username}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}"
        return f"mongodb://{cfg.host}:{cfg.port}/{cfg.database}"

    def fetch_collection(
        self,
        collection: str,
        query: Optional[dict] = None,
        projection: Optional[dict] = None,
        limit: int = 0,
    ) -> pd.DataFrame:
        """Fetch MongoDB collection → DataFrame."""
        from pymongo import MongoClient
        db   = self._client[self.config.database]
        col  = db[collection]
        cur  = col.find(query or {}, projection or {}, limit=limit)
        docs = list(cur)
        df   = pd.DataFrame(docs)
        if "_id" in df.columns:
            df["_id"] = df["_id"].astype(str)
        log.info("Fetched %d documents from MongoDB collection '%s'", len(df), collection)
        return df

    # ── Cassandra ──────────────────────────────────────────────────────────────

    def _connect_cassandra(self):
        try:
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
        except ImportError:
            raise ImportError("Install cassandra-driver: pip install cassandra-driver")

        auth = None
        if self.config.username and self.config.password:
            auth = PlainTextAuthProvider(self.config.username, self.config.password)

        cluster      = Cluster([self.config.host], port=self.config.port, auth_provider=auth)
        self._client = cluster.connect(self.config.database)
        log.info("Cassandra connected | host=%s keyspace=%s", self.config.host, self.config.database)

    def fetch_cassandra_table(self, table: str, cql: Optional[str] = None) -> pd.DataFrame:
        """Execute CQL and return DataFrame."""
        sql = cql or f"SELECT * FROM {table}"
        rows = self._client.execute(sql)
        df   = pd.DataFrame(rows)
        log.info("Fetched %d rows from Cassandra table '%s'", len(df), table)
        return df

    # ── DynamoDB ───────────────────────────────────────────────────────────────

    def _connect_dynamodb(self):
        try:
            import boto3
        except ImportError:
            raise ImportError("Install boto3: pip install boto3")

        region = self.config.region or "us-east-1"
        self._client = boto3.resource("dynamodb", region_name=region)
        log.info("DynamoDB connected | region=%s", region)

    def fetch_dynamodb_table(self, table_name: str, scan_kwargs: Optional[dict] = None) -> pd.DataFrame:
        """Scan a DynamoDB table and return DataFrame."""
        table    = self._client.Table(table_name)
        kwargs   = scan_kwargs or {}
        items: list[dict] = []

        while True:
            resp  = table.scan(**kwargs)
            items.extend(resp.get("Items", []))
            last  = resp.get("LastEvaluatedKey")
            if not last:
                break
            kwargs["ExclusiveStartKey"] = last

        df = pd.DataFrame(items)
        log.info("Fetched %d items from DynamoDB table '%s'", len(df), table_name)
        return df
