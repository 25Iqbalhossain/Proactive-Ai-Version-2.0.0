from __future__ import annotations

from smart_db_csv_builder.connectors.base import BaseConnector
from smart_db_csv_builder.models.schemas import ConnectionCredential, DBType, SchemaResponse


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
            all_keys: dict[str, set] = {}
            for doc in sample_docs:
                for key, value in doc.items():
                    all_keys.setdefault(key, set()).add(type(value).__name__)
            collections.append(
                {
                    "name": name,
                    "doc_count": self._db[name].estimated_document_count(),
                    "fields": {key: list(value) for key, value in all_keys.items()},
                    "sample": sample_docs[:2],
                }
            )

        return SchemaResponse(
            connection_id=conn_id,
            db_type=DBType.MONGODB,
            tables=[],
            collections=collections,
        )

    def fetch_collection(self, name: str, limit: int = 50_000) -> list[dict]:
        return list(self._db[name].find({}, {"_id": 0}).limit(limit))

    def close(self):
        if self._client:
            self._client.close()
