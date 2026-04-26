from __future__ import annotations

from smart_db_csv_builder.models.schemas import ConnectionCredential, SchemaResponse


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


def split_table_reference(table: str) -> list[str]:
    return [part.strip() for part in (table or "").split(".") if part and part.strip()]
