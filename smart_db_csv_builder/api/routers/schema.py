"""api/routers/schema.py"""

from fastapi import APIRouter, HTTPException
from smart_db_csv_builder.models.schemas import SchemaResponse
from smart_db_csv_builder.core.connection_store import connection_store

router = APIRouter()


@router.get("/{conn_id}", response_model=SchemaResponse)
def get_schema(conn_id: str):
    conn = connection_store.get(conn_id)
    if conn is None:
        raise HTTPException(404, "Connection not found")
    try:
        return conn.driver.get_schema(conn_id)
    except Exception as exc:
        raise HTTPException(500, f"Schema extraction failed: {exc}")
