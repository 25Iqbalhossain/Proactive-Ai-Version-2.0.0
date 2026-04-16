"""api/routers/connections.py"""

from fastapi import APIRouter, HTTPException
from smart_db_csv_builder.models.schemas import (
    ConnectionCredential,
    ConnectionResponse,
)
from smart_db_csv_builder.connectors.drivers import build_connector
from smart_db_csv_builder.core.connection_store import connection_store

router = APIRouter()


@router.post("/test", response_model=ConnectionResponse)
def test_connection(cred: ConnectionCredential):
    """Test credentials without storing. Returns status immediately."""
    driver = None
    try:
        driver = build_connector(cred)
        driver.test()
        return ConnectionResponse(
            id="test",
            name=cred.name,
            db_type=cred.db_type,
            status="connected",
            database=cred.database,
        )
    except Exception as exc:
        return ConnectionResponse(
            id="test",
            name=cred.name,
            db_type=cred.db_type,
            status="error",
            database=cred.database,
            message=str(exc),
        )
    finally:
        if driver:
            try:
                driver.close()
            except Exception:
                pass


@router.post("", response_model=ConnectionResponse)
def add_connection(cred: ConnectionCredential):
    """Test + register a connection. Returns the connection ID for future calls."""
    driver = None
    try:
        driver = build_connector(cred)
        driver.test()
    except Exception as exc:
        if driver:
            try:
                driver.close()
            except Exception:
                pass
        raise HTTPException(status_code=422, detail=f"Connection failed: {exc}")

    conn_id = connection_store.add(cred, driver)
    return ConnectionResponse(
        id=conn_id,
        name=cred.name,
        db_type=cred.db_type,
        status="connected",
        database=cred.database,
    )


@router.get("", response_model=list[ConnectionResponse])
def list_connections():
    items = connection_store.get_all()
    responses = []

    for c in items:
        cred = getattr(c, "cred", None) or getattr(c, "credential", None)

        responses.append(
            ConnectionResponse(
                id=getattr(c, "id", ""),
                name=getattr(c, "name", None) or (cred.name if cred else ""),
                db_type=getattr(c, "db_type", None) or (cred.db_type if cred else None),
                status=getattr(c, "status", "connected"),
                database=getattr(c, "database", None) or (cred.database if cred else None),
            )
        )

    return responses


@router.delete("/{conn_id}")
def remove_connection(conn_id: str):
    if not connection_store.remove(conn_id):
        raise HTTPException(status_code=404, detail="Connection not found")
    return {"removed": conn_id}