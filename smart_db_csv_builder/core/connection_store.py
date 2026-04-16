
from __future__ import annotations
import threading
import uuid
from dataclasses import dataclass

@dataclass
class ConnectionRecord:
    id: str
    cred: object
    driver: object

class ConnectionStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._items: dict[str, ConnectionRecord] = {}

    def add(self, cred, driver):
        conn_id = str(uuid.uuid4())
        with self._lock:
            self._items[conn_id] = ConnectionRecord(id=conn_id, cred=cred, driver=driver)
        return conn_id

    def get(self, conn_id):
        with self._lock:
            return self._items.get(conn_id)

    def get_all(self):
        with self._lock:
            return list(self._items.values())

    def remove(self, conn_id):
        with self._lock:
            rec = self._items.pop(conn_id, None)
        if rec is None:
            return False
        try:
            rec.driver.close()
        except Exception:
            pass
        return True

connection_store = ConnectionStore()
