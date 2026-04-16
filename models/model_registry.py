"""
Model registry and versioned persistence for trained recommendation models.
"""
from __future__ import annotations

import json
import pickle
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.settings import MODEL_STORE_DIR
from utils.logger import get_logger

log = get_logger(__name__)

_REGISTRY_FILE = "registry.json"


@dataclass
class ModelRecord:
    model_id: str
    algorithm: str
    version: int
    metrics: dict
    params: dict
    is_implicit: bool
    created_at: float
    promoted: bool = False
    file_path: str = ""
    notes: str = ""
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    def __init__(self, store_dir: str = MODEL_STORE_DIR):
        self.store_dir = Path(store_dir).expanduser().resolve()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.store_dir / _REGISTRY_FILE
        self._index: dict[str, dict] = self._load_index()

    def save(
        self,
        model: Any,
        algorithm: str,
        metrics: dict,
        params: dict,
        is_implicit: bool = False,
        notes: str = "",
        tags: list | None = None,
    ) -> str:
        self.refresh()
        model_id = str(uuid.uuid4())[:12]
        version = self._next_version(algorithm)
        filename = f"{algorithm.lower().replace(' ', '_')}_{version}_{model_id}.pkl"
        path = (self.store_dir / filename).resolve()

        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        record = ModelRecord(
            model_id=model_id,
            algorithm=algorithm,
            version=version,
            metrics=metrics,
            params=params,
            is_implicit=is_implicit,
            created_at=time.time(),
            file_path=str(path),
            notes=notes,
            tags=tags or [],
        )
        self._index[model_id] = record.to_dict()
        self._save_index()
        log.info("Model saved | id=%s algo=%s version=%d path=%s", model_id, algorithm, version, path)
        return model_id

    def load(self, model_id: str) -> Any:
        record = self._get_record(model_id)
        path = self._resolve_file_path(record["file_path"])
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        log.info("Model loaded | id=%s algo=%s", model_id, record["algorithm"])
        return model

    def promote(self, model_id: str) -> None:
        self.refresh()
        record = self._get_record(model_id)
        algorithm = record["algorithm"]
        for rid, rec in self._index.items():
            if rec["algorithm"] == algorithm and rec.get("promoted"):
                self._index[rid]["promoted"] = False
        self._index[model_id]["promoted"] = True
        self._save_index()
        log.info("Model promoted | id=%s algo=%s", model_id, algorithm)

    def get_promoted(self, algorithm: str) -> Optional[ModelRecord]:
        self.refresh()
        for record in self._index.values():
            if record["algorithm"] == algorithm and record.get("promoted"):
                return ModelRecord(**self._normalise_record(record))
        return None

    def get(self, model_id: str) -> Optional[ModelRecord]:
        self.refresh()
        record = self._index.get(model_id)
        if not record:
            return None
        return ModelRecord(**self._normalise_record(record))

    def latest_for_algorithm(self, algorithm: str) -> Optional[ModelRecord]:
        records = self.list_models(algorithm=algorithm)
        return records[0] if records else None

    def list_models(
        self,
        algorithm: Optional[str] = None,
        promoted_only: bool = False,
    ) -> list[ModelRecord]:
        self.refresh()
        records = [ModelRecord(**self._normalise_record(v)) for v in self._index.values()]
        if algorithm:
            records = [r for r in records if r.algorithm == algorithm]
        if promoted_only:
            records = [r for r in records if r.promoted]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def delete(self, model_id: str, delete_file: bool = True) -> None:
        self.refresh()
        record = self._get_record(model_id)
        if delete_file:
            path = self._resolve_file_path(record["file_path"])
            if path.exists():
                path.unlink()
        del self._index[model_id]
        self._save_index()
        log.info("Model deleted | id=%s", model_id)

    def refresh(self) -> None:
        self._index = self._load_index()

    def _get_record(self, model_id: str) -> dict:
        self.refresh()
        record = self._index.get(model_id)
        if not record:
            raise KeyError(f"Model ID '{model_id}' not found in registry.")
        return self._normalise_record(record)

    def _next_version(self, algorithm: str) -> int:
        versions = [r["version"] for r in self._index.values() if r["algorithm"] == algorithm]
        return max(versions, default=0) + 1

    def _load_index(self) -> dict:
        if not self._index_path.exists():
            return {}
        try:
            with open(self._index_path) as f:
                raw = json.load(f)
            return {
                model_id: self._normalise_record(record)
                for model_id, record in raw.items()
            }
        except Exception as e:
            log.warning("Registry index corrupted, starting fresh: %s", e)
            return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _normalise_record(self, record: dict) -> dict:
        out = dict(record)
        out["file_path"] = str(self._resolve_file_path(out.get("file_path", "")))
        return out

    def _resolve_file_path(self, raw_path: str) -> Path:
        path = Path(raw_path or "")
        if not raw_path:
            return self.store_dir
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == self.store_dir.name:
            return (self.store_dir.parent / path).resolve()
        return (self.store_dir / path.name).resolve()
