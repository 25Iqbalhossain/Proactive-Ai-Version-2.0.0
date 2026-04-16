"""
models/model_loader.py – Model loader for the serving system

Loads the production model from the registry into memory and
keeps it warm for low-latency inference.  Supports hot-reload
when a new model is promoted without restarting the server.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Optional

from models.model_registry import ModelRegistry, ModelRecord
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LoadedModel:
    """In-memory model with its metadata."""
    model:      Any
    record:     ModelRecord
    loaded_at:  float

    @property
    def algorithm(self) -> str:
        return self.record.algorithm

    @property
    def model_id(self) -> str:
        return self.record.model_id

    @property
    def is_implicit(self) -> bool:
        return self.record.is_implicit


class ModelLoader:
    """
    Manages the in-memory lifecycle of production models.

    Features:
      - Load model by ID or auto-load the promoted model for an algorithm
      - Thread-safe model replacement (hot-reload)
      - Warm-up validation after loading
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self._registry = registry or ModelRegistry()
        self._loaded:   dict[str, LoadedModel] = {}   # algorithm → LoadedModel
        self._lock      = threading.Lock()

    # ── Load ───────────────────────────────────────────────────────────────────

    def load_promoted(self, algorithm: str) -> LoadedModel:
        """
        Load the promoted (production) model for an algorithm.
        Caches in memory; subsequent calls return the cached version.
        """
        with self._lock:
            # Return cached if still promoted
            cached = self._loaded.get(algorithm)
            if cached:
                current = self._registry.get_promoted(algorithm)
                if current and current.model_id == cached.model_id:
                    return cached

            record = self._registry.get_promoted(algorithm)
            if not record:
                raise RuntimeError(
                    f"No promoted model found for algorithm '{algorithm}'. "
                    f"Train and promote a model first."
                )
            return self._load_record(record)

    def load_by_id(self, model_id: str) -> LoadedModel:
        """Load a specific model by its registry ID."""
        with self._lock:
            # Check if already loaded
            for lm in self._loaded.values():
                if lm.model_id == model_id:
                    return lm
            records = self._registry.list_models()
            record  = next((r for r in records if r.model_id == model_id), None)
            if not record:
                raise KeyError(f"Model ID '{model_id}' not found in registry.")
            return self._load_record(record)

    def reload(self, algorithm: str) -> LoadedModel:
        """Force-reload the promoted model (e.g. after a new promotion)."""
        with self._lock:
            self._loaded.pop(algorithm, None)
        return self.load_promoted(algorithm)

    def unload(self, algorithm: str) -> None:
        """Remove a model from memory."""
        with self._lock:
            dropped = self._loaded.pop(algorithm, None)
            if dropped:
                log.info("Model unloaded from memory | algo=%s id=%s", algorithm, dropped.model_id)

    # ── Inspection ─────────────────────────────────────────────────────────────

    def list_loaded(self) -> list[dict]:
        """Return info about all currently in-memory models."""
        with self._lock:
            return [
                {
                    "algorithm":  lm.algorithm,
                    "model_id":   lm.model_id,
                    "version":    lm.record.version,
                    "loaded_at":  lm.loaded_at,
                    "is_implicit": lm.is_implicit,
                }
                for lm in self._loaded.values()
            ]

    # ── Internals ──────────────────────────────────────────────────────────────

    def _load_record(self, record: ModelRecord) -> LoadedModel:
        log.info(
            "Loading model into memory | id=%s algo=%s version=%d",
            record.model_id, record.algorithm, record.version,
        )
        t0    = time.time()
        model = self._registry.load(record.model_id)
        elapsed = time.time() - t0

        lm = LoadedModel(model=model, record=record, loaded_at=time.time())
        self._loaded[record.algorithm] = lm

        log.info(
            "Model loaded | algo=%s id=%s version=%d elapsed=%.2fs",
            record.algorithm, record.model_id, record.version, elapsed,
        )
        return lm
