"""
ingestion/csv_loader.py – File-based data loader

Supports: CSV, Excel (.xlsx/.xls), JSON, Parquet
Accepts file paths or in-memory bytes/BytesIO objects (for API uploads).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# Type alias for flexible input
FileInput = Union[str, Path, bytes, io.BytesIO]


class FileLoader:
    """
    Load tabular data from disk or in-memory file objects.

    Supports CSV, Excel, JSON (records/lines), Parquet.
    """

    _SUPPORTED = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}

    def load(self, source: FileInput, file_format: str = "auto") -> pd.DataFrame:
        """
        Load a file and return a DataFrame.

        Parameters
        ----------
        source      : file path (str/Path) or raw bytes / BytesIO
        file_format : "auto" (infer from extension), "csv", "excel",
                      "json", "parquet"
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            ext = path.suffix.lower()
            buf = path
        else:
            buf = io.BytesIO(source) if isinstance(source, bytes) else source
            ext = ""   # format must be inferred from file_format arg

        fmt = self._resolve_format(file_format, ext)
        log.info("Loading file | format=%s source_type=%s", fmt, type(source).__name__)

        df = self._dispatch(buf, fmt)
        log.info("Loaded %d rows × %d columns", *df.shape)
        return df

    # ── Dispatch ───────────────────────────────────────────────────────────────

    def _dispatch(self, buf, fmt: str) -> pd.DataFrame:
        loaders = {
            "csv":     self._load_csv,
            "excel":   self._load_excel,
            "json":    self._load_json,
            "parquet": self._load_parquet,
        }
        loader = loaders.get(fmt)
        if not loader:
            raise ValueError(f"Unsupported format '{fmt}'. Choose from: {list(loaders)}")
        return loader(buf)

    # ── Format loaders ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_csv(buf) -> pd.DataFrame:
        try:
            return pd.read_csv(buf, low_memory=False)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}") from e

    @staticmethod
    def _load_excel(buf) -> pd.DataFrame:
        try:
            return pd.read_excel(buf, engine="openpyxl")
        except ImportError:
            raise ImportError("Install 'openpyxl' to load Excel files: pip install openpyxl")
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {e}") from e

    @staticmethod
    def _load_json(buf) -> pd.DataFrame:
        try:
            # Try records orient first, then lines format
            try:
                return pd.read_json(buf, orient="records")
            except Exception:
                if hasattr(buf, "seek"):
                    buf.seek(0)
                return pd.read_json(buf, lines=True)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {e}") from e

    @staticmethod
    def _load_parquet(buf) -> pd.DataFrame:
        try:
            return pd.read_parquet(buf)
        except ImportError:
            raise ImportError("Install 'pyarrow' or 'fastparquet' to load Parquet files.")
        except Exception as e:
            raise ValueError(f"Failed to parse Parquet: {e}") from e

    # ── Format resolution ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_format(file_format: str, ext: str) -> str:
        if file_format != "auto":
            return file_format.lower()
        ext_map = {
            ".csv":     "csv",
            ".xlsx":    "excel",
            ".xls":     "excel",
            ".json":    "json",
            ".jsonl":   "json",
            ".parquet": "parquet",
        }
        fmt = ext_map.get(ext)
        if not fmt:
            raise ValueError(
                f"Cannot infer format from extension '{ext}'. "
                f"Pass file_format explicitly: csv, excel, json, parquet."
            )
        return fmt
