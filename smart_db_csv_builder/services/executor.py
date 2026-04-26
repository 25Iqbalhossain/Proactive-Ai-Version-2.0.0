from __future__ import annotations

import logging
import os
import re
import tempfile
from collections import Counter
from typing import Callable

import pandas as pd
from pandas.api.types import is_numeric_dtype

from smart_db_csv_builder.connectors.drivers import build_select_sql
from smart_db_csv_builder.core.connection_store import connection_store
from smart_db_csv_builder.models.schemas import OutputFormat
from smart_db_csv_builder.services.llm_planner import CollectionFetch, MergePlan, TableQuery

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]


def _normalize_table_name(table: str) -> str:
    return table.split(".")[-1] if table else table


def _sanitize_source_name(source: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", source or "").strip("_").lower()
    return cleaned or "source"


def _make_unique_column_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def _apply_aliases(df: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns})


def _format_merge_key_number(value) -> str:
    as_float = float(value)
    if as_float.is_integer():
        return str(int(as_float))
    return format(as_float, "g")


def _normalize_merge_key_series(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.map(
            lambda value: pd.NA if pd.isna(value) else _format_merge_key_number(value)
        ).astype("string")

    normalized = series.astype("string").str.strip()
    numeric = pd.to_numeric(normalized, errors="coerce")
    numeric_mask = normalized.notna() & numeric.notna()

    if numeric_mask.any():
        normalized = normalized.copy()
        normalized.loc[numeric_mask] = numeric.loc[numeric_mask].map(_format_merge_key_number)

    return normalized


def _normalize_merge_keys(frames: list[pd.DataFrame], merge_keys: list[str]) -> list[pd.DataFrame]:
    normalized_frames: list[pd.DataFrame] = []

    for frame in frames:
        normalized = frame.copy()
        for key in merge_keys:
            if key in normalized.columns:
                normalized[key] = _normalize_merge_key_series(normalized[key])
        normalized_frames.append(normalized)

    return normalized_frames


def _prepare_frames_for_merge(raw_frames, merge_keys):
    merge_key_set = set(merge_keys)
    column_counts = Counter()

    for _, frame in raw_frames:
        column_counts.update([c for c in frame.columns if c not in merge_key_set])

    prepared = []
    source_maps = []
    taken = set(merge_keys)

    for source, df in raw_frames:
        prefix = _sanitize_source_name(source)
        rename_map = {}
        mapping = {}

        for col in df.columns:
            if col in merge_key_set:
                new_col = col
            else:
                base = f"{prefix}_{col}" if column_counts[col] > 1 else col
                new_col = _make_unique_column_name(base, taken)

            rename_map[col] = new_col
            mapping.setdefault(col, []).append(new_col)
            taken.add(new_col)

        prepared.append(df.rename(columns=rename_map))
        source_maps.append(mapping)

    return prepared, source_maps


def _resolve_output_columns(merged, final_columns, source_maps):
    if not final_columns:
        return list(merged.columns)

    ordered = []

    def add(col):
        if col in merged.columns and col not in ordered:
            ordered.append(col)

    for col in final_columns:
        add(col)
        for m in source_maps:
            for c in m.get(col, []):
                add(c)

    for col in merged.columns:
        add(col)

    return ordered


def execute_plan(
    plan: MergePlan,
    output_format: OutputFormat,
    max_rows_per_table: int = 50_000,
    progress_cb: ProgressCallback | None = None,
):
    def report(p, msg):
        logger.info("[%d%%] %s", p, msg)
        if progress_cb:
            progress_cb(p, msg)

    raw_frames = []

    for tq in plan.table_queries:
        conn = connection_store.get(tq.connection_id)
        if not conn:
            continue

        sql = build_select_sql(
            db_type=conn.db_type,
            table=tq.table,
            columns=tq.columns,
            where=tq.where,
            limit=max_rows_per_table,
        )

        try:
            rows = conn.driver.execute(sql)
            df = pd.DataFrame(rows)

            if df.empty:
                continue

            df = _apply_aliases(df, tq.alias_map)
            raw_frames.append((_normalize_table_name(tq.table), df))

        except Exception as exc:
            logger.error("SQL error: %s", exc)

    for cf in plan.collection_fetches:
        conn = connection_store.get(cf.connection_id)
        if not conn:
            continue

        try:
            docs = conn.driver.fetch_collection(cf.collection)
            df = pd.json_normalize(docs)

            df = _apply_aliases(df, cf.alias_map)
            raw_frames.append((cf.collection, df))

        except Exception as exc:
            logger.error("Mongo error: %s", exc)

    frames, source_maps = _prepare_frames_for_merge(raw_frames, plan.merge_keys)
    frames = _normalize_merge_keys(frames, plan.merge_keys)

    if not frames:
        raise RuntimeError("No data collected")

    report(60, "Merging data")

    merged = frames[0]

    for other in frames[1:]:
        keys = [k for k in plan.merge_keys if k in merged.columns and k in other.columns]

        if keys:
            merged = pd.merge(merged, other, on=keys, how="outer")
        else:
            merged = pd.concat([merged, other], ignore_index=True)

    ordered_cols = _resolve_output_columns(merged, plan.final_columns, source_maps)
    merged = merged[ordered_cols]
    merged = merged.drop_duplicates().reset_index(drop=True)

    output_files: dict[str, str] = {}

    fd, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    merged.to_csv(csv_path, index=False)
    output_files[OutputFormat.CSV.value] = csv_path

    fd, json_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    merged.to_json(json_path, orient="records", indent=2)
    output_files[OutputFormat.JSON.value] = json_path

    path = output_files[output_format.value]
    logger.info("Saved outputs: %s", output_files)

    return path, len(merged), len(merged.columns), output_files
