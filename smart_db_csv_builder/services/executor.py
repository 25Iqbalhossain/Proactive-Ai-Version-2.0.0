from __future__ import annotations

import logging
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas.api.types import is_numeric_dtype

from data_processing.dataset_analyzer import DatasetAnalyzer
from smart_db_csv_builder.connectors.drivers import build_select_sql
from smart_db_csv_builder.core.connection_store import connection_store
from smart_db_csv_builder.models.schemas import OutputFormat
from smart_db_csv_builder.services.llm_planner import CollectionFetch, MergePlan, TableQuery

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "built_datasets"
TRAINING_USER_HINTS = ("user", "customer", "member", "account", "client", "person", "subscriber", "owner", "employee", "patient", "buyer")
TRAINING_ITEM_HINTS = ("item", "product", "service", "offer", "asset", "object", "content", "listing", "catalog", "sku", "article", "book", "movie", "plan", "package")
TRAINING_RATING_HINTS = ("rating", "score", "stars", "grade", "value", "weight", "rank", "preference")
TRAINING_TIME_HINTS = ("time", "timestamp", "date", "created", "updated", "viewed", "purchased", "ordered", "event")

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


def _make_output_file_map(output_stem: str | None) -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^0-9a-zA-Z._-]+", "_", (output_stem or "").strip()).strip("._-")
    if not stem:
        stem = f"dataset_{uuid.uuid4().hex[:12]}"

    csv_path = OUTPUT_DIR / f"{stem}.csv"
    json_path = OUTPUT_DIR / f"{stem}.json"
    suffix = 2

    while csv_path.exists() or json_path.exists():
        csv_path = OUTPUT_DIR / f"{stem}_{suffix}.csv"
        json_path = OUTPUT_DIR / f"{stem}_{suffix}.json"
        suffix += 1

    return {
        OutputFormat.CSV.value: str(csv_path),
        OutputFormat.JSON.value: str(json_path),
    }


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

    for col in ("userID", "itemID", "rating", "timestamp"):
        add(col)

    for col in final_columns:
        add(col)
        for m in source_maps:
            for c in m.get(col, []):
                add(c)

    for col in merged.columns:
        add(col)

    return ordered


def _coalesce_requested_output_columns(
    merged: pd.DataFrame,
    final_columns: list[str],
    source_maps: list[dict[str, list[str]]],
    merge_keys: list[str],
) -> pd.DataFrame:
    if not final_columns:
        return merged

    merge_key_set = {key.lower() for key in merge_keys}
    drop_columns: list[str] = []

    for logical_name in final_columns:
        if logical_name.lower() in merge_key_set:
            continue

        candidate_columns: list[str] = []
        if logical_name in merged.columns:
            candidate_columns.append(logical_name)

        for source_map in source_maps:
            for actual_column in source_map.get(logical_name, []):
                if actual_column in merged.columns and actual_column not in candidate_columns:
                    candidate_columns.append(actual_column)

        if len(candidate_columns) <= 1:
            continue

        merged[logical_name] = merged[candidate_columns].bfill(axis=1).iloc[:, 0]
        drop_columns.extend(
            column_name
            for column_name in candidate_columns
            if column_name != logical_name
        )

    if drop_columns:
        merged = merged.drop(columns=list(dict.fromkeys(drop_columns)), errors="ignore")

    return merged


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _column_looks_like_time(column_name: str) -> bool:
    normalized = _normalize_name(column_name)
    return any(hint in normalized for hint in TRAINING_TIME_HINTS)


def _column_looks_like_rating(column_name: str) -> bool:
    normalized = _normalize_name(column_name)
    return any(hint in normalized for hint in TRAINING_RATING_HINTS)


def _candidate_id_score(series: pd.Series, column_name: str, role: str) -> float:
    normalized = _normalize_name(column_name)
    non_null = series.dropna()
    n_total = max(len(non_null), 1)
    n_unique = int(non_null.nunique())
    ratio = n_unique / n_total if n_total else 1.0
    score = 0.0

    if n_unique < 2:
        return -1_000.0
    if pd.api.types.is_float_dtype(series):
        score -= 80.0
    if _column_looks_like_time(column_name) or _column_looks_like_rating(column_name):
        score -= 120.0

    role_hints = TRAINING_USER_HINTS if role == "userID" else TRAINING_ITEM_HINTS
    if any(hint in normalized for hint in role_hints):
        score += 160.0
    if any(hint in normalized for hint in ("id", "key", "code", "ref", "uuid", "guid", "sku")):
        score += 35.0
    if normalized == "id":
        score -= 90.0

    if ratio < 0.98:
        score += 55.0
    if ratio < 0.90:
        score += 20.0
    if ratio < 0.50:
        score += 10.0
    if ratio >= 0.995 and n_total > 20:
        score -= 60.0

    if 3 <= n_unique <= 500_000:
        score += 10.0
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        score += 10.0

    return score


def _best_training_id_candidate(df: pd.DataFrame, role: str, exclude: set[str]) -> str | None:
    ranked: list[tuple[float, str]] = []
    for column_name in df.columns:
        if column_name in exclude:
            continue
        score = _candidate_id_score(df[column_name], column_name, role)
        if score > -100:
            ranked.append((score, column_name))

    if not ranked:
        return None

    ranked.sort(reverse=True)
    best_score, best_column = ranked[0]
    if best_score < 40:
        return None
    return best_column


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_names = [name for name, count in Counter(df.columns).items() if count > 1]
    if not duplicate_names:
        return df

    out = df.copy()
    for name in duplicate_names:
        matching = out.loc[:, out.columns == name]
        out = out.drop(columns=name)
        out[name] = matching.bfill(axis=1).iloc[:, 0]
    return out


def _standardize_training_columns(merged: pd.DataFrame) -> pd.DataFrame:
    analyzer = DatasetAnalyzer()
    detected = analyzer.detect_columns(merged)
    resolved = {
        "userID": detected.userID,
        "itemID": detected.itemID,
        "rating": detected.rating,
        "timestamp": detected.timestamp,
    }

    used_columns = {column_name for column_name in resolved.values() if column_name}
    for role_name in ("userID", "itemID"):
        if not resolved[role_name]:
            candidate = _best_training_id_candidate(merged, role_name, used_columns)
            if candidate:
                resolved[role_name] = candidate
                used_columns.add(candidate)

    rename_map = {}
    for canonical_name in ("userID", "itemID", "rating", "timestamp"):
        source_name = resolved.get(canonical_name)
        if source_name and source_name != canonical_name:
            rename_map[source_name] = canonical_name

    standardized = merged.rename(columns=rename_map)
    standardized = _coalesce_duplicate_columns(standardized)

    if "userID" not in standardized.columns or "itemID" not in standardized.columns:
        detected_after = analyzer.detect_columns(standardized)
        raise RuntimeError(
            "Built dataset is not training-ready: no reliable user-item interaction identifiers could be normalized. "
            f"Detected mapping was userID={detected_after.userID!r}, itemID={detected_after.itemID!r}, "
            f"rating={detected_after.rating!r}, timestamp={detected_after.timestamp!r}. "
            f"Available columns: {list(standardized.columns)}"
        )

    return standardized


def execute_plan(
    plan: MergePlan,
    output_format: OutputFormat,
    max_rows_per_table: int = 50_000,
    progress_cb: ProgressCallback | None = None,
    output_stem: str | None = None,
):
    def report(p, msg):
        logger.info("[%d%%] %s", p, msg)
        if progress_cb:
            progress_cb(p, msg)

    raw_frames = []
    failures: list[str] = []

    for tq in plan.table_queries:
        conn = connection_store.get(tq.connection_id)
        if not conn:
            failures.append(f"Connection '{tq.connection_id}' was not found for table '{tq.table}'.")
            continue
        if not tq.columns:
            failures.append(f"Table '{tq.table}' has no validated columns to select.")
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
            logger.error("SQL error for %s: %s", tq.table, exc)
            failures.append(f"Query failed for table '{tq.table}': {exc}")

    for cf in plan.collection_fetches:
        conn = connection_store.get(cf.connection_id)
        if not conn:
            failures.append(f"Connection '{cf.connection_id}' was not found for collection '{cf.collection}'.")
            continue

        try:
            docs = conn.driver.fetch_collection(cf.collection)
            df = pd.json_normalize(docs)

            df = _apply_aliases(df, cf.alias_map)
            raw_frames.append((cf.collection, df))

        except Exception as exc:
            logger.error("Mongo error for %s: %s", cf.collection, exc)
            failures.append(f"Collection fetch failed for '{cf.collection}': {exc}")

    if failures:
        raise RuntimeError(" ; ".join(failures))

    frames, source_maps = _prepare_frames_for_merge(raw_frames, plan.merge_keys)
    frames = _normalize_merge_keys(frames, plan.merge_keys)

    if not frames:
        raise RuntimeError("No data collected")

    report(60, "Merging data")

    merged = frames[0]

    for other in frames[1:]:
        keys = [k for k in plan.merge_keys if k in merged.columns and k in other.columns]

        if not keys:
            raise RuntimeError("Unable to merge result sets because no validated merge keys were present in both tables.")

        merged = pd.merge(merged, other, on=keys, how="outer")

    merged = _coalesce_requested_output_columns(
        merged=merged,
        final_columns=plan.final_columns,
        source_maps=source_maps,
        merge_keys=plan.merge_keys,
    )
    merged = _standardize_training_columns(merged)
    ordered_cols = _resolve_output_columns(merged, plan.final_columns, source_maps)
    merged = merged[ordered_cols]
    merged = merged.drop_duplicates().reset_index(drop=True)

    output_files = _make_output_file_map(output_stem)
    merged.to_csv(output_files[OutputFormat.CSV.value], index=False)
    merged.to_json(output_files[OutputFormat.JSON.value], orient="records", indent=2)

    path = output_files[output_format.value]
    logger.info("Saved outputs: %s", output_files)

    return path, len(merged), len(merged.columns), output_files
