"""
services/builder.py

Top-level orchestrator for a build job.
Runs in a background thread so the HTTP endpoint returns immediately.

Steps:
  1. Validate connection IDs
  2. Extract schemas from all connections
  3. Call LLM to produce MergePlan
  4. Execute MergePlan → dataset file
  5. Update job status at each step
"""

from __future__ import annotations

import logging
import os
import re
import traceback
from collections import Counter

from smart_db_csv_builder.core.connection_store import connection_store
from smart_db_csv_builder.core.job_store import Job, JobStatus
from smart_db_csv_builder.models.schemas import BuildMode, BuildRequest, DBType, SchemaResponse
from smart_db_csv_builder.services.llm_planner import (
    MergePlan,
    TableQuery,
    _auto_alias_role_columns,
    build_merge_plan,
)
from smart_db_csv_builder.services.executor import execute_plan, execute_raw_sql_query

logger = logging.getLogger(__name__)

STEPS = [
    "validate_connections",
    "extract_schemas",
    "llm_plan",
    "execute_queries",
    "write_output",
]

RELATIONSHIP_RE = re.compile(
    r"(?P<left_table>[\w.]+)\.(?P<left_col>[\w]+)\s*=\s*(?P<right_table>[\w.]+)\.(?P<right_col>[\w]+)"
)
MANUAL_ROLE_FIELDS = (
    ("target_field", "userID"),
    ("label_field", "itemID"),
)
GENERIC_JOIN_COLUMNS = {"id", "ref", "key", "code", "uuid", "guid"}
RAW_SQL_PREFIX_RE = re.compile(r"^(?:--[^\n]*\s+|/\*.*?\*/\s+)*(select|with)\b", re.IGNORECASE | re.DOTALL)
SQL_BUILD_DB_TYPES = {DBType.MYSQL, DBType.POSTGRES, DBType.MSSQL, DBType.SQLITE}


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_target_description(req: BuildRequest) -> str | None:
    if req.mode == BuildMode.QUERY:
        return _clean_text(req.query_text) or _clean_text(req.target_description)
    if req.mode == BuildMode.LLM:
        return _clean_text(req.llm_prompt) or _clean_text(req.target_description)
    return _clean_text(req.target_description)


def _sanitize_raw_sql_query(query_text: str | None) -> str | None:
    cleaned = _clean_text(query_text)
    if not cleaned:
        return None

    stripped = cleaned.rstrip().rstrip(";").rstrip()
    if not RAW_SQL_PREFIX_RE.match(stripped):
        return None

    if ";" in stripped:
        raise ValueError("Raw SQL query mode accepts a single SELECT/WITH statement only.")

    return stripped


def _resolve_raw_sql_query(req: BuildRequest) -> str | None:
    if req.mode != BuildMode.QUERY:
        return None
    return _sanitize_raw_sql_query(req.query_text)


def _table_lookup(
    schemas: list[SchemaResponse],
) -> tuple[dict[str, tuple[SchemaResponse, object]], dict[str, list[tuple[SchemaResponse, object]]]]:
    exact: dict[str, tuple[SchemaResponse, object]] = {}
    short: dict[str, list[tuple[SchemaResponse, object]]] = {}

    for schema in schemas:
        for table in schema.tables:
            exact[table.full_name.lower()] = (schema, table)
            short.setdefault(table.table_name.lower(), []).append((schema, table))

    return exact, short


def _parse_table_names(*values: str | None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not value:
            continue
        for part in re.split(r"[\n,]+", value):
            token = part.strip()
            key = token.lower()
            if token and key not in seen:
                names.append(token)
                seen.add(key)

    return names


def _parse_manual_field_spec(value: str | None) -> tuple[str | None, str | None]:
    cleaned = _clean_text(value)
    if not cleaned:
        return None, None

    parts = [part.strip() for part in cleaned.split(".") if part.strip()]
    if len(parts) <= 1:
        return None, parts[0] if parts else None

    return ".".join(parts[:-1]), parts[-1]


def _table_matches_reference(table, reference: str | None) -> bool:
    reference = (reference or "").strip().lower()
    if not reference:
        return True

    full_name = (table.full_name or "").lower()
    short_name = (table.table_name or "").lower()
    return reference in {full_name, short_name} or full_name.endswith(f".{reference}")


def _build_table_query_lookup(table_queries: list[TableQuery]) -> dict[str, TableQuery]:
    lookup: dict[str, TableQuery] = {}
    for table_query in table_queries:
        full_name = table_query.table.lower()
        lookup.setdefault(full_name, table_query)
        short_name = full_name.split(".")[-1]
        lookup.setdefault(short_name, table_query)
    return lookup


def _projected_columns(columns: list[str], alias_map: dict[str, str]) -> list[str]:
    projected: list[str] = []
    seen: set[str] = set()

    for column in columns:
        output_name = alias_map.get(column, column)
        if output_name not in seen:
            projected.append(output_name)
            seen.add(output_name)

    return projected


def _manual_join_name(left_name: str, right_name: str) -> str:
    canonical = {"userID", "itemID", "rating", "timestamp"}
    if left_name in canonical:
        return left_name
    if right_name in canonical:
        return right_name
    if left_name.lower() == right_name.lower():
        return left_name
    if left_name.lower() in GENERIC_JOIN_COLUMNS and right_name.lower() not in GENERIC_JOIN_COLUMNS:
        return right_name
    if right_name.lower() in GENERIC_JOIN_COLUMNS and left_name.lower() not in GENERIC_JOIN_COLUMNS:
        return left_name
    return left_name if len(left_name) >= len(right_name) else right_name


def _build_projected_column_index(
    projected_columns_per_query: list[list[str]],
) -> tuple[Counter[str], dict[str, str]]:
    projected_counter: Counter[str] = Counter()
    canonical_names: dict[str, str] = {}

    for projected_columns in projected_columns_per_query:
        for column_name in set(projected_columns):
            lowered = column_name.lower()
            projected_counter[lowered] += 1
            canonical_names.setdefault(lowered, column_name)

    return projected_counter, canonical_names


def _build_manual_plan(req: BuildRequest, schemas: list[SchemaResponse]) -> MergePlan:
    manual = req.manual_config or {}
    exact_lookup, short_lookup = _table_lookup(schemas)

    manual_tables = _parse_table_names(manual.get("tables"))
    relationship_matches = list(RELATIONSHIP_RE.finditer(manual.get("relationships") or ""))

    for match in relationship_matches:
        manual_tables.extend(_parse_table_names(match.group("left_table"), match.group("right_table")))

    selected: list[tuple[SchemaResponse, object]] = []
    selected_keys: set[tuple[str, str]] = set()

    def add_table(name: str):
        lowered = name.lower()
        matches = [exact_lookup[lowered]] if lowered in exact_lookup else short_lookup.get(lowered, [])
        for schema, table in matches:
            key = (schema.connection_id, table.full_name)
            if key not in selected_keys:
                selected.append((schema, table))
                selected_keys.add(key)

    for name in manual_tables:
        add_table(name)

    if not selected:
        for schema in schemas:
            for table in schema.tables:
                key = (schema.connection_id, table.full_name)
                if key not in selected_keys:
                    selected.append((schema, table))
                    selected_keys.add(key)

    if not selected:
        raise ValueError("Manual mode could not resolve any tables from the selected connections.")

    table_queries = []
    for schema, table in selected:
        column_names = [column.name for column in table.columns]
        table_queries.append(
            TableQuery(
                connection_id=schema.connection_id,
                table=table.full_name,
                columns=column_names,
                alias_map=_auto_alias_role_columns(column_names),
            )
        )

    query_lookup = _build_table_query_lookup(table_queries)

    for field_name, canonical_name in MANUAL_ROLE_FIELDS:
        table_ref, column_name = _parse_manual_field_spec(manual.get(field_name))
        if not column_name:
            continue

        matched = False
        for table_query, (_, table) in zip(table_queries, selected):
            if not _table_matches_reference(table, table_ref):
                continue
            for query_column in table_query.columns:
                if query_column.lower() == column_name.lower():
                    table_query.alias_map[query_column] = canonical_name
                    matched = True

        if not matched:
            raise ValueError(
                f"Manual mode field '{manual.get(field_name)}' could not be matched to any selected table column."
            )

    merge_keys: list[str] = []
    relationship_merge_keys: list[str] = []
    for match in relationship_matches:
        left_table = match.group("left_table").lower()
        right_table = match.group("right_table").lower()
        left_col = match.group("left_col")
        right_col = match.group("right_col")

        left_query = query_lookup.get(left_table)
        right_query = query_lookup.get(right_table)

        if not left_query or not right_query:
            continue

        if left_col not in left_query.columns or right_col not in right_query.columns:
            continue

        left_name = left_query.alias_map.get(left_col, left_col)
        right_name = right_query.alias_map.get(right_col, right_col)
        join_name = _manual_join_name(left_name, right_name)

        left_query.alias_map[left_col] = join_name
        right_query.alias_map[right_col] = join_name

        if join_name not in relationship_merge_keys:
            relationship_merge_keys.append(join_name)

    projected_columns_per_query = [
        _projected_columns(table_query.columns, table_query.alias_map)
        for table_query in table_queries
    ]
    projected_counter, canonical_names = _build_projected_column_index(projected_columns_per_query)

    for merge_key in relationship_merge_keys:
        lowered = merge_key.lower()
        if projected_counter[lowered] >= 2 and canonical_names[lowered] not in merge_keys:
            merge_keys.append(canonical_names[lowered])

    for projected_columns in projected_columns_per_query:
        for column_name in projected_columns:
            lowered = column_name.lower()
            if projected_counter[lowered] < 2:
                continue
            canonical_name = canonical_names[lowered]
            if canonical_name not in merge_keys:
                merge_keys.append(canonical_name)

    final_columns: list[str] = []
    for preferred_name in ("userID", "itemID", "rating", "timestamp"):
        if any(preferred_name == column_name for columns in projected_columns_per_query for column_name in columns):
            final_columns.append(preferred_name)
    for projected_columns in projected_columns_per_query:
        for column_name in projected_columns:
            if column_name not in final_columns:
                final_columns.append(column_name)

    raw_plan = {
        "mode": "manual",
        "description": _clean_text(manual.get("notes")) or "Manual dataset build",
        "tables": [table.full_name for _, table in selected],
        "relationships": _clean_text(manual.get("relationships")) or "",
        "target_field": _clean_text(manual.get("target_field")) or "",
        "label_field": _clean_text(manual.get("label_field")) or "",
        "notes": _clean_text(manual.get("notes")) or "",
        "merge_keys": merge_keys,
        "final_columns": final_columns,
        "table_queries": [
            {
                "connection_id": tq.connection_id,
                "table": tq.table,
                "columns": tq.columns,
                "alias_map": tq.alias_map,
                "where": tq.where,
            }
            for tq in table_queries
        ],
    }

    return MergePlan(
        table_queries=table_queries,
        collection_fetches=[],
        merge_keys=merge_keys,
        final_columns=final_columns,
        description=raw_plan["description"],
        raw_plan=raw_plan,
    )


def run_build_job(job: Job, req: BuildRequest) -> None:
    """
    Synchronous function meant to be called inside a thread pool executor.
    Mutates `job` in place so the polling endpoint sees live updates.
    """
    job.status = JobStatus.RUNNING

    for s in STEPS:
        job.set_step(s, "pending")

    try:
        # ── Step 1: Validate connections ──────────────────────────────────
        job.set_step("validate_connections", "running")
        job.progress = 5

        conns = []
        for cid in req.connection_ids:
            conn = connection_store.get(cid)
            if conn is None:
                raise ValueError(f"Connection '{cid}' not found. Please register it first.")
            conns.append(conn)

        job.set_step("validate_connections", "done",
                     f"{len(conns)} connection(s) verified")
        job.progress = 15

        raw_sql_query = _resolve_raw_sql_query(req)
        if raw_sql_query:
            sql_connections = [
                (cid, conn)
                for cid, conn in zip(req.connection_ids, conns)
                if conn.db_type in SQL_BUILD_DB_TYPES
            ]
            if len(sql_connections) != 1:
                raise ValueError(
                    "Raw SQL query mode requires exactly one SQL connection because the query is executed as a single final dataset query."
                )

            raw_connection_id, _ = sql_connections[0]
            job.set_step("extract_schemas", "done", "Skipped for raw SQL query")
            job.set_step("llm_plan", "done", "Raw SQL query detected; planner skipped")
            job.plan = {
                "mode": "query",
                "raw_sql": True,
                "connection_id": raw_connection_id,
                "description": "Raw SQL query executed directly",
                "required_columns": ["user_id", "item_id"],
                "optional_columns": ["interaction_value", "timestamp"],
                "sql": raw_sql_query,
            }
            job.progress = 55

            job.set_step("execute_queries", "running")

            def on_raw_progress(pct: int, msg: str):
                job.progress = 55 + int(pct * 0.40)
                job.set_step("execute_queries" if pct < 85 else "write_output", "running", msg)

            filepath, row_count, col_count, output_files = execute_raw_sql_query(
                connection_id=raw_connection_id,
                sql=raw_sql_query,
                output_format=req.output_format,
                max_rows=req.max_rows_per_table,
                progress_cb=on_raw_progress,
                output_stem=f"recommendation_dataset_{job.job_id[:8]}",
            )

            job.set_step("execute_queries", "done")
            job.set_step("write_output", "done", f"{row_count} rows × {col_count} columns")
            job.output_file = filepath
            job.output_files = output_files
            job.output_format = req.output_format
            job.row_count = row_count
            job.column_count = col_count
            job.progress = 100
            job.status = JobStatus.DONE

            logger.info("Job %s done - %s (%d rows)", job.job_id, filepath, row_count)
            return

        # ── Step 2: Extract schemas ───────────────────────────────────────
        job.set_step("extract_schemas", "running")

        schemas: list[SchemaResponse] = []
        for conn in conns:
            try:
                schema = conn.driver.get_schema(conn.id)
                schemas.append(schema)
                logger.info("Schema extracted from %s: %d tables, %d collections",
                            conn.name,
                            len(schema.tables),
                            len(schema.collections))
            except Exception as exc:
                raise RuntimeError(f"Schema extraction failed for '{conn.name}': {exc}") from exc

        job.set_step("extract_schemas", "done",
                     f"{sum(len(s.tables) for s in schemas)} tables, "
                     f"{sum(len(s.collections) for s in schemas)} collections")
        job.progress = 35

        # ── Step 3: LLM merge plan ────────────────────────────────────────
        job.set_step("llm_plan", "running")
        if req.mode == BuildMode.MANUAL:
            plan = _build_manual_plan(req, schemas)
            job.set_step("llm_plan", "done", plan.description or "Manual plan generated")
        else:
            has_chat_env = bool(os.getenv("CHAT_API_KEY") and os.getenv("CHAT_MODEL_NAME"))
            logger.info(
                "Generating LLM plan with providers: mistral=%s chat=%s groq=%s openai=%s",
                bool(req.mistral_api_key or os.getenv("MISTRAL_API_KEY")),
                has_chat_env,
                bool(req.groq_api_key),
                bool(req.openai_api_key or req.anthropic_api_key),
            )

            plan = build_merge_plan(
                schemas=schemas,
                rec_type=req.rec_system_type,
                target_description=_resolve_target_description(req),
                mistral_api_key=req.mistral_api_key,
                groq_api_key=req.groq_api_key,
                openai_api_key=req.openai_api_key,
            )

            job.set_step("llm_plan", "done", plan.description or "Plan generated")

        job.plan = plan.raw_plan
        job.progress = 55

        # ── Step 4 + 5: Execute + write ───────────────────────────────────
        job.set_step("execute_queries", "running")

        def on_progress(pct: int, msg: str):
            # Map executor's 0-100 onto job's 55-95 range
            job.progress = 55 + int(pct * 0.40)
            job.set_step("execute_queries" if pct < 85 else "write_output",
                         "running", msg)

        filepath, row_count, col_count, output_files = execute_plan(
            plan=plan,
            output_format=req.output_format,
            max_rows_per_table=req.max_rows_per_table,
            progress_cb=on_progress,
            output_stem=f"recommendation_dataset_{job.job_id[:8]}",
        )

        job.set_step("execute_queries", "done")
        job.set_step("write_output", "done", f"{row_count} rows × {col_count} columns")

        # ── Done ──────────────────────────────────────────────────────────
        job.output_file   = filepath
        job.output_files  = output_files
        job.output_format = req.output_format
        job.row_count     = row_count
        job.column_count  = col_count
        job.progress      = 100
        job.status        = JobStatus.DONE

        logger.info("Job %s done — %s (%d rows)", job.job_id, filepath, row_count)

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Job %s FAILED:\n%s", job.job_id, tb)
        job.status  = JobStatus.FAILED
        job.error   = str(exc)
        # Mark the currently-running step as error
        for step in job.steps:
            if step.get("status") == "running":
                step["status"] = "error"
                step["message"] = str(exc)
