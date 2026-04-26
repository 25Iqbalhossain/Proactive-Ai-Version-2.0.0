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

from smart_db_csv_builder.core.connection_store import connection_store
from smart_db_csv_builder.core.job_store import Job, JobStatus
from smart_db_csv_builder.models.schemas import BuildMode, BuildRequest, SchemaResponse
from smart_db_csv_builder.services.llm_planner import MergePlan, TableQuery, build_merge_plan
from smart_db_csv_builder.services.executor import execute_plan

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


def _parse_field_name(value: str | None) -> str | None:
    value = _clean_text(value)
    if not value:
        return None
    return value.split(".")[-1].strip() or None


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

    table_queries = [
        TableQuery(
            connection_id=schema.connection_id,
            table=table.full_name,
            columns=[col.name for col in table.columns],
            alias_map={},
        )
        for schema, table in selected
    ]

    merge_keys: list[str] = []
    for match in relationship_matches:
        left_table = match.group("left_table").lower()
        right_table = match.group("right_table").lower()
        left_col = match.group("left_col")
        right_col = match.group("right_col")

        left_query = next(
            (tq for tq in table_queries if tq.table.lower() == left_table or tq.table.lower().endswith(f".{left_table}")),
            None,
        )
        right_query = next(
            (tq for tq in table_queries if tq.table.lower() == right_table or tq.table.lower().endswith(f".{right_table}")),
            None,
        )

        if not left_query or not right_query:
            continue

        if left_col not in merge_keys:
            merge_keys.append(left_col)
        if right_col != left_col:
            right_query.alias_map[right_col] = left_col

    final_columns: list[str] = []
    for value in (_parse_field_name(manual.get("target_field")), _parse_field_name(manual.get("label_field"))):
        if value and value not in final_columns:
            final_columns.append(value)

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
                "Generating LLM plan with providers: chat=%s groq=%s openai=%s",
                has_chat_env,
                bool(req.groq_api_key),
                bool(req.openai_api_key or req.anthropic_api_key),
            )

            plan = build_merge_plan(
                schemas=schemas,
                rec_type=req.rec_system_type,
                target_description=_resolve_target_description(req),
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
