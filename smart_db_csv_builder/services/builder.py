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
import traceback

from smart_db_csv_builder.core.connection_store import connection_store
from smart_db_csv_builder.core.job_store import Job, JobStatus
from smart_db_csv_builder.models.schemas import BuildRequest, SchemaResponse
from smart_db_csv_builder.services.llm_planner import build_merge_plan
from smart_db_csv_builder.services.executor import execute_plan

logger = logging.getLogger(__name__)

STEPS = [
    "validate_connections",
    "extract_schemas",
    "llm_plan",
    "execute_queries",
    "write_output",
]


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
        logger.info(
            "Generating LLM plan with providers: groq=%s openai=%s",
            bool(req.groq_api_key),
            bool(req.openai_api_key or req.anthropic_api_key),
        )

        plan = build_merge_plan(
            schemas=schemas,
            rec_type=req.rec_system_type,
            target_description=req.target_description,
            groq_api_key=req.groq_api_key,
            openai_api_key=req.openai_api_key,
        )

        job.plan = plan.raw_plan
        job.set_step("llm_plan", "done", plan.description or "Plan generated")
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
            if step.status == "running":
                step.status  = "error"
                step.message = str(exc)
