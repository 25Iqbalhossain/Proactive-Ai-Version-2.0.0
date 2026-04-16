"""
Dataset Builder API — main entry point.

Endpoints:
  POST /api/connections/test      — Test a single DB credential
  POST /api/connections           — Register connections for this session
  GET  /api/connections           — List active connections
  DELETE /api/connections/{id}    — Remove a connection
  GET  /api/schema/{conn_id}      — Fetch schema from a connection
  POST /api/build                 — LLM-merges schemas → generates queries → exports dataset
  GET  /api/jobs/{job_id}         — Poll job status + progress
  GET  /api/jobs/{job_id}/download — Download finished CSV / JSON file
  GET  /api/health                — Health check
"""

import logging
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from smart_db_csv_builder.api.routers import connections, schema, build, jobs
from smart_db_csv_builder.core.job_store import job_store

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Dataset Builder API starting …")
    yield
    logger.info("👋 Dataset Builder API stopped")


app = FastAPI(
    title="Dataset Builder API",
    description="Multi-DB credential intake → schema extraction → LLM merge → CSV/JSON export",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(connections.router, prefix="/api/connections", tags=["connections"])
app.include_router(schema.router, prefix="/api/schema", tags=["schema"])
app.include_router(build.router, prefix="/api/build", tags=["build"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "active_jobs": len(job_store.jobs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
