"""api/routers/build.py"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter
from smart_db_csv_builder.models.schemas import BuildRequest, BuildResponse
from smart_db_csv_builder.core.job_store import job_store
from smart_db_csv_builder.services.builder import run_build_job

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=4)


@router.post("", response_model=BuildResponse)
async def start_build(req: BuildRequest):
    """Start a background dataset-build job. Poll /api/jobs/{job_id} for progress."""
    job = job_store.create()

    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, run_build_job, job, req)

    return BuildResponse(job_id=job.job_id)
