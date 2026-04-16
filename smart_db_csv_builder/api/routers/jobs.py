"""api/routers/jobs.py"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from smart_db_csv_builder.models.schemas import JobResponse, JobStatus
from smart_db_csv_builder.core.job_store import job_store

router = APIRouter()


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JobResponse(**job.to_dict())


@router.get("/{job_id}/download")
def download_dataset(job_id: str, output_format: str | None = None):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(400, f"Job is not complete yet (status={job.status})")

    requested_format = (output_format or (job.output_format.value if job.output_format else "csv")).lower()
    if requested_format not in {"csv", "json"}:
        raise HTTPException(400, "output_format must be 'csv' or 'json'")

    path = job.output_files.get(requested_format) if job.output_files else job.output_file
    if not path or not os.path.exists(path):
        raise HTTPException(500, "Output file missing")

    ext = f".{requested_format}"
    return FileResponse(
        path=path,
        media_type="text/csv" if ext == ".csv" else "application/json",
        filename=f"recommendation_dataset_{job_id[:8]}{ext}",
    )
