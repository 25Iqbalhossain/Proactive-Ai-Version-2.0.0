
from __future__ import annotations
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional
from smart_db_csv_builder.models.schemas import JobStatus, OutputFormat

@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    steps: list = field(default_factory=list)
    error: Optional[str] = None
    output_file: Optional[str] = None
    output_files: dict[str, str] = field(default_factory=dict)
    output_format: Optional[OutputFormat] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    plan: Optional[dict] = None

    def set_step(self, step: str, status: str, message: str = ""):
        for item in self.steps:
            if item["step"] == step:
                item["status"] = status
                item["message"] = message
                return
        self.steps.append({"step": step, "status": status, "message": message})

    def to_dict(self):
        return {
            'job_id': self.job_id,
            'status': self.status,
            'progress': self.progress,
            'steps': self.steps,
            'error': self.error,
            'output_file': self.output_file,
            'output_files': self.output_files,
            'output_format': self.output_format,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'plan': self.plan,
        }

class JobStore:
    def __init__(self):
        self._lock = threading.RLock()
        self.jobs: dict[str, Job] = {}

    def create(self):
        job = Job(job_id=str(uuid.uuid4()))
        with self._lock:
            self.jobs[job.job_id] = job
        return job

    def get(self, job_id):
        with self._lock:
            return self.jobs.get(job_id)

job_store = JobStore()
