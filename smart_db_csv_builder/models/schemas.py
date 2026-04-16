"""
models/schemas.py
All Pydantic request/response models for the Dataset Builder API.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────

class DBType(str, Enum):
    MYSQL      = "mysql"
    POSTGRES   = "postgres"
    MSSQL      = "mssql"
    SQLITE     = "sqlite"
    MONGODB    = "mongodb"
    REDIS      = "redis"


class OutputFormat(str, Enum):
    CSV  = "csv"
    JSON = "json"


class JobStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"


class BuildMode(str, Enum):
    QUERY = "query"
    MANUAL = "manual"
    LLM = "llm"


class RecSystemType(str, Enum):
    """Target recommendation system paradigm — influences what columns LLM prioritises."""
    COLLABORATIVE = "collaborative"   # user × item interactions
    CONTENT_BASED = "content_based"   # item features
    HYBRID        = "hybrid"          # both
    SEQUENTIAL    = "sequential"      # session / time-ordered events


# ── Connection ─────────────────────────────────────────────────────────────

class ConnectionCredential(BaseModel):
    """User-supplied credential for one database."""
    db_type:  DBType
    name:     str = Field(..., description="Friendly label shown in UI")

    # SQL / MSSQL / Postgres
    host:     Optional[str] = None
    port:     Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    # SQLite
    filepath: Optional[str] = None

    # MongoDB
    uri:      Optional[str] = None   # full URI (overrides host/port if given)

    # Redis
    db_index: Optional[int] = 0

    # TLS / extra
    ssl:      bool = False
    options:  dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_optional_strings(cls, data):
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        for key in ("host", "database", "filepath", "uri"):
            value = normalized.get(key)
            if isinstance(value, str):
                value = value.strip()
                normalized[key] = value or None
        for key in ("username", "password"):
            if normalized.get(key) == "":
                normalized[key] = None
        return normalized

    @model_validator(mode="after")
    def host_required_for_network_dbs(self):
        if self.db_type in (DBType.MYSQL, DBType.POSTGRES, DBType.MSSQL, DBType.MONGODB):
            if not self.host and not self.uri:
                raise ValueError(f"host is required for {self.db_type}")
        return self


class ConnectionResponse(BaseModel):
    id:       str
    name:     str
    db_type:  DBType
    status:   str       # "connected" | "error"
    message:  str = ""
    database: Optional[str] = None


# ── Schema ─────────────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name:          str
    data_type:     str
    nullable:      bool = True
    is_pk:         bool = False
    is_fk:         bool = False
    fk_ref_table:  Optional[str] = None
    fk_ref_column: Optional[str] = None
    sample_values: list[Any] = Field(default_factory=list)


class TableInfo(BaseModel):
    schema_name: Optional[str] = None
    table_name:  str
    row_count:   int = 0
    columns:     list[ColumnInfo] = Field(default_factory=list)

    @property
    def full_name(self) -> str:
        if self.schema_name:
            return f"{self.schema_name}.{self.table_name}"
        return self.table_name


class FKRelationship(BaseModel):
    from_table:  str
    from_column: str
    to_table:    str
    to_column:   str


class SchemaResponse(BaseModel):
    connection_id:   str
    db_type:         DBType
    tables:          list[TableInfo]
    relationships:   list[FKRelationship] = Field(default_factory=list)
    collections:     list[dict] = Field(default_factory=list)  # MongoDB


# ── Build ───────────────────────────────────────────────────────────────────

class BuildRequest(BaseModel):
    """
    User submits which connections + what kind of rec system they want.
    LLM then decides which tables/fields to merge and how.
    """

    connection_ids: list[str] = Field(..., min_items=1)

    mode: Optional[BuildMode] = Field(
        None,
        description="Build mode. If omitted, preserves the legacy planner behaviour."
    )

    rec_system_type: RecSystemType = RecSystemType.HYBRID

    output_format: OutputFormat = OutputFormat.CSV

    max_rows_per_table: int = Field(50_000, ge=100, le=500_000)

    target_description: Optional[str] = Field(
        None,
        description="Free-text hint, e.g. 'e-commerce product recs based on purchase history'"
    )

    query_text: Optional[str] = Field(
        None,
        description="Free-text query/instruction for query mode."
    )

    llm_prompt: Optional[str] = Field(
        None,
        description="Explicit LLM planner prompt for llm mode."
    )

    manual_config: Optional[dict[str, Optional[str]]] = Field(
        None,
        description="Minimal manual dataset configuration."
    )

    # 🔥 PRIMARY (Groq)
    groq_api_key: Optional[str] = Field(
        None,
        description="Primary LLM (Groq). Used first for generating merge plan."
    )

    # 🔥 FALLBACK (OpenAI)
    openai_api_key: Optional[str] = Field(
        None,
        description="Fallback LLM (OpenAI). Used if Groq fails."
    )

    # Backward compatibility for older frontend payloads.
    anthropic_api_key: Optional[str] = Field(
        None,
        description="Legacy field name. If openai_api_key is missing, this value is reused as the fallback provider key."
    )

    @model_validator(mode="after")
    def populate_openai_fallback(self):
        if not self.openai_api_key and self.anthropic_api_key:
            self.openai_api_key = self.anthropic_api_key
        if not self.target_description:
            if self.mode == BuildMode.QUERY and self.query_text:
                self.target_description = self.query_text
            elif self.mode == BuildMode.LLM and self.llm_prompt:
                self.target_description = self.llm_prompt
        return self

class BuildResponse(BaseModel):
    job_id: str


# ── Job ─────────────────────────────────────────────────────────────────────

class JobStep(BaseModel):
    step:    str
    status:  str   # pending | running | done | error
    message: str = ""


class JobResponse(BaseModel):
    job_id:       str
    status:       JobStatus
    progress:     int = 0          # 0-100
    steps:        list[JobStep] = Field(default_factory=list)
    error:        Optional[str] = None
    output_file:  Optional[str] = None   # filename when done
    output_files: dict[str, str] = Field(default_factory=dict)
    output_format: Optional[OutputFormat] = None
    row_count:    Optional[int] = None
    column_count: Optional[int] = None
    plan:         Optional[dict] = None  # LLM merge plan (for transparency)
