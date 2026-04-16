from __future__ import annotations
import json
import logging
import os
import re
import urllib.request
from typing import Optional

from smart_db_csv_builder.models.schemas import RecSystemType, SchemaResponse

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# =========================
# DATA CLASSES
# =========================
class TableQuery:
    def __init__(self, connection_id, table, columns, where="", alias_map=None):
        self.connection_id = connection_id
        self.table = table
        self.columns = columns
        self.where = where
        self.alias_map = alias_map or {}


class CollectionFetch:
    def __init__(self, connection_id, collection, fields, alias_map=None):
        self.connection_id = connection_id
        self.collection = collection
        self.fields = fields
        self.alias_map = alias_map or {}


class MergePlan:
    def __init__(
        self,
        table_queries,
        collection_fetches,
        merge_keys,
        final_columns,
        description="",
        raw_plan=None,
    ):
        self.table_queries = table_queries
        self.collection_fetches = collection_fetches
        self.merge_keys = merge_keys
        self.final_columns = final_columns
        self.description = description
        self.raw_plan = raw_plan or {}


# =========================
# 🔥 COLUMN VALIDATION
# =========================
def _validate_columns(table_queries, schemas):
    schema_map = {}

    for s in schemas:
        for t in s.tables:
            table_name = t.full_name.split(".")[-1]
            schema_map[table_name] = {c.name for c in t.columns}

    for tq in table_queries:
        table_name = tq.table.split(".")[-1]
        valid_cols = schema_map.get(table_name, set())

        original = tq.columns.copy()
        tq.columns = [c for c in tq.columns if c in valid_cols]

        removed = set(original) - set(tq.columns)
        if removed:
            logger.warning(
                f"Removed invalid columns from {table_name}: {removed}"
            )

    return table_queries


# =========================
# PROMPT
# =========================
def _build_prompt(schemas, rec_type, target_description):
    schema_text = []

    for s in schemas:
        lines = [f"\n### Connection: {s.connection_id} (type: {s.db_type})"]
        for tbl in s.tables:
            cols = "\n".join([f" - {c.name}" for c in tbl.columns])
            lines.append(f"\nTable: {tbl.full_name}\n{cols}")
        schema_text.append("\n".join(lines))

    all_schemas = "\n".join(schema_text)

    return f"""
You are a senior data engineer.

IMPORTANT RULES:
- You MUST ONLY use column names EXACTLY as shown below.
- DO NOT guess or hallucinate column names.
- If unsure, SKIP the column.

{all_schemas}

Return ONLY JSON:

{{
  "description": "",
  "merge_keys": [],
  "final_columns": [],
  "table_queries": [
    {{
      "connection_id": "",
      "table": "",
      "columns": [],
      "alias_map": {{}},
      "where": ""
    }}
  ],
  "collection_fetches": []
}}
"""


# =========================
# HTTP HELPER
# =========================
def _post_json(url, payload, headers, timeout=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# =========================
# GROQ CALL (PRIMARY)
# =========================
def _call_groq(prompt, api_key, model):
    data = _post_json(
        url="https://api.groq.com/openai/v1/chat/completions",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    if "choices" not in data:
        raise RuntimeError(f"Groq invalid response: {data}")

    return data["choices"][0]["message"]["content"]


# =========================
# OPENAI CALL (FALLBACK)
# =========================
def _call_openai(prompt, api_key, model):
    data = _post_json(
        url="https://api.openai.com/v1/chat/completions",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return data["choices"][0]["message"]["content"]


# =========================
# PARSE
# =========================
def _parse_plan(raw_text):
    text = re.sub(r"```(?:json)?", "", raw_text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON returned from LLM")
    return json.loads(match.group(0))


# =========================
# MAIN FUNCTION
# =========================
def build_merge_plan(
    schemas: list[SchemaResponse],
    rec_type: RecSystemType,
    target_description: Optional[str] = None,
    groq_api_key: Optional[str] = "",
    openai_api_key: Optional[str] = "",
    groq_model: str = "llama-3.3-70b-versatile",
    openai_model: str = DEFAULT_OPENAI_MODEL,
) -> MergePlan:

    prompt = _build_prompt(schemas, rec_type, target_description)
    logger.info("Sending schema prompt to LLM")

    raw = None
    errors = []

    
    if groq_api_key:
        try:
            logger.info("Trying Groq...")
            raw = _call_groq(prompt, groq_api_key, groq_model)
            logger.info("Groq success ")
        except Exception as e:
            logger.warning(f"Groq failed : {e}")
            errors.append(f"Groq: {e}")

    # 🔥  FALLBACK OPENAI
    if raw is None and openai_api_key:
        try:
            logger.info("Falling back to OpenAI...")
            raw = _call_openai(prompt, openai_api_key, openai_model)
            logger.info("OpenAI success ")
        except Exception as e:
            logger.error(f"OpenAI failed ❌: {e}")
            errors.append(f"OpenAI: {e}")

    #  BOTH FAILED
    if raw is None:
        raise RuntimeError("All LLM providers failed: " + " | ".join(errors))

    # Parse JSON
    plan_dict = _parse_plan(raw)

    table_queries = [
        TableQuery(
            connection_id=tq["connection_id"],
            table=tq["table"],
            columns=tq.get("columns", []),
            where=tq.get("where", ""),
            alias_map=tq.get("alias_map", {}),
        )
        for tq in plan_dict.get("table_queries", [])
    ]

    #  VALIDATION (critical fix)
    table_queries = _validate_columns(table_queries, schemas)

    return MergePlan(
        table_queries=table_queries,
        collection_fetches=[],
        merge_keys=plan_dict.get("merge_keys", []),
        final_columns=plan_dict.get("final_columns", []),
        description=plan_dict.get("description", ""),
        raw_plan=plan_dict,
    )