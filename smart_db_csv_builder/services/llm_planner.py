from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Optional

from smart_db_csv_builder.models.schemas import RecSystemType, SchemaResponse

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "")
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
WHERE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
QUOTED_STRING_RE = re.compile(r"'(?:''|[^'])*'")
COLUMN_ALIAS_RE = re.compile(r"^(?P<source>.+?)\s+as\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*)$", re.IGNORECASE)
FORBIDDEN_SQL_RE = re.compile(
    r"(;|--|/\*|\*/|\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|merge|exec|execute|call)\b)",
    re.IGNORECASE,
)
ALLOWED_WHERE_KEYWORDS = {
    "and",
    "or",
    "not",
    "in",
    "is",
    "null",
    "like",
    "between",
    "true",
    "false",
}
ALLOWED_WHERE_FUNCTIONS = {
    "abs",
    "cast",
    "coalesce",
    "date",
    "datetime",
    "extract",
    "ifnull",
    "isnull",
    "length",
    "lower",
    "ltrim",
    "nullif",
    "replace",
    "round",
    "rtrim",
    "substr",
    "substring",
    "trim",
    "upper",
}
USER_COLUMN_HINTS = ("user", "customer", "member", "account", "profile", "client", "person", "subscriber", "owner", "employee", "patient")
ITEM_COLUMN_HINTS = ("item", "product", "movie", "book", "content", "article", "sku", "listing", "service", "offer", "asset", "object", "plan", "package")
INTERACTION_COLUMN_HINTS = ("event", "action", "interaction", "rating", "score", "click", "view", "purchase", "cart", "order", "play", "watch", "like")
TIME_COLUMN_HINTS = ("time", "date", "created", "updated", "timestamp", "ts", "ordered", "purchased", "viewed")
METADATA_COLUMN_HINTS = ("title", "name", "category", "brand", "description", "text", "tag", "genre", "price", "type")
RATING_COLUMN_HINTS = ("rating", "score", "stars", "grade", "review", "preference")
INTERACTION_TABLE_HINTS = ("event", "interaction", "rating", "purchase", "click", "view", "order", "watch", "play")
GENERIC_KEY_HINTS = ("id", "key", "uuid", "guid", "code", "sku", "session", "order")


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


def _clean_identifier(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _strip_identifier_quotes(value: str) -> str:
    value = (value or "").strip()
    if value.startswith("[") and value.endswith("]") and len(value) > 2:
        return value[1:-1].strip()
    return value.strip("`\" ")


def _parse_column_reference(value) -> tuple[str, str]:
    raw = _clean_identifier(value)
    if not raw:
        return "", ""

    alias = ""
    match = COLUMN_ALIAS_RE.match(raw)
    if match:
        raw = match.group("source").strip()
        alias = _strip_identifier_quotes(match.group("alias"))

    source = _strip_identifier_quotes(raw)
    if "." in source:
        source = _strip_identifier_quotes(source.split(".")[-1])

    if not SAFE_IDENTIFIER_RE.match(source):
        return "", alias

    return source, alias


def _build_schema_indexes(schemas: list[SchemaResponse]):
    exact_tables: dict[str, dict[str, object]] = {}
    short_tables: dict[str, dict[str, list[object]]] = {}
    column_lookup: dict[tuple[str, str], dict[str, str]] = {}

    for schema in schemas:
        exact_tables.setdefault(schema.connection_id, {})
        short_tables.setdefault(schema.connection_id, {})

        for table in schema.tables:
            exact_tables[schema.connection_id][table.full_name.lower()] = table
            short_tables[schema.connection_id].setdefault(table.table_name.lower(), []).append(table)
            column_lookup[(schema.connection_id, table.full_name.lower())] = {
                column.name.lower(): column.name for column in table.columns
            }

    return exact_tables, short_tables, column_lookup


def _resolve_table(connection_id: str, table_name: str, exact_tables, short_tables):
    cleaned_name = _clean_identifier(table_name)
    if not cleaned_name:
        return None

    exact_match = exact_tables.get(connection_id, {}).get(cleaned_name.lower())
    if exact_match is not None:
        return exact_match

    short_matches = short_tables.get(connection_id, {}).get(cleaned_name.lower(), [])
    if len(short_matches) == 1:
        return short_matches[0]

    if len(short_matches) > 1:
        logger.warning(
            "Skipping ambiguous table reference '%s' on connection '%s'",
            cleaned_name,
            connection_id,
        )
    else:
        logger.warning(
            "Skipping unknown table reference '%s' on connection '%s'",
            cleaned_name,
            connection_id,
        )
    return None


def _sanitize_where_clause(
    where: str,
    valid_columns: dict[str, str],
    allowed_identifiers: set[str] | None = None,
) -> str:
    where = (where or "").strip()
    if not where:
        return ""
    if len(where) > 500 or FORBIDDEN_SQL_RE.search(where):
        logger.warning("Dropping unsafe WHERE clause: %s", where)
        return ""

    allowed_identifiers = {identifier.lower() for identifier in (allowed_identifiers or set())}
    scrubbed = QUOTED_STRING_RE.sub("''", where)
    for token in WHERE_IDENTIFIER_RE.findall(scrubbed):
        lowered = token.lower()
        if (
            lowered in ALLOWED_WHERE_KEYWORDS
            or lowered in ALLOWED_WHERE_FUNCTIONS
            or lowered in valid_columns
            or lowered in allowed_identifiers
        ):
            continue
        logger.warning("Dropping WHERE clause with unknown identifier '%s': %s", token, where)
        return ""

    return where


def _sanitize_alias_map(alias_map: dict, selected_columns: list[str]) -> dict[str, str]:
    if not isinstance(alias_map, dict):
        return {}

    selected_set = set(selected_columns)
    cleaned: dict[str, str] = {}
    used_targets: set[str] = set()

    for source, target in alias_map.items():
        source_name = _clean_identifier(source)
        target_name = _clean_identifier(target)

        if source_name not in selected_set:
            continue
        if not SAFE_IDENTIFIER_RE.match(target_name):
            logger.warning("Dropping unsafe alias target '%s' for column '%s'", target_name, source_name)
            continue
        if target_name in used_targets and cleaned.get(source_name) != target_name:
            logger.warning("Dropping duplicate alias target '%s'", target_name)
            continue

        cleaned[source_name] = target_name
        used_targets.add(target_name)

    return cleaned


def _projected_columns(selected_columns: list[str], alias_map: dict[str, str]) -> list[str]:
    projected: list[str] = []
    seen: set[str] = set()

    for column in selected_columns:
        resolved = alias_map.get(column, column)
        if resolved not in seen:
            projected.append(resolved)
            seen.add(resolved)

    return projected


def _auto_alias_role_columns(selected_columns: list[str]) -> dict[str, str]:
    role_specs = (
        ("userID", USER_COLUMN_HINTS),
        ("itemID", ITEM_COLUMN_HINTS),
        ("rating", RATING_COLUMN_HINTS),
        ("timestamp", TIME_COLUMN_HINTS),
    )
    alias_map: dict[str, str] = {}
    used_targets: set[str] = set()

    for target_name, hints in role_specs:
        candidates = [column for column in selected_columns if _column_matches(column, hints)]
        if not candidates or target_name in used_targets:
            continue
        best = max(candidates, key=lambda column_name: _column_preference_score(column_name, hints))
        if best != target_name:
            alias_map[best] = target_name
            used_targets.add(target_name)

    return alias_map


def _allowed_table_identifiers(table) -> set[str]:
    identifiers: set[str] = set()
    for value in (
        getattr(table, "full_name", "") or "",
        getattr(table, "table_name", "") or "",
        getattr(table, "schema_name", "") or "",
    ):
        lowered = value.lower()
        if not lowered:
            continue
        identifiers.add(lowered)
        identifiers.update(part for part in re.split(r"[^a-z0-9_]+", lowered) if part)
    return identifiers


def _merge_key_priority(column_name: str, rec_type: RecSystemType) -> tuple[int, int, int]:
    tokens = _identifier_tokens(column_name)
    lowered = (column_name or "").lower()
    priority = 0

    if _column_matches(column_name, USER_COLUMN_HINTS):
        priority += 300
    if _column_matches(column_name, ITEM_COLUMN_HINTS):
        priority += 280
    if rec_type == RecSystemType.SEQUENTIAL and "session" in tokens:
        priority += 220
    if any(hint in tokens for hint in GENERIC_KEY_HINTS) or lowered.endswith("_id"):
        priority += 160
    if "id" == lowered:
        priority -= 20

    return priority, len(tokens), -len(lowered)


def _is_likely_merge_key(column_name: str, rec_type: RecSystemType) -> bool:
    return _merge_key_priority(column_name, rec_type)[0] > 0


def _can_assign_alias(query: dict, source_column: str, target_column: str) -> bool:
    current_outputs = _projected_columns(query["columns"], query.get("alias_map", {}))
    current_output = query.get("alias_map", {}).get(source_column, source_column)
    return target_column == current_output or target_column not in current_outputs


def _preferred_join_output_name(left_name: str, right_name: str, rec_type: RecSystemType) -> str:
    return max((left_name, right_name), key=lambda name: _merge_key_priority(name, rec_type))


def _apply_relationship_merge_hints(
    schemas: list[SchemaResponse],
    sanitized_queries: list[dict],
    rec_type: RecSystemType,
) -> list[str]:
    query_lookup = {
        (query["connection_id"], query["table"].lower()): query
        for query in sanitized_queries
    }
    inferred_keys: list[str] = []
    seen_keys: set[str] = set()

    for schema in schemas:
        for rel in schema.relationships:
            left_query = query_lookup.get((schema.connection_id, rel.from_table.lower()))
            right_query = query_lookup.get((schema.connection_id, rel.to_table.lower()))

            if not left_query or not right_query:
                continue
            if rel.from_column not in left_query["columns"] or rel.to_column not in right_query["columns"]:
                continue

            left_output = left_query["alias_map"].get(rel.from_column, rel.from_column)
            right_output = right_query["alias_map"].get(rel.to_column, rel.to_column)
            preferred = _preferred_join_output_name(left_output, right_output, rec_type)

            if left_output != preferred and _can_assign_alias(left_query, rel.from_column, preferred):
                left_query["alias_map"][rel.from_column] = preferred
                left_output = preferred
            if right_output != preferred and _can_assign_alias(right_query, rel.to_column, preferred):
                right_query["alias_map"][rel.to_column] = preferred
                right_output = preferred

            if left_output == right_output and _is_likely_merge_key(left_output, rec_type):
                lowered = left_output.lower()
                if lowered not in seen_keys:
                    inferred_keys.append(left_output)
                    seen_keys.add(lowered)

    return inferred_keys


def _infer_merge_keys(projected_columns_per_query: list[list[str]], rec_type: RecSystemType) -> list[str]:
    projected_counter: Counter[str] = Counter()
    canonical_names: dict[str, str] = {}

    for projected_columns in projected_columns_per_query:
        for column in set(projected_columns):
            lowered = column.lower()
            projected_counter[lowered] += 1
            canonical_names.setdefault(lowered, column)

    ranked = []
    for lowered, count in projected_counter.items():
        canonical = canonical_names[lowered]
        if count < 2 or not _is_likely_merge_key(canonical, rec_type):
            continue
        ranked.append((_merge_key_priority(canonical, rec_type), count, canonical))

    ranked.sort(reverse=True)
    return [canonical for _, _, canonical in ranked]


def _final_column_priority(column_name: str, merge_keys: list[str], rec_type: RecSystemType) -> tuple[int, int]:
    lowered = (column_name or "").lower()
    if lowered in {key.lower() for key in merge_keys}:
        return 0, 0
    if _column_matches(column_name, USER_COLUMN_HINTS):
        return 1, 0
    if _column_matches(column_name, ITEM_COLUMN_HINTS):
        return 2, 0
    if _column_matches(column_name, RATING_COLUMN_HINTS):
        return 3, 0
    if _column_matches(column_name, INTERACTION_COLUMN_HINTS):
        return 4, 0
    if _column_matches(column_name, TIME_COLUMN_HINTS):
        return 5 if rec_type == RecSystemType.SEQUENTIAL else 6, 0
    if _column_matches(column_name, METADATA_COLUMN_HINTS):
        return 5 if rec_type == RecSystemType.CONTENT_BASED else 7, 0
    return 8, 0


def _infer_final_columns(
    projected_columns_per_query: list[list[str]],
    merge_keys: list[str],
    rec_type: RecSystemType,
) -> list[str]:
    first_seen: dict[str, tuple[int, str]] = {}

    for query_index, projected_columns in enumerate(projected_columns_per_query):
        for column in projected_columns:
            lowered = column.lower()
            first_seen.setdefault(lowered, (query_index, column))

    ordered = sorted(
        first_seen.items(),
        key=lambda item: (
            _final_column_priority(item[1][1], merge_keys, rec_type)[0],
            item[1][0],
            item[1][1].lower(),
        ),
    )
    return [column for _, (_, column) in ordered]


def _final_columns_are_usable(
    final_columns: list[str],
    merge_keys: list[str],
    rec_type: RecSystemType,
) -> bool:
    lowered = {column.lower() for column in final_columns}
    if any(key.lower() not in lowered for key in merge_keys):
        return False

    has_user = any(_column_matches(column, USER_COLUMN_HINTS) for column in final_columns)
    has_item = any(_column_matches(column, ITEM_COLUMN_HINTS) for column in final_columns)
    has_metadata = any(_column_matches(column, METADATA_COLUMN_HINTS) for column in final_columns)

    if rec_type in (RecSystemType.COLLABORATIVE, RecSystemType.HYBRID, RecSystemType.SEQUENTIAL):
        return has_user and has_item
    if rec_type == RecSystemType.CONTENT_BASED:
        return has_item and has_metadata
    return bool(final_columns)


def _sanitize_plan(plan_dict: dict, schemas: list[SchemaResponse], rec_type: RecSystemType) -> dict:
    exact_tables, short_tables, column_lookup = _build_schema_indexes(schemas)

    sanitized_queries = []
    projected_columns_per_query: list[list[str]] = []

    for raw_query in plan_dict.get("table_queries", []):
        connection_id = _clean_identifier(raw_query.get("connection_id"))
        if not connection_id:
            logger.warning("Skipping table query without connection_id: %s", raw_query)
            continue

        table = _resolve_table(connection_id, raw_query.get("table", ""), exact_tables, short_tables)
        if table is None:
            continue

        valid_column_lookup = column_lookup[(connection_id, table.full_name.lower())]
        selected_columns: list[str] = []
        seen_columns: set[str] = set()
        implicit_alias_map: dict[str, str] = {}
        for column in raw_query.get("columns", []):
            source_column, implicit_alias = _parse_column_reference(column)
            lookup_key = source_column.lower() if source_column else _clean_identifier(column).lower()
            canonical = valid_column_lookup.get(lookup_key)
            if canonical and canonical not in seen_columns:
                selected_columns.append(canonical)
                seen_columns.add(canonical)
                if implicit_alias and SAFE_IDENTIFIER_RE.match(implicit_alias):
                    implicit_alias_map.setdefault(canonical, implicit_alias)

        if not selected_columns:
            logger.warning(
                "Skipping table query for %s on %s because no valid columns remained after validation",
                table.full_name,
                connection_id,
            )
            continue

        alias_map = _auto_alias_role_columns(selected_columns)
        alias_map.update(implicit_alias_map)
        alias_map.update(_sanitize_alias_map(raw_query.get("alias_map", {}), selected_columns))
        where = _sanitize_where_clause(
            raw_query.get("where", ""),
            valid_column_lookup,
            allowed_identifiers=_allowed_table_identifiers(table),
        )

        sanitized_queries.append(
            {
                "connection_id": connection_id,
                "table": table.full_name,
                "columns": selected_columns,
                "alias_map": alias_map,
                "where": where,
            }
        )
        projected_columns_per_query.append(_projected_columns(selected_columns, alias_map))

    if not sanitized_queries:
        raise ValueError("LLM plan did not contain any valid table queries.")

    inferred_relationship_keys = _apply_relationship_merge_hints(schemas, sanitized_queries, rec_type)
    projected_columns_per_query = [
        _projected_columns(query["columns"], query["alias_map"])
        for query in sanitized_queries
    ]

    _validate_recommendation_shape(projected_columns_per_query, rec_type)

    projected_counter: Counter[str] = Counter()
    canonical_output_names: dict[str, str] = {}
    for projected_columns in projected_columns_per_query:
        for column in set(projected_columns):
            lowered = column.lower()
            projected_counter[lowered] += 1
            canonical_output_names.setdefault(lowered, column)

    merge_keys: list[str] = []
    seen_merge_keys: set[str] = set()
    for key in plan_dict.get("merge_keys", []):
        lowered = _clean_identifier(key).lower()
        if not lowered or lowered in seen_merge_keys:
            continue
        if projected_counter[lowered] >= 2:
            merge_keys.append(canonical_output_names[lowered])
            seen_merge_keys.add(lowered)
        else:
            logger.warning("Dropping invalid merge key '%s'", key)

    for key in inferred_relationship_keys:
        lowered = key.lower()
        if lowered not in seen_merge_keys and projected_counter[lowered] >= 2:
            merge_keys.append(canonical_output_names.get(lowered, key))
            seen_merge_keys.add(lowered)

    if len(sanitized_queries) > 1 and not merge_keys:
        for key in _infer_merge_keys(projected_columns_per_query, rec_type):
            lowered = key.lower()
            if lowered not in seen_merge_keys:
                merge_keys.append(canonical_output_names.get(lowered, key))
                seen_merge_keys.add(lowered)

    if len(sanitized_queries) > 1 and not merge_keys:
        raise ValueError("LLM plan did not produce any valid merge keys for the selected tables.")

    final_columns: list[str] = []
    seen_final_columns: set[str] = set()
    for column in plan_dict.get("final_columns", []):
        lowered = _clean_identifier(column).lower()
        if lowered and lowered in canonical_output_names and lowered not in seen_final_columns:
            final_columns.append(canonical_output_names[lowered])
            seen_final_columns.add(lowered)
        elif lowered:
            logger.warning("Dropping invalid final column '%s'", column)

    if not _final_columns_are_usable(final_columns, merge_keys, rec_type):
        final_columns = _infer_final_columns(projected_columns_per_query, merge_keys, rec_type)

    return {
        "description": (plan_dict.get("description") or "").strip(),
        "merge_keys": merge_keys,
        "final_columns": final_columns,
        "table_queries": sanitized_queries,
        "collection_fetches": [],
    }


def _identifier_tokens(value: str) -> set[str]:
    cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value or "")
    return {token for token in re.split(r"[^a-z0-9]+", cleaned.lower()) if token}


def _column_matches(column_name: str, hints: tuple[str, ...]) -> bool:
    tokens = _identifier_tokens(column_name)
    if any(hint in tokens for hint in hints):
        return True
    lowered = (column_name or "").lower()
    return any(hint in lowered for hint in hints)


def _column_preference_score(column_name: str, hints: tuple[str, ...]) -> tuple[int, int]:
    lowered = (column_name or "").lower()
    tokens = _identifier_tokens(column_name)
    score = 0

    for weight, hint in enumerate(hints[::-1], start=1):
        if lowered == hint:
            score += 200 + weight
        if hint in tokens:
            score += 120 + weight
        if lowered.endswith(f"_{hint}") or lowered.startswith(f"{hint}_"):
            score += 80 + weight
        if hint in lowered:
            score += 40 + weight

    if "id" in tokens:
        score += 5

    return score, -len(lowered)


def _pick_best_matching_column(table, hints: tuple[str, ...]) -> str | None:
    candidates = [column.name for column in table.columns if _column_matches(column.name, hints)]
    if not candidates:
        return None
    return max(candidates, key=lambda name: _column_preference_score(name, hints))


def _pick_metadata_columns(table, limit: int = 4) -> list[str]:
    ranked: list[tuple[tuple[int, int], str]] = []

    for column in table.columns:
        name = column.name
        if not _column_matches(name, METADATA_COLUMN_HINTS):
            continue
        ranked.append((_column_preference_score(name, METADATA_COLUMN_HINTS), name))

    ranked.sort(reverse=True)
    ordered: list[str] = []
    seen: set[str] = set()

    for _, name in ranked:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
        if len(ordered) >= limit:
            break

    return ordered


def _describe_table_roles(table) -> dict[str, object]:
    user_col = _pick_best_matching_column(table, USER_COLUMN_HINTS)
    item_col = _pick_best_matching_column(table, ITEM_COLUMN_HINTS)
    rating_col = _pick_best_matching_column(table, RATING_COLUMN_HINTS)
    time_col = _pick_best_matching_column(table, TIME_COLUMN_HINTS)
    metadata_cols = _pick_metadata_columns(table)
    interaction_cols = [
        column.name
        for column in table.columns
        if _column_matches(column.name, INTERACTION_COLUMN_HINTS) and column.name not in {rating_col, time_col}
    ]
    return {
        "user_col": user_col,
        "item_col": item_col,
        "rating_col": rating_col,
        "time_col": time_col,
        "metadata_cols": metadata_cols,
        "interaction_cols": interaction_cols,
    }


def _relationship_details_for_table(schema: SchemaResponse, table) -> list[tuple[str, object | None]]:
    table_lookup = {candidate.full_name.lower(): candidate for candidate in schema.tables}
    table_name = table.full_name.lower()
    details: list[tuple[str, object | None]] = []
    seen: set[tuple[str, str]] = set()

    for rel in schema.relationships:
        local_column = None
        other_table_name = None
        if rel.from_table.lower() == table_name:
            local_column = rel.from_column
            other_table_name = rel.to_table.lower()
        elif rel.to_table.lower() == table_name:
            local_column = rel.to_column
            other_table_name = rel.from_table.lower()

        if not local_column:
            continue

        key = (local_column.lower(), other_table_name or "")
        if key in seen:
            continue
        seen.add(key)
        details.append((local_column, table_lookup.get(other_table_name or "")))

    return details


def _relationship_count(schema: SchemaResponse | None, table) -> int:
    if schema is None:
        return 0
    return len(_relationship_details_for_table(schema, table))


def _relationship_role_score(local_column: str, other_table, role: str) -> tuple[float, int]:
    hints = USER_COLUMN_HINTS if role == "user" else ITEM_COLUMN_HINTS
    score = 0.0
    if _column_matches(local_column, hints):
        score += 220.0

    if other_table is not None:
        other_roles = _describe_table_roles(other_table)
        if _column_matches(other_table.full_name, hints) or _column_matches(other_table.table_name, hints):
            score += 260.0
        if role == "user" and other_roles["user_col"]:
            score += 140.0
        if role == "item" and other_roles["item_col"]:
            score += 140.0
        if role == "item":
            score += min(len(other_roles["metadata_cols"]), 4) * 24.0

    if local_column.lower().endswith("_id"):
        score += 25.0

    return score, -len(local_column)


def _infer_entity_columns_from_relationships(
    schema: SchemaResponse,
    table,
    rec_type: RecSystemType,
) -> dict[str, str | None]:
    details = _relationship_details_for_table(schema, table)
    unique_columns = []
    seen_columns: set[str] = set()
    for local_column, _ in details:
        lowered = local_column.lower()
        if lowered not in seen_columns:
            unique_columns.append(local_column)
            seen_columns.add(lowered)

    inferred = {"user_col": None, "item_col": None}
    if not unique_columns:
        return inferred

    role_rankings: dict[str, list[tuple[tuple[float, int], str]]] = {"user": [], "item": []}
    for local_column, other_table in details:
        for role in ("user", "item"):
            role_rankings[role].append((_relationship_role_score(local_column, other_table, role), local_column))

    for role in ("user", "item"):
        role_rankings[role].sort(reverse=True)

    for role_name, output_key in (("user", "user_col"), ("item", "item_col")):
        for (score, _), local_column in role_rankings[role_name]:
            if score <= 0:
                continue
            if local_column not in inferred.values():
                inferred[output_key] = local_column
                break

    if rec_type in (RecSystemType.COLLABORATIVE, RecSystemType.HYBRID, RecSystemType.SEQUENTIAL):
        remaining = [column for column in unique_columns if column not in inferred.values()]
        if inferred["user_col"] is None and remaining:
            inferred["user_col"] = remaining[0]
            remaining = [column for column in remaining if column != inferred["user_col"]]
        if inferred["item_col"] is None and remaining:
            inferred["item_col"] = remaining[0]

    return inferred


def _describe_table_roles_with_schema(
    schema: SchemaResponse,
    table,
    rec_type: RecSystemType,
) -> dict[str, object]:
    roles = _describe_table_roles(table)
    if rec_type in (RecSystemType.COLLABORATIVE, RecSystemType.HYBRID, RecSystemType.SEQUENTIAL):
        inferred = _infer_entity_columns_from_relationships(schema, table, rec_type)
        if not roles["user_col"]:
            roles["user_col"] = inferred["user_col"]
        if not roles["item_col"]:
            roles["item_col"] = inferred["item_col"]
    return roles


def _collect_table_features(table) -> dict[str, list[str]]:
    features = {
        "user": [],
        "item": [],
        "interaction": [],
        "time": [],
        "metadata": [],
    }

    for column in table.columns:
        name = column.name
        if _column_matches(name, USER_COLUMN_HINTS):
            features["user"].append(name)
        if _column_matches(name, ITEM_COLUMN_HINTS):
            features["item"].append(name)
        if _column_matches(name, INTERACTION_COLUMN_HINTS):
            features["interaction"].append(name)
        if _column_matches(name, TIME_COLUMN_HINTS):
            features["time"].append(name)
        if _column_matches(name, METADATA_COLUMN_HINTS):
            features["metadata"].append(name)

    return features


def _query_role_summary(columns: list[str]) -> dict[str, bool]:
    return {
        "has_user": any(_column_matches(column, USER_COLUMN_HINTS) for column in columns),
        "has_item": any(_column_matches(column, ITEM_COLUMN_HINTS) for column in columns),
        "has_rating": any(_column_matches(column, RATING_COLUMN_HINTS) for column in columns),
        "has_time": any(_column_matches(column, TIME_COLUMN_HINTS) for column in columns),
        "has_metadata": any(_column_matches(column, METADATA_COLUMN_HINTS) for column in columns),
    }


def _validate_recommendation_shape(
    projected_columns_per_query: list[list[str]],
    rec_type: RecSystemType,
) -> None:
    if not projected_columns_per_query:
        raise ValueError("LLM plan did not contain any validated projected columns.")

    summaries = [_query_role_summary(columns) for columns in projected_columns_per_query]

    if rec_type in (RecSystemType.COLLABORATIVE, RecSystemType.HYBRID, RecSystemType.SEQUENTIAL):
        if not any(summary["has_user"] and summary["has_item"] for summary in summaries):
            raise ValueError(
                "LLM plan did not include a usable interaction table with both user and item identifiers."
            )

    if rec_type == RecSystemType.CONTENT_BASED:
        if not any(summary["has_item"] and summary["has_metadata"] for summary in summaries):
            raise ValueError(
                "LLM plan did not include a usable item-content table with both item identifiers and metadata."
            )


def _score_table_for_rec_type(
    table,
    rec_type: RecSystemType,
    target_description: str | None,
    schema: SchemaResponse | None = None,
) -> float:
    features = _collect_table_features(table)
    tokens = _identifier_tokens(f"{table.full_name} {target_description or ''}")
    score = 0.0
    relationship_count = _relationship_count(schema, table)

    if rec_type in (RecSystemType.COLLABORATIVE, RecSystemType.HYBRID, RecSystemType.SEQUENTIAL):
        if features["user"]:
            score += 6
        if features["item"]:
            score += 6
        if features["user"] and features["item"]:
            score += 18
        if relationship_count >= 2:
            score += 18
        elif relationship_count == 1:
            score += 6
        score += min(len(features["interaction"]), 3) * 4
        score += min(len(features["time"]), 2) * 2
        if rec_type == RecSystemType.HYBRID:
            score += min(len(features["metadata"]), 3) * 1.5
        if rec_type == RecSystemType.SEQUENTIAL and features["time"]:
            score += 8
    else:
        if features["item"]:
            score += 10
        if features["metadata"]:
            score += min(len(features["metadata"]), 5) * 4
        if features["item"] and features["metadata"]:
            score += 12

    if any(token in tokens for token in ("event", "interaction", "rating", "purchase", "click")):
        score += 2
    if any(token in tokens for token in ("item", "product", "movie", "content")):
        score += 2
    if getattr(table, "row_count", 0):
        score += min(float(table.row_count), 100000.0) / 25000.0

    return score


def _pick_columns_for_table(table, rec_type: RecSystemType, required: list[str] | None = None) -> list[str]:
    features = _collect_table_features(table)
    ordered: list[str] = []
    seen: set[str] = set()

    def add(column_name: str):
        if column_name and column_name not in seen:
            ordered.append(column_name)
            seen.add(column_name)

    for column_name in required or []:
        add(column_name)

    for bucket in ("user", "item", "interaction", "time", "metadata"):
        for column_name in features[bucket]:
            add(column_name)

    for column in table.columns:
        add(column.name)
        if len(ordered) >= (8 if rec_type != RecSystemType.CONTENT_BASED else 10):
            break

    return ordered


def _relationship_candidates(schemas: list[SchemaResponse], primary_connection_id: str, primary_table_name: str):
    for schema in schemas:
        if schema.connection_id != primary_connection_id:
            continue
        for rel in schema.relationships:
            if rel.from_table.lower() == primary_table_name.lower():
                yield ("out", rel)
            elif rel.to_table.lower() == primary_table_name.lower():
                yield ("in", rel)


def _find_join_pair(
    schema: SchemaResponse,
    primary_table,
    other_table,
    role: str,
    rec_type: RecSystemType,
) -> tuple[str, str] | None:
    primary_name = primary_table.full_name.lower()
    other_name = other_table.full_name.lower()

    for rel in schema.relationships:
        from_table = rel.from_table.lower()
        to_table = rel.to_table.lower()

        if from_table == primary_name and to_table == other_name:
            return rel.from_column, rel.to_column
        if from_table == other_name and to_table == primary_name:
            return rel.to_column, rel.from_column

    primary_roles = _describe_table_roles_with_schema(schema, primary_table, rec_type)
    other_roles = _describe_table_roles_with_schema(schema, other_table, rec_type)
    primary_key = primary_roles["user_col"] if role == "user" else primary_roles["item_col"]
    other_key = other_roles["user_col"] if role == "user" else other_roles["item_col"]

    if primary_key and other_key:
        if primary_key.lower() == other_key.lower():
            return primary_key, other_key

    primary_columns = {column.name.lower(): column.name for column in primary_table.columns}
    other_columns = {column.name.lower(): column.name for column in other_table.columns}

    for lowered, original in primary_columns.items():
        if lowered in other_columns:
            if role == "user" and _column_matches(original, USER_COLUMN_HINTS):
                return original, other_columns[lowered]
            if role == "item" and _column_matches(original, ITEM_COLUMN_HINTS):
                return original, other_columns[lowered]

    return None


def _rank_primary_table(
    schema: SchemaResponse,
    table,
    rec_type: RecSystemType,
    target_description: str | None,
) -> tuple[float, float]:
    score = _score_table_for_rec_type(table, rec_type, target_description, schema=schema)
    roles = _describe_table_roles_with_schema(schema, table, rec_type)
    bonus = 0.0

    if rec_type in (RecSystemType.COLLABORATIVE, RecSystemType.HYBRID, RecSystemType.SEQUENTIAL):
        if roles["user_col"] and roles["item_col"]:
            bonus += 50.0
        if roles["rating_col"]:
            bonus += 6.0
        if roles["time_col"]:
            bonus += 4.0
        if any(token in _identifier_tokens(table.full_name) for token in INTERACTION_TABLE_HINTS):
            bonus += 10.0
    elif roles["item_col"] and roles["metadata_cols"]:
        bonus += 25.0

    if _relationship_count(schema, table) >= 2:
        bonus += 12.0

    return score + bonus, score


def _build_query_columns(
    schema: SchemaResponse,
    table,
    rec_type: RecSystemType,
    required: list[str] | None = None,
) -> list[str]:
    roles = _describe_table_roles_with_schema(schema, table, rec_type)
    required_columns = list(required or [])

    for column_name in (roles["user_col"], roles["item_col"], roles["rating_col"], roles["time_col"]):
        if column_name and column_name not in required_columns:
            required_columns.append(column_name)

    for column_name in roles["interaction_cols"][:2]:
        if column_name and column_name not in required_columns:
            required_columns.append(column_name)

    for column_name in roles["metadata_cols"][:4]:
        if column_name and column_name not in required_columns:
            required_columns.append(column_name)

    return _pick_columns_for_table(table, rec_type, required_columns)


def _build_fallback_plan(
    schemas: list[SchemaResponse],
    rec_type: RecSystemType,
    target_description: str | None,
    reason: str,
    raw_plan: dict | None = None,
    raw_text: str | None = None,
) -> dict:
    table_pool: list[tuple[SchemaResponse, object, float]] = []
    table_map: dict[tuple[str, str], tuple[SchemaResponse, object]] = {}

    for schema in schemas:
        for table in schema.tables:
            score = _score_table_for_rec_type(table, rec_type, target_description, schema=schema)
            table_pool.append((schema, table, score))
            table_map[(schema.connection_id, table.full_name.lower())] = (schema, table)

    if not table_pool:
        raise ValueError("No schema tables were available to build a fallback plan.")

    table_pool.sort(
        key=lambda item: _rank_primary_table(item[0], item[1], rec_type, target_description),
        reverse=True,
    )
    primary_schema, primary_table, _ = table_pool[0]
    primary_roles = _describe_table_roles_with_schema(primary_schema, primary_table, rec_type)
    primary_columns = _build_query_columns(primary_schema, primary_table, rec_type)
    primary_alias_map = _auto_alias_role_columns(primary_columns)
    if primary_roles["user_col"] and primary_roles["user_col"] in primary_columns:
        primary_alias_map[primary_roles["user_col"]] = "userID"
    if primary_roles["item_col"] and primary_roles["item_col"] in primary_columns:
        primary_alias_map[primary_roles["item_col"]] = "itemID"
    if primary_roles["rating_col"] and primary_roles["rating_col"] in primary_columns:
        primary_alias_map.setdefault(primary_roles["rating_col"], "rating")
    if primary_roles["time_col"] and primary_roles["time_col"] in primary_columns:
        primary_alias_map.setdefault(primary_roles["time_col"], "timestamp")
    selected_queries = [
        {
            "connection_id": primary_schema.connection_id,
            "table": primary_table.full_name,
            "columns": primary_columns,
            "alias_map": primary_alias_map,
            "where": "",
        }
    ]
    merge_keys: list[str] = []
    selected_tables = {primary_table.full_name.lower()}

    def add_related_table(role: str):
        primary_key = primary_roles["user_col"] if role == "user" else primary_roles["item_col"]
        if not primary_key:
            return

        related_options = []
        for schema, other_table, _ in table_pool:
            if schema.connection_id != primary_schema.connection_id:
                continue
            if other_table.full_name.lower() in selected_tables:
                continue

            other_roles = _describe_table_roles_with_schema(primary_schema, other_table, rec_type)
            join_pair = _find_join_pair(primary_schema, primary_table, other_table, role, rec_type)
            if not join_pair:
                continue

            primary_column, other_column = join_pair
            metadata_bonus = len(other_roles["metadata_cols"]) * (3.0 if role == "item" else 0.25)
            dimension_bonus = 4.0 if role == "user" else 8.0
            explicit_key_bonus = 12.0 if primary_column != other_column else 6.0
            role_alignment_bonus = _relationship_role_score(primary_column, other_table, role)[0]
            if role == "user" and other_roles["metadata_cols"]:
                role_alignment_bonus -= len(other_roles["metadata_cols"]) * 12.0
            if role == "item" and other_roles["user_col"] and not other_roles["metadata_cols"]:
                role_alignment_bonus -= 10.0
            related_score = _score_table_for_rec_type(other_table, rec_type, target_description, schema=schema)
            related_options.append(
                (
                    related_score + metadata_bonus + dimension_bonus + explicit_key_bonus + role_alignment_bonus,
                    other_table,
                    primary_column,
                    other_column,
                    other_roles,
                )
            )

        related_options.sort(key=lambda item: item[0], reverse=True)
        if not related_options:
            return

        _, other_table, primary_column, other_column, other_roles = related_options[0]
        required_columns = [other_column]
        for column_name in other_roles["metadata_cols"][:4]:
            if column_name not in required_columns:
                required_columns.append(column_name)
        other_columns = _build_query_columns(primary_schema, other_table, rec_type, required_columns)
        alias_map = _auto_alias_role_columns(other_columns)
        canonical_key = "userID" if role == "user" else "itemID"
        alias_map[other_column] = canonical_key
        if other_column != primary_column:
            selected_queries[0]["alias_map"][primary_column] = canonical_key
        selected_queries.append(
            {
                "connection_id": primary_schema.connection_id,
                "table": other_table.full_name,
                "columns": other_columns,
                "alias_map": alias_map,
                "where": "",
            }
        )
        selected_tables.add(other_table.full_name.lower())
        if canonical_key not in merge_keys:
            merge_keys.append(canonical_key)

    if rec_type in (RecSystemType.COLLABORATIVE, RecSystemType.HYBRID, RecSystemType.SEQUENTIAL):
        add_related_table("user")
        add_related_table("item")
    elif rec_type == RecSystemType.CONTENT_BASED:
        add_related_table("item")

    final_columns = []
    for query in selected_queries:
        for column in query["columns"]:
            output_name = query["alias_map"].get(column, column)
            if output_name not in final_columns:
                final_columns.append(output_name)

    description = (
        f"Fallback schema-driven plan generated because the LLM plan was unusable: {reason}"
    )
    fallback_plan = {
        "description": description,
        "merge_keys": merge_keys,
        "final_columns": final_columns,
        "table_queries": selected_queries,
        "collection_fetches": [],
        "_fallback": True,
        "_fallback_reason": reason,
    }
    if raw_plan is not None:
        fallback_plan["_llm_raw_plan"] = raw_plan
    if raw_text:
        fallback_plan["_llm_raw_text_excerpt"] = raw_text[:2000]

    logger.warning("%s", description)
    return fallback_plan


def _build_prompt(schemas, rec_type, target_description):
    schema_text = []
    relationship_text = []

    rec_guidance = {
        RecSystemType.COLLABORATIVE: "Prioritize user identifiers, item identifiers, interaction events, ratings, and timestamps.",
        RecSystemType.CONTENT_BASED: "Prioritize item identifiers, item attributes, descriptions, categories, tags, and metadata.",
        RecSystemType.HYBRID: "Prioritize user identifiers, item identifiers, interaction events, and item metadata needed for a hybrid recommender.",
        RecSystemType.SEQUENTIAL: "Prioritize session or user identifiers, item identifiers, event order, and timestamps.",
    }

    for schema in schemas:
        lines = [f"\n### Connection: {schema.connection_id} (type: {schema.db_type})"]
        for table in schema.tables:
            cols = "\n".join(f" - {column.name}" for column in table.columns)
            lines.append(f"\nTable: {table.full_name}\n{cols}")
        schema_text.append("\n".join(lines))

        if schema.relationships:
            rel_lines = [
                f"- {rel.from_table}.{rel.from_column} = {rel.to_table}.{rel.to_column}"
                for rel in schema.relationships
            ]
            relationship_text.append(
                f"\n### Relationships for {schema.connection_id}\n" + "\n".join(rel_lines)
            )

    all_schemas = "\n".join(schema_text)
    all_relationships = "\n".join(relationship_text)
    goal_text = ""
    if target_description and target_description.strip():
        goal_text = f"\nUSER GOAL:\n{target_description.strip()}\n"

    return f"""
You are a senior data engineer.

IMPORTANT RULES:
- You MUST ONLY use column names EXACTLY as shown below.
- DO NOT guess or hallucinate column names.
- If unsure, SKIP the column.
- Only create SELECT-style extraction plans for existing tables.
- Only use simple WHERE filters on columns that exist in the same table.
- Use merge_keys only when the same output column will exist in at least two table queries.
- For collaborative, hybrid, or sequential recommendation datasets, the plan MUST include one main interaction table containing both a user identifier and an item identifier.
- After selecting the main interaction table, only add user or item dimension tables that are directly related by a foreign key or the same identifier column.
- Prefer output columns that make a training-ready dataset: user id, item id, optional rating/score, optional timestamp, plus a small amount of useful user/item metadata.
- Do NOT return unrelated entity tables just because they look semantically relevant.
- Recommendation system focus: {rec_guidance.get(rec_type, rec_guidance[RecSystemType.HYBRID])}
{goal_text}

{all_schemas}
{all_relationships}

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


def _post_json(url, payload, headers, timeout=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        response_text = ""
        try:
            response_text = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            response_text = ""
        detail = f"HTTP {exc.code} {exc.reason}"
        if response_text:
            detail = f"{detail}: {response_text[:1000]}"
        raise RuntimeError(detail) from exc


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


def _parse_chat_model_name(model_name: str) -> tuple[str, str]:
    raw = (model_name or "").strip()
    if not raw:
        return "", ""
    if ":" not in raw:
        return "google_genai", raw
    provider, model = raw.split(":", 1)
    return provider.strip().lower(), model.strip()


def _call_google_genai(prompt, api_key, model):
    data = _post_json(
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        payload={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        },
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
    )

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Google GenAI invalid response: {data}")

    parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    if not text_parts:
        raise RuntimeError(f"Google GenAI returned no text content: {data}")

    return "\n".join(text_parts)


def _parse_plan(raw_text):
    text = re.sub(r"```(?:json)?", "", raw_text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON returned from LLM")
    return json.loads(match.group(0))


def build_merge_plan(
    schemas: list[SchemaResponse],
    rec_type: RecSystemType,
    target_description: Optional[str] = None,
    groq_api_key: Optional[str] = "",
    openai_api_key: Optional[str] = "",
    chat_api_key: Optional[str] = "",
    chat_model_name: str = DEFAULT_CHAT_MODEL_NAME,
    groq_model: str = DEFAULT_GROQ_MODEL,
    openai_model: str = DEFAULT_OPENAI_MODEL,
) -> MergePlan:
    chat_api_key = chat_api_key or os.getenv("CHAT_API_KEY", "")
    chat_model_name = chat_model_name or os.getenv("CHAT_MODEL_NAME", "")
    groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
    chat_provider, chat_model = _parse_chat_model_name(chat_model_name)

    if chat_model_name and not chat_api_key and not groq_api_key and not openai_api_key:
        raise RuntimeError(
            "CHAT_MODEL_NAME is configured but CHAT_API_KEY is missing. Set CHAT_API_KEY in the backend environment."
        )

    if not chat_api_key and not groq_api_key and not openai_api_key:
        raise RuntimeError(
            "No LLM API key configured. Set CHAT_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY in the backend "
            "environment or include one in the build request."
        )

    prompt = _build_prompt(schemas, rec_type, target_description)
    logger.info("Sending schema prompt to LLM")

    raw = None
    errors = []

    if chat_api_key and chat_model:
        try:
            if chat_provider != "google_genai":
                raise RuntimeError(
                    f"Unsupported CHAT_MODEL_NAME provider '{chat_provider}'. Expected 'google_genai:<model-name>'."
                )
            logger.info("Trying Google GenAI model %s...", chat_model)
            raw = _call_google_genai(prompt, chat_api_key, chat_model)
            logger.info("Google GenAI success")
        except Exception as exc:
            logger.warning("Google GenAI failed: %s", exc)
            errors.append(f"Google GenAI: {exc}")

    if groq_api_key:
        try:
            logger.info("Trying Groq...")
            raw = _call_groq(prompt, groq_api_key, groq_model)
            logger.info("Groq success")
        except Exception as exc:
            logger.warning("Groq failed: %s", exc)
            errors.append(f"Groq: {exc}")

    if raw is None and openai_api_key:
        try:
            logger.info("Falling back to OpenAI...")
            raw = _call_openai(prompt, openai_api_key, openai_model)
            logger.info("OpenAI success")
        except Exception as exc:
            logger.error("OpenAI failed: %s", exc)
            errors.append(f"OpenAI: {exc}")

    if raw is None:
        if errors:
            raise RuntimeError("All LLM providers failed: " + " | ".join(errors))
        raise RuntimeError(
            "No LLM response was produced. Check CHAT_API_KEY, GROQ_API_KEY, and OPENAI_API_KEY configuration and provider availability."
        )

    parsed_plan = _parse_plan(raw)
    try:
        plan_dict = _sanitize_plan(parsed_plan, schemas, rec_type)
    except ValueError as exc:
        logger.warning("LLM plan validation failed; using fallback planner. Reason: %s", exc)
        plan_dict = _build_fallback_plan(
            schemas=schemas,
            rec_type=rec_type,
            target_description=target_description,
            reason=str(exc),
            raw_plan=parsed_plan,
            raw_text=raw,
        )

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

    return MergePlan(
        table_queries=table_queries,
        collection_fetches=[],
        merge_keys=plan_dict.get("merge_keys", []),
        final_columns=plan_dict.get("final_columns", []),
        description=plan_dict.get("description", ""),
        raw_plan=plan_dict,
    )
