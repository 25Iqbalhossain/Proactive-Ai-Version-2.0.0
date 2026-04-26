import unittest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from smart_db_csv_builder.core.job_store import Job
from smart_db_csv_builder.models.schemas import (
    BuildMode,
    BuildRequest,
    ColumnInfo,
    DBType,
    FKRelationship,
    OutputFormat,
    RecSystemType,
    SchemaResponse,
    TableInfo,
)
from smart_db_csv_builder.services.builder import run_build_job
from smart_db_csv_builder.services.executor import execute_plan
from smart_db_csv_builder.services.llm_planner import (
    MergePlan,
    TableQuery,
    _parse_chat_model_name,
    build_merge_plan,
)


class _FakeDriver:
    def __init__(self, error=None, rows=None):
        self._error = error
        self._rows = rows or []

    def execute(self, sql):
        if self._error is not None:
            raise self._error
        return self._rows


class _FakeConnection:
    def __init__(self, db_type=DBType.SQLITE, driver=None):
        self.db_type = db_type
        self.driver = driver or _FakeDriver()


class SmartDbCsvBuilderLlmPlannerTests(unittest.TestCase):
    def _schemas(self):
        return [
            SchemaResponse(
                connection_id="conn-1",
                db_type=DBType.POSTGRES,
                tables=[
                    TableInfo(
                        schema_name="public",
                        table_name="events",
                        row_count=1000,
                        columns=[
                            ColumnInfo(name="user_id", data_type="integer"),
                            ColumnInfo(name="item_id", data_type="integer"),
                            ColumnInfo(name="rating", data_type="integer"),
                            ColumnInfo(name="created_at", data_type="timestamp"),
                            ColumnInfo(name="event_type", data_type="text"),
                        ],
                    ),
                    TableInfo(
                        schema_name="public",
                        table_name="users",
                        row_count=100,
                        columns=[
                            ColumnInfo(name="user_id", data_type="integer"),
                            ColumnInfo(name="country", data_type="text"),
                        ],
                    ),
                    TableInfo(
                        schema_name="public",
                        table_name="items",
                        row_count=200,
                        columns=[
                            ColumnInfo(name="item_id", data_type="integer"),
                            ColumnInfo(name="category", data_type="text"),
                            ColumnInfo(name="price", data_type="numeric"),
                        ],
                    ),
                ],
                relationships=[
                    FKRelationship(
                        from_table="public.events",
                        from_column="user_id",
                        to_table="public.users",
                        to_column="user_id",
                    ),
                    FKRelationship(
                        from_table="public.events",
                        from_column="item_id",
                        to_table="public.items",
                        to_column="item_id",
                    ),
                ],
            )
        ]

    def _generic_relationship_schemas(self):
        return [
            SchemaResponse(
                connection_id="conn-generic",
                db_type=DBType.POSTGRES,
                tables=[
                    TableInfo(
                        schema_name="public",
                        table_name="activity_log",
                        row_count=5000,
                        columns=[
                            ColumnInfo(name="actor_ref", data_type="integer"),
                            ColumnInfo(name="object_ref", data_type="integer"),
                            ColumnInfo(name="score_value", data_type="integer"),
                            ColumnInfo(name="updated_at", data_type="timestamp"),
                        ],
                    ),
                    TableInfo(
                        schema_name="public",
                        table_name="actors",
                        row_count=300,
                        columns=[
                            ColumnInfo(name="ref", data_type="integer"),
                            ColumnInfo(name="region", data_type="text"),
                        ],
                    ),
                    TableInfo(
                        schema_name="public",
                        table_name="objects",
                        row_count=200,
                        columns=[
                            ColumnInfo(name="ref", data_type="integer"),
                            ColumnInfo(name="category", data_type="text"),
                            ColumnInfo(name="title", data_type="text"),
                        ],
                    ),
                ],
                relationships=[
                    FKRelationship(
                        from_table="public.activity_log",
                        from_column="actor_ref",
                        to_table="public.actors",
                        to_column="ref",
                    ),
                    FKRelationship(
                        from_table="public.activity_log",
                        from_column="object_ref",
                        to_table="public.objects",
                        to_column="ref",
                    ),
                ],
            )
        ]

    @patch("smart_db_csv_builder.services.llm_planner._call_openai")
    def test_build_merge_plan_sanitizes_llm_generated_queries(self, mock_call_openai):
        mock_call_openai.return_value = """
        {
          "description": "Hybrid dataset",
          "merge_keys": ["user_id", "missing_key"],
          "final_columns": ["user_id", "country", "product_id", "missing_key"],
          "table_queries": [
            {
              "connection_id": "conn-1",
              "table": "events",
              "columns": ["USER_ID", "item_id", "missing_col"],
              "alias_map": {"item_id": "product_id", "user_id": "bad-name"},
              "where": "event_type = 'click'; DROP TABLE users"
            },
            {
              "connection_id": "conn-1",
              "table": "public.users",
              "columns": ["user_id", "country"],
              "where": "country = 'BD'"
            }
          ],
          "collection_fetches": []
        }
        """

        plan = build_merge_plan(
            schemas=self._schemas(),
            rec_type=RecSystemType.HYBRID,
            target_description="Build rec data",
            openai_api_key="test-key",
        )

        self.assertEqual(len(plan.table_queries), 2)
        self.assertEqual(plan.table_queries[0].table, "public.events")
        self.assertEqual(plan.table_queries[0].columns, ["user_id", "item_id"])
        self.assertEqual(plan.table_queries[0].alias_map.get("item_id"), "product_id")
        self.assertEqual(plan.table_queries[0].alias_map.get("user_id"), "userID")
        self.assertEqual(plan.table_queries[0].where, "")
        self.assertEqual(plan.table_queries[1].where, "country = 'BD'")
        self.assertEqual(plan.merge_keys, ["userID"])
        self.assertEqual(plan.final_columns[0], "userID")
        self.assertIn("country", plan.final_columns)
        self.assertIn("product_id", plan.final_columns)
        self.assertEqual(plan.raw_plan["table_queries"][0]["table"], "public.events")

    @patch("smart_db_csv_builder.services.llm_planner._call_openai")
    def test_build_merge_plan_accepts_dotted_columns_aliases_and_table_qualified_where(self, mock_call_openai):
        mock_call_openai.return_value = """
        {
          "description": "Complex hybrid dataset",
          "merge_keys": [],
          "final_columns": [],
          "table_queries": [
            {
              "connection_id": "conn-1",
              "table": "public.events",
              "columns": ["public.events.user_id", "events.item_id AS product_id", "events.created_at"],
              "where": "events.created_at >= '2024-01-01' and events.event_type = 'purchase'"
            },
            {
              "connection_id": "conn-1",
              "table": "users",
              "columns": ["users.user_id", "users.country"],
              "where": "lower(users.country) = 'bd'"
            }
          ],
          "collection_fetches": []
        }
        """

        plan = build_merge_plan(
            schemas=self._schemas(),
            rec_type=RecSystemType.HYBRID,
            target_description="Build purchase recommendation data from events with user geography",
            openai_api_key="test-key",
        )

        self.assertEqual(plan.table_queries[0].columns, ["user_id", "item_id", "created_at"])
        self.assertEqual(plan.table_queries[0].alias_map.get("item_id"), "product_id")
        self.assertEqual(plan.table_queries[0].alias_map.get("user_id"), "userID")
        self.assertEqual(plan.table_queries[0].alias_map.get("created_at"), "timestamp")
        self.assertEqual(
            plan.table_queries[0].where,
            "events.created_at >= '2024-01-01' and events.event_type = 'purchase'",
        )
        self.assertEqual(plan.table_queries[1].where, "lower(users.country) = 'bd'")
        self.assertEqual(plan.merge_keys, ["userID"])
        self.assertIn("userID", plan.final_columns)
        self.assertIn("product_id", plan.final_columns)
        self.assertIn("country", plan.final_columns)

    @patch("smart_db_csv_builder.services.llm_planner._call_openai")
    def test_build_merge_plan_infers_merge_keys_and_final_columns_for_partial_complex_plan(self, mock_call_openai):
        mock_call_openai.return_value = """
        {
          "description": "Partial complex plan",
          "merge_keys": [],
          "final_columns": ["country"],
          "table_queries": [
            {"connection_id": "conn-1", "table": "events", "columns": ["user_id", "item_id", "rating", "created_at"]},
            {"connection_id": "conn-1", "table": "users", "columns": ["user_id", "country"]},
            {"connection_id": "conn-1", "table": "items", "columns": ["item_id", "category", "price"]}
          ],
          "collection_fetches": []
        }
        """

        plan = build_merge_plan(
            schemas=self._schemas(),
            rec_type=RecSystemType.HYBRID,
            target_description="Build a training dataset with user behavior, item catalog context, and timestamps",
            openai_api_key="test-key",
        )

        self.assertEqual(plan.merge_keys, ["userID", "itemID"])
        self.assertEqual(plan.table_queries[0].table, "public.events")
        self.assertTrue({"userID", "itemID", "country", "category"}.issubset(set(plan.final_columns)))
        self.assertLess(plan.final_columns.index("userID"), plan.final_columns.index("country"))
        self.assertLess(plan.final_columns.index("itemID"), plan.final_columns.index("category"))

    @patch("smart_db_csv_builder.services.llm_planner._call_openai")
    def test_build_merge_plan_falls_back_when_llm_plan_has_no_valid_merge_keys(self, mock_call_openai):
        mock_call_openai.return_value = """
        {
          "description": "Bad plan",
          "merge_keys": ["missing_key"],
          "final_columns": [],
          "table_queries": [
            {"connection_id": "conn-1", "table": "events", "columns": ["user_id"]},
            {"connection_id": "conn-1", "table": "users", "columns": ["country"]}
          ],
          "collection_fetches": []
        }
        """

        plan = build_merge_plan(
            schemas=self._schemas(),
            rec_type=RecSystemType.HYBRID,
            target_description="Build rec data",
            openai_api_key="test-key",
        )

        self.assertTrue(plan.raw_plan.get("_fallback"))
        self.assertGreaterEqual(len(plan.table_queries), 1)

    @patch("smart_db_csv_builder.services.llm_planner._call_openai")
    def test_build_merge_plan_falls_back_when_llm_plan_omits_interaction_table(self, mock_call_openai):
        mock_call_openai.return_value = """
        {
          "description": "Bad plan without fact table",
          "merge_keys": ["user_id"],
          "final_columns": ["user_id", "country", "category"],
          "table_queries": [
            {"connection_id": "conn-1", "table": "users", "columns": ["user_id", "country"]},
            {"connection_id": "conn-1", "table": "items", "columns": ["item_id", "category"]}
          ],
          "collection_fetches": []
        }
        """

        plan = build_merge_plan(
            schemas=self._schemas(),
            rec_type=RecSystemType.HYBRID,
            target_description="Build a recommendation dataset with user behavior and item data",
            openai_api_key="test-key",
        )

        self.assertTrue(plan.raw_plan.get("_fallback"))
        self.assertEqual(plan.table_queries[0].table, "public.events")
        self.assertIn("userID", plan.merge_keys)
        self.assertIn("itemID", plan.merge_keys)
        self.assertTrue(any(tq.table == "public.users" for tq in plan.table_queries))
        self.assertTrue(any(tq.table == "public.items" for tq in plan.table_queries))

    @patch.dict(os.environ, {}, clear=True)
    def test_build_merge_plan_requires_at_least_one_llm_key(self):
        with self.assertRaisesRegex(RuntimeError, "No LLM API key configured"):
            build_merge_plan(
                schemas=self._schemas(),
                rec_type=RecSystemType.HYBRID,
                target_description="Build rec data",
            )

    def test_parse_chat_model_name_google_genai_provider(self):
        provider, model = _parse_chat_model_name("google_genai:gemini-2.5-flash-lite")
        self.assertEqual(provider, "google_genai")
        self.assertEqual(model, "gemini-2.5-flash-lite")

    @patch("smart_db_csv_builder.services.llm_planner._call_google_genai")
    def test_build_merge_plan_supports_chat_api_key_with_google_genai_model(self, mock_call_google_genai):
        mock_call_google_genai.return_value = """
        {
          "description": "Hybrid dataset",
          "merge_keys": ["user_id"],
          "final_columns": ["user_id", "country"],
          "table_queries": [
            {"connection_id": "conn-1", "table": "events", "columns": ["user_id", "item_id"]},
            {"connection_id": "conn-1", "table": "users", "columns": ["user_id", "country"]}
          ],
          "collection_fetches": []
        }
        """

        plan = build_merge_plan(
            schemas=self._schemas(),
            rec_type=RecSystemType.HYBRID,
            target_description="Build rec data",
            chat_api_key="test-google-key",
            chat_model_name="google_genai:gemini-2.5-flash-lite",
        )

        self.assertEqual(plan.merge_keys, ["userID"])
        self.assertEqual(len(plan.table_queries), 2)
        mock_call_google_genai.assert_called_once()

    @patch("smart_db_csv_builder.services.llm_planner._call_google_genai")
    def test_build_merge_plan_falls_back_when_llm_plan_has_no_valid_tables(self, mock_call_google_genai):
        mock_call_google_genai.return_value = """
        {
          "description": "Bad Gemini plan",
          "merge_keys": ["unknown_key"],
          "final_columns": [],
          "table_queries": [
            {"connection_id": "wrong-conn", "table": "unknown_table", "columns": ["ghost_col"]}
          ],
          "collection_fetches": []
        }
        """

        plan = build_merge_plan(
            schemas=self._schemas(),
            rec_type=RecSystemType.HYBRID,
            target_description="Build recommendation data from events and users",
            chat_api_key="test-google-key",
            chat_model_name="google_genai:gemini-2.5-flash-lite",
        )

        self.assertTrue(plan.raw_plan.get("_fallback"))
        self.assertGreaterEqual(len(plan.table_queries), 1)
        self.assertIn("Fallback schema-driven plan", plan.description)
        self.assertTrue(any(tq.table == "public.events" for tq in plan.table_queries))

    @patch("smart_db_csv_builder.services.llm_planner._call_openai")
    def test_build_merge_plan_fallback_uses_relationship_structure_for_generic_entity_keys(self, mock_call_openai):
        mock_call_openai.return_value = """
        {
          "description": "Unusable plan",
          "merge_keys": [],
          "final_columns": [],
          "table_queries": [
            {"connection_id": "conn-generic", "table": "actors", "columns": ["region"]}
          ],
          "collection_fetches": []
        }
        """

        plan = build_merge_plan(
            schemas=self._generic_relationship_schemas(),
            rec_type=RecSystemType.HYBRID,
            target_description="Build recommendation interactions from the generic activity log",
            openai_api_key="test-key",
        )

        self.assertTrue(plan.raw_plan.get("_fallback"))
        self.assertEqual(plan.table_queries[0].table, "public.activity_log")
        self.assertEqual(plan.table_queries[0].alias_map.get("actor_ref"), "userID")
        self.assertEqual(plan.table_queries[0].alias_map.get("object_ref"), "itemID")
        self.assertIn("userID", plan.merge_keys)
        self.assertIn("itemID", plan.merge_keys)
        self.assertIn("userID", plan.final_columns)
        self.assertIn("itemID", plan.final_columns)

    @patch.dict(os.environ, {"CHAT_MODEL_NAME": "google_genai:gemini-2.5-flash-lite"}, clear=True)
    def test_build_merge_plan_requires_chat_api_key_when_chat_model_is_configured(self):
        with self.assertRaisesRegex(RuntimeError, "CHAT_MODEL_NAME is configured but CHAT_API_KEY is missing"):
            build_merge_plan(
                schemas=self._schemas(),
                rec_type=RecSystemType.HYBRID,
                target_description="Build rec data",
            )

    @patch("smart_db_csv_builder.services.executor.connection_store.get")
    def test_execute_plan_raises_when_query_execution_fails(self, mock_get):
        mock_get.return_value = _FakeConnection(
            db_type=DBType.SQLITE,
            driver=_FakeDriver(error=RuntimeError("bad sql")),
        )

        plan = MergePlan(
            table_queries=[TableQuery("conn-1", "events", ["user_id"])],
            collection_fetches=[],
            merge_keys=[],
            final_columns=[],
            description="",
            raw_plan={},
        )

        with self.assertRaisesRegex(RuntimeError, "Query failed for table 'events'"):
            execute_plan(plan=plan, output_format=OutputFormat.CSV)

    @patch("smart_db_csv_builder.services.executor.connection_store.get")
    def test_execute_plan_persists_csv_and_json_outputs(self, mock_get):
        mock_get.return_value = _FakeConnection(
            db_type=DBType.SQLITE,
            driver=_FakeDriver(
                rows=[
                    {"user_id": 1, "item_id": 101},
                    {"user_id": 2, "item_id": 202},
                ]
            ),
        )

        plan = MergePlan(
            table_queries=[TableQuery("conn-1", "events", ["user_id", "item_id"])],
            collection_fetches=[],
            merge_keys=[],
            final_columns=[],
            description="",
            raw_plan={},
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "smart_db_csv_builder.services.executor.OUTPUT_DIR",
            Path(tmpdir),
        ):
            filepath, row_count, col_count, output_files = execute_plan(
                plan=plan,
                output_format=OutputFormat.CSV,
                output_stem="llm_dataset",
            )

            self.assertEqual(filepath, output_files["csv"])
            self.assertEqual(row_count, 2)
            self.assertEqual(col_count, 2)
            self.assertTrue(output_files["csv"].endswith("llm_dataset.csv"))
            self.assertTrue(output_files["json"].endswith("llm_dataset.json"))
            self.assertTrue(Path(output_files["csv"]).exists())
            self.assertTrue(Path(output_files["json"]).exists())
            self.assertEqual(len(json.loads(Path(output_files["json"]).read_text(encoding="utf-8"))), 2)

    @patch("smart_db_csv_builder.services.executor.connection_store.get")
    def test_execute_plan_coalesces_requested_output_columns_from_multiple_sources(self, mock_get):
        drivers = {
            "public.events": _FakeDriver(
                rows=[
                    {"user_id": 1, "item_id": 101},
                    {"user_id": 2, "item_id": 202},
                ]
            ),
            "public.users": _FakeDriver(
                rows=[
                    {"user_id": 1, "country": "BD"},
                    {"user_id": 2, "country": "US"},
                ]
            ),
            "public.items": _FakeDriver(
                rows=[
                    {"item_id": 101, "country": "Warehouse-BD", "category": "Books"},
                    {"item_id": 202, "country": "Warehouse-US", "category": "Games"},
                ]
            ),
        }

        class _DispatchDriver:
            def execute(self, sql):
                if '"public"."events"' in sql:
                    return drivers["public.events"].execute(sql)
                if '"public"."users"' in sql:
                    return drivers["public.users"].execute(sql)
                if '"public"."items"' in sql:
                    return drivers["public.items"].execute(sql)
                raise AssertionError(f"Unexpected SQL: {sql}")

        mock_get.return_value = _FakeConnection(
            db_type=DBType.SQLITE,
            driver=_DispatchDriver(),
        )

        plan = MergePlan(
            table_queries=[
                TableQuery("conn-1", "public.events", ["user_id", "item_id"]),
                TableQuery("conn-1", "public.users", ["user_id", "country"]),
                TableQuery("conn-1", "public.items", ["item_id", "country", "category"]),
            ],
            collection_fetches=[],
            merge_keys=["user_id", "item_id"],
            final_columns=["user_id", "item_id", "country", "category"],
            description="",
            raw_plan={},
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "smart_db_csv_builder.services.executor.OUTPUT_DIR",
            Path(tmpdir),
        ):
            filepath, row_count, col_count, _ = execute_plan(
                plan=plan,
                output_format=OutputFormat.CSV,
                output_stem="coalesced_dataset",
            )

            frame = __import__("pandas").read_csv(filepath)
            self.assertEqual(row_count, 2)
            self.assertEqual(col_count, 4)
            self.assertEqual(list(frame.columns), ["userID", "itemID", "country", "category"])
            self.assertEqual(frame.loc[0, "country"], "BD")
            self.assertEqual(frame.loc[1, "country"], "US")

    @patch("smart_db_csv_builder.services.executor.connection_store.get")
    def test_execute_plan_normalizes_generic_entity_columns_for_training(self, mock_get):
        class _DispatchDriver:
            def execute(self, sql):
                if '"public"."activity_log"' in sql:
                    return [
                        {"actor_ref": 1, "object_ref": 10, "score_value": 5, "updated_at": "2026-01-01"},
                        {"actor_ref": 1, "object_ref": 11, "score_value": 3, "updated_at": "2026-01-02"},
                        {"actor_ref": 2, "object_ref": 10, "score_value": 4, "updated_at": "2026-01-03"},
                    ]
                raise AssertionError(f"Unexpected SQL: {sql}")

        mock_get.return_value = _FakeConnection(
            db_type=DBType.SQLITE,
            driver=_DispatchDriver(),
        )

        plan = MergePlan(
            table_queries=[
                TableQuery("conn-1", "public.activity_log", ["actor_ref", "object_ref", "score_value", "updated_at"])
            ],
            collection_fetches=[],
            merge_keys=[],
            final_columns=["actor_ref", "object_ref", "score_value", "updated_at"],
            description="",
            raw_plan={},
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "smart_db_csv_builder.services.executor.OUTPUT_DIR",
            Path(tmpdir),
        ):
            filepath, row_count, col_count, _ = execute_plan(
                plan=plan,
                output_format=OutputFormat.CSV,
                output_stem="normalized_dataset",
            )

            frame = __import__("pandas").read_csv(filepath)
            self.assertEqual(row_count, 3)
            self.assertEqual(col_count, 4)
            self.assertIn("userID", frame.columns)
            self.assertIn("itemID", frame.columns)
            self.assertIn("rating", frame.columns)
            self.assertIn("timestamp", frame.columns)

    @patch("smart_db_csv_builder.services.executor.connection_store.get")
    def test_execute_plan_fails_early_when_built_dataset_has_no_second_entity_id(self, mock_get):
        mock_get.return_value = _FakeConnection(
            db_type=DBType.SQLITE,
            driver=_FakeDriver(
                rows=[
                    {"id": 1001, "rating": 4, "updated_at": "2026-01-01"},
                    {"id": 1002, "rating": 5, "updated_at": "2026-01-02"},
                    {"id": 1003, "rating": 3, "updated_at": "2026-01-03"},
                ]
            ),
        )

        plan = MergePlan(
            table_queries=[TableQuery("conn-1", "public.audit_log", ["id", "rating", "updated_at"])],
            collection_fetches=[],
            merge_keys=[],
            final_columns=["id", "rating", "updated_at"],
            description="",
            raw_plan={},
        )

        with self.assertRaisesRegex(RuntimeError, "Built dataset is not training-ready"):
            execute_plan(plan=plan, output_format=OutputFormat.CSV)

    def test_run_build_job_marks_running_step_as_error(self):
        job = Job(job_id="job-1")
        req = BuildRequest(
            connection_ids=["missing-connection"],
            mode=BuildMode.LLM,
            rec_system_type=RecSystemType.HYBRID,
            llm_prompt="build a hybrid recommendation dataset",
        )

        run_build_job(job, req)

        self.assertEqual(str(job.status), "JobStatus.FAILED")
        self.assertEqual(job.error, "Connection 'missing-connection' not found. Please register it first.")
        self.assertEqual(job.steps[0]["step"], "validate_connections")
        self.assertEqual(job.steps[0]["status"], "error")

    def test_build_request_normalizes_content_rec_system_alias(self):
        req = BuildRequest(
            connection_ids=["conn-1"],
            mode=BuildMode.LLM,
            llm_prompt="build a content dataset",
            rec_system_type="content",
        )

        self.assertEqual(req.rec_system_type, RecSystemType.CONTENT_BASED)
        self.assertEqual(req.target_description, "build a content dataset")

    def test_build_request_rejects_query_mode_without_text(self):
        with self.assertRaisesRegex(ValueError, "query mode requires query_text or target_description"):
            BuildRequest(
                connection_ids=["conn-1"],
                mode=BuildMode.QUERY,
            )

    def test_build_request_rejects_empty_manual_mode(self):
        with self.assertRaisesRegex(ValueError, "manual mode requires at least one manual_config field"):
            BuildRequest(
                connection_ids=["conn-1"],
                mode=BuildMode.MANUAL,
                manual_config={"tables": "   ", "notes": ""},
            )


if __name__ == "__main__":
    unittest.main()
