import unittest

from smart_db_csv_builder.connectors.factory import build_select_sql
from smart_db_csv_builder.core.connection_store import ConnectionRecord
from smart_db_csv_builder.models.schemas import ConnectionCredential, DBType


class SmartDbCsvBuilderSqlTests(unittest.TestCase):
    def test_postgres_select_sql_preserves_schema_name(self):
        sql = build_select_sql(
            DBType.POSTGRES,
            "public.orders",
            ["id", "created_at"],
            "status = 'paid'",
            25,
        )
        self.assertEqual(
            sql,
            'SELECT "id", "created_at" FROM "public"."orders" WHERE status = \'paid\' LIMIT 25',
        )

    def test_mysql_select_sql_quotes_schema_and_table(self):
        sql = build_select_sql(
            DBType.MYSQL,
            "analytics.events",
            ["user_id", "event_type"],
            "",
            10,
        )
        self.assertEqual(
            sql,
            "SELECT `user_id`, `event_type` FROM `analytics`.`events` LIMIT 10",
        )

    def test_mssql_select_sql_uses_top_and_bracketed_schema(self):
        sql = build_select_sql(
            DBType.MSSQL,
            "dbo.Users",
            ["Id", "Email"],
            "[IsActive] = 1",
            5,
        )
        self.assertEqual(
            sql,
            "SELECT TOP 5 [Id], [Email] FROM [dbo].[Users] WHERE [IsActive] = 1",
        )

    def test_sqlite_select_sql_quotes_table_name(self):
        sql = build_select_sql(
            DBType.SQLITE,
            "events",
            ["user_id"],
            "",
            3,
        )
        self.assertEqual(sql, 'SELECT "user_id" FROM "events" LIMIT 3')

    def test_connection_record_exposes_credential_metadata(self):
        cred = ConnectionCredential(
            db_type=DBType.SQLITE,
            name="Local DB",
            filepath=":memory:",
        )
        record = ConnectionRecord(id="conn-1", cred=cred, driver=object())

        self.assertEqual(record.name, "Local DB")
        self.assertEqual(record.db_type, DBType.SQLITE)
        self.assertIsNone(record.database)


if __name__ == "__main__":
    unittest.main()
