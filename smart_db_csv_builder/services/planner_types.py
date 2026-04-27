from __future__ import annotations


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
