import json
import unittest

from src.app.shared.api.openapi import OPENAPI_SCHEMA_PATH, render_openapi_schema
from src.main import app


class OpenAPIContractTests(unittest.TestCase):
    def test_committed_schema_is_up_to_date(self):
        """openapi.json is the contract WeLearn-client is generated from"""
        committed = OPENAPI_SCHEMA_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            json.loads(committed),
            json.loads(render_openapi_schema(app)),
            "openapi.json is out of date, run `make openapi-poetry` and commit it",
        )

    def test_operation_ids_are_unique(self):
        operation_ids = [
            operation["operationId"]
            for path in app.openapi()["paths"].values()
            for operation in path.values()
        ]
        self.assertEqual(len(operation_ids), len(set(operation_ids)))

    def test_mcp_operation_ids_are_kept(self):
        operation_ids = {
            operation["operationId"]
            for path in app.openapi()["paths"].values()
            for operation in path.values()
        }
        self.assertTrue({"get_corpus_list", "search_by_document"} <= operation_ids)
