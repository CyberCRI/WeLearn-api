import unittest
import uuid
from unittest import mock

import sqlalchemy
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, insert
from sqlalchemy.orm import sessionmaker
from welearn_database.data.models import (
    Base,
    Category,
    Corpus,
    DbSchemaEnum,
    QtyDocumentInQdrantPerCorpus,
    QtyDocumentPerCorpus,
)

from src.app.core.config import settings
from src.app.data_collection.services.data_collection import get_data_collection_service
from src.app.metric.infra.queries import get_document_qty_table_info_sync
from src.main import app

METRIC_URL = f"{settings.API_V1_STR}/metric"
HEADERS = {"X-API-Key": "test"}


def _attach_schemas(engine: Engine) -> None:
    with engine.begin() as conn:
        for schema in DbSchemaEnum:  # type: ignore
            conn.execute(sqlalchemy.text(f"ATTACH ':memory:' AS {schema.value}"))


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class MetricDataCollectionRoutesTests(unittest.TestCase):
    def setUp(self):
        self.data_collection = mock.MagicMock()
        self.data_collection.register_document_click = mock.AsyncMock()
        self.data_collection.register_syllabus_download = mock.AsyncMock()
        self.data_collection.register_form_answer = mock.AsyncMock()
        app.dependency_overrides[get_data_collection_service] = (
            lambda: self.data_collection
        )
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_data_collection_service, None)

    def test_clicked_document_registers_click(self):
        doc_id, message_id = uuid.uuid4(), uuid.uuid4()

        response = self.client.post(
            f"{METRIC_URL}/clicked_document",
            json={"doc_id": str(doc_id), "message_id": str(message_id)},
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "updated")
        self.data_collection.register_document_click.assert_awaited_once_with(
            doc_id, message_id
        )

    def test_clicked_document_rejects_invalid_ids(self):
        response = self.client.post(
            f"{METRIC_URL}/clicked_document",
            json={"doc_id": "nope", "message_id": "nope"},
            headers=HEADERS,
        )
        self.assertEqual(response.status_code, 422)
        self.data_collection.register_document_click.assert_not_awaited()

    def test_syllabus_downloaded_passes_session_id(self):
        session_id = uuid.uuid4()

        response = self.client.post(
            f"{METRIC_URL}/syllabus_downloaded",
            headers={**HEADERS, "X-Session-Id": str(session_id)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "registered")
        self.data_collection.register_syllabus_download.assert_awaited_once_with(
            session_id
        )

    def test_syllabus_downloaded_without_session(self):
        response = self.client.post(
            f"{METRIC_URL}/syllabus_downloaded", headers=HEADERS
        )
        self.assertEqual(response.status_code, 200)
        self.data_collection.register_syllabus_download.assert_awaited_once_with(None)

    def _form_body(self):
        return {
            "form_name": "satisfaction",
            "question": "Did you like it?",
            "answer": "yes",
            "answer_type": "text",
        }

    def test_form_data_registered(self):
        session_id = uuid.uuid4()

        response = self.client.post(
            f"{METRIC_URL}/form_data",
            json=self._form_body(),
            headers={**HEADERS, "X-Session-Id": str(session_id)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "form data registered")
        self.data_collection.register_form_answer.assert_awaited_once_with(
            form_name="satisfaction",
            question="Did you like it?",
            answer="yes",
            answer_type="text",
            session_id=session_id,
        )

    def test_form_data_missing_session_returns_400(self):
        response = self.client.post(
            f"{METRIC_URL}/form_data", json=self._form_body(), headers=HEADERS
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), "missing session id")
        self.data_collection.register_form_answer.assert_not_awaited()

    @mock.patch("src.app.metric.api.router.logger")
    def test_form_data_service_error_returns_500(self, logger_mock):
        self.data_collection.register_form_answer.side_effect = RuntimeError("db")

        response = self.client.post(
            f"{METRIC_URL}/form_data",
            json=self._form_body(),
            headers={**HEADERS, "X-Session-Id": str(uuid.uuid4())},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(), "an error occured when registering the form data"
        )
        logger_mock.error.assert_called_once()

    def test_form_data_invalid_answer_type_returns_422(self):
        body = {**self._form_body(), "answer_type": "unknown"}
        response = self.client.post(
            f"{METRIC_URL}/form_data",
            json=body,
            headers={**HEADERS, "X-Session-Id": str(uuid.uuid4())},
        )
        self.assertEqual(response.status_code, 422)


class DocumentQtyQueryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        _attach_schemas(self.engine)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(self.engine)

        category_id = uuid.uuid4()
        self.parent_id = uuid.uuid4()
        self.child_id = uuid.uuid4()
        self.inactive_id = uuid.uuid4()
        with self.session_factory() as s:
            s.add(Category(id=category_id, title="cat"))
            s.add_all(
                [
                    Corpus(
                        id=self.parent_id,
                        source_name="parent",
                        main_url="https://parent",
                        is_fix=True,
                        is_active=True,
                        category_id=category_id,
                    ),
                    Corpus(
                        id=self.child_id,
                        parent_corpus_id=self.parent_id,
                        source_name="child",
                        main_url="https://child",
                        is_fix=True,
                        is_active=True,
                        category_id=category_id,
                    ),
                    Corpus(
                        id=self.inactive_id,
                        source_name="inactive",
                        main_url="https://inactive",
                        is_fix=True,
                        is_active=False,
                        category_id=category_id,
                    ),
                ]
            )
            s.flush()
            s.execute(
                insert(QtyDocumentInQdrantPerCorpus),
                [
                    {"source_name": "parent", "count": 3},
                    {"source_name": "child", "count": 2},
                    {"source_name": "inactive", "count": 9},
                ],
            )
            s.execute(
                insert(QtyDocumentPerCorpus),
                [
                    {"source_name": "parent", "count": 10},
                    {"source_name": "child", "count": 5},
                    {"source_name": "inactive", "count": 9},
                ],
            )
            s.commit()

    def tearDown(self):
        self.engine.dispose()

    def test_sub_corpora_are_aggregated_into_parent_and_inactive_excluded(self):
        with mock.patch(
            "src.app.metric.infra.queries.session_maker", self.session_factory
        ):
            rows = get_document_qty_table_info_sync()

        self.assertEqual(len(rows), 1)
        corpus, qty_in_qdrant, qty_total = rows[0]
        self.assertEqual(corpus.id, self.parent_id)
        self.assertEqual(corpus.source_name, "parent")
        self.assertEqual(qty_in_qdrant, 5)
        self.assertEqual(qty_total, 15)
