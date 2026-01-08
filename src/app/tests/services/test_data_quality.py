import unittest
from unittest.mock import patch
from uuid import uuid4

import sqlalchemy
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker
from welearn_database.data.enumeration import Step
from welearn_database.data.models import (
    Base,
    Category,
    Corpus,
    DbSchemaEnum,
    ErrorDataQuality,
    ProcessState,
    WeLearnDocument,
)

from src.app.services.constants import APP_NAME
from src.app.services.data_quality import DataQualityChecker


def handle_schema_with_sqlite(db_engine: Engine):
    """
    Create the schema for the sqlite database in memory
    :param db_engine:  The database engine
    :return:
    """
    with db_engine.begin() as conn:
        for schema_name in DbSchemaEnum:  # type: ignore
            conn.execute(sqlalchemy.text(f"ATTACH ':memory:' AS {schema_name.value}"))


class DummyScoredPoint:
    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


class TestRemoveDuplicates(unittest.TestCase):
    def setUp(self):
        self.checker = DataQualityChecker(log_background_task=None)

        self.engine = create_engine("sqlite://")
        s_maker = sessionmaker(self.engine)
        handle_schema_with_sqlite(self.engine)

        self.test_session = s_maker()
        Base.metadata.create_all(self.test_session.get_bind())

        self.category_name = "categroy_test0"
        self.category_id = uuid4()

        self.category = Category(id=self.category_id, title=self.category_name)

        corpus_source_name = "test_corpus"

        self.corpus_test = Corpus(
            id=uuid4(),
            source_name=corpus_source_name,
            is_fix=True,
            is_active=True,
            category_id=self.category_id,
        )

        self.doc_test_id = uuid4()
        self.doc_test = WeLearnDocument(
            id=self.doc_test_id,
            url="https://example.org",
            corpus_id=self.corpus_test.id,
            title="test",
            lang="en",
            full_content="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi volutpat aliquam sollicitudin.",
            description="test",
            details={"test": "test"},
        )

        self.test_session.add(self.category)
        self.test_session.add(self.corpus_test)
        self.test_session.add(self.doc_test)

        self.test_session.commit()

    def test_basic(self):
        points = [
            DummyScoredPoint(1, {"text": "a"}),
            DummyScoredPoint(2, {"text": "b"}),
            DummyScoredPoint(3, {"text": "a"}),
        ]
        result = self.checker.remove_duplicates(["text"], points)
        self.assertEqual(len(result), 2)
        self.assertSetEqual(set(p.payload["text"] for p in result), {"a", "b"})

    def test_empty_keys(self):
        points = [DummyScoredPoint(1, {"text": "a"})]
        result = self.checker.remove_duplicates([], points)
        self.assertListEqual(result, points)

    def test_empty_keys_strict(self):
        points = [DummyScoredPoint(1, {"text": "a"})]
        with self.assertRaises(ValueError):
            self.checker.remove_duplicates([], points, strict=True)

    def test_missing_key(self):
        points = [
            DummyScoredPoint(1, {"text": "a"}),
            DummyScoredPoint(2, {"other": "b"}),
        ]
        result = self.checker.remove_duplicates(["text"], points)
        self.assertListEqual(result, points)

    def test_missing_key_strict(self):
        points = [
            DummyScoredPoint(1, {"text": "a"}),
            DummyScoredPoint(2, {"other": "b"}),
        ]
        with self.assertRaises(ValueError):
            self.checker.remove_duplicates(["text"], points, strict=True)

    def test_non_str_value(self):
        points = [
            DummyScoredPoint(1, {"text": 123}),
            DummyScoredPoint(2, {"text": "a"}),
        ]
        result = self.checker.remove_duplicates(["text"], points)
        self.assertListEqual(result, points)

    def test_non_str_value_strict(self):
        points = [DummyScoredPoint(1, {"text": 123})]
        with self.assertRaises(TypeError):
            self.checker.remove_duplicates(["text"], points, strict=True)

    def test_empty_payload(self):
        for payload in [{}, None]:
            points = [DummyScoredPoint(1, payload)]
            result = self.checker.remove_duplicates(["text"], points)
            self.assertListEqual(result, points)

    def test_no_duplicates(self):
        points = [
            DummyScoredPoint(1, {"text": "a"}),
            DummyScoredPoint(2, {"text": "b"}),
        ]
        result = self.checker.remove_duplicates(["text"], points)
        self.assertEqual(len(result), 2)
        self.assertListEqual(result, points)

    def test_multiple_keys(self):
        points = [
            DummyScoredPoint(1, {"a": "x", "b": "y"}),
            DummyScoredPoint(2, {"a": "x", "b": "y"}),
            DummyScoredPoint(3, {"a": "x", "b": "z"}),
        ]
        result = self.checker.remove_duplicates(["a", "b"], points)
        self.assertEqual(len(result), 2)
        self.assertListEqual(result, [points[0], points[2]])

    def test_all_duplicates(self):
        points = [
            DummyScoredPoint(1, {"text": "a"}),
            DummyScoredPoint(2, {"text": "a"}),
            DummyScoredPoint(3, {"text": "a"}),
        ]
        result = self.checker.remove_duplicates(["text"], points)
        self.assertEqual(len(result), 1)
        self.assertListEqual(result, [points[0]])

    @patch("src.app.services.sql_service.wl_sql.session_maker")
    def test__log_duplicates_points_in_db(self, mocked_session_maker):
        s_maker = sessionmaker(self.engine)
        mocked_session_maker.return_value = s_maker()

        dsp_id = uuid4()

        points = [
            DummyScoredPoint(
                1,
                {
                    "document_title": self.doc_test.title,
                    "document_desc": self.doc_test.description,
                    "document_url": self.doc_test.url,
                    "document_id": self.doc_test_id,
                },
            ),
            DummyScoredPoint(
                1,
                {
                    "document_title": self.doc_test.title,
                    "document_desc": self.doc_test.description,
                    "document_url": self.doc_test.url + "1",
                    "document_id": dsp_id,
                },
            ),
            DummyScoredPoint(
                3,
                {
                    "text": "b",
                    "document_id": self.doc_test_id,
                    "document_title": "lorem",
                    "document_desc": "description",
                    "document_url": "https://example.com",
                },
            ),
        ]
        deduped = [points[0], points[2]]
        self.checker._log_duplicates_points_in_db(points, deduped)

        errors = (
            self.test_session.query(ErrorDataQuality)
            .filter_by(document_id=dsp_id)
            .all()
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].document_id, dsp_id)
        self.assertEqual(errors[0].error_raiser, APP_NAME)

        states = (
            self.test_session.query(ProcessState).filter_by(document_id=dsp_id).all()
        )
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].document_id, dsp_id)
        self.assertEqual(states[0].title, Step.DOCUMENT_IS_INVALID.value.lower())
