import unittest
from unittest.mock import patch
from uuid import uuid4

import sqlalchemy
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import IntegrityError
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

from src.app.services.data_quality import DataQualityChecker
from src.app.shared.domain.constants import APP_NAME


def handle_schema_with_sqlite(db_engine: Engine):
    """
    Create the schema for the sqlite database in memory
    :param db_engine:  The database engine
    :return:
    """
    with db_engine.begin() as conn:
        for schema_name in DbSchemaEnum:  # type: ignore
            conn.execute(sqlalchemy.text(f"ATTACH ':memory:' AS {schema_name.value}"))


class DummyContainer:
    """Minimal stand-in for FullContainer used by DataQualityChecker."""
    def __init__(self, id, deduplicate_hash):
        self.id = id
        self.deduplicate_hash = deduplicate_hash


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

    def test_empty_list(self):
        self.assertListEqual(self.checker.remove_duplicates([]), [])
    def test_no_duplicates(self):
        containers = [DummyContainer(1, "a"), DummyContainer(2, "b")]
        result = self.checker.remove_duplicates(containers)
        self.assertListEqual(result, containers)
    def test_consecutive_duplicates_keep_last(self):
        containers = [
            DummyContainer(1, "a"),
            DummyContainer(2, "a"),
            DummyContainer(3, "b"),
        ]
        result = self.checker.remove_duplicates(containers)
        self.assertListEqual(result, [containers[1], containers[2]])
    def test_all_duplicates(self):
        containers = [
            DummyContainer(1, "a"),
            DummyContainer(2, "a"),
            DummyContainer(3, "a"),
        ]
        result = self.checker.remove_duplicates(containers)
        self.assertListEqual(result, [containers[2]])
    def test_non_consecutive_duplicates_are_kept(self):
        # groupby only groups consecutive items
        containers = [
            DummyContainer(1, "a"),
            DummyContainer(2, "b"),
            DummyContainer(3, "a"),
        ]
        result = self.checker.remove_duplicates(containers)
        self.assertListEqual(result, containers)
    @patch("src.app.services.sql_db.queries.session_maker")
    def test__log_duplicates_containers_in_db(self, mocked_session_maker):
        s_maker = sessionmaker(self.engine)
        mocked_session_maker.return_value = s_maker()
        dsp_id = uuid4()
        self.checker._log_duplicates_containers_in_db([DummyContainer(dsp_id, "a")])
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
    @patch("src.app.services.data_quality.write_process_state")
    @patch("src.app.services.data_quality.write_new_data_quality_error")
    def test__log_duplicates_containers_in_db_integrity_error(
        self, mock_write_error, mock_write_state
    ):
        mock_write_error.side_effect = IntegrityError("stmt", {}, Exception("dup"))
        self.checker._log_duplicates_containers_in_db([DummyContainer(uuid4(), "a")])
        mock_write_error.assert_called_once()
        mock_write_state.assert_not_called()
