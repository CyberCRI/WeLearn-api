import unittest
from unittest.mock import patch
from uuid import uuid4

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from welearn_database.data.models import (
    Base,
    Category,
    ContextDocument,
    Corpus,
    DbSchemaEnum,
    DocumentSlice,
    EmbeddingModel,
    QtyDocumentInQdrant,
    WeLearnDocument,
)

from src.app.search.infra import queries
from src.app.search.infra.queries import (
    get_collections_info_sync,
    get_corpus_and_sub_corpus_repartition,
    get_documents_according_slice_ids,
    get_nb_docs_sync,
    get_subject,
)

QUERIES_SESSION_MAKER = "src.app.search.infra.queries.session_maker"


class SearchQueriesSqliteTests(unittest.TestCase):
    """Run the search queries against an in-memory sqlite database."""

    def setUp(self):
        self.engine = create_engine("sqlite://")
        self.addCleanup(self.engine.dispose)
        with self.engine.begin() as conn:
            for schema_name in DbSchemaEnum:  # type: ignore
                conn.execute(
                    sqlalchemy.text(f"ATTACH ':memory:' AS {schema_name.value}")
                )
        Base.metadata.create_all(self.engine)

        # every call to session_maker() gets a fresh session on the test engine
        patcher = patch(QUERIES_SESSION_MAKER, side_effect=self._new_session)
        self.mocked_session_maker = patcher.start()
        self.addCleanup(patcher.stop)

        # the parent-corpus cache is module global: isolate it per test
        dict_patcher = patch.dict(queries.main_corpus_dict, clear=True)
        dict_patcher.start()
        self.addCleanup(dict_patcher.stop)

        self.category = Category(id=uuid4(), title="cat_title")
        self.parent_corpus = Corpus(
            id=uuid4(),
            source_name="parent",
            is_fix=True,
            is_active=True,
            category_id=self.category.id,
        )
        self.child_corpus = Corpus(
            id=uuid4(),
            source_name="child",
            is_fix=True,
            is_active=True,
            category_id=self.category.id,
            parent_corpus_id=self.parent_corpus.id,
        )
        self.inactive_corpus = Corpus(
            id=uuid4(),
            source_name="inactive",
            is_fix=True,
            is_active=False,
            category_id=self.category.id,
        )
        self.other_category = Category(id=uuid4(), title="other_cat")
        self.other_corpus = Corpus(
            id=uuid4(),
            source_name="other",
            is_fix=True,
            is_active=True,
            category_id=self.other_category.id,
        )
        self.embedding_model = EmbeddingModel(id=uuid4(), title="model", lang="mul")
        self.other_embedding_model = EmbeddingModel(
            id=uuid4(), title="other", lang="mul"
        )
        with Session(self.engine, expire_on_commit=False) as s:
            s.add_all(
                [
                    self.category,
                    self.other_category,
                    self.parent_corpus,
                    self.child_corpus,
                    self.inactive_corpus,
                    self.other_corpus,
                    self.embedding_model,
                    self.other_embedding_model,
                ]
            )
            s.commit()

    def _new_session(self):
        return Session(self.engine, expire_on_commit=False)

    def _add_document(self, title: str, corpus: Corpus) -> WeLearnDocument:
        doc = WeLearnDocument(
            id=uuid4(),
            url=f"https://example.org/{title}",
            corpus_id=corpus.id,
            title=title,
            lang="en",
            full_content="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi volutpat.",
            description="desc",
            details={},
        )
        with Session(self.engine, expire_on_commit=False) as s:
            s.add(doc)
            s.commit()
        return doc

    def _add_slice(self, doc: WeLearnDocument, order: int) -> DocumentSlice:
        wl_slice = DocumentSlice(
            id=uuid4(),
            document_id=doc.id,
            body=f"body {order}",
            order_sequence=order,
            embedding_model_name="model",
            embedding_model_id=self.embedding_model.id,
        )
        with Session(self.engine, expire_on_commit=False) as s:
            s.add(wl_slice)
            s.commit()
        return wl_slice

    def test_get_collections_info_sync_returns_active_parent_corpora(self):
        rows = get_collections_info_sync()

        by_name = {r.source_name: r for r in rows}
        # child corpus (has a parent) and inactive corpus are excluded
        self.assertCountEqual(by_name.keys(), ["parent", "other"])
        self.assertEqual(by_name["parent"].category_name, "cat_title")
        self.assertEqual(by_name["other"].category_name, "other_cat")
        self.assertTrue(all(r.is_active for r in rows))

    def test_get_nb_docs_sync_returns_first_row(self):
        # the mapped class is read-only (it maps a view): insert through SQL
        with self.engine.begin() as conn:
            conn.execute(
                sqlalchemy.insert(QtyDocumentInQdrant.__table__).values(
                    document_in_qdrant=42
                )
            )

        result = get_nb_docs_sync()

        self.assertEqual(result[0], 42)

    def test_get_nb_docs_sync_returns_none_when_empty(self):
        self.assertIsNone(get_nb_docs_sync())

    def test_get_corpus_and_sub_corpus_repartition_with_parent(self):
        relation = get_corpus_and_sub_corpus_repartition(corpus=self.child_corpus)

        self.assertEqual(relation.corpus.id, self.parent_corpus.id)
        self.assertEqual(relation.sub_corpus.id, self.child_corpus.id)
        self.assertIn(self.child_corpus.id, queries.main_corpus_dict)

    def test_get_corpus_and_sub_corpus_repartition_without_parent(self):
        relation = get_corpus_and_sub_corpus_repartition(corpus=self.parent_corpus)

        self.assertEqual(relation.corpus.id, self.parent_corpus.id)
        self.assertIsNone(relation.sub_corpus)
        self.assertIsNone(queries.main_corpus_dict[self.parent_corpus.id])

    def test_get_documents_according_slice_ids_groups_by_document(self):
        doc_a = self._add_document("a", self.parent_corpus)
        doc_b = self._add_document("b", self.child_corpus)
        a1 = self._add_slice(doc_a, 0)
        a2 = self._add_slice(doc_a, 1)
        b1 = self._add_slice(doc_b, 0)
        self._add_slice(doc_b, 1)  # not requested

        result = get_documents_according_slice_ids([a1.id, a2.id, b1.id])

        by_doc_id = {doc.id: slices for doc, slices in result.items()}
        self.assertCountEqual(by_doc_id.keys(), [doc_a.id, doc_b.id])
        self.assertCountEqual([s.id for s in by_doc_id[doc_a.id]], [a1.id, a2.id])
        self.assertEqual([s.id for s in by_doc_id[doc_b.id]], [b1.id])
        # corpus is eagerly loaded so it is usable outside the session
        corpora = {doc.id: doc.corpus.source_name for doc in result}
        self.assertEqual(corpora[doc_b.id], "child")

    def test_get_documents_according_slice_ids_empty(self):
        self.assertEqual(get_documents_according_slice_ids([]), {})


class GetSubjectTests(unittest.TestCase):
    """get_subject relies on postgres-only column types (ARRAY, ENUM):
    check the query it builds with a mocked session instead of sqlite."""

    @patch(QUERIES_SESSION_MAKER)
    def test_get_subject_returns_first_match(self, mocked_session_maker):
        session = mocked_session_maker.return_value.__enter__.return_value
        expected = ContextDocument(title="maths")
        session.query.return_value.filter.return_value.first.return_value = expected
        model_ids = [uuid4()]

        result = get_subject(subject="maths", embedding_models_ids=model_ids)

        self.assertIs(result, expected)
        session.query.assert_called_once_with(ContextDocument)
        filter_args = session.query.return_value.filter.call_args.args
        self.assertEqual(len(filter_args), 3)
        compiled = [
            str(c.compile(compile_kwargs={"literal_binds": True}))
            for c in filter_args[:2]
        ]
        self.assertIn("'subject'", compiled[0])
        self.assertIn("'maths'", compiled[1])

    @patch(QUERIES_SESSION_MAKER)
    def test_get_subject_returns_none_when_not_found(self, mocked_session_maker):
        session = mocked_session_maker.return_value.__enter__.return_value
        session.query.return_value.filter.return_value.first.return_value = None

        self.assertIsNone(get_subject(subject="none", embedding_models_ids=[uuid4()]))
