import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import sqlalchemy
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker
from welearn_database.data.models import (
    Base,
    ContextDocument,
    DbSchemaEnum,
    EmbeddingModel,
)

from src.app.core.config import settings
from src.app.micro_learning.api.router import get_full_journey
from src.app.micro_learning.infra.queries import get_context_documents, get_subjects
from src.app.micro_learning.services.helpers import (
    choose_readability_according_journey_section_type,
    collection_and_model_id_according_lang,
)
from src.app.search.services.search import get_search_service
from src.app.shared.domain.collections import Collection
from src.app.shared.domain.documents import JourneySectionType
from src.main import app

ROUTER = "src.app.micro_learning.api.router"
HELPERS = "src.app.micro_learning.services.helpers"
COLLECTION = Collection(name="collection_welearn_en_model", lang="en", model="model")


class ReadabilityTests(unittest.TestCase):
    def test_introduction_is_easy_range(self):
        r = choose_readability_according_journey_section_type(
            JourneySectionType.INTRODUCTION
        )
        self.assertEqual((r.gte, r.lte), (60, 100))

    def test_target_is_hard_range(self):
        r = choose_readability_according_journey_section_type(JourneySectionType.TARGET)
        self.assertEqual((r.gte, r.lte), (0, 60))

    def test_unknown_type_raises(self):
        with self.assertRaises(NotImplementedError):
            choose_readability_according_journey_section_type("subject")


class CollectionAndModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sp = mock.MagicMock()
        self.sp.get_collection_by_language = mock.AsyncMock(return_value=COLLECTION)
        self.models = [EmbeddingModel(id=uuid.uuid4(), title="model", lang="en")]

    async def test_lang_is_forwarded(self):
        with mock.patch(
            f"{HELPERS}.get_embeddings_model_id_according_name",
            return_value=self.models,
        ) as get_models:
            result = await collection_and_model_id_according_lang(lang="fr", sp=self.sp)

        self.assertEqual(result, (COLLECTION, self.models))
        self.sp.get_collection_by_language.assert_awaited_once_with("fr")
        get_models.assert_called_once_with("model")

    async def test_default_language_when_lang_is_none(self):
        with mock.patch(
            f"{HELPERS}.get_embeddings_model_id_according_name",
            return_value=self.models,
        ):
            await collection_and_model_id_according_lang(lang=None, sp=self.sp)

        self.sp.get_collection_by_language.assert_awaited_once_with()

    async def test_no_collection_raises_404(self):
        self.sp.get_collection_by_language.return_value = None

        with self.assertRaises(HTTPException) as ctx:
            await collection_and_model_id_according_lang(lang="xx", sp=self.sp)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("'xx'", ctx.exception.detail)

    async def test_unknown_embedding_model_raises(self):
        with mock.patch(
            f"{HELPERS}.get_embeddings_model_id_according_name", return_value=[]
        ):
            with self.assertRaises(ValueError) as ctx:
                await collection_and_model_id_according_lang(lang="en", sp=self.sp)
        self.assertIn("model", str(ctx.exception))


class MicroLearningQueriesTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        with self.engine.begin() as conn:
            for schema in DbSchemaEnum:  # type: ignore
                conn.execute(sqlalchemy.text(f"ATTACH ':memory:' AS {schema.value}"))
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()

    def _insert_context_doc(self, conn, title, context_type, model_id):
        # The postgres ARRAY column (sdg_related) cannot be bound from Python on
        # sqlite, so rows are inserted with raw SQL.
        conn.execute(
            sqlalchemy.text(
                "INSERT INTO document_related.context_document (id, url, title, "
                "full_content, embedding_model_id, sdg_related, created_at, "
                "updated_at, embedding, context_type) VALUES (:id, :url, :title, "
                "'content', :model_id, '{1}', '2024-01-01 00:00:00', "
                "'2024-01-01 00:00:00', x'00', :context_type)"
            ),
            {
                "id": uuid.uuid4().hex,
                "url": f"https://example.org/{title}",
                "title": title,
                "model_id": model_id.hex,
                "context_type": context_type,
            },
        )

    def test_get_subjects_filters_by_type_and_model(self):
        model_a, model_b = uuid.uuid4(), uuid.uuid4()
        with self.session_factory() as s:
            s.add_all(
                [
                    EmbeddingModel(id=model_a, title="a", lang="en"),
                    EmbeddingModel(id=model_b, title="b", lang="fr"),
                ]
            )
            s.commit()
        with self.engine.begin() as conn:
            self._insert_context_doc(conn, "subject_a", "subject", model_a)
            self._insert_context_doc(conn, "subject_b", "subject", model_b)
            self._insert_context_doc(conn, "intro_a", "introduction", model_a)

        with mock.patch(
            "src.app.micro_learning.infra.queries.session_maker", self.session_factory
        ):
            subjects = get_subjects(embedding_models_ids=[model_a])
            none_found = get_subjects(embedding_models_ids=[uuid.uuid4()])

        self.assertEqual([d.title for d in subjects], ["subject_a"])
        self.assertEqual(none_found, [])

    def test_get_context_documents_builds_expected_filter(self):
        # sdg_related.contains() is a PostgreSQL array operator, so the query is
        # checked through a mocked session and compiled for PostgreSQL.
        session = mock.MagicMock()
        expected = [mock.sentinel.doc]
        session.query.return_value.filter.return_value.all.return_value = expected
        session_maker = mock.MagicMock()
        session_maker.return_value.__enter__.return_value = session
        model_id = uuid.uuid4()

        with mock.patch(
            "src.app.micro_learning.infra.queries.session_maker", session_maker
        ):
            result = get_context_documents(
                journey_part=["introduction", "target"],
                sdg=4,
                embedding_models_ids=[model_id],
            )

        self.assertIs(result, expected)
        session.query.assert_called_once_with(ContextDocument)
        clauses = session.query.return_value.filter.call_args.args
        self.assertEqual(len(clauses), 3)
        compiled = [
            str(
                c.compile(
                    dialect=postgresql.dialect(),
                    compile_kwargs={"literal_binds": False},
                )
            )
            for c in clauses
        ]
        self.assertIn("context_type IN", compiled[0])
        self.assertIn("sdg_related @>", compiled[1])
        self.assertIn("embedding_model_id IN", compiled[2])


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class MicroLearningRouterErrorsTests(unittest.TestCase):
    def setUp(self):
        self.sp = mock.MagicMock()
        self.sp.search = mock.AsyncMock(return_value=[])
        app.dependency_overrides[get_search_service] = lambda: self.sp
        self.client = TestClient(app)
        self.models = [EmbeddingModel(id=uuid.uuid4(), title="model", lang="en")]

    def tearDown(self):
        app.dependency_overrides.pop(get_search_service, None)

    def test_subject_list_empty_returns_404(self):
        with (
            mock.patch(
                f"{ROUTER}.collection_and_model_id_according_lang",
                new=mock.AsyncMock(return_value=(COLLECTION, self.models)),
            ),
            mock.patch(f"{ROUTER}.get_subjects", return_value=[]) as get_subj,
        ):
            response = self.client.get(
                f"{settings.API_V1_STR}/micro_learning/subject_list",
                params={"lang": "en"},
                headers={"X-API-Key": "test"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "No subjects found.")
        get_subj.assert_called_once_with(
            embedding_models_ids=[m.id for m in self.models]
        )

    def test_full_journey_unknown_sdg_returns_404(self):
        with (
            mock.patch(
                f"{ROUTER}.collection_and_model_id_according_lang",
                new=mock.AsyncMock(return_value=(COLLECTION, self.models)),
            ),
            mock.patch(f"{ROUTER}.get_context_documents", return_value=[]),
        ):
            response = self.client.get(
                f"{settings.API_V1_STR}/micro_learning/full_journey",
                params={"sdg": 42, "subject": "s"},
                headers={"X-API-Key": "test"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"], "SDG '42' not found in meta documents."
        )


class FullJourneyInvalidDataTests(unittest.IsolatedAsyncioTestCase):
    """Data errors in get_full_journey are raised as plain Python exceptions."""

    def setUp(self):
        self.sp = mock.MagicMock()
        self.sp.search = mock.AsyncMock(return_value=[])
        self.sp.flavored_with_subject.return_value = [0.1]
        models = [EmbeddingModel(id=uuid.uuid4(), title="model", lang="en")]
        patches = [
            mock.patch(
                f"{ROUTER}.collection_and_model_id_according_lang",
                new=mock.AsyncMock(return_value=(COLLECTION, models)),
            ),
            mock.patch(f"{ROUTER}.convert_embedding_bytes", return_value=[0.1]),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _patch_docs(self, context_docs, subject):
        ctx = mock.patch(f"{ROUTER}.get_context_documents", return_value=context_docs)
        subj = mock.patch(f"{ROUTER}.get_subject", return_value=subject)
        ctx.start()
        subj.start()
        self.addCleanup(ctx.stop)
        self.addCleanup(subj.stop)

    async def _call(self):
        return await get_full_journey(sdg=1, subject="climate", lang="en", sp=self.sp)

    def _sdg_doc(self, embedding=b"\x00", context_type="introduction"):
        return SimpleNamespace(
            title="sdg doc",
            full_content="content",
            embedding=embedding,
            context_type=context_type,
        )

    async def test_missing_subject_raises(self):
        self._patch_docs([self._sdg_doc()], None)
        with self.assertRaises(ValueError) as ctx:
            await self._call()
        self.assertIn("Subject 'climate' not found", str(ctx.exception))
        self.sp.search.assert_not_awaited()

    async def test_subject_embedding_not_bytes_raises(self):
        self._patch_docs([self._sdg_doc()], SimpleNamespace(embedding="text"))
        with self.assertRaises(ValueError) as ctx:
            await self._call()
        self.assertIn("received type: str", str(ctx.exception))

    async def test_sdg_embedding_not_bytes_raises(self):
        self._patch_docs(
            [self._sdg_doc(embedding=[1, 2])], SimpleNamespace(embedding=b"\x00")
        )
        with self.assertRaises(ValueError) as ctx:
            await self._call()
        self.assertIn("received type: list", str(ctx.exception))
        self.sp.search.assert_not_awaited()

    async def test_unknown_context_type_raises(self):
        self._patch_docs(
            [self._sdg_doc(context_type="subject")],
            SimpleNamespace(embedding=b"\x00"),
        )
        with self.assertRaises(NotImplementedError) as ctx:
            await self._call()
        self.assertIn("'subject'", str(ctx.exception))

    async def test_sections_without_search_results_stay_empty(self):
        self._patch_docs(
            [self._sdg_doc(context_type="target")],
            SimpleNamespace(embedding=b"\x00"),
        )
        result = await self._call()
        self.assertEqual(result, {"target": []})
        self.sp.search.assert_awaited_once()
        self.assertEqual(
            self.sp.search.await_args.kwargs["collection_info"], COLLECTION.name
        )
        self.assertEqual(self.sp.search.await_args.kwargs["nb_results"], 10)
