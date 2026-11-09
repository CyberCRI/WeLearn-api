import os
import uuid
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, mock

import numpy as np
import torch
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client.http import models as http_models
from welearn_database.data.models import Corpus, DocumentSlice, WeLearnDocument

from src.app.search.domain.models import EnhancedSearchQuery, SearchMethods
from src.app.search.services import search as search_module
from src.app.search.services.search import (
    SearchService,
    concatenate_same_doc_id_slices,
    get_subject_vector,
)
from src.app.shared.domain.collections import Collection, CorpusRelation
from src.app.shared.domain.exceptions import CollectionNotFoundError, ModelNotFoundError

SEARCH_MODULE = "src.app.search.services.search"
LONG_CONTENT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi volutpat."
)


class FakeTokenizer:
    model_max_length = 512

    def __init__(self):
        self.calls = []

    def __call__(self, inputs, padding, truncation, return_tensors):
        self.calls.append((inputs, padding, truncation, return_tensors))
        return {"input_ids": torch.zeros((len(inputs), 3), dtype=torch.long)}


class FakeModel:
    """Return a (batch, seq, dim) tensor whose CLS token is [3, 4] for each input."""

    def __init__(self):
        self.received = None

    def __call__(self, **kwargs):
        self.received = kwargs
        batch = kwargs["input_ids"].shape[0]
        hidden = torch.zeros((batch, 3, 2))
        hidden[:, 0] = torch.tensor([3.0, 4.0])
        return (hidden,)


class ModelCacheMixin:
    """SearchService.model is a class-level cache: isolate it per test."""

    def setUp(self):
        patcher = mock.patch.dict(SearchService.model, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = mock.MagicMock()
        self.client.query_points = mock.AsyncMock()
        self.client.query_points_groups = mock.AsyncMock()
        self.service = SearchService(client=self.client)


class FlavoredWithSubjectTests(TestCase):
    def test_adds_weighted_subject_embedding(self):
        result = SearchService.flavored_with_subject(
            sdg_emb=np.array([1.0, 2.0]),
            subject_emb=np.array([0.5, -1.0]),
            discipline_factor=2,
        )
        np.testing.assert_allclose(result, [2.0, 0.0])

    def test_default_factor_is_two(self):
        result = SearchService.flavored_with_subject(np.zeros(2), np.ones(2))
        np.testing.assert_allclose(result, [2.0, 2.0])


class CollectionTests(ModelCacheMixin, IsolatedAsyncioTestCase):
    async def test_get_collection_by_language_not_found(self):
        self.service.collections = ("collection_welearn_en_model",)

        with self.assertRaises(CollectionNotFoundError) as ctx:
            await self.service.get_collection_by_language("fr")

        self.assertIn("fr", ctx.exception.message)

    async def test_get_collection_by_language_uses_cached_collections(self):
        self.service.collections = ("collection_welearn_mul_model",)
        self.client.get_collections = mock.AsyncMock()

        collection = await self.service.get_collection_by_language("mul")

        self.assertEqual(collection.model, "model")
        self.client.get_collections.assert_not_called()


class SplitInputTests(ModelCacheMixin, TestCase):
    def test_invalid_seq_len_raises(self):
        with self.assertRaises(ValueError):
            self.service._split_input_seq_len(0, "some input")

    def test_short_input_is_returned_as_is(self):
        self.assertEqual(
            self.service._split_input_seq_len(100, "short input"), ["short input"]
        )

    def test_long_input_is_split_on_words(self):
        result = self.service._split_input_seq_len(10, "aaaa bbbb cccc dddd eeeeee")

        self.assertEqual(result, ["aaaa bbbb", "cccc dddd", "eeeeee"])
        self.assertTrue(all(len(chunk) <= 10 for chunk in result))


class GetModelTests(ModelCacheMixin, TestCase):
    def test_returns_cached_model_without_loading(self):
        cached = {"max_seq_length": 1, "instance": "m", "tokenizer": "t"}
        SearchService.model["cached-model"] = cached

        with mock.patch(f"{SEARCH_MODULE}.AutoModel") as auto_model:
            self.assertIs(self.service._get_model("cached-model"), cached)
            auto_model.from_pretrained.assert_not_called()

    @mock.patch(f"{SEARCH_MODULE}.os.path.isdir", return_value=True)
    @mock.patch(f"{SEARCH_MODULE}.AutoTokenizer")
    @mock.patch(f"{SEARCH_MODULE}.AutoModel")
    def test_loads_model_and_tokenizer_and_caches_them(
        self, auto_model, auto_tokenizer, _isdir
    ):
        tokenizer = mock.MagicMock(model_max_length=256)
        auto_tokenizer.from_pretrained.return_value = tokenizer
        model = auto_model.from_pretrained.return_value

        result = self.service._get_model("my-model")

        expected_path = "../models/embedding/my-model/"
        auto_model.from_pretrained.assert_called_once_with(expected_path)
        auto_tokenizer.from_pretrained.assert_called_once_with(expected_path)
        model.eval.assert_called_once()
        self.assertEqual(
            result,
            {"max_seq_length": 256, "instance": model, "tokenizer": tokenizer},
        )
        self.assertIs(SearchService.model["my-model"], result)

    @mock.patch(f"{SEARCH_MODULE}.os.path.isdir", return_value=False)
    @mock.patch(f"{SEARCH_MODULE}.AutoModel")
    def test_missing_model_dir_logs_the_resolved_path(self, auto_model, _isdir):
        with (
            self.assertLogs(SEARCH_MODULE, level="ERROR") as logs,
            self.assertRaises(ModelNotFoundError) as ctx,
        ):
            self.service._get_model("missing")

        auto_model.from_pretrained.assert_not_called()
        self.assertNotIn("missing", SearchService.model)
        # the client sees the model name, never the filesystem path
        self.assertEqual(
            ctx.exception.message, "Embedding model 'missing' is not available"
        )
        self.assertEqual(ctx.exception.msg_code, "MODEL_NOT_FOUND")
        log = "\n".join(logs.output)
        self.assertIn(os.path.abspath("../models/embedding/missing/"), log)
        self.assertIn(f"cwd={os.getcwd()}", log)
        self.assertIn("reason=directory not found", log)

    @mock.patch(f"{SEARCH_MODULE}.os.path.isdir", return_value=True)
    @mock.patch(f"{SEARCH_MODULE}.AutoTokenizer")
    @mock.patch(f"{SEARCH_MODULE}.AutoModel")
    def test_load_failure_raises_model_not_found_with_reason(
        self, auto_model, auto_tokenizer, _isdir
    ):
        for error in (OSError("no config.json"), ValueError("bad repo id")):
            with self.subTest(error=type(error).__name__):
                auto_model.from_pretrained.side_effect = error

                with (
                    self.assertLogs(SEARCH_MODULE, level="ERROR") as logs,
                    self.assertRaises(ModelNotFoundError) as ctx,
                ):
                    self.service._get_model("broken")

                self.assertNotIn("broken", SearchService.model)
                self.assertIs(ctx.exception.__cause__, error)
                self.assertEqual(
                    ctx.exception.message,
                    "Embedding model 'broken' could not be loaded",
                )
                log = "\n".join(logs.output)
                self.assertIn(os.path.abspath("../models/embedding/broken/"), log)
                self.assertIn(f"reason={type(error).__name__}: {error}", log)


class ComputeEmbeddingsTests(ModelCacheMixin, TestCase):
    def test_returns_normalized_cls_embeddings(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()

        result = self.service._compute_embeddings(model, tokenizer, ["a", "b"])

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, [[0.6, 0.8], [0.6, 0.8]], rtol=1e-6)
        self.assertEqual(tokenizer.calls, [(["a", "b"], True, True, "pt")])
        self.assertIn("input_ids", model.received)


class EmbedQueryTests(ModelCacheMixin, IsolatedAsyncioTestCase):
    def _cache_model(self, name="m", seq_len=512):
        SearchService.model[name] = {
            "max_seq_length": seq_len,
            "instance": "instance",
            "tokenizer": "tokenizer",
        }

    async def test_embed_query_returns_mean_of_chunks(self):
        self._cache_model(seq_len=10)
        with mock.patch.object(
            self.service,
            "_compute_embeddings",
            return_value=np.array([[1.0, 0.0], [0.0, 1.0]]),
        ) as compute:
            result = await self.service._embed_query("aaaa bbbb cccc dddd", "m")

        compute.assert_called_once_with(
            "instance", "tokenizer", ["aaaa bbbb", "cccc dddd"]
        )
        np.testing.assert_allclose(result, [0.5, 0.5])

    async def test_embed_query_loads_model_when_not_cached(self):
        def load(curr_model):
            self._cache_model(curr_model)
            return SearchService.model[curr_model]

        with (
            mock.patch.object(self.service, "_get_model", side_effect=load) as get_m,
            mock.patch.object(
                self.service, "_compute_embeddings", return_value=np.ones((1, 2))
            ),
        ):
            result = await self.service._embed_query("query", "new-model")

        get_m.assert_called_once_with("new-model")
        np.testing.assert_allclose(result, [1.0, 1.0])

    async def test_embed_query_wraps_errors(self):
        self._cache_model()
        with mock.patch.object(
            self.service, "_compute_embeddings", side_effect=Exception("boom")
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await self.service._embed_query("query", "m")

        self.assertEqual(
            ctx.exception.args, ("Not able to create embed", "EMBED_ERROR")
        )

    async def test_get_query_embed_without_subject(self):
        embedding = np.array([0.1, 0.2])
        with (
            mock.patch.object(
                self.service, "_embed_query", mock.AsyncMock(return_value=embedding)
            ) as embed,
            mock.patch.object(SearchService, "flavored_with_subject") as flavored,
        ):
            result = await self.service.get_query_embed(model="m", query="q")

        embed.assert_awaited_once_with("q", "m")
        flavored.assert_not_called()
        self.assertIs(result, embedding)

    async def test_get_query_embed_with_subject_flavors_embedding(self):
        # B-1: real ndarray subject vector, flavored_with_subject not mocked
        embedding = np.array([0.1, 0.2])
        subject_vector = np.array([1.0, -1.0], dtype=np.float32)
        with mock.patch.object(
            self.service, "_embed_query", mock.AsyncMock(return_value=embedding)
        ):
            result = await self.service.get_query_embed(
                model="m",
                query="q",
                subject_vector=subject_vector,
                subject_influence_factor=3.0,
            )

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(result, [3.1, -2.8], rtol=1e-6)


class QdrantSearchTests(ModelCacheMixin, IsolatedAsyncioTestCase):
    async def test_search_returns_points_and_forwards_params(self):
        points = [http_models.ScoredPoint(id=1, version=1, score=0.9)]
        self.client.query_points.return_value = SimpleNamespace(points=points)

        result = await self.service.search(
            collection_info="col", embedding=[0.1], filters="f", nb_results=5
        )

        self.assertEqual(result, points)
        kwargs = self.client.query_points.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], "col")
        self.assertEqual(kwargs["query_filter"], "f")
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["score_threshold"], 0.5)
        self.assertTrue(kwargs["with_vectors"])
        self.assertEqual(kwargs["with_payload"], self.service.payload_keys)

    async def test_search_returns_empty_on_response_error(self):
        self.client.query_points.side_effect = (
            qdrant_exceptions.ResponseHandlingException(Exception("down"))
        )

        result = await self.service.search(collection_info="col", embedding=[0.1])

        self.assertEqual(result, [])

    async def test_search_group_by_document_returns_first_hit_of_each_group(self):
        hit_a = http_models.ScoredPoint(id=1, version=1, score=0.9)
        hit_b = http_models.ScoredPoint(id=2, version=1, score=0.8)
        self.client.query_points_groups.return_value = SimpleNamespace(
            groups=[
                SimpleNamespace(hits=[hit_a]),
                SimpleNamespace(hits=[hit_b]),
            ]
        )

        result = await self.service.search_group_by_document(
            collection_info="col", embedding=np.array([0.1]), nb_results=7
        )

        self.assertEqual(result, [hit_a, hit_b])
        kwargs = self.client.query_points_groups.call_args.kwargs
        self.assertEqual(kwargs["group_by"], "document_id")
        self.assertEqual(kwargs["group_size"], 1)
        self.assertEqual(kwargs["limit"], 7)
        self.assertEqual(kwargs["collection_name"], "col")

    async def test_search_group_by_document_returns_empty_on_errors(self):
        for exc in (
            qdrant_exceptions.ApiException("api"),
            qdrant_exceptions.ResponseHandlingException(Exception("down")),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.client.query_points_groups.side_effect = exc
                result = await self.service.search_group_by_document(
                    collection_info="col", embedding=np.array([0.1])
                )
                self.assertEqual(result, [])

    async def test_simple_search_handler_embeds_and_searches(self):
        model = {"instance": "instance", "tokenizer": "tokenizer"}
        embedding = np.array([[0.1, 0.2]])
        with (
            mock.patch.object(self.service, "_get_model", return_value=model) as get_m,
            mock.patch.object(
                self.service, "_compute_embeddings", return_value=embedding
            ) as compute,
            mock.patch.object(
                self.service, "search", mock.AsyncMock(return_value=["hit"])
            ) as search,
        ):
            for query, expected_input in (
                ("water", ["water"]),
                (["a", "b"], ["a", "b"]),
            ):
                with self.subTest(query=query):
                    result = await self.service.simple_search_handler(
                        EnhancedSearchQuery(query=query)
                    )
                    self.assertEqual(result, ["hit"])
                    compute.assert_called_with("instance", "tokenizer", expected_input)

        get_m.assert_called_with(curr_model="granite-embedding-107m-multilingual")
        search.assert_awaited_with(
            collection_info="collection_welearn_mul_granite-embedding-107m-multilingual",
            embedding=embedding,
            nb_results=30,
        )


def _make_doc(title: str) -> WeLearnDocument:
    return WeLearnDocument(
        id=uuid.uuid4(),
        url=f"https://example.org/{title}",
        title=title,
        lang="en",
        full_content=LONG_CONTENT,
        description=f"desc {title}",
        details={},
    )


def _make_slice(doc: WeLearnDocument, order: int, body: str) -> DocumentSlice:
    return DocumentSlice(
        id=uuid.uuid4(), document_id=doc.id, order_sequence=order, body=body
    )


def _point(wl_slice: DocumentSlice, score: float, vector: list[float]):
    return http_models.ScoredPoint(
        id=str(wl_slice.id), version=1, score=score, vector=vector
    )


class SearchHandlerTests(ModelCacheMixin, IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.collection = Collection(
            lang="mul", model="m", name="collection_welearn_mul_m"
        )
        self.embedding = np.array([0.1, 0.2])

        self.doc_a = _make_doc("a")
        self.doc_b = _make_doc("b")
        self.a1 = _make_slice(self.doc_a, 1, "a-second")
        self.a0 = _make_slice(self.doc_a, 0, "a-first")
        self.b0 = _make_slice(self.doc_b, 0, "b-first")
        self.points = [
            _point(self.a1, 0.9, [1.0, 0.0]),
            _point(self.a0, 0.8, [1.0, 0.0]),
            _point(self.b0, 0.7, [0.0, 1.0]),
        ]
        self.db_ret = {self.doc_a: [self.a1, self.a0], self.doc_b: [self.b0]}
        self.sdg_map = {
            self.a1.id: [SimpleNamespace(sdg_number=4)],
            self.a0.id: [SimpleNamespace(sdg_number=4), SimpleNamespace(sdg_number=13)],
        }
        self.corpus = Corpus(source_name="corpus")
        self.sub_corpus = Corpus(source_name="sub")

        self.patches = {
            "subject_vector": mock.patch(
                f"{SEARCH_MODULE}.get_subject_vector",
                return_value=np.array([0.5, 0.5]),
            ),
            "docs": mock.patch(
                f"{SEARCH_MODULE}.get_documents_according_slice_ids",
                return_value=self.db_ret,
            ),
            "sdgs": mock.patch(
                f"{SEARCH_MODULE}.batch_fetch_sdgs", return_value=self.sdg_map
            ),
            "corpus": mock.patch(
                f"{SEARCH_MODULE}.get_corpus_and_sub_corpus_repartition",
                return_value=CorpusRelation(
                    corpus=self.corpus, sub_corpus=self.sub_corpus
                ),
            ),
            "collection": mock.patch.object(
                self.service,
                "get_collection_by_language",
                mock.AsyncMock(return_value=self.collection),
            ),
            "embed": mock.patch.object(
                self.service,
                "get_query_embed",
                mock.AsyncMock(return_value=self.embedding),
            ),
            "search": mock.patch.object(
                self.service, "search", mock.AsyncMock(return_value=self.points)
            ),
            "group": mock.patch.object(
                self.service,
                "search_group_by_document",
                mock.AsyncMock(return_value=self.points),
            ),
        }
        self.mocks = {name: p.start() for name, p in self.patches.items()}
        for p in self.patches.values():
            self.addCleanup(p.stop)

    def _query(self, **kwargs):
        defaults = dict(
            query="water",
            corpora=("corpus",),
            subject="maths",
            influence_factor=3,
            nb_results=12,
            sdg_filter=[4],
        )
        defaults.update(kwargs)
        return EnhancedSearchQuery(**defaults)

    @staticmethod
    def _filter_keys(filters) -> set[str]:
        return {condition.key for condition in filters.must}

    async def test_by_slices_returns_ranked_documents(self):
        result = await self.service.search_handler(
            background_tasks=mock.MagicMock(), qp=self._query()
        )

        self.mocks["collection"].assert_awaited_once_with(lang="mul")
        self.mocks["subject_vector"].assert_called_once_with("maths", "m")
        self.mocks["embed"].assert_awaited_once()
        embed_kwargs = self.mocks["embed"].await_args.kwargs
        self.assertEqual(embed_kwargs["model"], "m")
        self.assertEqual(embed_kwargs["query"], "water")
        self.assertEqual(embed_kwargs["subject_influence_factor"], 3)
        np.testing.assert_array_equal(embed_kwargs["subject_vector"], [0.5, 0.5])
        search_kwargs = self.mocks["search"].call_args.kwargs
        self.assertEqual(search_kwargs["collection_info"], "collection_welearn_mul_m")
        self.assertEqual(search_kwargs["nb_results"], 12)
        self.assertIs(search_kwargs["embedding"], self.embedding)
        self.assertIn("slice_sdg", self._filter_keys(search_kwargs["filters"]))
        self.assertNotIn("document_sdg", self._filter_keys(search_kwargs["filters"]))
        self.mocks["group"].assert_not_called()
        self.mocks["docs"].assert_called_once_with([p.id for p in self.points])
        self.assertCountEqual(
            self.mocks["sdgs"].call_args.args[0], [self.a1.id, self.a0.id, self.b0.id]
        )

        self.assertEqual(
            [r.payload.document_id for r in result], [self.doc_a.id, self.doc_b.id]
        )
        first = result[0]
        self.assertAlmostEqual(first.score, 0.85)
        self.assertEqual(first.payload.slice_content, "a-first\n\n[...]a-second")
        self.assertEqual(first.payload.document_corpus, "corpus")
        self.assertEqual(first.payload.document_sub_corpus, "sub")
        self.assertEqual(first.payload.document_sdg, [4, 13])
        self.assertEqual(result[1].payload.document_sdg, [])

    async def test_subject_flavors_query_embedding_end_to_end(self):
        # B-1: only the DB and the embedding model are mocked, so the subject
        # vector goes through get_subject_vector, get_query_embed and
        # flavored_with_subject for real
        self.patches["subject_vector"].stop()
        self.patches["embed"].stop()
        subject = np.array([1.0, -1.0], dtype=np.float32)
        with (
            mock.patch(
                f"{SEARCH_MODULE}.get_embeddings_model_id_according_name",
                return_value=[SimpleNamespace(id=uuid.uuid4())],
            ),
            mock.patch(
                f"{SEARCH_MODULE}.get_subject",
                return_value=SimpleNamespace(embedding=subject.tobytes()),
            ),
            mock.patch.object(
                self.service,
                "_embed_query",
                mock.AsyncMock(return_value=np.array([0.1, 0.2])),
            ),
        ):
            result = await self.service.search_handler(
                background_tasks=mock.MagicMock(), qp=self._query()
            )

        embedding = self.mocks["search"].call_args.kwargs["embedding"]
        self.assertIsInstance(embedding, np.ndarray)
        np.testing.assert_allclose(embedding, [3.1, -2.8], rtol=1e-6)
        self.assertEqual(len(result), 2)

    async def test_by_document_uses_grouped_search_and_document_sdg(self):
        result = await self.service.search_handler(
            background_tasks=mock.MagicMock(),
            qp=self._query(),
            method=SearchMethods.BY_DOCUMENT,
        )

        self.mocks["search"].assert_not_called()
        group_kwargs = self.mocks["group"].call_args.kwargs
        self.assertIn("document_sdg", self._filter_keys(group_kwargs["filters"]))
        self.assertNotIn("slice_sdg", self._filter_keys(group_kwargs["filters"]))
        self.assertEqual(len(result), 2)

    async def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            await self.service.search_handler(
                background_tasks=mock.MagicMock(), qp=self._query(), method="unknown"
            )
        self.mocks["docs"].assert_not_called()

    async def test_duplicates_are_removed(self):
        twin = _make_doc("a")
        twin.description = self.doc_a.description
        twin_slice = _make_slice(twin, 0, "twin")
        self.points.append(_point(twin_slice, 0.6, [0.5, 0.5]))
        self.db_ret[twin] = [twin_slice]
        background_tasks = mock.MagicMock()

        result = await self.service.search_handler(
            background_tasks=background_tasks, qp=self._query()
        )

        self.assertEqual(len(result), 2)
        background_tasks.add_task.assert_called_once()

    async def test_duplicate_removal_failure_still_returns_results(self):
        with mock.patch(f"{SEARCH_MODULE}.DataQualityChecker") as dqc:
            dqc.return_value.remove_duplicates.side_effect = RuntimeError("db down")
            result = await self.service.search_handler(
                background_tasks=mock.MagicMock(), qp=self._query()
            )

        self.assertEqual(
            [r.payload.document_id for r in result], [self.doc_a.id, self.doc_b.id]
        )

    async def test_no_hits_returns_empty_list(self):
        self.mocks["search"].return_value = []
        self.mocks["docs"].return_value = {}
        self.mocks["sdgs"].return_value = {}

        result = await self.service.search_handler(
            background_tasks=mock.MagicMock(), qp=self._query()
        )

        self.assertEqual(result, [])


class ConcatenateSkipTests(TestCase):
    def test_results_without_payload_or_document_id_are_skipped(self):
        doc_id = uuid.uuid4()
        kept = SimpleNamespace(
            payload=SimpleNamespace(document_id=doc_id, slice_content="kept")
        )
        results = [
            SimpleNamespace(payload=None),
            SimpleNamespace(
                payload=SimpleNamespace(document_id=None, slice_content="x")
            ),
            kept,
        ]

        self.assertEqual(concatenate_same_doc_id_slices(results), [kept])


class GetSubjectVectorTests(TestCase):
    def test_no_subject_returns_none_without_db_calls(self):
        with mock.patch(
            f"{SEARCH_MODULE}.get_embeddings_model_id_according_name"
        ) as get_models:
            self.assertIsNone(get_subject_vector(None, "m"))
            self.assertIsNone(get_subject_vector("", "m"))
        get_models.assert_not_called()

    @mock.patch(f"{SEARCH_MODULE}.get_subject")
    @mock.patch(f"{SEARCH_MODULE}.get_embeddings_model_id_according_name")
    def test_unknown_embedding_model_returns_none(self, get_models, get_subject):
        get_models.return_value = [None]

        self.assertIsNone(get_subject_vector("maths", "m"))
        get_models.assert_called_once_with("m")
        get_subject.assert_not_called()

    @mock.patch(f"{SEARCH_MODULE}.get_subject", return_value=None)
    @mock.patch(f"{SEARCH_MODULE}.get_embeddings_model_id_according_name")
    def test_subject_not_in_db_returns_none(self, get_models, get_subject):
        model_id = uuid.uuid4()
        get_models.return_value = [SimpleNamespace(id=model_id), None]

        self.assertIsNone(get_subject_vector("maths", "m"))
        get_subject.assert_called_once_with(
            subject="maths", embedding_models_ids=[model_id]
        )

    @mock.patch(f"{SEARCH_MODULE}.get_subject")
    @mock.patch(f"{SEARCH_MODULE}.get_embeddings_model_id_according_name")
    def test_subject_found_returns_decoded_vector(self, get_models, get_subject):
        get_models.return_value = [SimpleNamespace(id=uuid.uuid4())]
        vector = np.array([0.25, -1.5, 3.0], dtype=np.float32)
        get_subject.return_value = SimpleNamespace(embedding=vector.tobytes())

        result = get_subject_vector("maths", "m")

        # an ndarray, not a list: flavored_with_subject computes `factor * vector`
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, vector)


class GetQdrantTests(IsolatedAsyncioTestCase):
    async def test_get_qdrant_and_search_service_factory(self):
        qdrant = object()
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(qdrant=qdrant))
        )

        self.assertIs(await search_module.get_qdrant(request), qdrant)
        service = await search_module.get_search_service(qdrant)
        self.assertIsInstance(service, SearchService)
        self.assertIs(service.client, qdrant)
