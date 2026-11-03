import hashlib
import unittest
import uuid
from types import SimpleNamespace

from qdrant_client.http.models import ScoredPoint

from src.app.shared.domain.collections import CorpusRelation
from src.app.shared.domain.documents import (
    ContainedSlice,
    EnhancedDocumentSlice,
    FullContainer,
)


def _make_document(**overrides):
    values = {
        "id": uuid.uuid4(),
        "title": "Doc title",
        "description": "Doc description",
        "external_id": "ext-1",
        "url": "https://example.org/doc",
        "details": {"author": "someone"},
        "lang": "en",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_slice(order, body):
    return SimpleNamespace(id=uuid.uuid4(), body=body, order_sequence=order)


def _make_point(slice_id, score, vector):
    return ScoredPoint(id=str(slice_id), version=1, score=score, vector=vector)


class ContainedSliceTests(unittest.TestCase):
    def test_attributes_are_stored(self):
        slice_id = uuid.uuid4()
        cs = ContainedSlice(
            id_=slice_id,
            content="content",
            order=3,
            vector=[0.1, 0.2],
            score=0.5,
            sdgs=[1, 2],
        )
        self.assertEqual(cs.id, slice_id)
        self.assertEqual(cs.content, "content")
        self.assertEqual(cs.order, 3)
        self.assertEqual(cs.vector, [0.1, 0.2])
        self.assertEqual(cs.score, 0.5)
        self.assertEqual(cs.sdgs, [1, 2])


class FullContainerTests(unittest.TestCase):
    def setUp(self):
        self.document = _make_document()
        self.slice_a = _make_slice(order=2, body="second part")
        self.slice_b = _make_slice(order=1, body="first part")
        # slice without a matching scored point: must be ignored
        self.slice_orphan = _make_slice(order=0, body="orphan")

        self.points = [
            _make_point(self.slice_a.id, 0.8, [1.0, 2.0, 3.0]),
            _make_point(self.slice_b.id, 0.4, [3.0, 4.0, 5.0]),
        ]
        self.sdg_map = {
            self.slice_a.id: [SimpleNamespace(sdg_number=5)],
            self.slice_b.id: [
                SimpleNamespace(sdg_number=5),
                SimpleNamespace(sdg_number=3),
            ],
        }
        self.corpus = SimpleNamespace(source_name="corpus_a")
        self.sub_corpus = SimpleNamespace(source_name="sub_corpus_a")

    def _container(self, sub_corpus=None, sdg_map=None):
        return FullContainer(
            wl_document=self.document,
            wl_slices=[self.slice_a, self.slice_orphan, self.slice_b],
            scored_points=self.points,
            sdg_map=self.sdg_map if sdg_map is None else sdg_map,
            corpus_relation=CorpusRelation(self.corpus, sub_corpus),
        )

    def test_copies_document_fields(self):
        container = self._container()
        self.assertEqual(container.id, self.document.id)
        self.assertEqual(container.title, "Doc title")
        self.assertEqual(container.description, "Doc description")
        self.assertEqual(container.external_id, "ext-1")
        self.assertEqual(container.url, "https://example.org/doc")
        self.assertEqual(container.details, {"author": "someone"})
        self.assertEqual(container.lang, "en")
        self.assertIsNone(container.mmr_rank)

    def test_only_slices_with_scored_points_are_kept(self):
        container = self._container()
        ids = [s.id for s in container.contained_slices]
        self.assertEqual(ids, [self.slice_a.id, self.slice_b.id])
        first = container.contained_slices[0]
        self.assertEqual(first.content, "second part")
        self.assertEqual(first.order, 2)
        self.assertEqual(first.score, 0.8)
        self.assertEqual(first.vector, [1.0, 2.0, 3.0])
        self.assertEqual(first.sdgs, [5])

    def test_slice_without_sdg_gets_empty_list(self):
        container = self._container(sdg_map={})
        self.assertTrue(all(s.sdgs == [] for s in container.contained_slices))
        self.assertEqual(container.n_most_common_sdg(2), [])

    def test_corpus_and_sub_corpus(self):
        container = self._container(sub_corpus=self.sub_corpus)
        self.assertIs(container.corpus, self.corpus)
        self.assertIs(container.sub_corpus, self.sub_corpus)
        self.assertIsNone(self._container().sub_corpus)

    def test_mean_score_and_vector(self):
        container = self._container()
        self.assertAlmostEqual(container.mean_score, 0.6)
        self.assertEqual(container.mean_vector, [2.0, 3.0, 4.0])

    def test_mmr_rank_setter(self):
        container = self._container()
        container.mmr_rank = 7
        self.assertEqual(container.mmr_rank, 7)

    def test_n_most_common_sdg(self):
        container = self._container()
        self.assertEqual(container.n_most_common_sdg(1), [5])
        self.assertEqual(sorted(container.n_most_common_sdg(2)), [3, 5])
        self.assertEqual(container.n_most_common_sdg(0), [])
        self.assertEqual(container.n_most_common_sdg(-1), [])

    def test_overview_orders_slices(self):
        container = self._container()
        self.assertEqual(container.overview(), "first part\n\n[...]second part")

    def test_deduplicate_hash_is_order_independent(self):
        container = self._container()
        expected = hashlib.sha256()
        for value in sorted([b"Doc title", b"Doc description"]):
            expected.update(value)
        self.assertEqual(container.deduplicate_hash, expected.hexdigest())

        swapped = FullContainer(
            wl_document=_make_document(
                title="Doc description", description="Doc title"
            ),
            wl_slices=[],
            scored_points=[],
            sdg_map={},
            corpus_relation=CorpusRelation(self.corpus),
        )
        self.assertEqual(swapped.deduplicate_hash, container.deduplicate_hash)

        other = FullContainer(
            wl_document=_make_document(title="Other"),
            wl_slices=[],
            scored_points=[],
            sdg_map={},
            corpus_relation=CorpusRelation(self.corpus),
        )
        self.assertNotEqual(other.deduplicate_hash, container.deduplicate_hash)

    def test_to_enhanced_document_slice(self):
        container = self._container(sub_corpus=self.sub_corpus)
        result = container.to_enhanced_document_slice()

        self.assertIsInstance(result, EnhancedDocumentSlice)
        self.assertAlmostEqual(result.score, 0.6)
        self.assertIsNone(result.id)
        payload = result.payload
        self.assertEqual(payload.document_corpus, "corpus_a")
        self.assertEqual(payload.document_sub_corpus, "sub_corpus_a")
        self.assertEqual(payload.document_desc, "Doc description")
        self.assertEqual(payload.document_id, self.document.id)
        self.assertEqual(payload.document_title, "Doc title")
        self.assertEqual(payload.document_lang, "en")
        self.assertEqual(sorted(payload.document_sdg), [3, 5])
        self.assertEqual(payload.document_url, "https://example.org/doc")
        self.assertEqual(payload.document_details, {"author": "someone"})
        self.assertEqual(payload.document_external_id, "ext-1")
        self.assertIsNone(payload.slice_sdg)
        self.assertEqual(payload.slice_content, "first part\n\n[...]second part")

    def test_to_enhanced_document_slice_without_sub_corpus(self):
        result = self._container().to_enhanced_document_slice()
        self.assertIsNone(result.payload.document_sub_corpus)
