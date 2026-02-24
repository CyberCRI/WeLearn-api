import os
from typing import List
from unittest import IsolatedAsyncioTestCase, mock

from qdrant_client.models import CollectionDescription, CollectionsResponse, ScoredPoint

from src.app.models.collections import Collection
from src.app.services.search import SearchService, concatenate_same_doc_id_slices

os.environ["USE_CACHED_SETTINGS"] = "False"


class FakeQdrantClient:
    async def get_collections(self):
        class Collections:
            collections = [
                type("C", (), {"name": "collection_welearn_fr_exists"}),
                type("C", (), {"name": "collection_welearn_en_exists"}),
            ]

        return Collections()


class alternate_mock_method(object):
    def __init__(self, url: str, timeout: int, port: int, **kwargs):
        return

    async def get_aliases(self, *args, **kwargs):
        pass


USER_QUERY = "query1"

collections = CollectionsResponse(
    collections=[
        CollectionDescription(name="collection_welearn_fr_exists"),
        CollectionDescription(name="collection_welearn_en_exists"),
    ]
)


def fake_callback_function(embedding, nb_results, filters, collection_info):
    return f"{embedding}, {nb_results}, {filters}, {collection_info}"


class SearchServiceTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.qdrant = FakeQdrantClient()
        self.sp = SearchService(client=self.qdrant)

    async def test_get_collection_by_language(self):
        collection = await self.sp.get_collection_by_language("fr")

        self.assertEqual(collection.name, "collection_welearn_fr_exists")
        self.assertEqual(collection.lang, "fr")
        self.assertEqual(collection.model, "exists")

    def test_get_info_from_collection_name(self):
        collection = self.sp._get_info_from_collection_name(
            "collection_welearn_fr_exists"
        )

        self.assertEqual(collection.name, "collection_welearn_fr_exists")
        self.assertEqual(collection.lang, "fr")
        self.assertEqual(collection.model, "exists")

    async def test_get_collection_by_language_with_collection(self):
        with mock.patch.object(
            SearchService,
            "get_collections",
            return_value=(
                "collection_welearn_en_exists",
                "collection_welearn_fr_exists",
                "wiki_fr_exists",
            ),
        ):
            collection = await self.sp.get_collection_by_language("fr")
            exp_collection = Collection(
                name="collection_welearn_fr_exists",
                lang="fr",
                model="exists",
            )
            self.assertEqual(collection.name, "collection_welearn_fr_exists")
            self.assertEqual(collection, exp_collection)

    def test_concatenate_same_doc_id_slices(self):

        qdrant_docs: List[ScoredPoint] = [
            ScoredPoint(
                id=1,
                version=1,
                score=0.5,
                payload={"document_id": "1", "slice_content": "content1"},
            ),
            ScoredPoint(
                id=1,
                version=1,
                score=0.8,
                payload={"document_id": "1", "slice_content": "content2"},
            ),
            ScoredPoint(
                id=2,
                version=1,
                score=0.8,
                payload={"document_id": "2", "slice_content": "content3"},
            ),
        ]
        results = concatenate_same_doc_id_slices(qdrant_results=qdrant_docs)
        expected_result = [
            ScoredPoint(
                id=1,
                version=1,
                score=0.5,
                payload={"document_id": "1", "slice_content": "content1\n\ncontent2"},
            ),
            ScoredPoint(
                id=2,
                version=1,
                score=0.8,
                payload={"document_id": "2", "slice_content": "content3"},
            ),
        ]
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].payload, expected_result[0].payload)
        self.assertEqual(results[1].payload, expected_result[1].payload)
