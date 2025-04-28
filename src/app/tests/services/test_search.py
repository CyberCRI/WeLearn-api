import os
from typing import List
from unittest import IsolatedAsyncioTestCase, mock

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import CollectionDescription, CollectionsResponse, ScoredPoint

from src.app.models.collections import Collection
from src.app.services.search import (
    DBClientSingleton,
    SearchService,
    concatenate_same_doc_id_slices,
)

os.environ["USE_CACHED_SETTINGS"] = "False"


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


@mock.patch("qdrant_client.AsyncQdrantClient", alternate_mock_method)
@mock.patch.object(AsyncQdrantClient, "get_collections", return_value=collections)
class SearchServiceTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.sp = SearchService()

    def test_db_singleton(self, *mocks):
        db_sing1 = DBClientSingleton()
        db_sing2 = DBClientSingleton()

        self.assertEqual(db_sing1, db_sing2)

    async def test_get_collection_by_language(self, *mocks):
        collections = await self.sp.get_collection_by_language("fr")

        self.assertEqual(collections.name, "collection_welearn_fr_exists")

    def test_get_info_from_collection_name(self, *mocks):
        collection = self.sp._get_info_from_collection_name(
            "collection_welearn_fr_exists"
        )

        self.assertEqual(collection.name, "collection_welearn_fr_exists")
        self.assertEqual(collection.lang, "fr")
        self.assertEqual(collection.model, "exists")

    async def test_get_collection_by_language_with_collection(self, *mocks):
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

    def test_concatenate_same_doc_id_slices(self, *mocks):

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
