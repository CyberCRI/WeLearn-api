import os
from unittest import IsolatedAsyncioTestCase, mock

from qdrant_client import AsyncQdrantClient

from src.app.services.search import (
    DBClientSingleton,
    SearchService,
    concatenate_same_doc_id_slices,
    search_items_method,
)

os.environ["USE_CACHED_SETTINGS"] = "False"


class AliasItem:
    def __init__(self, alias_name):
        self.alias_name = alias_name


class Aliases:
    def __init__(self):
        self.aliases = [
            AliasItem(alias_name="collection_fr_exists"),
            AliasItem(alias_name="collection_en_exists"),
        ]


class alternate_mock_method(object):
    def __init__(self, url: str, timeout: int, port: int, **kwargs):
        return

    async def get_aliases(self, *args, **kwargs):
        pass


USER_QUERY = "query1"


def fake_callback_function(embedding, nb_results, filters, collection_info):
    return f"{embedding}, {nb_results}, {filters}, {collection_info}"


@mock.patch("qdrant_client.AsyncQdrantClient", alternate_mock_method)
@mock.patch.object(AsyncQdrantClient, "get_aliases", return_value=Aliases())
class SearchServiceTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.sp = SearchService()

    def test_db_singleton(self, *mocks):
        db_sing1 = DBClientSingleton()
        db_sing2 = DBClientSingleton()

        self.assertEqual(db_sing1, db_sing2)

    async def test_search_pipeline_collection(self, *mocks):
        collection = await self.sp.get_collection_name("collection", "fr")

        self.assertEqual(collection, "collection_fr_exists")

    async def test_get_collection_by_language(self, *mocks):
        collections = await self.sp.get_collection_by_language("fr")

        self.assertEqual(collections, ["collection_fr_exists"])

    async def test_get_collection_by_language_without_sel_collection(
        self, *mocks
    ):
        await self.sp.get_collection_by_language("fr")
        assert mocks[0].called

    async def test_get_collection_name(self, *mocks):
        collection = await self.sp.get_collection_name("collection", "fr")
        self.assertEqual(collection, "collection_fr_exists")

    async def test_get_collection_name_without_sel_collection(self, *mocks):
        await self.sp.get_collection_name("collection", "fr")
        assert mocks[0].called

    def test_get_info_from_collection_name(self, *mocks):
        collection = self.sp._get_info_from_collection_name("collection_fr_exists")

        self.assertEqual(collection.alias, "collection_fr_exists")
        self.assertEqual(collection.lang, "fr")
        self.assertEqual(collection.model, "exists")
        self.assertEqual(collection.name, "collection")

    async def test_get_collection_by_language_with_collection(self, *mocks):
        with mock.patch.object(
            SearchService,
            "get_collections",
            return_value=(
                "conversation_en_exists",
                "conversation_fr_exists",
                "wiki_fr_exists",
            ),
        ):
            collections = await self.sp.get_collection_by_language(
                "fr"            )
            self.assertEqual(collections, ["wiki_fr_exists"])

    def test_concatenate_same_doc_id_slices(self, *mocks):
        class FakeQdrantDoc:
            def __init__(self, id, payload) -> None:
                self.id = id
                self.payload = payload

        qdrant_docs = [
            FakeQdrantDoc(1, {"document_id": "1", "slice_content": "content1"}),
            FakeQdrantDoc(2, {"document_id": "1", "slice_content": "content2"}),
            FakeQdrantDoc(3, {"document_id": "2", "slice_content": "content3"}),
        ]
        results = concatenate_same_doc_id_slices(qdrant_results=qdrant_docs)
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0].payload.get("slice_content"), "content1\n\ncontent2"
        )
        self.assertEqual(results[1].payload.get("slice_content"), "content3")

    def test_search_itmes_method(self, *mocks):
        results = search_items_method(
            callback_function=fake_callback_function,
            embedding=USER_QUERY,
            nb_results=1,
            collection="collection",
            filters=None,
        )

        self.assertEqual(results, "query1, 1, None, collection")
