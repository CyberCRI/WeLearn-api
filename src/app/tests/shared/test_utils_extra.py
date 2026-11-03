import uuid
from io import BytesIO
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from docx import Document as DocxDocument
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from src.app.shared.domain.documents import (
    EnhancedDocumentSlice,
    EnhancedDocumentSlicePayloadModel,
)
from src.app.shared.utils.utils import (
    _extract_docx_content,
    _extract_pdf_content,
    extract_doc_info,
    get_files_content,
)


def _upload(content: bytes, content_type: str, filename: str = "f") -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


class ExtractDocInfoTests(TestCase):
    def test_plain_dict_documents_are_used_as_payload(self):
        docs = [
            {
                "document_title": "Dict doc",
                "document_url": "http://dict",
                "slice_content": "dict content",
            }
        ]
        self.assertEqual(
            extract_doc_info(docs),
            [{"title": "Dict doc", "url": "http://dict", "content": "dict content"}],
        )

    def test_pydantic_payload_is_read_by_attribute(self):
        doc = EnhancedDocumentSlice(
            score=0.5,
            payload=EnhancedDocumentSlicePayloadModel(
                document_corpus="c",
                document_desc="d",
                document_details={},
                document_id=uuid.uuid4(),
                document_lang="en",
                document_sdg=[1],
                document_title="Model doc",
                document_url="http://model",
                slice_content="model content",
                slice_sdg=1,
            ),
        )
        self.assertEqual(
            extract_doc_info([doc]),
            [{"title": "Model doc", "url": "http://model", "content": "model content"}],
        )

    def test_payload_without_any_info_is_skipped(self):
        empty_object_payload = SimpleNamespace(payload=SimpleNamespace())
        empty_dict = {"other": "value"}
        self.assertEqual(extract_doc_info([empty_object_payload, empty_dict]), [])

    def test_object_without_payload_is_skipped(self):
        self.assertEqual(extract_doc_info([SimpleNamespace(), "not a doc"]), [])


class GetFilesContentTests(IsolatedAsyncioTestCase):
    async def test_returns_content_of_every_file(self):
        files = [
            _upload(b"  first file  ", "text/plain", "a.txt"),
            _upload(b"second file", "text/plain", "b.txt"),
        ]
        result = await get_files_content(files)
        self.assertEqual(sorted(result), ["first file", "second file"])

    async def test_no_files_raises_400(self):
        with self.assertRaises(HTTPException) as ctx:
            await get_files_content([])
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "added files are empty")

    async def test_unsupported_file_propagates_http_error(self):
        with self.assertRaises(HTTPException) as ctx:
            await get_files_content([_upload(b"x", "image/png")])
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Unsupported file type")


class ExtractorsTests(IsolatedAsyncioTestCase):
    @patch(
        "src.app.shared.utils.utils.extract_txt_from_pdf_with_tika",
        new_callable=AsyncMock,
    )
    async def test_pdf_uses_tika_when_available(self, tika_mock):
        tika_mock.return_value = "tika text"
        file = _upload(b"%PDF", "application/pdf")

        result = await _extract_pdf_content(file)

        self.assertEqual(result, "tika text")
        tika_mock.assert_awaited_once()
        self.assertIs(tika_mock.await_args.args[0], file.file)
        self.assertEqual(tika_mock.await_args.args[1], "https://tika.example.com")

    @patch("src.app.shared.utils.utils.PdfReader")
    @patch(
        "src.app.shared.utils.utils.extract_txt_from_pdf_with_tika",
        new_callable=AsyncMock,
    )
    async def test_pdf_falls_back_to_pypdf_when_tika_fails(
        self, tika_mock, reader_mock
    ):
        tika_mock.side_effect = RuntimeError("tika down")
        page_1 = MagicMock()
        page_1.extract_text.return_value = "page one"
        page_2 = MagicMock()
        page_2.extract_text.return_value = None
        reader_mock.return_value.pages = [page_1, page_2]
        file = _upload(b"%PDF", "application/pdf")

        result = await _extract_pdf_content(file)

        self.assertEqual(result, "page one\n")
        reader_mock.assert_called_once_with(file.file)

    async def test_docx_paragraphs_are_joined(self):
        document = DocxDocument()
        document.add_paragraph("Hello")
        document.add_paragraph("World")
        buffer = BytesIO()
        document.save(buffer)
        buffer.seek(0)
        file = UploadFile(file=buffer, filename="doc.docx")

        result = await _extract_docx_content(file)

        self.assertEqual(result, "Hello\nWorld")
