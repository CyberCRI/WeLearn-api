from io import BytesIO
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi import HTTPException, UploadFile
from qdrant_client.models import ScoredPoint

from src.app.services.tutor.utils import (
    build_system_message,
    extract_doc_info,
    get_file_content,
)


class TestTutorUtils(TestCase):
    def test_build_system_message_with_all_params(self):
        message = build_system_message(
            role="tutor",
            backstory="You are a tutor",
            goal="help students learn",
            instructions="Follow the syllabus",
            expected_output="Detailed explanation",
        )
        expected_message = (
            "You are tutor. You are a tutor\nYour personal goal is: help students learn."
            "You must accomplish your goal by following these steps: Follow the syllabus"
            "\nThis is the expected criteria for your final answer: Detailed explanation"
            "\nYou MUST return the actual complete content as the final answer, not a summary."
        )
        self.assertEqual(message, expected_message)

    def test_build_system_message_without_optional_params(self):
        message = build_system_message(
            role="tutor",
            backstory="You are a tutor",
            goal="help students learn",
        )
        expected_message = "You are tutor. You are a tutor\nYour personal goal is: help students learn."
        self.assertEqual(message, expected_message)

    def test_extract_doc_info(self):
        # Create mock documents
        doc1 = Mock(spec=ScoredPoint)
        doc1.payload = Mock(
            document_title="Test Doc 1",
            document_url="http://test1.com",
            slice_content="Content 1",
        )

        doc2 = Mock(spec=ScoredPoint)
        doc2.payload = Mock(
            document_title="Test Doc 2",
            document_url="http://test2.com",
            slice_content="Content 2",
        )

        doc3 = Mock(spec=ScoredPoint)
        doc3.payload = None  # Test case for None payload

        documents = [doc1, doc2, doc3]

        result = extract_doc_info(documents)

        expected = [
            {"title": "Test Doc 1", "url": "http://test1.com", "content": "Content 1"},
            {"title": "Test Doc 2", "url": "http://test2.com", "content": "Content 2"},
        ]

        self.assertEqual(result, expected)

    @patch("src.app.services.tutor.utils._extract_docx_content")
    async def test_get_file_content_docx(self, mock_extract_docx):
        mock_extract_docx.return_value = "Hello, world!"
        file = UploadFile(
            file=BytesIO(b"test content"),
            filename="test.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        content = await get_file_content(file)
        self.assertEqual(content, "Hello, world!")
        mock_extract_docx.assert_called_once()

    @patch("src.app.services.tutor.utils._extract_pdf_content")
    async def test_get_file_content_pdf(self, mock_extract_pdf):
        mock_extract_pdf.return_value = "Hello, world!"
        file = UploadFile(
            file=BytesIO(b"test content"),
            filename="test.pdf",
            content_type="application/pdf",
        )
        content = await get_file_content(file)
        self.assertEqual(content, "Hello, world!")
        mock_extract_pdf.assert_called_once()

    @patch("src.app.services.tutor.utils._extract_text_content")
    async def test_get_file_content_txt(self, mock_extract_text):
        mock_extract_text.return_value = "Hello, world!"
        file = UploadFile(
            file=BytesIO(b"test content"),
            filename="test.txt",
            content_type="text/plain",
        )
        content = await get_file_content(file)
        self.assertEqual(content, "Hello, world!")
        mock_extract_text.assert_called_once()

    async def test_get_file_content_unsupported(self):
        file = UploadFile(
            file=BytesIO(b"test content"),
            filename="test.unsupported",
            content_type="application/unsupported",
        )
        with self.assertRaises(HTTPException):
            await get_file_content(file)

    async def test_get_file_content_empty(self):
        file = UploadFile(
            file=open("test.empty.txt", "rb"),
            filename="test.empty.txt",
            content_type="text/plain",
        )
        with self.assertRaises(HTTPException):
            await get_file_content(file)
