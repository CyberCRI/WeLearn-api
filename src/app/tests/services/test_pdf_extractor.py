import io
import unittest
from unittest.mock import MagicMock, patch

from src.app.services import pdf_extractor
from src.app.services.pdf_extractor import (
    _parse_tika_content,
    _send_pdf_to_tika,
    extract_txt_from_pdf_with_tika,
)


class TestPDFExtractor(unittest.TestCase):
    def test_replace_ligatures(self):
        text = "ﬁrst ﬂight"
        cleaned_text = pdf_extractor.replace_ligatures(text)
        self.assertEqual(cleaned_text, "first flight")

    def test_delete_accents(self):
        text = "re ´sume´"
        cleaned_text = pdf_extractor.delete_accents(text)
        self.assertEqual(cleaned_text, "resume")

    def test_remove_hyphens(self):
        text = "well-\nknown"
        cleaned_text = pdf_extractor.remove_hyphens(text)
        self.assertEqual(cleaned_text, "wellknown\n")

    @patch("src.app.services.pdf_extractor.get_new_https_session")
    def test_send_pdf_to_tika(self, mock_get_session):
        # Mock de la session HTTP
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "X-TIKA:content": "<html>Mock Content</html>"
        }
        mock_response.raise_for_status.return_value = None
        mock_session.put.return_value = mock_response
        mock_get_session.return_value.__enter__.return_value = mock_session

        # Appel de la méthode
        pdf_content = io.BytesIO(b"Mock PDF content")
        tika_base_url = "http://mock-tika-url"
        result = _send_pdf_to_tika(pdf_content, tika_base_url)

        # Assertions
        mock_session.put.assert_called_once_with(
            url=f"{tika_base_url}/tika",
            files={"file": pdf_content},
            headers={
                "Accept": "application/json",
                "Content-type": "application/octet-stream",
                "X-Tika-PDFOcrStrategy": "no_ocr",
            },
        )
        self.assertEqual(result, {"X-TIKA:content": "<html>Mock Content</html>"})

    def test_parse_tika_content(self):
        tika_content = {
            "X-TIKA:content": """
            <html>
                <div class="page">Page 1 content</div>
                <div class="page">Page 2 content</div>
            </html>
            """
        }

        result = _parse_tika_content(tika_content)

        expected_result = [["Page 1 content"], ["Page 2 content"]]
        self.assertEqual(result, expected_result)

    @patch("src.app.services.pdf_extractor._send_pdf_to_tika")
    @patch("src.app.services.pdf_extractor._parse_tika_content")
    def test_extract_txt_from_pdf_with_tika(
        self, mock_parse_tika_content, mock_send_pdf_to_tika
    ):
        pdf_content = io.BytesIO(b"%PDF-1.4 simulated content")
        tika_base_url = "http://localhost:9998"

        mock_send_pdf_to_tika.return_value = {
            "X-TIKA:content": "<div class='page'>Page 1 content</div>"
        }
        mock_parse_tika_content.return_value = [["Page 1 content"]]

        result = extract_txt_from_pdf_with_tika(pdf_content, tika_base_url)

        self.assertEqual(result, "Page 1 content")
        mock_send_pdf_to_tika.assert_called_once_with(pdf_content, tika_base_url)
        mock_parse_tika_content.assert_called_once_with(
            mock_send_pdf_to_tika.return_value
        )
