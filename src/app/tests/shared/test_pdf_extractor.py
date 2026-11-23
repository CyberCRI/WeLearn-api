import io
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.app.shared.infra import pdf_extractor
from src.app.shared.infra.pdf_extractor import (
    _parse_tika_content,
    _send_pdf_to_tika,
    _tika_page_lines,
    extract_layout_from_pdf_with_tika,
    extract_txt_from_pdf_with_tika,
    layout_text,
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


class TestPDFExtractorAsync(unittest.IsolatedAsyncioTestCase):
    @patch("src.app.shared.infra.pdf_extractor.get_new_https_async_client")
    async def test_send_pdf_to_tika(self, mock_get_client):
        # Mock du client HTTPX asynchrone
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.json = Mock(
            return_value={"X-TIKA:content": "<html>Mock Content</html>"}
        )
        mock_response.raise_for_status.return_value = None
        mock_client.__aenter__.return_value = mock_client
        mock_client.put.return_value = mock_response
        mock_get_client.return_value = mock_client

        pdf_content = io.BytesIO(b"Mock PDF content")
        tika_base_url = "http://mock-tika-url"
        result = await _send_pdf_to_tika(pdf_content, tika_base_url)

        mock_client.put.assert_awaited_once_with(
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

    @patch(
        "src.app.shared.infra.pdf_extractor._send_pdf_to_tika", new_callable=AsyncMock
    )
    @patch("src.app.shared.infra.pdf_extractor._parse_tika_content")
    async def test_extract_txt_from_pdf_with_tika(
        self, mock_parse_tika_content, mock_send_pdf_to_tika
    ):
        pdf_content = io.BytesIO(b"%PDF-1.4 simulated content")
        tika_base_url = "http://localhost:9998"

        mock_send_pdf_to_tika.return_value = {
            "X-TIKA:content": "<div class='page'>Page 1 content</div>"
        }
        mock_parse_tika_content.return_value = [["Page 1 content"]]

        result = await extract_txt_from_pdf_with_tika(pdf_content, tika_base_url)

        self.assertEqual(result, "Page 1 content")
        mock_send_pdf_to_tika.assert_awaited_once_with(pdf_content, tika_base_url)
        mock_parse_tika_content.assert_called_once_with(
            mock_send_pdf_to_tika.return_value
        )


class LayoutTextTests(unittest.TestCase):
    def test_joins_the_lines_the_page_width_cut(self):
        text = layout_text(
            [
                [
                    "This course aims to train ",
                    "students on human rights. ",
                    "It also works as a real-",
                    "situation lab.",
                ]
            ]
        )
        self.assertEqual(
            text,
            "This course aims to train students on human rights.\n"
            "It also works as a real-situation lab.",
        )

    def test_keeps_headings_lists_and_blocks(self):
        text = layout_text(
            [
                [
                    "Objective of the Course ",
                    " ",
                    "Theme 1: Treaties",
                    "Theme 2: Right to life",
                    "",
                    "",
                ]
            ]
        )
        self.assertEqual(
            text, "Objective of the Course\n\nTheme 1: Treaties\nTheme 2: Right to life"
        )

    def test_keeps_accents_and_replaces_ligatures(self):
        self.assertEqual(
            layout_text([["Eugénie Mérieau, ﬁnal exam"]]), "Eugénie Mérieau, final exam"
        )

    def test_leaves_out_the_lines_found_on_every_page(self):
        pages = [["2026", "Summary", "First page."], ["2026", "Second page."]]
        self.assertEqual(layout_text(pages), "Summary\nFirst page.\n\nSecond page.")

    def test_a_page_break_ends_the_block(self):
        self.assertEqual(
            layout_text(
                [
                    ["engage in a discussion on their chosen"],
                    ["methods and strategies."],
                ]
            ),
            "engage in a discussion on their chosen\n\nmethods and strategies.",
        )


class TikaLayoutTests(unittest.IsolatedAsyncioTestCase):
    TIKA = {
        "X-TIKA:content": (
            '<html><body><div class="page"><p>Objective \n \n</p><p>This course aims \n</p>'
            "<p>to train students. \n</p><p>\n</p><p>Summary \n</p></div></body></html>"
        )
    }

    def test_reads_one_line_per_paragraph_of_tika(self):
        self.assertEqual(
            _tika_page_lines(self.TIKA),
            [
                [
                    "Objective ",
                    " ",
                    "This course aims ",
                    "to train students. ",
                    "",
                    "Summary ",
                ]
            ],
        )

    @patch(
        "src.app.shared.infra.pdf_extractor._send_pdf_to_tika", new_callable=AsyncMock
    )
    async def test_extracts_a_pdf_with_its_layout(self, send):
        send.return_value = self.TIKA

        text = await extract_layout_from_pdf_with_tika(
            io.BytesIO(b"%PDF"), "http://tika"
        )

        self.assertEqual(
            text, "Objective\n\nThis course aims to train students.\n\nSummary"
        )
