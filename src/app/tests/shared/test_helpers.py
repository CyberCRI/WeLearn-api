from unittest import TestCase, mock

import numpy
from langdetect.language import Language

from src.app.shared.domain.exceptions import LanguageNotSupportedError
from src.app.shared.utils.embeddings import convert_embedding_bytes
from src.app.shared.utils.json_parsing import extract_json_from_response
from src.app.shared.utils.language import detect_language_from_entry


class SharedHelpersTests(TestCase):
    def test_detect_language_from_entry_no_lang(self):
        with mock.patch("src.app.shared.utils.language.detect_langs", return_value=[]):
            with self.assertRaises(LanguageNotSupportedError):
                detect_language_from_entry("test")

    def test_detect_language_from_entry(self):
        with mock.patch(
            "src.app.shared.utils.language.detect_langs",
            return_value=[Language("en", 1.0)],
        ):
            self.assertEqual(detect_language_from_entry("test again"), "en")

    def test_extract_json_from_response(self):
        response = 'Here is the JSON: {"key": "value"}'
        expected = {"key": "value"}
        self.assertEqual(extract_json_from_response(response), expected)

    def test_extract_json_from_response_no_json(self):
        response = "Here is the JSON: "
        with self.assertRaises(ValueError):
            extract_json_from_response(response)

    def test_convert_embedding_bytes(self):
        x = numpy.random.rand(
            5,
        )
        ret = convert_embedding_bytes(embeddings_byte=x.tobytes(), dtype=numpy.float64)

        self.assertEqual(x.tolist(), ret.tolist())
