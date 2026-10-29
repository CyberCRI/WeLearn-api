from unittest import TestCase

import numpy

from src.app.shared.utils.embeddings import convert_embedding_bytes
from src.app.shared.utils.json_parsing import extract_json_from_response


class SharedHelpersTests(TestCase):
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
