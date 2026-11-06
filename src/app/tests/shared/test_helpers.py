from unittest import TestCase

import numpy

from src.app.shared.utils.embeddings import convert_embedding_bytes


class SharedHelpersTests(TestCase):
    def test_convert_embedding_bytes(self):
        x = numpy.random.rand(
            5,
        )
        ret = convert_embedding_bytes(embeddings_byte=x.tobytes(), dtype=numpy.float64)

        self.assertEqual(x.tolist(), ret.tolist())
