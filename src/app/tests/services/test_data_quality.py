import unittest

from src.app.services.data_quality import remove_duplicates


class DummyScoredPoint:
    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


class TestRemoveDuplicates(unittest.TestCase):
    def test_basic(self):
        points = [
            DummyScoredPoint(1, {"text": "a"}),
            DummyScoredPoint(2, {"text": "b"}),
            DummyScoredPoint(3, {"text": "a"}),
        ]
        result = remove_duplicates(["text"], points)
        self.assertEqual(len(result), 2)
        self.assertSetEqual(set(p.payload["text"] for p in result), {"a", "b"})

    def test_empty_keys(self):
        points = [DummyScoredPoint(1, {"text": "a"})]
        result = remove_duplicates([], points)
        self.assertListEqual(result, points)

    def test_empty_keys_strict(self):
        points = [DummyScoredPoint(1, {"text": "a"})]
        with self.assertRaises(ValueError):
            remove_duplicates([], points, strict=True)

    def test_missing_key(self):
        points = [
            DummyScoredPoint(1, {"text": "a"}),
            DummyScoredPoint(2, {"other": "b"}),
        ]
        result = remove_duplicates(["text"], points)
        self.assertListEqual(result, points)

    def test_missing_key_strict(self):
        points = [
            DummyScoredPoint(1, {"text": "a"}),
            DummyScoredPoint(2, {"other": "b"}),
        ]
        with self.assertRaises(ValueError):
            remove_duplicates(["text"], points, strict=True)

    def test_non_str_value(self):
        points = [
            DummyScoredPoint(1, {"text": 123}),
            DummyScoredPoint(2, {"text": "a"}),
        ]
        result = remove_duplicates(["text"], points)
        self.assertListEqual(result, points)

    def test_non_str_value_strict(self):
        points = [DummyScoredPoint(1, {"text": 123})]
        with self.assertRaises(TypeError):
            remove_duplicates(["text"], points, strict=True)

    def test_empty_payload(self):
        for payload in [{}, None]:
            points = [DummyScoredPoint(1, payload)]
            result = remove_duplicates(["text"], points)
            self.assertListEqual(result, points)

    def test_no_duplicates(self):
        points = [
            DummyScoredPoint(1, {"text": "a"}),
            DummyScoredPoint(2, {"text": "b"}),
        ]
        result = remove_duplicates(["text"], points)
        self.assertEqual(len(result), 2)
        self.assertListEqual(result, points)

    def test_multiple_keys(self):
        points = [
            DummyScoredPoint(1, {"a": "x", "b": "y"}),
            DummyScoredPoint(2, {"a": "x", "b": "y"}),
            DummyScoredPoint(3, {"a": "x", "b": "z"}),
        ]
        result = remove_duplicates(["a", "b"], points)
        self.assertEqual(len(result), 2)
        self.assertListEqual(result, [points[0], points[2]])

    def test_all_duplicates(self):
        points = [
            DummyScoredPoint(1, {"text": "a"}),
            DummyScoredPoint(2, {"text": "a"}),
            DummyScoredPoint(3, {"text": "a"}),
        ]
        result = remove_duplicates(["text"], points)
        self.assertEqual(len(result), 1)
        self.assertListEqual(result, [points[0]])
