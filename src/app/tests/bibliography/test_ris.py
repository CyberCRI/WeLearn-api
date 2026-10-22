from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase

from src.app.bibliography.services.ris import (
    compute_authors_for_ris,
    compute_publication_date_for_ris,
    compute_ris_doctype,
    ris_line,
    welearn_document_to_ris,
)


class RisTests(TestCase):
    def test_ris_line(self):
        self.assertEqual(ris_line("TI", "My Title"), "TI  - My Title")
        self.assertEqual(ris_line("ID", 123), "ID  - 123")
        self.assertIsNone(ris_line("TI", None))
        self.assertIsNone(ris_line("TI", ""))

    def test_compute_ris_doctype(self):
        self.assertEqual(compute_ris_doctype("book"), "BOOK")
        self.assertEqual(compute_ris_doctype("chapter"), "CHAP")
        self.assertEqual(compute_ris_doctype("article"), "JFULL")
        self.assertEqual(compute_ris_doctype("report"), "ELEC")
        self.assertEqual(compute_ris_doctype(None), "ELEC")

    def test_compute_publication_date_for_ris(self):
        ret = compute_publication_date_for_ris("1704067200")
        self.assertRegex(ret, r"^\d{4}/\d{2}/\d{2}$")
        self.assertEqual(compute_publication_date_for_ris("abc"), "")

    def test_compute_authors_for_ris(self):
        authors = [{"name": "Alice"}, {"name": "Bob"}]
        self.assertEqual(
            compute_authors_for_ris(cast(Any, authors)), ["AU  - Alice", "AU  - Bob"]
        )
        self.assertEqual(compute_authors_for_ris(cast(Any, [])), [])

    def test_welearn_document_to_ris_complete(self):
        doc = SimpleNamespace(
            id="12345678-1234-5678-1234-567812345678",
            title="SDG education",
            url="https://example.org/doc",
            description="Short description",
            corpus=SimpleNamespace(source_name="test_corpus"),
            lang="en",
            doi="10.1000/test",
            details={
                "type": "article",
                "publication_date": "1704067200",
                "authors": [{"name": "Alice"}, {"name": "Bob"}],
                "publisher": "UNESCO",
                "license_url": "https://license.example.org",
            },
        )

        ris = welearn_document_to_ris(cast(Any, doc))

        self.assertIn("TY  - JFULL", ris)
        self.assertIn("ID  - 12345678-1234-5678-1234-567812345678", ris)
        self.assertIn("TI  - SDG education", ris)
        self.assertIn("UR  - https://example.org/doc", ris)
        self.assertIn("AB  - Short description", ris)
        self.assertIn("DB  - test_corpus", ris)
        self.assertIn("LA  - en", ris)
        self.assertIn("DO  - 10.1000/test", ris)
        self.assertIn("AU  - Alice", ris)
        self.assertIn("AU  - Bob", ris)
        self.assertIn("PB  - UNESCO", ris)
        self.assertIn("C1  - https://license.example.org", ris)
        self.assertRegex(ris, r"\nPY  - \d{4}/\d{2}/\d{2}\n")
        self.assertTrue(ris.endswith("ER  - "))

    def test_welearn_document_to_ris_minimal(self):
        doc = SimpleNamespace(
            id="12345678-1234-5678-1234-567812345678",
            title="Minimal doc",
            url="https://example.org/minimal",
            description="dfqsdfsqdf",
            corpus=SimpleNamespace(source_name="test_corpus"),
            lang="fr",
            doi=None,
            details=None,
        )

        ris = welearn_document_to_ris(cast(Any, doc))

        self.assertIn("TY  - ELEC", ris)
        self.assertIn("TI  - Minimal doc", ris)
        self.assertIn("DB  - test_corpus", ris)
        self.assertIn("LA  - fr", ris)
        self.assertNotIn("\nDO  - ", ris)
        self.assertNotIn("\nPB  - ", ris)
        self.assertNotIn("\nC1  - ", ris)
        self.assertTrue(ris.endswith("ER  - "))
