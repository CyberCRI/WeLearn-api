import json
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from src.app.tests.tutor.llm.harness import check, load_cases


class Reading(BaseModel):
    title: str
    topics: list[str]


class LoadCasesTests(unittest.TestCase):
    def test_reads_the_cases_of_a_prompt_and_their_text_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "read"
            folder.mkdir()
            (folder / "course.txt").write_text(
                "Sociolinguistique, L1", encoding="utf-8"
            )
            (folder / "b.json").write_text(
                json.dumps(
                    {
                        "lang": "fr",
                        "data": {"course": "file:course.txt", "note": "inline"},
                        "expect": {"title": {"not_empty": True}},
                    }
                ),
                encoding="utf-8",
            )
            (folder / "a.json").write_text(
                json.dumps({"id": "first", "lang": "en", "data": {"course": "x"}}),
                encoding="utf-8",
            )

            cases = load_cases("read", Path(tmp))

        self.assertEqual([case.id for case in cases], ["first", "b"])
        self.assertEqual(
            cases[1].data, {"course": "Sociolinguistique, L1", "note": "inline"}
        )
        self.assertEqual(cases[1].prompt, "read")
        self.assertEqual(cases[0].variables, {})
        self.assertEqual(cases[0].expect, {})

    def test_no_folder_means_no_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_cases("ideas", Path(tmp)), [])


class CheckTests(unittest.TestCase):
    answer = Reading(
        title="Introduction à la sociolinguistique", topics=["Langue", "Société"]
    )

    def test_passes_when_every_expectation_holds(self):
        self.assertEqual(
            check(
                self.answer,
                {
                    "title": {"contains": "SOCIOLINGUISTIQUE", "not_empty": True},
                    "topics": {"min_items": 2, "max_items": 4, "contains": "société"},
                    "topics.0": {"equals": "Langue"},
                },
            ),
            [],
        )

    def test_lists_the_failed_expectations(self):
        failures = check(
            self.answer.model_dump(),
            {
                "title": {"equals": "Other"},
                "topics": {"min_items": 3, "shape": 1},
                "objectives": {"not_empty": True},
                "topics.9": {"equals": "x"},
            },
        )
        self.assertEqual(len(failures), 5)
        self.assertTrue(failures[0].startswith("title: equals 'Other' failed"))
        self.assertIn("topics: unknown check 'shape'", failures)
        self.assertIn("objectives: missing", failures)
        self.assertIn("topics.9: missing", failures)
