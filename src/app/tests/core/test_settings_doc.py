import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.app.core import settings_doc

CONFIG = """
class Other:
    IGNORED: bool = True


class Settings(BaseSettings):
    NAME: str = "WeLearn"
    ENV: str

    # turns the thing on. More details that the table leaves out
    THING_ENABLED: bool = False
    # quotas of the thing, per day
    THING_DAILY_LIMIT: int = 5
    OTHER_DAILY_LIMIT: int = 2

    TIMEOUT_SECONDS: float = 60
    NO_DEFAULT: int
"""

VALUES = """
image:
  tag: "x"

config:
  nonSensitive:
    # a comment above a value
    THING_DAILY_LIMIT: "7"
    URL: https://example.org/a # trailing comment

other:
  nonSensitive:
    NOT_CONFIG: "1"
"""


def write_root(root: Path, env_values: dict[str, str]) -> None:
    (root / "src/app/core").mkdir(parents=True)
    (root / "k8s/welearn-api").mkdir(parents=True)
    (root / settings_doc.CONFIG).write_text(CONFIG)
    (root / settings_doc.VALUES).write_text(VALUES)
    for env in settings_doc.ENVIRONMENTS:
        values = env_values.get(env, "")
        (root / settings_doc.ENV_VALUES.format(env=env)).write_text(
            f"config:\n  nonSensitive:\n{values}"
        )


class ReadSettingsTests(unittest.TestCase):
    def test_reads_flags_and_limits_with_the_comment_above_them(self):
        settings = settings_doc.read_settings(CONFIG)

        self.assertEqual(
            [(s.name, s.type, s.default) for s in settings],
            [
                ("THING_ENABLED", "bool", "false"),
                ("THING_DAILY_LIMIT", "int", "5"),
                ("OTHER_DAILY_LIMIT", "int", "2"),
                ("TIMEOUT_SECONDS", "float", "60"),
            ],
        )
        descriptions = [s.description for s in settings]
        # the first sentence only, and a comment block shared until a blank line
        self.assertEqual(
            descriptions,
            [
                "Turns the thing on.",
                "Quotas of the thing, per day",
                "Quotas of the thing, per day",
                "",
            ],
        )


class ReadValuesTests(unittest.TestCase):
    def test_reads_config_non_sensitive_only(self):
        self.assertEqual(
            settings_doc.read_values(VALUES),
            {"THING_DAILY_LIMIT": "7", "URL": "https://example.org/a"},
        )

    def test_refuses_what_it_cannot_read(self):
        with self.assertRaises(ValueError):
            settings_doc.read_values("config:\n  nonSensitive:\n    - not a key\n")


class PageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_root(self.root, {"dev": '    THING_ENABLED: "true"\n'})

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = settings_doc.main(list(args), root=self.root)
        return code, out.getvalue()

    def test_an_environment_overrides_the_base_values_and_the_code_default(self):
        self.assertEqual(self.run_main()[0], 0)
        page = (self.root / settings_doc.PAGE).read_text()

        # set for dev only, the code default elsewhere (in italics)
        self.assertIn(
            "| `THING_ENABLED` | `true` | _false_ | _false_ | `false` |", page
        )
        # set by the base values for every environment
        self.assertIn("| `THING_DAILY_LIMIT` | `7` | `7` | `7` | `5` |", page)
        self.assertIn("| `TIMEOUT_SECONDS` | _60_ | _60_ | _60_ | `60` | — |", page)

    def test_check_fails_until_the_page_is_written(self):
        self.assertEqual(self.run_main("--check")[0], 1)
        self.run_main()
        self.assertEqual(self.run_main("--check")[0], 0)

        (self.root / settings_doc.ENV_VALUES.format(env="prod")).write_text(
            'config:\n  nonSensitive:\n    THING_ENABLED: "true"\n'
        )
        self.assertEqual(self.run_main("--check")[0], 1)

    def test_release_notes_get_the_section_once(self):
        notes = self.root / "notes.md"
        notes.write_text("## What changed\n\n- a fix\n")

        code, first = self.run_main("--release", str(notes))
        notes.write_text(first)
        _, second = self.run_main("--release", str(notes))

        self.assertEqual(code, 0)
        self.assertTrue(first.startswith("## What changed"))
        self.assertEqual(first.count(settings_doc.START), 1)
        self.assertEqual(first, second)
        self.assertTrue(self.run_main("--release")[1].startswith(settings_doc.START))


class CommittedPageTests(unittest.TestCase):
    def test_committed_page_is_up_to_date(self):
        """docs/settings.md follows config.py and the Helm values"""
        with redirect_stderr(io.StringIO()):
            code = settings_doc.main(["--check"])
        self.assertEqual(
            code,
            0,
            "docs/settings.md is out of date, run `make docs-settings` and commit it",
        )
