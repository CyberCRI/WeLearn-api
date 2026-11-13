"""Evaluation harness for the tutor prompts (`make eval-tutor`, never in CI).

A case is a JSON file in `cases/<prompt>/`:

    {
      "id": "fr-sociology-txt",
      "lang": "fr",
      "variables": {"max_ideas": 4},
      "data": {"course": "file:sociology.txt"},
      "expect": {"title": {"contains": "sociolinguistique"}, "topics": {"min_items": 3}}
    }

- `data` values are text, or `file:<name>` for a UTF-8 text file next to the case.
- `expect` maps a dotted path of the answer to checks: `equals`, `contains` (case-insensitive,
  in a text or in a list of texts), `min_items`, `max_items`, `not_empty`.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

CASES_DIR = Path(__file__).parent / "cases"

# the prompts evaluated, with the schema their answer is read as. each LLM endpoint
# registers its prompts here (e.g. "read": CourseReading in SY-08)
EVALUATED_PROMPTS: dict[str, type[BaseModel]] = {}


@dataclass
class EvalCase:
    prompt: str
    id: str
    lang: str
    data: dict[str, str]
    variables: dict[str, Any] = field(default_factory=dict)
    expect: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_cases(prompt: str, cases_dir: Path = CASES_DIR) -> list[EvalCase]:
    cases = []
    for path in sorted((cases_dir / prompt).glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        data = {
            tag: (
                (path.parent / value[len("file:") :]).read_text(encoding="utf-8")
                if value.startswith("file:")
                else value
            )
            for tag, value in raw["data"].items()
        }
        cases.append(
            EvalCase(
                prompt=prompt,
                id=raw.get("id", path.stem),
                lang=raw["lang"],
                data=data,
                variables=raw.get("variables", {}),
                expect=raw.get("expect", {}),
            )
        )
    return cases


def _at(answer: Any, dotted: str) -> Any:
    value = answer
    for key in dotted.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _contains(value: Any, expected: str) -> bool:
    needle = expected.lower()
    if isinstance(value, list):
        return any(needle in str(item).lower() for item in value)
    return needle in str(value).lower()


CHECKS = {
    "equals": lambda value, expected: value == expected,
    "contains": _contains,
    "min_items": lambda value, expected: len(value) >= expected,
    "max_items": lambda value, expected: len(value) <= expected,
    "not_empty": lambda value, expected: bool(value) == expected,
}


def check(answer: BaseModel | dict, expect: dict[str, dict[str, Any]]) -> list[str]:
    """The failed expectations, as readable lines (empty when all pass)."""
    data = answer.model_dump() if isinstance(answer, BaseModel) else answer
    failures = []
    for path, checks in expect.items():
        try:
            value = _at(data, path)
        except (KeyError, IndexError, TypeError, ValueError):
            failures.append(f"{path}: missing")
            continue
        for name, expected in checks.items():
            if name not in CHECKS:
                failures.append(f"{path}: unknown check {name!r}")
            elif not CHECKS[name](value, expected):
                failures.append(f"{path}: {name} {expected!r} failed on {value!r}")
    return failures
