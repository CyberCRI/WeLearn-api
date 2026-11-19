"""Evaluation harness for the tutor prompts (`make eval-tutor`, never in CI).

A case is a JSON file in `cases/<prompt>/`:

    {
      "id": "fr-sociology-txt",
      "lang": "fr",
      "variables": {"max_ideas": 4},
      "data": {"course": "file:sociology.txt"},
      "expect": {"title": {"contains": "sociolinguistique"}, "topics": {"min_items": 3}}
    }

- `data` values are text, or `file:<name>` for a UTF-8 text file next to the case. A tag
  sent as several numbered blocks (the resources of `selection`) takes a list of
  `{"text": ..., "attrs": {...}}`, each `text` read the same way.
- `variables` are added to the ones the code always gives the prompt (`PROMPT_VARIABLES`,
  e.g. the GreenComp list of `adaptation`).
- `expect` maps a dotted path of the answer to checks: `equals`, `contains` (case-insensitive,
  in a text or in a list of texts), `min_items`, `max_items`, `not_empty`, and `has_item`
  (a list holds an object with these field values, e.g. `{"status": "established"}`; a list
  of values allows any of them).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.app.tutor.domain.adaptation import AdaptationAnswer, VerificationAnswer
from src.app.tutor.domain.connections import IdeasAnswer, SelectionAnswer
from src.app.tutor.domain.course import CourseReading
from src.app.tutor.domain.greencomp import greencomp_for_prompt
from src.app.tutor.services.llm import DataBlock

CASES_DIR = Path(__file__).parent / "cases"

# the prompts evaluated, with the schema their answer is read as. each LLM endpoint
# registers its prompts here
EVALUATED_PROMPTS: dict[str, type[BaseModel]] = {
    "read": CourseReading,
    "ideas": IdeasAnswer,
    "selection": SelectionAnswer,
    "adaptation": AdaptationAnswer,
    "verification": VerificationAnswer,
}

# variables the code always gives a prompt (not written in each case)
PROMPT_VARIABLES: dict[str, dict[str, Any]] = {
    "adaptation": {"greencomp": greencomp_for_prompt()},
}


@dataclass
class EvalCase:
    prompt: str
    id: str
    lang: str
    data: dict[str, str | list[DataBlock]]
    variables: dict[str, Any] = field(default_factory=dict)
    expect: dict[str, dict[str, Any]] = field(default_factory=dict)


def _text(value: str, folder: Path) -> str:
    if value.startswith("file:"):
        return (folder / value[len("file:") :]).read_text(encoding="utf-8")
    return value


def load_cases(prompt: str, cases_dir: Path = CASES_DIR) -> list[EvalCase]:
    cases = []
    for path in sorted((cases_dir / prompt).glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        data = {
            tag: (
                _text(value, path.parent)
                if isinstance(value, str)
                else [
                    DataBlock(_text(block["text"], path.parent), block.get("attrs", {}))
                    for block in value
                ]
            )
            for tag, value in raw["data"].items()
        }
        cases.append(
            EvalCase(
                prompt=prompt,
                id=raw.get("id", path.stem),
                lang=raw["lang"],
                data=data,
                variables={
                    **PROMPT_VARIABLES.get(prompt, {}),
                    **raw.get("variables", {}),
                },
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


def _has_item(items: Any, expected: dict[str, Any]) -> bool:
    """An item whose fields have these values (a list gives the values allowed)."""

    def matches(item: Any) -> bool:
        return all(
            (
                item.get(key) in wanted
                if isinstance(wanted, list)
                else item.get(key) == wanted
            )
            for key, wanted in expected.items()
        )

    return any(isinstance(item, dict) and matches(item) for item in items)


CHECKS = {
    "equals": lambda value, expected: value == expected,
    "contains": _contains,
    "has_item": _has_item,
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
