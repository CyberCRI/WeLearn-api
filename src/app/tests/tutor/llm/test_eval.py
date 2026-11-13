"""Runs every evaluated prompt on its cases with the real model: `make eval-tutor`."""

import asyncio

import pytest

from src.app.tests.tutor.llm.harness import EVALUATED_PROMPTS, check, load_cases
from src.app.tutor.services.llm import structured_call

pytestmark = pytest.mark.llm

CASES = [case for prompt in EVALUATED_PROMPTS for case in load_cases(prompt)]


@pytest.mark.skipif(not CASES, reason="no evaluated prompt has cases yet")
@pytest.mark.parametrize("case", CASES, ids=[f"{c.prompt}:{c.id}" for c in CASES])
def test_prompt_case(case):
    answer = asyncio.run(
        structured_call(
            case.prompt,
            case.lang,
            EVALUATED_PROMPTS[case.prompt],
            case.data,
            case.variables,
        )
    )
    assert check(answer, case.expect) == []
