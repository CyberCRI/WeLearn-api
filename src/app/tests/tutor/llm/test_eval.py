"""
Runs every evaluated prompt on its cases with the real model: `make eval-tutor`.

Skipped unless TUTOR_EVAL=1, so the normal test run and the CI never call the model.
"""

import os
import unittest

from src.app.tests.tutor.llm.harness import EVALUATED_PROMPTS, check, load_cases
from src.app.tutor.services.llm import structured_call


@unittest.skipUnless(
    os.getenv("TUTOR_EVAL") == "1", "calls the real LLM: run with `make eval-tutor`"
)
class PromptEvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_cases(self):
        cases = [case for prompt in EVALUATED_PROMPTS for case in load_cases(prompt)]
        if not cases:
            self.skipTest("no evaluated prompt has cases yet")

        for case in cases:
            with self.subTest(case=f"{case.prompt}:{case.id}"):
                answer = await structured_call(
                    case.prompt,
                    case.lang,
                    EVALUATED_PROMPTS[case.prompt],
                    case.data,
                    case.variables,
                )
                self.assertEqual(check(answer, case.expect), [])
