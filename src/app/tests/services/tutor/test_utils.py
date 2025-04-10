from unittest import TestCase

from src.app.services.tutor.utils import build_system_message


class TestTutorUtils(TestCase):
    def test_build_system_message(self):
        message = build_system_message(
            role="tutor",
            backstory="You are a tutor",
            goal="help students learn",
            instructions="Follow the syllabus",
            expected_output="Detailed explanation",
        )
        expected_message = (
            "You are tutor. You are a tutor\nYour personal goal is: help students learn."
            "You must accomplish your goal by following these steps: Follow the syllabus"
            "\nThis is the expected criteria for your final answer: Detailed explanation"
            "\nYou MUST return the actual complete content as the final answer, not a summary."
        )

        self.assertEqual(message, expected_message)
