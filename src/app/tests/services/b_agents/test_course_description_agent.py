import unittest
from unittest.mock import AsyncMock, patch

from src.app.baml_client.async_client import types
from src.app.tutor.service.b_agents.course_description_agent import (
    CourseDescriptionAgent,
)


class TestCourseDescriptionAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.agent = CourseDescriptionAgent()

        self.metadata = types.CourseMetadata(
            output_language="en",
            discipline="computer science",
            level="licence",
            topic="fake topic",
            num_sessions=4,
            session_type="seminaire",
            session_mode=types.SessionMode.PRESENTIEL,
            class_size=10,
            session_duration=1.5,
        )

        self.context_text = "This is fake contextual text."

        self.course_description = types.CourseDescription(
            text="This is a generated course description.",
            word_count=8,
        )

    @patch(
        "src.app.tutor.service.b_agents.course_description_agent.b.GenerateCourseDescription",
        new_callable=AsyncMock,
    )
    async def test_generate_calls_baml(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.course_description

        result = await self.agent.generate(
            mode="mode_1",
            metadata=self.metadata,
            context_text=self.context_text,
            output_language="Français",
        )

        mock_generate.assert_awaited_once_with(
            mode="mode_1",
            metadata=self.metadata,
            context_text=self.context_text,
            output_language="Français",
        )

        self.assertEqual(result, self.course_description)

    @patch(
        "src.app.tutor.service.b_agents.course_description_agent.b.GenerateCourseDescription",
        new_callable=AsyncMock,
    )
    async def test_generate_uses_default_language(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.course_description

        await self.agent.generate(
            mode="mode_1",
            metadata=self.metadata,
            context_text=self.context_text,
        )

        mock_generate.assert_awaited_once_with(
            mode="mode_1",
            metadata=self.metadata,
            context_text=self.context_text,
            output_language="Français",
        )

    @patch(
        "src.app.tutor.service.b_agents.course_description_agent.b.GenerateCourseDescription",
        new_callable=AsyncMock,
    )
    async def test_generate_returns_course_description(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.course_description

        result = await self.agent.generate(
            mode="mode_1",
            metadata=self.metadata,
            context_text=self.context_text,
        )

        self.assertIsInstance(
            result,
            types.CourseDescription,
        )

        self.assertEqual(
            result.text,
            self.course_description.text,
        )

        self.assertEqual(
            result.word_count,
            self.course_description.word_count,
        )

    @patch(
        "src.app.tutor.service.b_agents.course_description_agent.b.GenerateCourseDescription",
        new_callable=AsyncMock,
    )
    async def test_generate_with_custom_language(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.course_description

        await self.agent.generate(
            mode="mode_1",
            metadata=self.metadata,
            context_text=self.context_text,
            output_language="English",
        )

        mock_generate.assert_awaited_once_with(
            mode="mode_1",
            metadata=self.metadata,
            context_text=self.context_text,
            output_language="English",
        )

    @patch(
        "src.app.tutor.service.b_agents.course_description_agent.b.GenerateCourseDescription",
        new_callable=AsyncMock,
    )
    async def test_generate_propagates_exception(
        self,
        mock_generate,
    ):
        mock_generate.side_effect = RuntimeError("BAML generation failed")

        with self.assertRaisesRegex(
            RuntimeError,
            "BAML generation failed",
        ):
            await self.agent.generate(
                mode="mode_1",
                metadata=self.metadata,
                context_text=self.context_text,
            )
