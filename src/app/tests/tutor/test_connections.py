import unittest
from unittest import mock

from pydantic import ValidationError

from src.app.shared.domain.exceptions import UnusableLLMOutputError
from src.app.tutor.domain.connections import (
    ConfirmedCourse,
    ConnectionIdea,
    CourseTopic,
    IdeaAnswer,
    IdeasAnswer,
)
from src.app.tutor.services import connections
from src.app.tutor.services.connections import (
    course_block,
    course_parts,
    find_connection_ideas,
    part_of_course,
    to_ideas,
)
from src.app.tutor.services.llm import render_prompt

COURSE = ConfirmedCourse(
    title="Introduction à la macroéconomie",
    discipline="économie",
    level="Licence 1",
    overview="Un cours sur les grands indicateurs.",
    topics=[
        CourseTopic(
            title="Les grandeurs de la macroéconomie",
            subtopics=["Le PIB : définition et mesure", "L’inflation"],
        )
    ],
    objectives=["Comprendre leur définition et leur mesure."],
    competencies=["Rédiger une note de conjoncture"],
)
PARTS = [
    "Les grandeurs de la macroéconomie",
    "Le PIB : définition et mesure",
    "L’inflation",
    "Comprendre leur définition et leur mesure.",
    "Rédiger une note de conjoncture",
]


def idea(**fields):
    values = {
        "title": "Les limites du PIB",
        "anchor": "Le PIB",
        "anchor_label": "Mesure du PIB",
        "why": "Le PIB ne mesure pas les dégâts environnementaux.",
        "queries": ["limites du PIB", "GDP limits"],
    }
    return {**values, **fields}


class CourseTests(unittest.TestCase):
    def test_the_course_block_has_one_labelled_line_or_list_per_field(self):
        self.assertEqual(
            course_block(COURSE),
            "Title: Introduction à la macroéconomie\n"
            "Discipline: économie\n"
            "Level: Licence 1\n"
            "Overview: Un cours sur les grands indicateurs.\n"
            "Topics:\n"
            "- Les grandeurs de la macroéconomie "
            "(Le PIB : définition et mesure, L’inflation)\n"
            "Learning objectives:\n"
            "- Comprendre leur définition et leur mesure.\n"
            "Competencies:\n"
            "- Rédiger une note de conjoncture",
        )

    def test_the_course_block_says_when_no_objective_is_stated(self):
        course = ConfirmedCourse(title="Ecology", sessions=["Week 1", "Week 2"])
        self.assertEqual(
            course_block(course),
            "Title: Ecology\n"
            "Learning objectives: none stated in the syllabus\n"
            "Course outline:\n- Week 1\n- Week 2",
        )

    def test_course_parts_are_every_topic_subtopic_objective_competency_and_session(
        self,
    ):
        course = COURSE.model_copy(update={"sessions": ["Semaine 1"]})
        self.assertEqual(course_parts(course), [*PARTS, "Semaine 1"])

    def test_an_empty_course_is_refused(self):
        with self.assertRaises(ValidationError):
            ConfirmedCourse(discipline="économie", level="L1")
        ConfirmedCourse(sessions=["Week 1"])


class PartOfCourseTests(unittest.TestCase):
    def test_the_part_that_contains_the_anchor(self):
        self.assertEqual(
            part_of_course("le pib", PARTS), "Le PIB : définition et mesure"
        )

    def test_the_most_precise_part_a_longer_anchor_contains(self):
        self.assertEqual(
            part_of_course(
                "Les grandeurs de la macroéconomie : L’inflation et ses causes", PARTS
            ),
            "Les grandeurs de la macroéconomie",
        )

    def test_quote_styles_and_spaces_aside(self):
        self.assertEqual(part_of_course("L'inflation", PARTS), "L’inflation")

    def test_nothing_when_not_in_the_course(self):
        self.assertIsNone(part_of_course("La biodiversité", PARTS))
        self.assertIsNone(part_of_course(None, PARTS))
        self.assertIsNone(part_of_course("Le PIB", ["", "  "]))


class IdeasAnswerTests(unittest.TestCase):
    def test_an_incomplete_idea_is_left_out(self):
        answer = IdeasAnswer.model_validate(
            {
                "ideas": [
                    idea(),
                    idea(title="Sans question", queries=[]),
                    idea(title="Sans raison", why="  "),
                    idea(title=None),
                ]
            }
        )
        self.assertEqual([i.title for i in answer.ideas], ["Les limites du PIB"])

    def test_no_idea_is_an_answer_only_with_its_reason(self):
        answer = IdeasAnswer.model_validate(
            {"ideas": [], "none_reason": "Un cours de grammaire latine."}
        )
        self.assertEqual(answer.ideas, [])
        with self.assertRaises(ValidationError):
            IdeasAnswer.model_validate({"ideas": [], "none_reason": " "})
        # ideas that are all incomplete: nothing usable either
        with self.assertRaises(ValidationError):
            IdeasAnswer.model_validate({"ideas": [idea(why=None)]})


class ToIdeasTests(unittest.TestCase):
    def test_ideas_are_numbered_and_tied_to_the_course_as_written(self):
        answer = IdeasAnswer(
            ideas=[
                IdeaAnswer(**idea(queries=["q1", "q2", "q3"], use="Mesurer.")),
                # a part that is not in the course: no anchor, so no label
                IdeaAnswer(
                    **idea(title="Ailleurs", anchor="La biodiversité", queries=["q"])
                ),
            ]
        )

        response = to_ideas(answer, PARTS)

        self.assertIsNone(response.none_reason)
        first, second = response.ideas
        self.assertEqual(first.id, "idea-1")
        self.assertEqual(first.anchor, "Le PIB : définition et mesure")
        self.assertEqual(first.anchor_label, "Mesure du PIB")
        self.assertEqual(first.queries, ["q1", "q2"])
        self.assertEqual(first.use, "Mesurer.")
        self.assertEqual(second.id, "idea-2")
        self.assertIsNone(second.anchor)
        self.assertIsNone(second.anchor_label)
        self.assertIsNone(second.use)

    def test_a_query_is_cut_to_a_search_and_one_sent_back_longer_is_refused(self):
        answer = IdeasAnswer(ideas=[IdeaAnswer(**idea(queries=["q" * 400]))])
        self.assertEqual(to_ideas(answer, PARTS).ideas[0].queries, ["q" * 300])
        with self.assertRaises(ValidationError):
            ConnectionIdea(**idea(id="idea-1", use=None, queries=["q" * 301]))

    def test_a_long_label_is_dropped(self):
        label = "un libellé bien trop long pour nommer une partie du cours"
        answer = IdeasAnswer(ideas=[IdeaAnswer(**idea(anchor_label=label))])
        self.assertIsNone(to_ideas(answer, PARTS).ideas[0].anchor_label)

    def test_at_most_four_ideas(self):
        answer = IdeasAnswer(
            ideas=[IdeaAnswer(**idea(title=f"Idée {n}")) for n in range(6)]
        )
        response = to_ideas(answer, PARTS)
        self.assertEqual(
            [i.id for i in response.ideas], [f"idea-{n}" for n in (1, 2, 3, 4)]
        )

    def test_the_reason_for_none_only_comes_with_no_idea(self):
        none = IdeasAnswer(ideas=[], none_reason="Aucun lien solide.")
        self.assertEqual(to_ideas(none, PARTS).none_reason, "Aucun lien solide.")
        both = IdeasAnswer(ideas=[IdeaAnswer(**idea())], none_reason="Aucun.")
        self.assertIsNone(to_ideas(both, PARTS).none_reason)


class FindConnectionIdeasTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_the_course_as_data_and_ties_the_ideas_to_it(self):
        answer = IdeasAnswer(ideas=[IdeaAnswer(**idea())])
        with mock.patch.object(
            connections, "structured_call", new=mock.AsyncMock(return_value=answer)
        ) as call:
            response = await find_connection_ideas(COURSE, "fr")

        call.assert_awaited_once_with(
            "ideas",
            "fr",
            IdeasAnswer,
            {"course": course_block(COURSE)},
            {"max_ideas": 4},
        )
        self.assertEqual(response.ideas[0].anchor, "Le PIB : définition et mesure")

    async def test_an_unusable_answer_is_an_error(self):
        with mock.patch.object(
            connections,
            "structured_call",
            new=mock.AsyncMock(side_effect=UnusableLLMOutputError()),
        ):
            with self.assertRaises(UnusableLLMOutputError):
                await find_connection_ideas(COURSE, "en")


class IdeasPromptTests(unittest.TestCase):
    def test_the_second_search_is_in_english_whatever_the_language(self):
        french = render_prompt("ideas", "fr", {"max_ideas": 4})
        english = render_prompt("ideas", "en", {"max_ideas": 4})
        self.assertIn("at most 4 possible sustainability connections", french)
        self.assertIn("the first in French, the second in English", french)
        self.assertIn("both in English", english)
        self.assertIn('"none_reason"', english)
        self.assertNotIn("{{", french + english)
