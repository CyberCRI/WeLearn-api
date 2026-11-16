import unittest
from unittest import mock

from src.app.shared.domain.exceptions import (
    UnusableLLMOutputError,
    UnusableReadingError,
)
from src.app.tutor.domain.course import (
    CourseOutline,
    CourseReading,
    CourseReadResponse,
    ReadObjective,
    ReadSession,
    ReadTopic,
)
from src.app.tutor.services import course_reading
from src.app.tutor.services.course_reading import (
    clean_reading,
    core,
    is_course_title,
    merge_duplicate_topics,
    normalise,
    numbered_topics,
    prefer_numbered_topics,
    read_course,
    text_parts,
)

# ported from the prototype's courseReading.spec.ts (teacher's name made up)
HRA = (
    "HUMAN RIGHTS ADVOCACY\nFORMS OF ACTION, STRATEGIES, METHODS\nObjective of the Course\n"
    "This course aims to train students on how to successfully engage in human rights\n"
    "advocacy in various settings (courts, NGOs, United Nations).\nSummary\n"
    "This course provides an overview of the terrain of international human rights "
    "advocacy,\nincluding forms of action, strategies, and methods.\nSYLLABUS\n"
    "Professor: Alex MARTIN\n"
    "The course is divided into four overarching themes: ‘Core\nHuman Rights Treaties and "
    "Mechanisms’, ‘Civil and Political Rights’, ‘Socio-economic\nRights’, and ‘The Future "
    "of Human Rights’.\n"
    "Theme 1: Main Human Rights Treaties and Mechanisms, Universal Periodic Review,\n"
    "Advocacy\n"
    "Theme 2: Right to life, freedom from torture, freedom from persecution\n"
    "Theme 3: Freedom of expression, freedom of assembly\n"
    "Theme 4: Right to an adequate standard of living, right to housing\nBibliography"
)
HRA_THEMES = [
    "Main Human Rights Treaties and Mechanisms, Universal Periodic Review, Advocacy",
    "Right to life, freedom from torture, freedom from persecution",
    "Freedom of expression, freedom of assembly",
    "Right to an adequate standard of living, right to housing",
]
THREE_THEMES = (
    "Theme 1: Right to life\nTheme 2: Freedom of expression\nTheme 3: Right to housing"
)


def topic(title, subtopics=(), source=None):
    return ReadTopic(title=title, subtopics=list(subtopics), source=source)


def titles(topics):
    return [t.title for t in topics]


class CompareTextsTests(unittest.TestCase):
    def test_normalise_ignores_case_spaces_quotes_and_dashes_not_accents(self):
        self.assertEqual(
            normalise("L’Économie  du\n«bien» – commun"),
            normalise('l\'économie du "bien" - commun'),
        )
        self.assertNotEqual(normalise("Maîtriser"), normalise("Maitriser"))
        self.assertEqual(normalise("struc-\nture"), "structure")

    def test_core_drops_list_marker_and_final_punctuation(self):
        self.assertEqual(core("- Analyse a network."), core("Analyse a network"))
        self.assertEqual(core("2) Size pipes;"), "sizepipes")


class TextPartsTests(unittest.TestCase):
    def test_cuts_a_long_text_in_consecutive_parts(self):
        self.assertEqual(len(text_parts("x" * 60_001)), 2)
        self.assertEqual(text_parts("abcdefg", size=3), ["abc", "def", "g"])
        self.assertEqual(text_parts("short", size=10), ["short"])
        self.assertEqual(text_parts("", size=10), [""])


class CourseTitleTests(unittest.TestCase):
    def test_never_a_person_a_date_or_a_label(self):
        for value in (
            "Professor: Alex MARTIN",
            "Enseignante : Marie Dupont",
            "2026",
            "2",
            "Session: July 2026",
            "Syllabus",
            None,
            "",
        ):
            with self.subTest(value=value):
                self.assertFalse(is_course_title(value))
        self.assertTrue(is_course_title("HUMAN RIGHTS ADVOCACY"))
        self.assertTrue(is_course_title("Introduction à la gestion"))


class NumberedTopicsTests(unittest.TestCase):
    def test_reads_numbered_themes_with_a_heading_cut_by_a_line_break(self):
        self.assertEqual(titles(numbered_topics(HRA)), HRA_THEMES)
        self.assertEqual(numbered_topics(HRA)[0].source, HRA_THEMES[0])

    def test_needs_three_items_numbered_from_one_and_ignores_weeks(self):
        self.assertEqual(numbered_topics("Theme 1: A\nTheme 2: B"), [])
        self.assertEqual(
            numbered_topics("Chapter 4: A\nChapter 5: B\nChapter 6: C"), []
        )
        self.assertEqual(numbered_topics("Week 1: A\nWeek 2: B\nWeek 3: C"), [])

    def test_french_labels(self):
        text = "Thème 1 : Langue\nThème 2 : Variation\nThème 3 – Normes"
        self.assertEqual(
            titles(numbered_topics(text)), ["Langue", "Variation", "Normes"]
        )


class PreferNumberedTopicsTests(unittest.TestCase):
    def test_keeps_the_syllabus_themes_when_the_model_regrouped_them(self):
        regrouped = [
            topic("Core Human Rights Treaties and Mechanisms", ["Advocacy"]),
            topic("Civil and Political Rights", ["Right to life"]),
            topic("Civil and Political Rights", ["Freedom of expression"]),
            topic("Socio-economic Rights", ["Right to housing"]),
        ]
        self.assertEqual(titles(prefer_numbered_topics(regrouped, HRA)), HRA_THEMES)

    def test_keeps_the_model_topics_when_they_follow_the_numbering(self):
        faithful = [
            topic(
                "Main Human Rights Treaties",
                ["UPR"],
                "Main Human Rights Treaties and Mechanisms",
            ),
            topic("Right to life", source="Right to life, freedom from torture"),
            topic(
                "Freedom of expression",
                source="Freedom of expression, freedom of assembly",
            ),
            topic(
                "Adequate standard of living",
                source="Right to an adequate standard of living",
            ),
        ]
        self.assertEqual(prefer_numbered_topics(faithful, HRA), faithful)

    def test_keeps_the_model_topics_under_the_syllabus_own_higher_headings(self):
        text = (
            "Partie I: Introduction\nChapitre 1: Concepts\nChapitre 2: Science\n"
            "Partie II : Marchés\nChapitre 3: Offre"
        )
        under_parts = [
            topic("Introduction", ["Concepts", "Science"], "Introduction"),
            topic("Marchés", ["Offre"], "Marchés"),
        ]
        self.assertEqual(prefer_numbered_topics(under_parts, text), under_parts)

    def test_merges_a_topic_named_twice_without_numbering(self):
        self.assertEqual(
            prefer_numbered_topics(
                [topic("Rights", ["A"]), topic("rights", ["B", "A"])],
                "no numbering here",
            ),
            [topic("Rights", ["A", "B"])],
        )

    def test_never_borrows_another_numbered_theme_as_a_subtopic(self):
        grouped = [
            topic(
                "Civil rights",
                ["Freedom of expression", "Habeas corpus"],
                "Theme 1: Right to life",
            ),
            topic("Civil rights"),
        ]
        self.assertEqual(
            prefer_numbered_topics(grouped, THREE_THEMES),
            [
                topic("Right to life", ["Habeas corpus"], "Right to life"),
                topic("Freedom of expression", [], "Freedom of expression"),
                topic("Right to housing", [], "Right to housing"),
            ],
        )

    def test_puts_back_the_heading_when_the_model_names_a_category(self):
        categories = [
            topic("Civil Rights", source="Theme 1: Right to life"),
            topic("Freedom of expression", ["Press"], "Theme 2: Freedom of expression"),
            topic(
                "Socio-economic Rights",
                ["Right to housing"],
                "Theme 3: Right to housing",
            ),
        ]
        self.assertEqual(
            prefer_numbered_topics(categories, THREE_THEMES),
            [
                topic("Right to life", [], "Right to life"),
                topic(
                    "Freedom of expression",
                    ["Press"],
                    "Theme 2: Freedom of expression",
                ),
                topic("Right to housing", [], "Right to housing"),
            ],
        )

    def test_merge_does_not_change_the_given_topics(self):
        topics = [topic("A", ["x"]), topic("a", ["y"])]
        merge_duplicate_topics(topics)
        self.assertEqual(topics[0].subtopics, ["x"])


class AnswerParsingTests(unittest.TestCase):
    def test_trims_texts_and_drops_empty_values(self):
        reading = CourseReading.model_validate(
            {
                "title": "  Ecology ",
                "discipline": " ",
                "summary": "  A course.  ",
                "courses_found": ["", " UE 1 "],
                "objectives": [
                    {"text": " Analyse ", "scope": "course"},
                    {"text": " "},
                ],
                "topics": [
                    {"title": " Rights ", "subtopics": [" A ", ""], "source": " "},
                    {"title": "  "},
                ],
                "sessions": [{"label": "Week 8"}, {"label": " ", "title": None}],
            }
        )

        self.assertEqual(reading.title, "Ecology")
        self.assertIsNone(reading.discipline)
        self.assertEqual(reading.summary, "A course.")
        self.assertEqual(reading.courses_found, ["UE 1"])
        self.assertEqual(
            reading.objectives, [ReadObjective(text="Analyse", scope="course")]
        )
        self.assertEqual(reading.topics, [topic("Rights", ["A"])])
        self.assertEqual(reading.sessions, [ReadSession(label="Week 8")])

    def test_outline_is_cleaned_the_same_way(self):
        outline = CourseOutline.model_validate(
            {"summary": " ", "topics": [{"title": " Soils "}, {"title": ""}]}
        )

        self.assertIsNone(outline.summary)
        self.assertEqual(outline.topics, [topic("Soils")])

    def test_extracted_text_is_returned_as_it_is(self):
        response = CourseReadResponse(
            text="  Theme 1: Soils\n",
            is_text_truncated=False,
            is_partial=False,
            reading=CourseReading(),
        )

        self.assertEqual(response.text, "  Theme 1: Soils\n")


class CleanReadingTests(unittest.TestCase):
    def test_drops_a_title_that_is_not_a_course(self):
        cleaned = clean_reading(
            CourseReading(title="Professor: Alex Martin", discipline_suggested="Law")
        )

        self.assertIsNone(cleaned.title)
        self.assertEqual(cleaned.discipline_suggested, "Law")

    def test_suggested_discipline_only_when_none_is_stated(self):
        cleaned = clean_reading(
            CourseReading(discipline="Economics", discipline_suggested="Management")
        )
        self.assertEqual(cleaned.discipline, "Economics")
        self.assertIsNone(cleaned.discipline_suggested)

    def test_unknown_scope_and_material_type_are_kept_as_unknown(self):
        reading = CourseReading.model_validate(
            {
                "material_type": "handbook",
                "objectives": [{"text": "Analyse", "scope": "faculty"}],
            }
        )
        self.assertEqual(reading.material_type, "other")
        self.assertEqual(reading.objectives[0].scope, "unspecified")


class ReadCourseTests(unittest.IsolatedAsyncioTestCase):
    def _call(self, *answers):
        return mock.patch.object(
            course_reading,
            "structured_call",
            new=mock.AsyncMock(side_effect=list(answers)),
        )

    async def test_short_document_is_read_once(self):
        reading = CourseReading(title="Ecology", topics=[topic("Soils")])
        with self._call(reading) as call:
            response = await read_course("short text", "fr")

        call.assert_awaited_once_with(
            "read", "fr", CourseReading, {"course": "short text"}, {"has_focus": False}
        )
        self.assertEqual(response.text, "short text")
        self.assertFalse(response.is_text_truncated)
        self.assertFalse(response.is_partial)
        self.assertEqual(response.reading.title, "Ecology")
        self.assertEqual(titles(response.reading.topics), ["Soils"])

    async def test_long_document_topics_are_completed_part_by_part(self):
        text = "a" * 10 + "b" * 10 + "c" * 5
        with (
            mock.patch.object(
                course_reading,
                "text_parts",
                return_value=[text[:10], text[10:20], text[20:]],
            ),
            self._call(
                CourseReading(title="Ecology", topics=[topic("Soils")]),
                CourseOutline(
                    summary="Ignored", topics=[topic("soils"), topic("Water")]
                ),
                CourseOutline(summary="Ignored", topics=[topic("Air")]),
            ) as call,
        ):
            response = await read_course(text, "en")

        prompts = [c.args[0] for c in call.await_args_list]
        self.assertEqual(prompts, ["read", "read_outline", "read_outline"])
        self.assertEqual(call.await_args_list[1].args[3], {"course_part": "b" * 10})
        self.assertEqual(titles(response.reading.topics), ["Soils", "Water", "Air"])
        self.assertTrue(response.is_text_truncated)
        self.assertFalse(response.is_partial)

    async def test_without_topics_the_outline_reads_every_part(self):
        with self._call(
            CourseReading(title="Ecology"),
            CourseOutline(summary="From the outline", topics=[topic("Soils")]),
        ) as call:
            response = await read_course("short", "en")

        self.assertEqual(call.await_args_list[1].args[0], "read_outline")
        self.assertEqual(call.await_args_list[1].args[3], {"course_part": "short"})
        self.assertEqual(response.reading.summary, "From the outline")
        self.assertEqual(titles(response.reading.topics), ["Soils"])

    async def test_a_part_that_cannot_be_read_makes_the_reading_partial(self):
        with (
            mock.patch.object(course_reading, "text_parts", return_value=["p1", "p2"]),
            self._call(
                CourseReading(topics=[topic("Soils")]), UnusableLLMOutputError()
            ),
            self.assertLogs(course_reading.logger, level="WARNING"),
        ):
            response = await read_course("p1p2", "en")

        self.assertTrue(response.is_partial)
        self.assertEqual(titles(response.reading.topics), ["Soils"])

    async def test_an_unreadable_document_raises_unusable_reading(self):
        with self._call(UnusableLLMOutputError()):
            with self.assertRaises(UnusableReadingError) as raised:
                await read_course("text", "en")

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.msg_code, "UNUSABLE_READING")

    async def test_focus_goes_in_a_data_block_never_in_the_prompt(self):
        with self._call(CourseReading(topics=[topic("Law")])) as call:
            await read_course("text", "fr", "  UE 102 – Droit  ")

        _, _, _, data, variables = call.await_args.args
        self.assertEqual(data, {"course": "text", "focus": "UE 102 – Droit"})
        self.assertEqual(variables, {"has_focus": True})

    async def test_numbered_themes_of_the_whole_text_win(self):
        with self._call(CourseReading(topics=[topic("Civil rights")])):
            response = await read_course(THREE_THEMES, "en")

        self.assertEqual(
            titles(response.reading.topics),
            ["Right to life", "Freedom of expression", "Right to housing"],
        )


class PromptFilesTests(unittest.TestCase):
    def test_read_prompts_render_with_and_without_focus(self):
        from src.app.tutor.services.llm import render_prompt

        for name in ("read", "read_outline"):
            for lang in ("fr", "en"):
                with self.subTest(name=name, lang=lang):
                    plain = render_prompt(name, lang, {"has_focus": False})
                    focused = render_prompt(name, lang, {"has_focus": True})
                    self.assertIn('"topics": the main topics', plain)
                    self.assertNotIn("<focus>", plain)
                    self.assertIn("<focus>", focused)
