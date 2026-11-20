import unittest
import uuid
from unittest import mock

from src.app.search.domain.models import SearchMethods
from src.app.shared.domain.documents import (
    EnhancedDocumentSlice,
    EnhancedDocumentSlicePayloadModel,
)
from src.app.shared.domain.exceptions import LLMUnavailableError, UnusableLLMOutputError
from src.app.tutor.domain.connections import (
    MAX_RESOURCE_CHARS,
    ConfirmedCourse,
    ConnectionIdea,
    SelectedResource,
    SelectionAnswer,
)
from src.app.tutor.services import evidence
from src.app.tutor.services.course_prompt import course_block, source_of
from src.app.tutor.services.evidence import (
    find_connection_evidence,
    has_link,
    idea_block,
    is_about_process,
    names_concept,
    resource_blocks,
    search_candidates,
    speaks_of_sustainability,
    to_evidence,
)
from src.app.tutor.services.llm import DataBlock, render_prompt
from src.app.tutor.services.resource_checks import (
    clean_quote,
    has_unsupported_figure,
    is_quoted_in,
    verified_quote,
    words_of,
)

COURSE = ConfirmedCourse(title="Introduction à la macroéconomie", topics=[])
IDEA = ConnectionIdea(
    id="idea-1",
    title="Les limites du PIB",
    anchor="Le PIB : définition et mesure",
    anchor_label=None,
    why="Le PIB ne mesure pas les dégâts environnementaux.",
    use=None,
    queries=["limites du PIB", "GDP limits"],
)


def doc(name, content, url=None, score=0.9, **details):
    return EnhancedDocumentSlice(
        score=score,
        payload=EnhancedDocumentSlicePayloadModel(
            document_corpus="conversation",
            document_desc="",
            document_details=details,
            document_id=uuid.uuid5(uuid.NAMESPACE_URL, name),
            document_lang="fr",
            document_sdg=[],
            document_title=f"Title {name}",
            document_url=url or f"https://example.org/{name}",
            slice_content=content,
            slice_sdg=None,
        ),
    )


def item(document, evidence, **fields):
    values = {
        "document": document,
        "relevance": "direct",
        "same_sense": True,
        "establishes": True,
        "concept_terms": ["PIB"],
        "sustainability_terms": ["ressources naturelles"],
        "usefulness": "Montre ce que le PIB ignore.",
        "evidence": evidence,
    }
    return SelectedResource(**{**values, **fields})


def selection(*resources, **fields):
    return SelectionAnswer(
        **{"keep": True, "fit": "genuine", "resources": list(resources), **fields}
    )


class QuoteTests(unittest.TestCase):
    TEXT = (
        "Continuous operation is a big hindrance for MES systems, as this mode makes it "
        "more diffi- cult to boost product concentration. Key to circularity is "
        "identifying material flows with the aim 82 J. Jonker and N. Faber being to "
        "close material loops."
    )

    def setUp(self):
        self.source = doc("m", self.TEXT)

    def test_finds_a_sentence_despite_what_pdf_extraction_adds(self):
        # a word cut by a hyphen
        self.assertTrue(
            is_quoted_in(
                "Continuous operation is a big hindrance for MES systems, as this mode "
                "makes it more difficult to boost product concentration.",
                self.source,
            )
        )
        # a page header inside the sentence
        self.assertTrue(
            is_quoted_in(
                "Key to circularity is identifying material flows with the aim being "
                "to close material loops.",
                self.source,
            )
        )

    def test_never_a_changed_sentence(self):
        self.assertFalse(
            is_quoted_in(
                "Key to circularity is identifying energy flows with the aim being to "
                "close material loops.",
                self.source,
            )
        )
        self.assertFalse(
            is_quoted_in("Continuous operation is a small hindrance.", self.source)
        )
        self.assertFalse(is_quoted_in("  ", self.source))

    def test_a_few_inserted_words_at_most(self):
        quote = "Key to circularity is identifying the material flows of the plant."

        def source(inserted):
            return doc(
                "w",
                "Key to circularity "
                + "header " * inserted
                + "is identifying the material flows of the plant. More text follows.",
            )

        self.assertTrue(is_quoted_in(quote, source(8)))
        self.assertFalse(is_quoted_in(quote, source(9)))

    def test_a_long_quote_may_leave_out_a_few_words_of_the_sentence(self):
        # the price of skipping a page header: "fossil energy and of" is left out
        source = doc(
            "p",
            "Prices of fossil energy and of food rose sharply in the country last year.",
        )
        self.assertTrue(
            is_quoted_in(
                "Prices of food rose sharply in the country last year.", source
            )
        )
        # a word changed by the model is caught
        self.assertFalse(
            is_quoted_in(
                "Prices of fossil fuels and of food rose sharply in the country.",
                source,
            )
        )

    def test_word_boundaries_are_set_aside_in_the_first_check(self):
        # a known trade-off of reading the text without spaces, harmless for a sentence
        self.assertTrue(is_quoted_in("the rapist", doc("t", "A therapist listens.")))

    def test_both_checks_read_characters_the_same_way(self):
        self.assertEqual(words_of("Le CŒUR, l’œuvre"), ["le", "coeur", "l", "oeuvre"])
        self.assertEqual(
            words_of("struc-\nture et diffi- cult"), ["structure", "et", "difficult"]
        )
        # an accent written as two characters (e + combining accent) is the same letter
        self.assertTrue(
            is_quoted_in("le problème", doc("e", "Le proble\u0300me du cœur."))
        )
        # "œ" read as "oe" in the loose check too, with a page header inside the quote
        source = doc(
            "c",
            "Au cœur de la transition 12 Rapport annuel se trouve la sobriété énergétique.",
        )
        self.assertTrue(
            is_quoted_in(
                "Au coeur de la transition se trouve la sobriété énergétique.", source
            )
        )

    def test_keeps_only_the_true_sentence_of_a_stitched_quote(self):
        self.assertEqual(
            verified_quote(
                "Key to circularity is identifying material flows with the aim being to "
                "close material loops. Moreover, circular models always cut costs by "
                "half for every company.",
                self.source,
            ),
            "Key to circularity is identifying material flows with the aim being to "
            "close material loops.",
        )
        self.assertIsNone(
            verified_quote(
                "Circular models always cut costs by half for every company in the sector.",
                self.source,
            )
        )
        self.assertIsNone(verified_quote(None, self.source))

    def test_a_quote_loses_its_quotation_marks(self):
        self.assertEqual(clean_quote("« Le PIB ignore. »."), "Le PIB ignore.")
        self.assertEqual(clean_quote('"Text"'), "Text")
        self.assertIsNone(clean_quote(None))
        self.assertIsNone(clean_quote(" « » "))


class ResourceCheckTests(unittest.TestCase):
    def test_a_direct_resource_names_the_course_concept_in_its_own_words(self):
        self.assertTrue(
            names_concept(
                "The material balance closes around the reactor.", ["material balance"]
            )
        )
        self.assertFalse(
            names_concept("Recycling saves resources.", ["material balance"])
        )
        self.assertFalse(names_concept("Recycling saves resources.", []))
        # a term too short to mean anything
        self.assertFalse(names_concept("A B C.", ["b"]))

    def test_speaks_of_sustainability_even_with_translated_terms(self):
        english = doc("e", "Inequalities by energy sources and environmental quality.")
        self.assertTrue(
            speaks_of_sustainability(english, ["inégalités de consommation"])
        )
        attachment = doc(
            "m", "Secure attachment predicts choosing primary care as a specialty."
        )
        self.assertFalse(
            speaks_of_sustainability(attachment, ["comportement écologique"])
        )
        self.assertTrue(speaks_of_sustainability(attachment, ["primary care"]))

    def test_a_figure_must_be_in_the_resources(self):
        docs = [doc("a", "Le PIB a baissé de 2,5 % en 2020.")]
        self.assertFalse(has_unsupported_figure("Une baisse de 2,5 %.", docs))
        self.assertTrue(has_unsupported_figure("Une baisse de 12 %.", docs))
        self.assertTrue(has_unsupported_figure("Un coût de 3 milliards.", docs))
        self.assertFalse(has_unsupported_figure("Pas de chiffre.", docs))

    def test_a_reason_about_the_process_is_not_a_reason(self):
        self.assertTrue(
            is_about_process(
                "This directly connects the course to the sustainability connection "
                "chosen by the teacher, using evidence from the documents."
            )
        )
        self.assertTrue(
            is_about_process("Ce lien relie le cours aux documents fournis.")
        )
        self.assertFalse(
            is_about_process(
                "Class 12 already examines AI and democracy; a climate-policy case lets "
                "students apply accountability to environmental decisions."
            )
        )

    def test_a_resource_needs_a_real_link(self):
        self.assertTrue(has_link(doc("a", "x")))
        self.assertFalse(has_link(doc("b", "x", url="not a link")))


class WhatTheModelSeesTests(unittest.TestCase):
    def test_a_resource_text_is_cut_for_the_prompt(self):
        [block] = resource_blocks([doc("l", "x" * (MAX_RESOURCE_CHARS + 50))])
        self.assertEqual(len(block.text), MAX_RESOURCE_CHARS)

    def test_the_idea_block(self):
        self.assertEqual(
            idea_block(IDEA),
            "Title: Les limites du PIB\n"
            "Part of the course concerned: Le PIB : définition et mesure\n"
            "Why: Le PIB ne mesure pas les dégâts environnementaux.",
        )
        whole = IDEA.model_copy(update={"anchor": None, "use": "Mesurer."})
        self.assertIn("concerned: the whole course", idea_block(whole))
        self.assertIn("Use of the course's concept: Mesurer.", idea_block(whole))

    def test_each_resource_with_its_title_and_source(self):
        resource = doc("a", "Texte.", publisher="INSEE", journal=None)
        self.assertEqual(source_of(resource), "conversation · INSEE · fr")
        self.assertEqual(
            resource_blocks([resource]),
            [
                DataBlock(
                    text="Texte.",
                    attrs={"title": "Title a", "source": "conversation · INSEE · fr"},
                )
            ],
        )

    def test_the_selection_prompt_counts_the_resources(self):
        prompt = render_prompt("selection", "fr", {"count": 3})
        self.assertIn("The 3 documents in the <resource> blocks", prompt)
        self.assertIn("precise disciplinary concept or mechanism", prompt)
        self.assertIn(
            'A document on the general sustainability theme is not "direct"', prompt
        )
        self.assertIn("That a sentence appears in the document proves nothing", prompt)
        self.assertNotIn("country or region", prompt)


class ToEvidenceTests(unittest.TestCase):
    CANDIDATES = [
        doc(
            "a", "Le PIB ignore la dégradation des ressources naturelles. Autre phrase."
        ),
        doc("b", "Un texte sans rapport."),
        doc("c", "Une phrase sans lien.", url="not a link"),
    ]

    def judge(self, answer, candidates=None):
        return to_evidence(answer, IDEA, candidates or self.CANDIDATES)

    def test_keeps_only_resources_whose_quote_is_in_them_with_a_real_link(self):
        result = self.judge(
            selection(
                item(1, "« Le PIB ignore la dégradation des ressources naturelles. »"),
                # the quote is not in the document
                item(2, "Une phrase inventée."),
                # no real link
                item(3, "Une phrase sans lien."),
                # a document that was not given
                item(9, "x"),
                why="Le PIB laisse de côté l’environnement.",
                concept="mesure du PIB",
            )
        )

        self.assertEqual(result.status, "supported")
        self.assertEqual(result.why, "Le PIB laisse de côté l’environnement.")
        self.assertEqual(result.concept, "mesure du PIB")
        self.assertEqual(result.candidate_count, 3)
        [resource] = result.resources
        self.assertEqual(resource.document, self.CANDIDATES[0])
        self.assertEqual(resource.usefulness, "Montre ce que le PIB ignore.")
        self.assertEqual(
            resource.evidence, "Le PIB ignore la dégradation des ressources naturelles."
        )
        self.assertEqual(result.background, [])

    def test_context_only_is_background_and_the_idea_keeps_its_words(self):
        result = self.judge(
            selection(
                item(
                    1,
                    "Le PIB ignore la dégradation des ressources naturelles.",
                    relevance="context",
                    usefulness="Donne le contexte.",
                ),
                # an unknown judgement is never kept, even with a quote found in it
                SelectedResource.model_validate(
                    {
                        "document": 2,
                        "relevance": "tangential",
                        "usefulness": "Rien.",
                        "evidence": "Un texte sans rapport.",
                    }
                ),
                why="Une affirmation que seul un document de contexte permettrait.",
            )
        )

        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.resources, [])
        self.assertEqual(result.why, IDEA.why)
        self.assertEqual(
            [r.usefulness for r in result.background], ["Donne le contexte."]
        )

    def test_a_theme_only_quote_is_context_never_direct(self):
        candidates = [
            doc(
                "r",
                "Recycling saves resources and energy across many industries today.",
            )
        ]
        result = self.judge(
            selection(
                item(
                    1,
                    "Recycling saves resources and energy across many industries today.",
                    concept_terms=["material balance"],
                )
            ),
            candidates,
        )
        self.assertEqual(result.resources, [])
        self.assertEqual(len(result.background), 1)

    def test_direct_only_when_it_establishes_the_claim_in_the_same_sense(self):
        candidates = [
            doc("a", "Le recyclage des déchets ménagers progresse en Europe."),
            doc("b", "Le PIB ignore la dégradation des ressources naturelles."),
            doc("c", "Le PIB ne compte que la production marchande."),
        ]
        quote = "Le PIB ignore la dégradation des ressources naturelles."

        def judged(n=2, evidence=quote, **fields):
            return self.judge(
                selection(
                    item(
                        n,
                        evidence,
                        concept_terms=["PIB", "recyclage"],
                        sustainability_terms=["ressources naturelles", "déchets"],
                        **fields,
                    )
                ),
                candidates,
            )

        # the key word in another sense is left out entirely
        other_sense = judged(
            1,
            "Le recyclage des déchets ménagers progresse en Europe.",
            same_sense=False,
        )
        self.assertEqual((other_sense.resources, other_sense.background), ([], []))
        # established for another system only: context, not support
        self.assertEqual(len(judged(establishes=False).background), 1)
        self.assertEqual(judged(establishes=None).resources, [])
        # the document does not speak of sustainability in its own words: context
        no_sustainability = judged(3, "Le PIB ne compte que la production marchande.")
        self.assertEqual(no_sustainability.resources, [])
        self.assertEqual(len(no_sustainability.background), 1)
        self.assertEqual(len(judged().resources), 1)

    def test_never_states_a_figure_the_resources_do_not_contain(self):
        result = self.judge(
            selection(
                item(1, "Autre phrase.", concept_terms=["autre phrase"]),
                # the same document twice: once only
                item(1, "Autre phrase.", usefulness="Doublon."),
                why="Le PIB baisserait de 12 % avec les dégâts.",
            )
        )
        self.assertEqual(result.why, IDEA.why)
        self.assertEqual(len(result.resources), 1)
        # a figure in the usefulness: the resource is left out
        figure = self.judge(
            selection(
                item(
                    1,
                    "Le PIB ignore la dégradation des ressources naturelles.",
                    usefulness="Il baisse de 3 %.",
                )
            )
        )
        self.assertEqual(figure.status, "unsupported")

    def test_keeps_the_idea_reason_when_the_answer_explains_its_process(self):
        result = self.judge(
            selection(
                item(1, "Le PIB ignore la dégradation des ressources naturelles."),
                why="Ce lien relie le cours aux documents fournis.",
            )
        )
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.why, IDEA.why)

    def test_at_most_three_resources(self):
        candidates = [
            doc(n, "Le PIB ignore la dégradation des ressources naturelles.")
            for n in "abcde"
        ]
        result = self.judge(
            selection(
                *(
                    item(n, "Le PIB ignore la dégradation des ressources naturelles.")
                    for n in range(1, 6)
                )
            ),
            candidates,
        )
        self.assertEqual(len(result.resources), 3)

    def test_a_connection_judged_forced_or_not_kept(self):
        forced = self.judge(selection(fit="analogy"))
        self.assertEqual(forced.status, "forced")
        self.assertEqual(
            self.judge(selection(fit="off_discipline", keep=False)).status, "forced"
        )
        not_kept = self.judge(selection(keep=False))
        self.assertEqual(not_kept.status, "unsupported")
        self.assertEqual(not_kept.why, IDEA.why)

    def test_unknown_values_in_the_answer_are_no_values(self):
        answer = SelectionAnswer.model_validate(
            {"fit": "maybe", "keep": True, "resources": [{"relevance": "indirect"}]}
        )
        self.assertIsNone(answer.fit)
        self.assertIsNone(answer.resources[0].relevance)


class SearchCandidatesTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_searches_merged_by_document_best_first(self):
        a_low, a_high = doc("a", "A", score=0.5), doc("a", "A", score=0.8)
        others = [doc(n, n, score=0.1 * i) for i, n in enumerate("bcdefg", start=1)]
        search = mock.MagicMock()
        search.search_handler = mock.AsyncMock(
            side_effect=[[a_low, *others[:3]], [a_high, *others[3:]]]
        )
        background = mock.MagicMock()

        with mock.patch.object(
            evidence, "resolve_corpora", new=mock.AsyncMock(return_value=("c1", "c2"))
        ):
            found = await search_candidates(IDEA, search, background)

        self.assertEqual(len(found), 6)
        self.assertEqual(found[0], a_high)
        self.assertEqual(
            [d.score for d in found], sorted((d.score for d in found), reverse=True)
        )
        first, second = search.search_handler.await_args_list
        qp = first.kwargs["qp"]
        self.assertEqual(qp.query, "limites du PIB")
        self.assertEqual(second.kwargs["qp"].query, "GDP limits")
        self.assertEqual((qp.nb_results, qp.relevance_factor), (6, 0.75))
        # every visible corpus, every language
        self.assertEqual(qp.corpora, ("c1", "c2"))
        self.assertIsNone(qp.lang)
        self.assertEqual(first.kwargs["method"], SearchMethods.BY_DOCUMENT)
        self.assertIs(first.kwargs["background_tasks"], background)


class FindConnectionEvidenceTests(unittest.IsolatedAsyncioTestCase):
    CANDIDATES = [doc("a", "Le PIB ignore la dégradation des ressources naturelles.")]

    def _patch(self, candidates, call):
        search = mock.patch.object(
            evidence, "search_candidates", new=mock.AsyncMock(return_value=candidates)
        )
        llm = mock.patch.object(evidence, "structured_call", new=call)
        return search, llm

    async def test_judges_the_candidates_with_the_course_and_the_idea(self):
        call = mock.AsyncMock(
            return_value=selection(
                item(1, "Le PIB ignore la dégradation des ressources naturelles.")
            )
        )
        search, llm = self._patch(self.CANDIDATES, call)
        with search, llm:
            result = await find_connection_evidence(
                COURSE, IDEA, "fr", mock.MagicMock(), mock.MagicMock()
            )

        self.assertEqual(result.status, "supported")
        name, lang, schema, data, variables = call.await_args.args
        self.assertEqual((name, lang, schema), ("selection", "fr", SelectionAnswer))
        self.assertEqual(data["course"], course_block(COURSE))
        self.assertEqual(data["idea"], idea_block(IDEA))
        self.assertEqual(data["resource"], resource_blocks(self.CANDIDATES))
        self.assertEqual(variables, {"count": 1})

    async def test_nothing_found_is_unsupported_without_a_judgement(self):
        call = mock.AsyncMock()
        search, llm = self._patch([], call)
        with search, llm:
            result = await find_connection_evidence(
                COURSE, IDEA, "en", mock.MagicMock(), mock.MagicMock()
            )

        call.assert_not_awaited()
        self.assertEqual(result.status, "unsupported")
        self.assertEqual((result.candidate_count, result.why), (0, IDEA.why))

    async def test_an_unusable_judgement_leaves_the_idea_unchecked(self):
        for error in (UnusableLLMOutputError(), LLMUnavailableError()):
            with self.subTest(error=type(error).__name__):
                call = mock.AsyncMock(side_effect=error)
                search, llm = self._patch(self.CANDIDATES, call)
                with search, llm:
                    result = await find_connection_evidence(
                        COURSE, IDEA, "en", mock.MagicMock(), mock.MagicMock()
                    )
                self.assertEqual(result.status, "unchecked")

        self.assertEqual(result.status, "unchecked")
        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(result.resources, [])
