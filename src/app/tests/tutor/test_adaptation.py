import unittest
import uuid
from unittest import mock

from src.app.shared.domain.documents import (
    EnhancedDocumentSlice,
    EnhancedDocumentSlicePayloadModel,
)
from src.app.shared.domain.exceptions import UnusableLLMOutputError
from src.app.tutor.domain.adaptation import (
    AdaptationAnswer,
    AdaptationResource,
    ChosenConnection,
    VerificationAnswer,
)
from src.app.tutor.domain.connections import (
    ConfirmedCourse,
    CourseTopic,
    EvidencedResource,
)
from src.app.tutor.domain.greencomp import GREENCOMP_IDS, greencomp_for_prompt
from src.app.tutor.services import adaptation
from src.app.tutor.services.adaptation import (
    MAX_PROBLEMS,
    RoledResource,
    Verification,
    adapt_course,
    connection_block,
    names_in,
    problems_block,
    resource_blocks,
    restates_change,
    roled_resources,
    tidy_note,
    to_proposal,
    to_verification,
    unbacked_names,
    with_checks,
)
from src.app.tutor.services.connections import course_block
from src.app.tutor.services.llm import render_prompt

QUOTE = "Le PIB ignore la dégradation des ressources naturelles."
COURSE = ConfirmedCourse(
    title="Introduction à la macroéconomie",
    topics=[
        CourseTopic(
            title="Les grandeurs de la macroéconomie",
            subtopics=["Le PIB : définition et mesure", "L’inflation"],
        )
    ],
    objectives=["Comprendre leur définition et leur mesure."],
)


def doc(name, content):
    return EnhancedDocumentSlice(
        score=0.9,
        payload=EnhancedDocumentSlicePayloadModel(
            document_corpus="conversation",
            document_desc="",
            document_details={},
            document_id=uuid.uuid5(uuid.NAMESPACE_URL, name),
            document_lang="fr",
            document_sdg=[],
            document_title=f"Title {name}",
            document_url=f"https://example.org/{name}",
            slice_content=content,
            slice_sdg=None,
        ),
    )


RESOURCE = EvidencedResource(
    document=doc("a", QUOTE), usefulness="Montre ce que le PIB ignore.", evidence=QUOTE
)


def connection(
    anchor="Le PIB : définition et mesure", resources=(RESOURCE,), background=()
):
    return ChosenConnection(
        title="Les limites du PIB",
        anchor=anchor,
        why="Le PIB ne mesure pas les dégâts environnementaux.",
        resources=list(resources),
        background=list(background),
    )


def answer(**fields):
    values = {
        "kind": "topic",
        "original": "Le PIB : définition et mesure",
        "suggested": "Ajouter un exemple de ce que le PIB ne compte pas.",
        "why": "Les étudiants voient les limites de l’indicateur.",
        "documents": [1],
        "evidence": [{"document": 1, "quote": QUOTE}],
        "support": "strong",
    }
    return AdaptationAnswer.model_validate({**values, **fields})


def propose(chosen=None, **fields):
    chosen = chosen or connection()
    return to_proposal(answer(**fields), COURSE, chosen, roled_resources(chosen))


class ProposalTests(unittest.TestCase):
    def test_reads_the_change_the_original_as_written_and_what_supports_it(self):
        proposal = propose(original="le PIB")

        self.assertEqual(proposal.kind, "topic")
        self.assertEqual(proposal.original, "Le PIB : définition et mesure")
        self.assertEqual(proposal.support, "strong")
        self.assertEqual(
            proposal.resources,
            [AdaptationResource(**RESOURCE.model_dump(), role="direct", quote=QUOTE)],
        )
        self.assertIsNone(proposal.checks)

    def test_never_presents_as_changed_something_the_course_does_not_contain(self):
        no_anchor = connection(anchor=None)
        self.assertIsNone(
            propose(no_anchor, original="Séance 9 : la croissance verte").original
        )
        self.assertIsNone(propose(no_anchor, original=None).original)

    def test_keeps_a_quote_only_when_it_is_really_in_that_resource(self):
        proposal = propose(
            evidence=[
                {"document": 1, "quote": f"« {QUOTE} »"},
                {"document": 1, "quote": "Une autre phrase."},
            ]
        )
        self.assertEqual(proposal.resources[0].quote, QUOTE)
        # an invented sentence is not kept, and "strong" becomes "partial"
        invented = propose(
            evidence=[{"document": 1, "quote": "Le PIB baisse avec la pollution."}]
        )
        self.assertIsNone(invented.resources[0].quote)
        self.assertEqual(invented.support, "partial")

    def test_an_addition_naming_no_part_sits_in_its_connections_part(self):
        self.assertEqual(
            propose(
                connection(anchor="L’inflation"), kind="teaching", original=None
            ).original,
            "L’inflation",
        )
        # never for an assessment change, nor with an anchor not in the course
        self.assertIsNone(
            propose(
                connection(anchor="L’inflation"), kind="assessment", original=None
            ).original
        )
        self.assertIsNone(
            propose(
                connection(anchor="La biodiversité"), kind="teaching", original=None
            ).original
        )

    def test_keeps_the_change_with_its_connection(self):
        def place(original):
            return propose(original=original).original

        self.assertEqual(
            place("Le PIB : définition et mesure"), "Le PIB : définition et mesure"
        )
        # the topic that contains it: the same place, at a broader level
        self.assertEqual(
            place("Les grandeurs de la macroéconomie"),
            "Les grandeurs de la macroéconomie",
        )
        # a sibling subtopic: the change goes where its connection is
        self.assertEqual(place("L’inflation"), "Le PIB : définition et mesure")
        # a revised objective is never moved
        self.assertEqual(
            propose(
                kind="objective", original="Comprendre leur définition et leur mesure."
            ).original,
            "Comprendre leur définition et leur mesure.",
        )

    def test_a_revised_objective_revises_one_of_the_course(self):
        self.assertIsNone(propose(kind="objective", original="Explain incentives."))

    def test_strongly_supported_only_by_a_verified_sentence_of_a_direct_resource(self):
        context_only = connection(resources=(), background=(RESOURCE,))
        proposal = propose(context_only)
        self.assertEqual(proposal.support, "partial")
        self.assertEqual(proposal.resources[0].role, "context")

    def test_never_supported_without_a_cited_resource(self):
        self.assertEqual(propose(documents=[], evidence=[]).support, "none")
        self.assertEqual(propose(documents=[4], evidence=[]).support, "none")
        self.assertEqual(propose(support="partial").support, "partial")
        self.assertEqual(propose(support="maybe").support, "none")

    def test_refuses_an_unknown_kind_no_change_or_a_figure_not_in_the_resources(self):
        self.assertIsNone(propose(kind="activity"))
        self.assertIsNone(propose(suggested=""))
        self.assertIsNone(propose(why="Le PIB baisserait de 12 %."))

    def test_reads_notes_apart_and_a_greencomp_competence_only_from_the_framework(self):
        guidance = {
            "learning": "  analyser  les chaînes d’approvisionnement",
            "greencomp": {
                "id": "critical_thinking",
                "how": "les étudiants interrogent un indicateur :",
            },
            "teaching": "Une discussion de cas ?",
            "check": None,
        }
        notes = propose(guidance=guidance).guidance
        self.assertEqual(notes.learning, "Analyser les chaînes d’approvisionnement.")
        self.assertEqual(notes.greencomp.id, "critical_thinking")
        self.assertEqual(
            notes.greencomp.how, "Les étudiants interrogent un indicateur."
        )
        self.assertEqual(notes.teaching, "Une discussion de cas ?")
        self.assertIsNone(notes.check)

        loose = propose(
            guidance={
                "learning": "Le PIB baisserait de 12 %.",
                "greencomp": {"id": "green_skills", "how": "Un lien."},
                "teaching": "",
            }
        ).guidance
        self.assertEqual(
            (loose.learning, loose.greencomp, loose.teaching, loose.check),
            (None, None, None, None),
        )
        # a competence without its link explained is not kept
        self.assertIsNone(
            propose(
                guidance={"greencomp": {"id": "critical_thinking"}}
            ).guidance.greencomp
        )

    def test_tidy_note(self):
        self.assertEqual(tidy_note("  a  note :"), "A note.")
        self.assertEqual(tidy_note("Done!"), "Done!")
        self.assertEqual(tidy_note(" : "), "")


class VerificationTests(unittest.TestCase):
    ARTIFICIAL = (
        "In this artificial photosynthesis system, NADH regenerates the enzyme cascade "
        "for CO2 fixation."
    )

    def setUp(self):
        direct = EvidencedResource(
            document=doc("n", self.ARTIFICIAL), usefulness="U.", evidence="x"
        )
        self.resources = [
            RoledResource(direct, "direct"),
            RoledResource(direct, "context"),
        ]

    def verify(self, change="", **fields):
        return to_verification(
            VerificationAnswer.model_validate({"claims": [], **fields}),
            self.resources,
            change,
        )

    def test_a_claim_is_established_only_with_a_quote_of_a_direct_resource(self):
        change = (
            "NADH carries electrons in the Calvin cycle; NADH regenerates the cascade; "
            "context says so; Calvin uses NADH."
        )
        result = self.verify(
            change,
            claims=[
                {
                    "claim": "NADH carries electrons in the Calvin cycle",
                    "status": "established",
                    "document": 1,
                    "quote": "NADH carries electrons in the Calvin cycle.",
                },
                {
                    "claim": "NADH regenerates the cascade",
                    "status": "established",
                    "document": 1,
                    "quote": self.ARTIFICIAL,
                },
                {
                    "claim": "Context says so",
                    "status": "established",
                    "document": 2,
                    "quote": "x",
                },
                {
                    "claim": "Calvin uses NADH",
                    "status": "incorrect",
                    "problem": "the Calvin cycle uses NADPH",
                },
                {"status": "incorrect"},
            ],
            fit="forced",
            fit_problem="an analogy",
        )
        self.assertEqual(
            result,
            Verification(
                problems=[
                    "An analogy.",
                    "NADH carries electrons in the Calvin cycle.",
                    "Context says so.",
                    "The Calvin cycle uses NADPH.",
                ],
                has_incorrect=True,
                has_unestablished=True,
                is_forced=True,
            ),
        )

    def test_claims_from_the_documents_are_left_out_and_problems_are_few(self):
        change = "Discuss how moral development shapes choices about waste sorting."
        self.assertTrue(
            restates_change("Moral development shapes waste sorting choices", change)
        )
        self.assertFalse(
            restates_change("Kohlberg stage six rests on universal principles", change)
        )
        self.assertFalse(restates_change("a b c", change))
        result = self.verify(
            change,
            claims=[
                {
                    "claim": "Kohlberg stage six rests on universal principles",
                    "status": "not_established",
                },
                {
                    "claim": "Moral development shapes waste sorting",
                    "status": "not_established",
                },
                *(
                    {
                        "claim": f"Moral development shapes sorting {n}",
                        "status": "not_established",
                    }
                    for n in ("one", "two", "three", "four")
                ),
            ],
        )
        self.assertEqual(len(result.problems), MAX_PROBLEMS)
        self.assertEqual(result.problems[0], "Moral development shapes waste sorting.")

    def test_no_claim_is_a_clean_check_but_no_list_is_unusable(self):
        self.assertEqual(self.verify(), Verification([], False, False, False))
        with self.assertRaises(ValueError):
            VerificationAnswer.model_validate({"fit": "genuine"})


class NamesTests(unittest.TestCase):
    def setUp(self):
        nadh = EvidencedResource(
            document=doc("n", "In this system, NADH regenerates the enzyme cascade."),
            usefulness="U.",
            evidence="x",
        )
        self.direct = [RoledResource(nadh, "direct")]
        self.context = [RoledResource(nadh, "context")]

    def test_finds_the_names_a_change_states(self):
        self.assertEqual(
            names_in(
                "Discuss the Bull’s Head by Picasso and Rauschenberg. NADH and CO2 matter."
            ),
            ["Bull’s", "Head", "Picasso", "Rauschenberg", "NADH"],
        )
        self.assertEqual(names_in("Les ODD et le SDG 13, avec GreenComp."), [])
        self.assertEqual(
            names_in(
                "Les émissions de CO₂ comptent. Compare Earth-based fusion with NADPH."
            ),
            ["NADPH"],
        )

    def test_a_name_needs_a_direct_resource_or_the_course(self):
        self.assertEqual(
            unbacked_names(
                "Use NADH and the PIB with Rauschenberg.", self.direct, COURSE
            ),
            ["Rauschenberg"],
        )
        self.assertEqual(unbacked_names("Use NADH.", self.context, COURSE), ["NADH"])
        # a name in another language than the resource is found by its stem
        heat = EvidencedResource(
            document=doc("h", "The heatwave in Australia and the Pacific."),
            usefulness="U.",
            evidence="x",
        )
        self.assertEqual(
            unbacked_names(
                "En Australie et dans le Pacifique, selon Rauschenberg.",
                [RoledResource(heat, "direct")],
                COURSE,
            ),
            ["Rauschenberg"],
        )


class WithChecksTests(unittest.TestCase):
    CLEAN = Verification([], False, False, False)

    def setUp(self):
        nadh = EvidencedResource(
            document=doc("n", "NADH regenerates the enzyme cascade."),
            usefulness="U.",
            evidence="x",
        )
        self.direct = [RoledResource(nadh, "direct")]
        self.context = [RoledResource(nadh, "context")]
        chosen = connection(resources=(nadh,))
        self.proposal = to_proposal(
            answer(
                suggested="Show how NADH works.",
                why="W.",
                evidence=[
                    {"document": 1, "quote": "NADH regenerates the enzyme cascade."}
                ],
            ),
            COURSE,
            chosen,
            roled_resources(chosen),
        )

    def support(self, verification, resources=None, proposal=None):
        return with_checks(
            proposal or self.proposal, verification, resources or self.direct, COURSE
        )

    def test_never_raised_and_lowered_by_what_the_check_found(self):
        checked = self.support(self.CLEAN)
        self.assertEqual(checked.support, "strong")
        self.assertEqual(checked.checks.status, "checked")
        self.assertEqual(checked.checks.unbacked_names, [])
        partial = self.proposal.model_copy(update={"support": "partial"})
        self.assertEqual(self.support(self.CLEAN, proposal=partial).support, "partial")
        unestablished = Verification(["P."], False, True, False)
        result = self.support(unestablished)
        self.assertEqual((result.support, result.checks.problems), ("partial", ["P."]))
        self.assertEqual(
            self.support(Verification([], True, False, False)).support, "none"
        )

    def test_forced_keeps_some_support_only_with_a_direct_quote(self):
        forced = Verification([], False, False, True)
        self.assertEqual(self.support(forced).support, "partial")
        unquoted = self.proposal.model_copy(
            update={
                "resources": [
                    r.model_copy(update={"quote": None})
                    for r in self.proposal.resources
                ]
            }
        )
        self.assertEqual(self.support(forced, proposal=unquoted).support, "none")

    def test_a_name_only_a_context_resource_gives_and_no_check(self):
        result = self.support(self.CLEAN, resources=self.context)
        self.assertEqual(
            (result.support, result.checks.unbacked_names), ("partial", ["NADH"])
        )
        unchecked = self.support(None)
        self.assertEqual(
            (unchecked.support, unchecked.checks.status), ("partial", "unchecked")
        )


class WhatTheModelSeesTests(unittest.TestCase):
    def test_blocks(self):
        chosen = connection(background=(RESOURCE,))
        self.assertEqual(
            connection_block(chosen),
            "Title: Les limites du PIB\n"
            "Part of the course concerned: Le PIB : définition et mesure\n"
            "Why: Le PIB ne mesure pas les dégâts environnementaux.",
        )
        self.assertIn("the whole course", connection_block(connection(anchor=None)))
        blocks = resource_blocks(roled_resources(chosen))
        self.assertEqual([b.attrs["role"] for b in blocks], ["direct", "context"])
        self.assertEqual(blocks[0].text, QUOTE)
        self.assertEqual(problems_block(["A.", "B."]), "- A.\n- B.")

    def test_the_greencomp_list_has_the_twelve_competences(self):
        lines = greencomp_for_prompt().splitlines()
        self.assertEqual(len(lines), 12)
        self.assertEqual(len(GREENCOMP_IDS), 12)
        self.assertTrue(lines[4].startswith("- critical_thinking: assess information"))

    def test_the_prompts_render_with_and_without_resources_or_problems(self):
        variables = {"greencomp": greencomp_for_prompt()}
        first = render_prompt(
            "adaptation", "fr", {**variables, "count": 2, "has_problems": False}
        )
        again = render_prompt(
            "adaptation", "en", {**variables, "count": 2, "has_problems": True}
        )
        bare = render_prompt(
            "adaptation", "en", {**variables, "count": 0, "has_problems": False}
        )
        self.assertIn("The 2 documents in the <resource> blocks", first)
        self.assertIn("No competence is preferred", first)
        self.assertIn("- critical_thinking: assess information", first)
        self.assertNotIn("<problems>", first)
        self.assertIn("The claims in <problems>", again)
        self.assertIn("This connection has no WeLearn resource", bare)
        self.assertNotIn("<resource>", bare)
        check = render_prompt("verification", "fr", {"count": 2})
        self.assertIn("The 2 documents in the <resource> blocks", check)
        self.assertIn("in French", check)


class AdaptCourseTests(unittest.IsolatedAsyncioTestCase):
    GOOD = answer()
    SECOND = answer(suggested="Un exemple plus général sur le PIB.")

    def verification(self, *claims):
        return VerificationAnswer.model_validate(
            {"claims": list(claims), "fit": "genuine"}
        )

    WRONG_CLAIM = {
        "claim": "Ajouter un exemple de ce que le PIB ne compte pas",
        "status": "not_established",
        "problem": "Non établi.",
    }

    async def run_with(self, *answers, chosen=None):
        call = mock.AsyncMock(side_effect=list(answers))
        with mock.patch.object(adaptation, "structured_call", new=call):
            result = await adapt_course(COURSE, chosen or connection(), "fr")
        return result, call

    async def test_a_clean_check_keeps_the_first_proposal(self):
        result, call = await self.run_with(self.GOOD, self.verification())

        self.assertEqual(call.await_count, 2)
        name, lang, _, data, variables = call.await_args_list[0].args
        self.assertEqual((name, lang), ("adaptation", "fr"))
        self.assertEqual(data["course"], course_block(COURSE))
        self.assertEqual(len(data["resource"]), 1)
        self.assertNotIn("problems", data)
        self.assertEqual(variables["count"], 1)
        self.assertFalse(variables["has_problems"])
        self.assertIn("is_usable", call.await_args_list[0].kwargs)
        self.assertEqual(call.await_args_list[1].args[0], "verification")
        self.assertIn(
            "Proposed change: Ajouter", call.await_args_list[1].args[3]["proposal"]
        )
        self.assertEqual((result.support, result.checks.status), ("strong", "checked"))

    async def test_problems_ask_again_once_and_the_checked_second_replaces_the_first(
        self,
    ):
        result, call = await self.run_with(
            self.GOOD,
            self.verification(self.WRONG_CLAIM),
            self.SECOND,
            self.verification(),
        )

        self.assertEqual(call.await_count, 4)
        _, _, _, data, variables = call.await_args_list[2].args
        self.assertEqual(data["problems"], "- Non établi.")
        self.assertTrue(variables["has_problems"])
        self.assertEqual(result.suggested, "Un exemple plus général sur le PIB.")
        self.assertEqual(result.checks.problems, [])

    async def test_a_second_proposal_that_cannot_be_checked_is_not_kept(self):
        result, _ = await self.run_with(
            self.GOOD,
            self.verification(self.WRONG_CLAIM),
            self.SECOND,
            UnusableLLMOutputError(),
        )
        self.assertEqual(result.suggested, self.GOOD.suggested)
        self.assertEqual(result.checks.problems, ["Non établi."])
        self.assertEqual(result.support, "partial")

    async def test_an_unusable_second_proposal_keeps_the_first(self):
        result, call = await self.run_with(
            self.GOOD, self.verification(self.WRONG_CLAIM), UnusableLLMOutputError()
        )
        self.assertEqual(call.await_count, 3)
        self.assertEqual(result.suggested, self.GOOD.suggested)

    async def test_an_unusable_check_leaves_the_proposal_unchecked(self):
        result, _ = await self.run_with(self.GOOD, UnusableLLMOutputError())
        self.assertEqual(
            (result.support, result.checks.status), ("partial", "unchecked")
        )

    async def test_without_resources_nothing_is_checked(self):
        bare = connection(resources=())
        result, call = await self.run_with(
            answer(documents=[], evidence=[], support="none"), chosen=bare
        )
        self.assertEqual(call.await_count, 1)
        _, _, _, data, variables = call.await_args.args
        self.assertNotIn("resource", data)
        self.assertEqual(variables["count"], 0)
        self.assertEqual(result.support, "none")
        self.assertIsNone(result.checks)

    async def test_a_proposal_citing_no_resource_is_not_checked(self):
        # the connection has resources, but the change relies on none of them
        with self.assertLogs(adaptation.logger, level="INFO") as logs:
            result, call = await self.run_with(answer(documents=[], evidence=[]))

        self.assertEqual(call.await_count, 1)
        self.assertEqual(result.support, "none")
        self.assertIsNone(result.checks)
        self.assertIn("calls=1 check=none", logs.output[-1])

    async def test_one_log_line_with_counts_only(self):
        with self.assertLogs(adaptation.logger, level="INFO") as logs:
            await self.run_with(
                self.GOOD,
                self.verification(self.WRONG_CLAIM),
                self.SECOND,
                self.verification(),
            )

        line = logs.output[-1]
        self.assertIn(
            "tutor_adaptation_done calls=4 check=checked first_problems=1 "
            "first_serious=False redone=True "
            "second_kept=True final_problems=0 unbacked_names=0 support=strong kind=topic",
            line,
        )
        # never the course, the change or a problem
        for text in ("PIB", "Non établi", "Ajouter"):
            self.assertNotIn(text, line)

    async def test_no_usable_proposal_is_an_error(self):
        with self.assertRaises(UnusableLLMOutputError):
            await self.run_with(UnusableLLMOutputError())

    async def test_the_usable_check_refuses_what_to_proposal_refuses(self):
        call = mock.AsyncMock(return_value=self.GOOD)
        with mock.patch.object(adaptation, "structured_call", new=call):
            await adaptation.propose(
                COURSE, connection(), roled_resources(connection()), "en"
            )
        is_usable = call.await_args.kwargs["is_usable"]
        self.assertTrue(is_usable(self.GOOD))
        self.assertFalse(is_usable(answer(kind="activity")))
