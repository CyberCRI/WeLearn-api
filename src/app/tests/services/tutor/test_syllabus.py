from src.app.tutor.service.models import (
    DraftAssessment,
    DraftCompetency,
    DraftOutcome,
    DraftSession,
    SyllabusDraft,
)
from src.app.tutor.service.syllabus import (
    compute_limits,
    detect_syllabus_lang,
    render_markdown,
    render_references,
    split_references,
    trim_chatter,
)


def make_draft() -> SyllabusDraft:
    return SyllabusDraft(
        course_title="Économie circulaire",
        description="Desc",
        objectives=["O1", "O2", "O3", "O4", "O5"],
        outcomes=[
            DraftOutcome(text=f"Out {i}", objective_numbers=[1]) for i in range(1, 10)
        ],
        competencies=[
            DraftCompetency(text="Travail en équipe", outcome_numbers=[1]),
            DraftCompetency(
                text="Systémique", greencomp_code="2.1", outcome_numbers=[2]
            ),
            DraftCompetency(text="Non liée", greencomp_code="2.2", outcome_numbers=[]),
            DraftCompetency(text="Critique", greencomp_code="2.2", outcome_numbers=[3]),
            DraftCompetency(text="Doublon", greencomp_code="2.2", outcome_numbers=[1]),
            DraftCompetency(
                text="Faux code", greencomp_code="9.9", outcome_numbers=[1]
            ),
        ],
        assessment=[
            DraftAssessment(method="Projet", weight="100%", outcome_numbers=[1, 99])
        ],
        schedule=[DraftSession(topics="A | B", outcome_numbers=[1], class_plan="x\ny")],
    )


def test_compute_limits():
    assert compute_limits(None).sessions is None
    assert compute_limits(None).outcomes == 5  # sized like a semester course
    assert compute_limits("4 semaines").outcomes == 3
    assert compute_limits("8 semaines").outcomes == 4  # grows with duration
    assert compute_limits("8 semaines").objectives == 3
    assert compute_limits("12 semaines").sessions == 12
    assert compute_limits("30h").sessions == 10
    assert compute_limits("1 semestre").sessions == 12
    assert compute_limits("40 weeks").outcomes == 7  # ceiling
    assert compute_limits("40 weeks").objectives == 4


def test_render_markdown_fr():
    md = render_markdown(make_draft(), "fr", compute_limits("12 semaines"))
    assert md.startswith("# Économie circulaire")
    assert "## 3. Résultats d'apprentissage" in md and "Learning" not in md
    assert "**RA5**" in md and "**RA6**" not in md  # capped at 5 outcomes
    assert "4. O4" not in md  # capped at 3 objectives
    assert "Non liée" not in md and "Doublon" not in md  # unlinked / duplicate code
    # official name in the syllabus language, GreenComp listed first
    assert "**GreenComp 2.1 – Pensée systémique** : Systémique *(RA2)*" in md
    assert "**GreenComp 2.2 – Pensée critique** : Critique" in md
    assert md.index("GreenComp 2.1") < md.index("Travail en équipe")
    assert "- Faux code" in md  # unknown code -> plain competency
    assert "RA99" not in md
    assert "| 1 | A / B | RA1 | x<br>y |" in md


def test_references_are_deterministic():
    refs = render_references(
        [
            {"title": "Doc A", "url": "https://a.org"},
            {"title": "Doc A", "url": "https://a.org"},  # second slice of same doc
            {"title": "Doc B", "url": "https://b.org"},
        ],
        "fr",
    )
    assert refs.startswith("## 7. Références")
    assert refs.count("https://a.org") == 1
    assert '<a href="https://b.org" target="_blank">[lien]</a>' in refs
    assert "Aucune" in render_references([], "fr")


def test_split_and_trim():
    md = (
        "Voici votre syllabus :\n# T\n## 6. Programme\n| a |\n|---|\n"
        "Ce syllabus est aligné avec GreenComp.\n## 7. Références\n- Doc <a>x</a>"
    )
    body, refs = split_references(md)
    assert refs == "## 7. Références\n- Doc <a>x</a>"
    assert trim_chatter(body) == "# T\n## 6. Programme\n| a |\n|---|"
    assert split_references("# T\nno refs") == ("# T\nno refs", "")


def test_detect_syllabus_lang():
    assert (
        detect_syllabus_lang(render_markdown(make_draft(), "fr", compute_limits(None)))
        == "fr"
    )
    assert (
        detect_syllabus_lang(render_markdown(make_draft(), "en", compute_limits(None)))
        == "en"
    )
    assert detect_syllabus_lang("# free text") == "unknown"
