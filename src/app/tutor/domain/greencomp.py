"""
The twelve GreenComp competences (European Commission, 2022, "GreenComp: the European
sustainability competence framework"). A competence is a learning outcome, not a teaching
method. Each comes with a short definition for the model, so a competence is chosen for what
students actually do, not for a shared word. The labels shown to teachers live in the client.
"""

from typing import Literal, get_args

GreenCompId = Literal[
    "valuing_sustainability",
    "supporting_fairness",
    "promoting_nature",
    "systems_thinking",
    "critical_thinking",
    "problem_framing",
    "futures_literacy",
    "adaptability",
    "exploratory_thinking",
    "political_agency",
    "collective_action",
    "individual_initiative",
]
GREENCOMP_IDS: tuple[str, ...] = get_args(GreenCompId)

DEFINITIONS: dict[str, str] = {
    "valuing_sustainability": (
        "reflect on one's own and others' values and evaluate how they align with "
        "sustainability"
    ),
    "supporting_fairness": "support equity and justice for current and future generations",
    "promoting_nature": (
        "acknowledge that humans are part of nature and respect the needs of other species "
        "and ecosystems"
    ),
    "systems_thinking": (
        "approach a sustainability problem from all sides, considering time, space and "
        "context, and understand how elements interact within and between systems"
    ),
    "critical_thinking": (
        "assess information and arguments, identify assumptions and challenge the status quo"
    ),
    "problem_framing": (
        "frame a challenge as a sustainability problem (difficulty, people involved, time "
        "and geographical scope) to choose a suitable approach"
    ),
    "futures_literacy": (
        "envision alternative sustainable futures, develop scenarios and identify the steps "
        "to reach them"
    ),
    "adaptability": (
        "manage transitions and make decisions in the face of uncertainty, ambiguity and risk"
    ),
    "exploratory_thinking": (
        "link different disciplines and experiment with new ideas or methods"
    ),
    "political_agency": (
        "navigate the political system, identify responsibility and demand effective policies"
    ),
    "collective_action": "act for change together with others",
    "individual_initiative": (
        "identify one's own potential to act and actively contribute to sustainability"
    ),
}


def greencomp_for_prompt() -> str:
    """The list the model chooses from: one `- id: definition` line per competence."""
    return "\n".join(f"- {id_}: {DEFINITIONS[id_]}" for id_ in GREENCOMP_IDS)
