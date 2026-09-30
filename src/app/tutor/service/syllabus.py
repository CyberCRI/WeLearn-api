"""Single-pass syllabus generation (old tutor).

The LLM fills a SyllabusDraft; headings, limits, GreenComp filtering and the
references section are handled here in code, so they can't be hallucinated.
"""

import re
from dataclasses import dataclass
from math import ceil
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.app.tutor.service.models import SyllabusDraft
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

GREENCOMP_COMPETENCIES = (
    "Here are the GreenComp competencies: "
    "url: https://joint-research-centre.ec.europa.eu/greencomp-european-sustainability-competence-framework_en "
    "1.1 Valuing sustainability: To reflect on personal values; identify and explain how values vary among people "
    "and over time, while critically evaluating how they align with sustainability values. "
    "1.2 Supporting fairness: To support equity and justice for current and future generations and learn from previous "
    "generations for sustainability. "
    "1.3 Promoting nature: To acknowledge that humans are part of nature; and to respect the needs and rights of other "
    "species and of nature itself in order to restore and regenerate healthy and resilient ecosystems. "
    "2.1 Systems thinking: To approach a sustainability problem from all sides; to consider time, space and context in "
    "order to understand how elements interact within and between systems. "
    "2.2 Critical thinking: To assess information and arguments, identify assumptions, challenge the status quo, and "
    "reflect on how personal, social and cultural backgrounds influence thinking and conclusions. "
    "2.3 Problem framing: To formulate current or potential challenges as a sustainability problem in terms of "
    "difficulty, people involved, time and geographical scope, in order to identify suitable approaches to anticipating "
    "and preventing problems, and to mitigating and adapting to already existing problems. "
    "3.1 Futures literacy: To envision alternative sustainable futures by imagining and developing alternative scenarios "
    "and identifying the steps needed to achieve a preferred sustainable future. "
    "3.2 Adaptability: To manage transitions and challenges in complex sustainability situations and make decisions "
    "related to the future in the face of uncertainty, ambiguity and risk. "
    "3.3 Exploratory thinking: To adopt a relational way of thinking by exploring and linking different disciplines, "
    "using creativity and experimentation with novel ideas or methods. "
    "4.1 Political agency: To navigate the political system, identify political responsibility and accountability for "
    "unsustainable behaviour, and demand effective policies for sustainability. "
    "4.2 Collective action: To act for change in collaboration with others. "
    "4.3 Individual initiative: To identify own potential for sustainability and to actively contribute to improving "
    "prospects for the community and the planet."
)
# official names; French from the JRC's French edition of GreenComp (JRC128040)
GREENCOMP_NAMES = {
    "1.1": {
        "en": "Valuing sustainability",
        "fr": "Accorder de la valeur à la durabilité",
    },
    "1.2": {"en": "Supporting fairness", "fr": "Encourager l'équité"},
    "1.3": {"en": "Promoting nature", "fr": "Promouvoir la nature"},
    "2.1": {"en": "Systems thinking", "fr": "Pensée systémique"},
    "2.2": {"en": "Critical thinking", "fr": "Pensée critique"},
    "2.3": {"en": "Problem framing", "fr": "Cadrage des problèmes"},
    "3.1": {"en": "Futures literacy", "fr": "Littératie des futurs"},
    "3.2": {"en": "Adaptability", "fr": "Adaptabilité"},
    "3.3": {"en": "Exploratory thinking", "fr": "Pensée exploratoire"},
    "4.1": {"en": "Political agency", "fr": "Agentivité politique"},
    "4.2": {"en": "Collective action", "fr": "Action collective"},
    "4.3": {"en": "Individual initiative", "fr": "Initiative individuelle"},
}
GREENCOMP_NAMES_GUIDE = "; ".join(
    f"{code} {names['en']} / {names['fr']}" for code, names in GREENCOMP_NAMES.items()
)
MIN_GREENCOMP = 1

# (up to N sessions, objectives, outcomes, competencies incl. GreenComp).
# Teaching-centre guidance is 3-5 / 4-8 outcomes per full course, ECTS 6-8 per module,
# so outcomes grow slowly with length and stay under 8.
COUNTS_BY_SESSIONS = [
    (4, 2, 3, 3),
    (8, 3, 4, 3),
    (12, 3, 5, 4),
    (16, 4, 6, 4),
    (20, 4, 7, 4),
]
COUNTS_GUIDE = "; ".join(
    f"up to {n} sessions: {o} objectives, {lo} outcomes, {c} competencies"
    for n, o, lo, c in COUNTS_BY_SESSIONS
)

HEADINGS = {
    "en": {
        "description": "1. Course Description",
        "objectives": "2. Learning Objectives",
        "outcomes": "3. Learning Outcomes",
        "competencies": "4. Competencies Developed",
        "assessment": "5. Assessment Methods",
        "schedule": "6. Course Schedule",
        "references": "7. References",
        "table": "| Week | Topics | Learning Outcomes | Class Plan |",
        "lo": "LO",
        "objectives_ref": "Objectives:",
        "link": "link",
        "no_references": "No WeLearn resource was used.",
    },
    "fr": {
        "description": "1. Description du cours",
        "objectives": "2. Objectifs d'apprentissage",
        "outcomes": "3. Résultats d'apprentissage",
        "competencies": "4. Compétences développées",
        "assessment": "5. Modalités d'évaluation",
        "schedule": "6. Programme du cours",
        "references": "7. Références",
        "table": "| Semaine | Thèmes | Résultats d'apprentissage | Plan de séance |",
        "lo": "RA",
        "objectives_ref": "Objectifs :",
        "link": "lien",
        "no_references": "Aucune ressource WeLearn n'a été utilisée.",
    },
}
LANG_NAMES = {"en": "English", "fr": "French"}

# matches a references heading in LLM/user markdown: "## 7. Références", "**References**", ...
_REFERENCES_HEADING = re.compile(
    r"^\s*(#+\s*|\*\*\s*)?(\d+\.?\s*)?(r[ée]f[ée]rences?|bibliograph)", re.IGNORECASE
)


@dataclass
class Limits:
    sessions: int | None
    objectives: int
    outcomes: int
    competencies: int


def compute_limits(duration: str | None) -> Limits:
    # ponytail: first number + unit in free text ("12 semaines", "30h"); a structured
    # duration field in the client would remove the guessing.
    match = re.search(r"(\d+)\s*([a-zA-Zéè]*)", duration or "")
    sessions = None
    if match:
        sessions, unit = int(match[1]), match[2].lower()
        if unit.startswith(("h", "hour", "heure")):
            sessions = ceil(sessions / 3)
        elif unit.startswith(("mo", "month")):
            sessions *= 4
        elif unit.startswith("sem") and "semaine" not in unit:  # semestre/semester
            sessions *= 12
        sessions = max(1, min(sessions, 20))
    n = sessions or 12  # unknown duration: size it like a standard semester course
    _, objectives, outcomes, competencies = next(
        row for row in COUNTS_BY_SESSIONS if n <= row[0]
    )
    return Limits(sessions, objectives, outcomes, competencies)


def detect_syllabus_lang(markdown: str) -> str:
    """Language of a syllabus rendered by this module, from its headings."""
    for lang, h in HEADINGS.items():
        if f"## {h['objectives']}" in markdown:
            return lang
    return "unknown"


def split_references(markdown: str) -> tuple[str, str]:
    """Split a syllabus into (body, references section). References may be ''."""
    lines = markdown.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if _REFERENCES_HEADING.match(lines[i]):
            return "\n".join(lines[:i]).rstrip(), "\n".join(lines[i:]).strip()
    return markdown.rstrip(), ""


def trim_chatter(body: str) -> str:
    """Drop LLM chatter before the first heading and after the schedule table."""
    lines = body.strip().split("\n")
    start = next((i for i, line in enumerate(lines) if line.startswith("#")), 0)
    end = max(
        (i for i, line in enumerate(lines) if line.lstrip().startswith("|")),
        default=len(lines) - 1,
    )
    return "\n".join(lines[start : end + 1]).strip()


def render_references(resources: list[dict], lang: str) -> str:
    h = HEADINGS.get(lang, HEADINGS["en"])
    seen, items = set(), []
    for res in resources:
        url, title = res.get("url") or "", res.get("title") or res.get("url") or ""
        key = url or title
        if not key or key in seen:
            continue
        seen.add(key)
        link = f' <a href="{url}" target="_blank">[{h["link"]}]</a>' if url else ""
        items.append(f"- {title}{link}")
    return f"## {h['references']}\n\n" + ("\n".join(items) or h["no_references"])


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", "<br>").strip()


def _refs(prefix: str, numbers: list[int], max_n: int) -> str:
    return ", ".join(f"{prefix}{n}" for n in numbers if 1 <= n <= max_n)


def render_markdown(draft: SyllabusDraft, lang: str, limits: Limits) -> str:
    h = HEADINGS.get(lang, HEADINGS["en"])
    objectives = draft.objectives[: limits.objectives]
    outcomes = draft.outcomes[: limits.outcomes]
    n_obj, n_lo, lo = len(objectives), len(outcomes), h["lo"]

    greencomp, others, seen_codes = [], [], set()
    for comp in draft.competencies:
        linked = _refs(lo, comp.outcome_numbers, n_lo)
        suffix = f" *({linked})*" if linked else ""
        code = (comp.greencomp_code or "").strip()
        if code not in GREENCOMP_NAMES:
            others.append(f"- {comp.text}{suffix}")
        elif linked and code not in seen_codes:
            # GreenComp only when it supports a listed outcome, each code once
            seen_codes.add(code)
            name = GREENCOMP_NAMES[code].get(lang, GREENCOMP_NAMES[code]["en"])
            greencomp.append(f"- **GreenComp {code} – {name}** : {comp.text}{suffix}")
    if not greencomp:
        logger.warning(
            "Syllabus draft has no GreenComp competency linked to an outcome"
        )
    greencomp = greencomp[: limits.competencies]
    competencies = greencomp + others[: limits.competencies - len(greencomp)]

    sections = [f"# {draft.course_title}", f"## {h['description']}", draft.description]
    sections += [
        f"## {h['objectives']}",
        "\n".join(f"{i}. {o}" for i, o in enumerate(objectives, 1)),
    ]
    sections += [
        f"## {h['outcomes']}",
        "\n".join(
            f"- **{lo}{i}** {o.text}"
            + (
                f" *({h['objectives_ref']} {refs})*"
                if (refs := _refs("", o.objective_numbers, n_obj))
                else ""
            )
            for i, o in enumerate(outcomes, 1)
        ),
    ]
    sections += [f"## {h['competencies']}", "\n".join(competencies)]
    sections += [
        f"## {h['assessment']}",
        "\n".join(
            f"- **{a.method}** ({a.weight})"
            + (f" — {refs}" if (refs := _refs(lo, a.outcome_numbers, n_lo)) else "")
            for a in draft.assessment
        ),
    ]
    table = [h["table"], "|---|---|---|---|"] + [
        f"| {i} | {_cell(s.topics)} | {_refs(lo, s.outcome_numbers, n_lo)} | {_cell(s.class_plan)} |"
        for i, s in enumerate(draft.schedule, 1)
    ]
    sections += [f"## {h['schedule']}", "\n".join(table)]
    return "\n\n".join(sections)


SYSTEM_PROMPT = (
    "You are an experienced university professor and pedagogical engineer, expert in "
    "competency-based course design, active learning and sustainability education (UN SDGs, "
    "EU GreenComp framework). You design one realistic, coherent syllabus for a real teacher. "
    "Your institute's pedagogy is active and student-centred: students learn by doing, discussing, "
    "investigating and creating, and the teacher acts as a facilitator.\n\n"
    "Rules:\n"
    "- Ground the course in the teacher's documents (summaries and themes). Use the WeLearn "
    "resources only as supporting content; never invent sources, and do not write references.\n"
    "- Weave sustainability into the course content where it connects to the discipline and topics.\n"
    "- Write every field as the final syllabus the teacher hands to students. Never comment on your "
    "design choices, these rules or the frameworks (no sentences like 'sustainability is not forced' or "
    "'a GreenComp competency is integrated'); frameworks appear only in the competencies section.\n"
    "- Learning objectives are broad and teacher-centred (what the course covers). Learning outcomes "
    "are student-centred, observable and measurable, start with a strong action verb, and name the "
    "activity or assessment through which they are demonstrated. Each outcome serves at least one objective.\n"
    "- Competencies are transferable skills, each linked to the outcomes that develop it. Map each "
    "competency to the GreenComp competency it corresponds to (e.g. critical analysis of discourses -> 2.2, "
    "understanding interactions between systems -> 2.1) and set its code; at least one competency must be "
    "GreenComp, each code at most once. Leave the code empty only for skills with no GreenComp equivalent "
    "(e.g. a disciplinary method). The competency text describes how it applies in this course; do not "
    "repeat the GreenComp code or name in it. Never list the whole framework.\n"
    "- Assessment methods evaluate the outcomes; weights add up to 100%. Favour authentic assessment "
    "(projects, case analyses, presentations, portfolios, peer and self-assessment) over exams alone.\n"
    "- The schedule has one row per session; each session targets outcomes from the list. Every outcome "
    "is targeted at least once. No placeholders or ellipses.\n"
    "- Class plans are student-centred: most of each session is active work such as project- or "
    "problem-based learning, case studies, debates and role plays, peer instruction, think-pair-share, "
    "flipped classroom, fieldwork, workshops and co-construction. Teacher input is short (15 minutes "
    "at most) and serves the activity; a session is never mainly a lecture.\n"
    "- Quality over quantity: aim for exactly the counts given by the user, with sharp, distinct items.\n"
    "- Write every field in the requested language."
)


def build_user_prompt(
    message: Any, limits: Limits, disciplinary_skills: list[str]
) -> str:
    lang = LANG_NAMES.get(message.lang, message.lang)
    themes = ", ".join(t["theme"] for t in message.themes)
    summaries = "\n\n".join(message.summary)
    # ponytail: slices truncated to 1500 chars to bound prompt size (latency)
    resources = "\n\n".join(
        f"- {r['title']}: {r['content'][:1500]}"
        for r in message.resources
        if r.get("title")
    )
    sessions = (
        f"exactly {limits.sessions} sessions"
        if limits.sessions
        else "a realistic number of sessions (usually 10 to 12)"
    )
    parts = [
        f"Language: {lang}. Every field must be written in {lang}.",
        f"Course title: {message.course_title or 'propose a short title'}",
        f"Level: {message.level or 'not specified'}",
        f"Duration: {message.duration or 'not specified'}",
        f"Teacher's description: {message.description or 'none'}",
        f"Aim for exactly: {limits.objectives} learning objectives, {limits.outcomes} learning outcomes, "
        f"{limits.competencies} competencies (at least {MIN_GREENCOMP} from GreenComp). "
        f"Schedule: {sessions}.",
    ]
    if disciplinary_skills:
        parts.append(
            "The course should also build these disciplinary skills:\n- "
            + "\n- ".join(disciplinary_skills)
        )
    parts += [
        f"TEACHER'S DOCUMENTS (summaries):\n{summaries}",
        f"THEMES:\n{themes}",
        f"WELEARN RESOURCES:\n{resources or 'none'}",
        GREENCOMP_COMPETENCIES,
    ]
    return "\n\n".join(parts)


async def generate_syllabus(
    message: Any,
    model: BaseChatModel,
    disciplinary_skills: list[str],
    config: RunnableConfig,
) -> str:
    limits = compute_limits(message.duration)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{user_prompt}")]
    )
    chain = prompt | model.with_structured_output(SyllabusDraft)
    result = await chain.ainvoke(
        {"user_prompt": build_user_prompt(message, limits, disciplinary_skills)},
        config=config,
    )
    draft = SyllabusDraft.model_validate(result, from_attributes=True)
    if message.course_title:
        draft.course_title = message.course_title
    return (
        render_markdown(draft, message.lang, limits)
        + "\n\n"
        + render_references(message.resources, message.lang)
    )
