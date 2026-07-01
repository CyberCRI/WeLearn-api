// Competency Mapping Agent
// Maps learning outcomes to GreenComp competencies

function MapCompetencies(
  outcomes: LearningOutcomes,
  greencomp_framework: map<string, string>?,
  output_language: string
) -> CompetencyMappings {
  client CustomFast
  prompt #"
    <system>
    You are an expert in the GreenComp framework (European sustainability competences). You map learning outcomes to relevant GreenComp competencies with clear pedagogical rationale.
    </system>

    <task>
    For each learning outcome, identify 1-3 relevant GreenComp competencies and provide brief rationale for the mapping.
    </task>

    <inputs>
    <outcomes>
    {% for outcome in outcomes.outcomes %}
    {{ outcome.number }}. {{ outcome.text }}
    (Related to objectives: {{ outcome.related_objectives | join(", ") }})
    {% endfor %}
    </outcomes>

    <greencomp_framework>
    Area 1: Embodying sustainability values
    - C1: Valuing sustainability (reflect on personal values, identify and explain how they guide actions)
    - C2: Supporting fairness (support equity and justice for current and future generations)
    - C3: Promoting nature (acknowledge humans are part of nature, respect needs of other species)

    Area 2: Embracing complexity in sustainability
    - C4: Systems thinking (approach sustainability challenges as complex systems)
    - C5: Critical thinking (assess information and arguments, identify assumptions and limitations)
    - C6: Problem framing (formulate current and future sustainability challenges as problems)

    Area 3: Envisioning sustainable futures
    - C7: Futures literacy (envision alternative sustainable futures)
    - C8: Adaptability (manage transitions and challenges in complex sustainability situations)
    - C9: Exploratory thinking (adopt relational thinking, explore and link different disciplines)

    Area 4: Acting for sustainability
    - C10: Political agency (navigate political systems, identify political responsibility)
    - C11: Collective action (act for change collaboratively)
    - C12: Individual initiative (identify own potential for sustainability, take action)
    </greencomp_framework>
    </inputs>

    <requirements>
    - Map each outcome to 1-3 GreenComp competencies (most relevant)
    - Provide brief rationale (1-2 sentences) explaining WHY this competency fits
    - Use official GreenComp codes and names (e.g., "C4: Systems thinking")
    - Ensure diversity across outcomes (don't map everything to C4, distribute across framework)
    - Be pedagogically honest - only map if there's genuine alignment
    </requirements>

    <examples>
    Good mapping:
    Outcome: "Analyze supply chain efficiency using lifecycle thinking and environmental impact metrics"
    Competencies: ["C4: Systems thinking", "C9: Exploratory thinking"]
    Rationale: "This outcome develops systems thinking (C4) by requiring students to view supply chains as interconnected systems with environmental feedback loops. It also promotes exploratory thinking (C9) by linking business operations (supply chain) with environmental science (lifecycle analysis)."
    </examples>

    <output_format>
    Return a JSON object with:
    {
      "mappings": [
        {
          "outcome_number": 1,
          "greencomp_competencies": ["C4: Systems thinking", "C9: Exploratory thinking"],
          "rationale": "Brief explanation..."
        },
        ...
      ]
    }
    </output_format>

    <output_language>
    Generate all outputs in {{ output_language }}.
    </output_language>
  "#
}
