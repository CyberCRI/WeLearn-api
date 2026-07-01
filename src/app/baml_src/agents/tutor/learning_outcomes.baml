// Learning Outcomes Agent
// Generates measurable learning outcomes (1-3 per objective)

function GenerateLearningOutcomes(
  objectives: LearningObjectives,
  sustainability_map: SustainabilityIntegration,
  metadata: CourseMetadata,
  output_language: string
) -> LearningOutcomes {
  client CustomFast
  prompt #"
    <system>
    You are a pedagogical assessment expert. You create measurable, observable learning outcomes that specify what students will be able to demonstrate upon completing the course.
    </system>

    <task>
    For each learning objective (including any suggested sustainability objectives), generate 1-3 specific, measurable outcomes. Naturally integrate sustainability themes informed by the sustainability mapping.
    </task>

    <inputs>
    <objectives>
    {% for obj in objectives.objectives %}
    {{ obj.number }}. {{ obj.text }} ({{ obj.bloom_level }})
    {% endfor %}
    </objectives>

    <sustainability_connections>
    {% for conn in sustainability_map.connections %}
    Objective {{ conn.objective_number }}: {{ conn.sdg_themes | join(", ") }}
    Connection: {{ conn.connection_explanation }}
    {% endfor %}
    </sustainability_connections>

    <metadata>
    - Discipline: {{ metadata.discipline }}
    - Level: {{ metadata.level }}
    - Duration: {{ metadata.num_sessions }} sessions
    </metadata>
    </inputs>

    <requirements>
    Each outcome should be:
    - Measurable and observable (use action verbs: demonstrate, analyze, create, evaluate, design, etc.)
    - Specific enough to guide activity design
    - Achievable within course duration
    - Related to 1-2 objectives (can span multiple)
    - Include assessment method (how to measure: exam, project, presentation, quiz, etc.)

    Integration of sustainability:
    - Weave sustainability naturally into outcomes (don't create separate "sustainability outcomes")
    - Use the sustainability connections to inform outcome design
    - Example: "Analyze supply chain efficiency using lifecycle thinking and environmental impact metrics"
      vs separate: "Analyze supply chains" + "Understand environmental impacts"

    Number of outcomes:
    - 1-3 outcomes per objective (more complex objectives may need more outcomes)
    - Total outcomes: typically 1.5x-2x number of objectives
    </requirements>

    <output_format>
    Return a JSON object with:
    {
      "outcomes": [
        {
          "number": 1,
          "text": "Students will demonstrate...",
          "related_objectives": [1, 2],
          "assessment_method": "Case study analysis and written report"
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
