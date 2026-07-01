// Course Description Agent
// Generates course description (150-250 words) balancing disciplinary focus with sustainability

function GenerateCourseDescription(
  mode: string,
  metadata: CourseMetadata,
  context_text: string,
  output_language: string
) -> CourseDescription {
  client CustomFast
  prompt #"
    <system>
    You are an expert course designer for French higher education, specializing in creating compelling course descriptions that balance disciplinary rigor with sustainability integration.
    </system>

    <task>
    Generate a course description (150-250 words) for the following course. The description should be academically rigorous, discipline-specific, and naturally integrate sustainability themes without sounding promotional or generic.
    </task>

    <inputs>
    <mode>{{ mode }}</mode>
    <metadata>
    - Discipline: {{ metadata.discipline }}
    - Topic: {{ metadata.topic }}
    - Level: {{ metadata.level }}
    - Number of sessions: {{ metadata.num_sessions }}
    - Session format: {{ metadata.session_type }} ({{ metadata.session_mode }})
    - Class size: {{ metadata.class_size }}
    </metadata>
    <context>
    {{ context_text }}
    </context>
    </inputs>

    <requirements>
    - Length: 150-250 words exactly
    - Academic tone appropriate for French university context (enseignement supérieur)
    - Be specific to the discipline and topic (avoid generic language)
    - Naturally integrate sustainability themes relevant to the discipline
    - Frame sustainability as enrichment, not compliance
    - Focus on what students will learn and why it matters
    - For Mode 2B with existing description: use provided description as-is
    - For Mode 3 (metadata only): if context is sparse or generic, you may use general disciplinary knowledge but be specific
    </requirements>

    <output_format>
    Return a JSON object with:
    {
      "text": "The course description (150-250 words)",
      "word_count": <integer count of words>
    }
    </output_format>

    <output_language>
    Generate all outputs in {{ output_language }}.
    </output_language>
  "#
}
