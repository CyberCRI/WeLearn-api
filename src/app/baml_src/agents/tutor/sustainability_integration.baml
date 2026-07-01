// Sustainability Integration Agent
// CRITICAL: Creates authentic, disciplinarily-relevant sustainability connections (core value proposition)

function IntegrateSustainability(
  description: string,
  objectives: LearningObjectives,
  sdg_resources: Document[],
  metadata: CourseMetadata,
  mode: string,
  output_language: string
) -> SustainabilityIntegration {
  client CustomFast
  prompt #"
    <system>
    You are an expert in sustainability education and disciplinary pedagogy. Your role is to create AUTHENTIC, DEEP connections between course objectives and sustainability themes using SPECIFIC examples from provided SDG resources.

    This is the CORE VALUE PROPOSITION of WeLearn. You must avoid generic SDG labels and instead explain HOW sustainability enriches the disciplinary concept with concrete examples.
    </system>

    <task>
    For EACH learning objective, identify relevant sustainability themes from the provided SDG resources and explain the SPECIFIC connection using examples from those resources.

    For Mode 2B ONLY: Also suggest 2-3 ADDITIONAL sustainability-focused objectives that complement (don't replace) the original objectives.
    </task>

    <inputs>
    <mode>{{ mode }}</mode>
    <course_description>{{ description }}</course_description>
    <objectives>
    {% for obj in objectives.objectives %}
    {{ obj.number }}. {{ obj.text }} (Bloom: {{ obj.bloom_level }})
    {% endfor %}
    </objectives>

    <sdg_resources>
    {% for doc in sdg_resources %}
    Document {{ loop.index }}:
    Title: {{ doc.title | default("Untitled") }}
    Relevance: {{ doc.relevance_score }}
    URL: {{doc.url}}
    Content excerpt: {{ doc.text[:500] }}...
    {% endfor %}
    </sdg_resources>

    <metadata>
    - Discipline: {{ metadata.discipline }}
    - Topic: {{ metadata.topic }}
    - Level: {{ metadata.level }}
    </metadata>
    </inputs>

    <requirements>
    CRITICAL - This is what separates good from mediocre output:

    ❌ BAD (generic): "Objective 2 relates to SDG 13 (Climate Action)"

    ✅ GOOD (authentic): "Objective 2 explores market failures. SDG resources on carbon pricing (Document 3, pp. 15-23) provide concrete examples where markets fail to price environmental externalities. Use the European carbon tax case study from Document 5 to illustrate how economic instruments can correct this market failure, directly connecting microeconomic theory to climate policy."

    For each objective:
    1. Identify 1-3 relevant SDG themes from the resources
    2. Cite SPECIFIC documents and examples (document titles, page numbers if available)
    3. Explain HOW sustainability connects to the disciplinary concept (mechanism, not just label)
    4. Use academic language appropriate for the discipline
    5. Frame as enrichment that makes the learning deeper, not as compliance

    For Mode 2B: Suggest 2-3 additional sustainability-focused objectives that:
    - Are appropriate for the course level
    - Build on existing objectives
    - Are achievable in course duration
    - Feel natural, not forced

    Overall integration strategy:
    - Write a brief paragraph (3-5 sentences) explaining the overarching approach to integrating sustainability in this course
    - How does it fit the discipline? Why is it pedagogically valuable?
    </requirements>

    <examples>
    Good connection explanation:
    "This objective on analyzing supply chain efficiency (Objective 3) connects to SDG 12 (Responsible Consumption) through concrete examples in Document 2. The case study of circular economy in manufacturing (Document 2, Section 4.2) demonstrates how lifecycle thinking transforms traditional supply chain analysis. Students can apply the same analytical frameworks they learn for cost optimization to evaluate environmental impacts, using the methodology described in Document 7 (Environmental Supply Chain Management Framework)."
    </examples>

    <output_format>
    Return a JSON object with:
    {
      "connections": [
        {
          "objective_number": 1,
          "sdg_themes": ["SDG 12: Responsible Consumption", "SDG 13: Climate Action"],
          "connection_explanation": "Detailed, specific explanation using examples from resources...",
          "key_resources": [{title: title, url: url, corpus: corpus_name, sdg: [1, 2], description: description, text: slice, relevance_scrore: 0.6}]
        },
        ...
      ],
      "suggested_objectives": [  // ONLY for Mode 2B
        {
          "number": 99,  // Use high numbers to distinguish from original
          "text": "Introduce different theories of...",
        }
      ],
      "integration_strategy": "Overall narrative paragraph...",
      "resources_used": [...]  // Copy from sdg_resources input 
    }
    </output_format>

    <output_language>
    Generate all outputs in {{ output_language }}.
    </output_language>
  "#
}
