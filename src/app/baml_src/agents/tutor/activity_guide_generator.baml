// Activity Guide Generator Agent
// Personalizes activity template for specific course context

function GenerateActivityGuide(
  activity_template: ActivityTemplate,
  session: SessionPlan,
  metadata: CourseMetadata,
  sustainability_map: SustainabilityIntegration,
  output_language: string
) -> ActivityGuide {
  client CustomFast
  prompt #"
    <system>
    You are a pedagogical designer who creates ready-to-use activity guides for professors. You take generic activity templates and personalize them with course-specific content, examples, and sustainability integration.
    </system>

    <task>
    Transform this generic activity template into a detailed, ready-to-use guide for this specific course. The professor should be able to follow this guide directly to implement the activity.
    </task>

    <inputs>
    <activity_template>
    Name: {{ activity_template.name }}
    Description: {{ activity_template.description }}
    Estimated duration: {{ activity_template.estimated_duration }}h
    </activity_template>

    <session>
    Session {{ session.session_number }}
    Objectives: {{ session.objectives | join(", ") }}
    Duration: {{ session.duration }}h
    Class size: {{ session.class_size }}
    </session>

    <course_context>
    Discipline: {{ metadata.discipline }}
    Topic: {{ metadata.topic }}
    Level: {{ metadata.level }}
    Format: {{ metadata.session_type }} - {{ metadata.session_mode }}
    </course_context>

    <sustainability_context>
    {% for conn in sustainability_map.connections %}
    Objective {{ conn.objective_number }}: {{ conn.connection_explanation[:200] }}...
    {% endfor %}
    </sustainability_context>
    </inputs>

    <requirements>
    Create a detailed guide with:

    1. Activity description: Adapted to this course context (mention discipline, topic)

    2. Learning objectives: Which objective numbers (from session.objectives) this addresses

    3. Detailed steps: Concrete procedure professor can follow
       - Be specific (not "discuss the topic" but "pose question X, give 5min for pairs to discuss, then collect 3 key points")
       - Include timing for each step
       - Use course-specific examples from the discipline

    4. Teacher role: What professor does during activity

    5. Student role: What students do, how they participate

    6. Resources needed: Specific materials, handouts, tools
       - Course-specific resources (not generic)
       - Include digital tools if mode is Distanciel/Hybride

    7. Timing breakdown: Detailed time allocation
       - Example: {"introduction": 10, "pair_work": 20, "group_discussion": 30, "debrief": 10}
       - Total should match activity template's estimated_duration

    8. Evaluation method: How to assess if learning happened
       - Formative assessment integrated into activity

    9. Sustainability integration: HOW sustainability themes are woven into this activity
       - Use sustainability connections for relevant objectives
       - Be specific and natural (not tacked on)
    </requirements>

    <examples>
    GENERIC: "Students discuss the topic in groups"
    SPECIFIC: "Students analyze the case study of [discipline-specific example] in groups of 4. Each group receives a worksheet with 3 guiding questions: (1) What are the key [disciplinary concepts]? (2) How do [sustainability themes] factor into the analysis? (3) What trade-offs exist? Groups have 20 minutes to discuss and prepare a 2-minute summary."

    GENERIC: "Use sustainability principles"
    SPECIFIC: "During the case analysis, students explicitly apply lifecycle thinking from SDG 12 resources. They map the full value chain, identifying environmental externalities at each stage. This connects to Objective 3's focus on system boundaries in [discipline]."
    </examples>

    <output_format>
    Return a JSON object with:
    {
      "activity_name": "Adapted name if needed",
      "description": "Course-specific description...",
      "learning_objectives": [1, 3],  // Objective numbers
      "steps": ["Step 1...", "Step 2...", ...],
      "teacher_role": "What professor does...",
      "student_role": "What students do...",
      "resources_needed": ["Resource 1", "Resource 2", ...],
      "timing_breakdown": {"phase1": 15, "phase2": 30, ...},
      "evaluation_method": "How to assess...",
      "sustainability_integration": "How sustainability is woven in..."
    }
    </output_format>

    <output_language>
    Generate all outputs in {{ output_language }}.
    </output_language>
  "#
}
