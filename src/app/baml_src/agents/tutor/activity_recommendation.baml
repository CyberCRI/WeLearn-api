// Activity Recommendation Agent
// Ranks and selects 2-4 pedagogical activities for session based on learning outcomes

function RecommendActivities(
  session: SessionPlan,
  learning_outcomes: LearningOutcomes,
  filtered_activities: ActivityTemplate[],
  greencomp_methods: string[],
  output_language: string
) -> ActivityRecommendations {
  client CustomFast
  prompt #"
    <system>
    You are a pedagogical activity design expert. You select and rank activities based on learning outcomes, pedagogical fit, variety, and feasibility.
    IMPORTANT: Activities should help students achieve the LEARNING OUTCOMES (what students will be able to demonstrate) rather than just covering objectives.
    </system>

    <task>
    From the pre-filtered activities, select 2-4 that best enable students to achieve the learning outcomes for this session. Ensure variety (mix of active/reflective, individual/group) and feasibility (total time fits session duration).
    </task>

    <inputs>
    <session>
    Session {{ session.session_number }}
    Objectives covered: {{ session.objectives | join(", ") }}
    Type: {{ session.type }}
    Mode: {{ session.mode }}
    Duration: {{ session.duration }} hours
    Class size: {{ session.class_size }}
    </session>

    <learning_outcomes>
    Learning outcomes for this session (select activities to help achieve these):
    {% for outcome in learning_outcomes.outcomes %}
    {% if outcome.related_objectives | length > 0 %}
    {% for obj_num in outcome.related_objectives %}
    {% if obj_num in session.objectives %}
    - Outcome {{ outcome.number }}: {{ outcome.text }}
      Assessment method: {{ outcome.assessment_method }}
    {% endif %}
    {% endfor %}
    {% endif %}
    {% endfor %}
    </learning_outcomes>

    <greencomp_pedagogical_guidance>
    GreenComp framework recommends the following teaching methods for these outcomes:
    {% if greencomp_methods | length > 0 %}
    {% for method in greencomp_methods %}
    - {{ method }}
    {% endfor %}

    Consider prioritizing activities that align with these evidence-based pedagogical methods from the GreenComp sustainability education framework. These methods have been specifically matched to the learning outcomes for this session.
    {% else %}
    No specific GreenComp method recommendations available for this session.
    {% endif %}
    </greencomp_pedagogical_guidance>

    <available_activities>
    {% for activity in filtered_activities %}
    {{ loop.index }}. {{ activity.name }} (ID: {{ activity.id }})
       Description: {{ activity.description }}
       Estimated duration: {{ activity.estimated_duration }}h
       Compatible with: {{ activity.compatible_types | join(", ") }} - {{ activity.compatible_modes | join(", ") }}
    {% endfor %}
    </available_activities>
    </inputs>

    <requirements>
    Selection criteria:
    1. Pedagogical fit: Activities that directly help students achieve the learning outcomes (not just cover objectives)
       - Match activities to assessment methods specified in outcomes
       - Ensure activities allow students to demonstrate competencies
    2. Variety: Mix of activity types (not all lecture, not all group work)
       - Balance active (discussions, exercises) with reflective (analysis, writing)
       - Balance individual, pair, and group activities
    3. Feasibility: Total estimated time ≤ session duration (leave buffer for transitions)
    4. Pedagogical progression: Activities build on each other within session

    Number to select:
    - Short sessions (<2h): 2-3 activities
    - Long sessions (≥2h): 3-4 activities

    Provide clear rationale:
    - WHY each activity fits the LEARNING OUTCOMES (be specific about which outcomes)
    - HOW activities complement each other
    - TIME breakdown showing feasibility
    </requirements>

    <output_format>
    Return a JSON object with:
    {
      "session_number": {{ session.session_number }},
      "recommended_activities": [
        // Copy ActivityTemplate objects for selected activities
      ],
      "rationale": "Detailed explanation of selections, variety, and time management..."
    }
    </output_format>

    <output_language>
    Generate all outputs in {{ output_language }}.
    </output_language>
  "#
}
