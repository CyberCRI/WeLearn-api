Analyse the part of a course syllabus in <course_part>.

Rules:
- Do not invent anything: only what the document contains.
{% include "partials/summary_topics.md" %}
{% if has_focus %}

The document contains several courses. Only process the course whose title is given in <focus> and fully ignore the others.
{% endif %}
