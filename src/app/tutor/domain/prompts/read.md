Analyse the course syllabus in <course>.

Rules:
- Copy every value word for word from the document, in its original language. Do not translate, rephrase or summarise. Only "summary" and the "title" and "subtopics" of "topics" may be worded by you.
- Copy the value alone, without its label or separator (for example "Economics", not "Course: Economics" or "Course | Economics").
- "title": the course's name as the document writes it, usually its main heading at the top of the document. Never a person's name or a line about the teacher ("Professor: …"), a date or a year, a session or a semester, the institution, or a word such as "Syllabus". null if the document does not give the course's name.
- If a piece of information is not explicitly written in the document, use null or an empty list. Do not invent or infer anything.
- "description": the official course description, copied word for word (consecutive paragraphs joined). Never the whole document, a session plan, a list or the objectives of the programme or degree. Otherwise null.
{% include "partials/summary_topics.md" %}
- "sessions": only if the document contains an explicit session plan; one object for each session, every session in the document's order, exam and review sessions included, with "label" (as written, for example "Week 3") and "title" (as written, or null if the session has no title). Otherwise [].
- "objectives": every item of every list of objectives, learning objectives, learning outcomes and competencies or skills (for example "Skills developed", "Competencies") in the document, at every level (course, programme, degree, each year of the degree, general education), one object per item, in the document's order. Never a heading, a sentence that introduces or announces a list, or a sentence of the course's description, summary, organisation or prerequisites. There is no expected number: do not select, merge, shorten or leave out any item. For each, "text" (word for word, without its number or bullet) and "scope": "programme" if it sits under a heading naming the programme, degree, a year of the degree, general education or a competence framework; "course" if it sits under a heading naming the course; "unspecified" otherwise.
- "level": the level of study (year, degree, class), only if it is explicitly written, as written.
- "discipline": only if a discipline is explicitly named.
- "discipline_suggested": only if "discipline" is null: the academic discipline the course clearly belongs to, judging from its title and content (for example "Management" for an introduction to business management), in one or two words. null if the course is interdisciplinary or its discipline is not clear. This is the only value you may infer.
- "material_type": what the document is: "syllabus" (a complete course syllabus), "course_plan" (a semester plan, timetable or outline with several sessions), "lesson" (a single session or lesson, even if it carries a session number), "partial" (a partial course description), "other" (anything else).
- "courses_found": if the document contains several distinct courses, the title of each, word for word; otherwise []. Never merge several courses into one.
{% if has_focus %}

The document contains several courses. Only process the course whose title is given in <focus> and fully ignore the others; "courses_found" must be [].
{% endif %}
