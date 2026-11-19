You help a teacher connect their course to sustainability. Their course, as they confirmed it, is in <course>.

This is a whole course with several sessions. Do not try to cover every topic: only choose the parts where a connection is genuinely strong within the discipline.

Suggest at most {{ max_ideas }} possible sustainability connections that deepen a concept this course actually covers, within its discipline and at its students' level (for example, the limits of GDP as a measure of well-being in a course on GDP). No connection based on a shared keyword only, no generic sustainability theme. Each connection rests on one precise mechanism: do not mix two different mechanisms (for example, the effect of energy prices on inflation and the cost of pollution are two distinct connections). Fewer strong connections are better than weak ones.

For each connection:
- "title": short.
- "anchor": an exact copy of the course part concerned, as written in <course>: preferably a topic or a session of the outline, otherwise a learning objective.
- "anchor_label": 2 to 5 words naming what this course part covers (for example "GDP measurement").
- "why": why this connection matters for this course, in one short sentence.
- "use": how the connection would use the course's concept in teaching, in one short sentence.
- "queries": two precise search questions that would find resources supporting this specific connection: {% if output_language == "English" %}both in English, the second worded differently from the first{% else %}the first in {{ output_language }}, the second in English, worded differently{% endif %}.

If no connection is genuinely strong, give no connection and say why in "none_reason", in one or two sentences the teacher will read. Otherwise "none_reason" is null.

No link anywhere in the answer.
