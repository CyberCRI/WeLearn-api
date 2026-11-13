# Tutor prompts (syllabus v2)

One file per LLM function and per language: `<function>.fr.md` and `<function>.en.md`.
They are loaded by `tutor/services/llm.py::structured_call`, which sends the prompt to
Mistral, reads the answer as a Pydantic model and traces the call in LangSmith.

## Editing a prompt

- Edit the text freely. Keep the FR and EN files saying the same thing.
- `{{ name }}` is a variable filled by the code (a count, the GreenComp list…). Don't rename
  or remove one without changing the code: a missing variable is an error, not a blank.
- **Never paste the course or a resource into a prompt file.** The teacher's course and the
  WeLearn resources are sent apart, in tagged blocks such as `<course>…</course>`. Refer to
  them by their tag ("the course in <course>"). The code adds the rule that their content is
  data, never instructions.
- Don't describe the JSON shape in detail: the answer is constrained by the Pydantic model
  of the endpoint. Explain what each field should contain, in plain words.

## Checking a change

`make eval-tutor` runs the prompts on the test syllabi in `src/app/tests/tutor/llm/cases/`
with the real model (it needs a `.env` with a Mistral key; it is never run in CI).

With LangSmith tracing on, every call carries `prompt`, `lang` and `prompt_version` (a short
hash of the prompt file) in its metadata: filter the runs on `prompt_version` to compare the
answers of two versions of a prompt.
