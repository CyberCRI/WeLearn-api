# Evaluation cases

One folder per prompt (`read/`, `ideas/`, ...), one JSON file per case, and the text files
the cases point to. The format is described in `../harness.py`.

- A case points to its text with `"file:<name>"` in `data`: the link is that reference, not
  the file names. Still, name a case after its main text (`fr-sociology.json` +
  `fr-sociology.txt`); a case without `id` takes its file name as id. One text can serve
  several cases (another `lang`, a `focus`...).
- Text files hold the **extracted text** (UTF-8), what the prompt receives. Reading PDF and
  DOCX files, and the scanned-PDF error, are tested by the route tests, with the model mocked.
- `about` (optional, ignored by the harness) says what the case checks.
- Every case is loaded by a CI test (`test_llm_harness.py`), without the model: a typo in a
  case or a missing text file fails there. The cases themselves only run with
  `make eval-tutor`, for the prompts registered in `harness.py::EVALUATED_PROMPTS`.

The `read/`, `ideas/`, `selection/`, `adaptation/` and `verification/` cases are
**fictional** syllabi, courses and resources written as examples (an `ideas/` text is a
course as the prompt receives it: `connections.py::course_block`; the later steps reuse
those courses and the made-up resources of `selection/`). The
test syllabi of SY-06 (anonymised, FR and EN, several disciplines) are added next to them. Never put a real,
non-anonymised syllabus here: this folder is committed.
