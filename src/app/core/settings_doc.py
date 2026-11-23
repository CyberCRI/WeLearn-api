"""Writes docs/settings.md: the API's feature flags and limits, environment by environment.

The value of a setting in an environment is the one its Helm values give
(k8s/welearn-api/values.<env>.yaml, then values.yaml), else the default of
src/app/core/config.py::Settings. Only the flags (bool) and the limits (int,
float) are listed; their description is the comment above them in config.py.
Standard library only, and config.py is read, not imported: no env var needed.

    python src/app/core/settings_doc.py            write the page
    python src/app/core/settings_doc.py --check    fail when it is out of date
    python src/app/core/settings_doc.py --release [notes.md]
        print a release's notes with the section in them: between its markers
        when the notes already have it (a new run replaces it), else at their end
"""

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = "src/app/core/config.py"
VALUES = "k8s/welearn-api/values.yaml"
ENV_VALUES = "k8s/welearn-api/values.{env}.yaml"
PAGE = "docs/settings.md"
ENVIRONMENTS = ["dev", "staging", "prod"]
START = "<!-- settings:start -->"
END = "<!-- settings:end -->"
TYPES = {"bool", "int", "float"}


def first_sentence(text: str) -> str:
    """The first sentence of a comment, capitalised: enough for a table."""
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].strip()
    return sentence[:1].upper() + sentence[1:]


@dataclass
class Setting:
    name: str
    type: str
    default: str
    description: str


def as_text(value: object) -> str:
    """A value as the env var would carry it: booleans in lowercase."""
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip()
    return text.lower() if text.lower() in ("true", "false") else text


def read_settings(source: str) -> list[Setting]:
    """The bool, int and float fields of Settings that have a default.

    A comment block describes the fields right under it, until a blank line.
    """
    tree = ast.parse(source)
    settings_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Settings"
    )
    lines = source.splitlines()
    settings = []
    for node in settings_class.body:
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Name)
            and node.annotation.id in TYPES
            and isinstance(node.value, ast.Constant)
        ):
            continue
        settings.append(
            Setting(
                name=node.target.id,
                type=node.annotation.id,
                default=as_text(node.value.value),
                description=first_sentence(comment_above(lines, node.lineno - 1)),
            )
        )
    return settings


def comment_above(lines: list[str], index: int) -> str:
    """The comment block above a field, or above the fields it shares it with."""
    comment: list[str] = []
    i = index - 1
    while i >= 0:
        line = lines[i].strip()
        if line.startswith("#"):
            comment.insert(0, line.lstrip("#").strip())
        elif comment or not re.match(r"^[A-Z_]+\s*:", line):
            break
        i -= 1
    return " ".join(comment)


def read_values(source: str) -> dict[str, str]:
    """The `config.nonSensitive` entries of a Helm values file.

    Its entries are flat `KEY: value` lines: anything else there is refused
    rather than misread.
    """
    values: dict[str, str] = {}
    section: list[str] = []
    for raw in source.splitlines():
        line = raw.split(" #")[0].rstrip() if not raw.strip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            section = [line.strip()]
        elif indent == 2 and section and section[0] == "config:":
            section = ["config:", line.strip()]
        elif indent == 4 and section == ["config:", "nonSensitive:"]:
            match = re.match(r"^([A-Z0-9_]+):\s*(.*)$", line.strip())
            if not match:
                raise ValueError(f"unexpected line in config.nonSensitive: {raw!r}")
            values[match[1]] = as_text(match[2].strip().strip("'\""))
    return values


def read_all(root: Path = ROOT) -> tuple[list[Setting], dict[str, dict[str, str]]]:
    settings = read_settings((root / CONFIG).read_text())
    base = read_values((root / VALUES).read_text())
    by_env = {
        env: {**base, **read_values((root / ENV_VALUES.format(env=env)).read_text())}
        for env in ENVIRONMENTS
    }
    return settings, by_env


def cell(text: str) -> str:
    return text.replace("|", "\\|")


def table(settings: list[Setting], by_env: dict[str, dict[str, str]]) -> str:
    header = f"| Setting | {' | '.join(ENVIRONMENTS)} | Default | What it is |"
    rule = f"|---|{'|'.join(':-:' for _ in ENVIRONMENTS)}|:-:|---|"
    rows = []
    for setting in settings:
        # a value from the code's default is in italics: no Helm value sets it there
        values = [
            (
                f"`{by_env[env][setting.name]}`"
                if setting.name in by_env[env]
                else f"_{setting.default}_"
            )
            for env in ENVIRONMENTS
        ]
        rows.append(
            f"| `{setting.name}` | {' | '.join(values)} | `{setting.default}` "
            f"| {cell(setting.description) or '—'} |"
        )
    return "\n".join([header, rule, *rows])


def tables(settings: list[Setting], by_env: dict[str, dict[str, str]]) -> str:
    flags = [s for s in settings if s.type == "bool"]
    limits = [s for s in settings if s.type != "bool"]
    return f"""### Feature flags

{table(flags, by_env)}

### Limits

{table(limits, by_env)}

A value in `monospace` is set by the environment's Helm values; one in _italics_ is
the default of `{CONFIG}`, since no Helm value sets it there.
These are the Azure environments. Scaleway's values live in CyberCRI/argocd-apps."""


def settings_page(settings: list[Setting], by_env: dict[str, dict[str, str]]) -> str:
    return f"""# Settings by environment

<!-- generated by `make docs-settings` from {CONFIG} and k8s/welearn-api/values*.yaml:
edit those files, not this one -->

The API's feature flags and limits in each environment. A setting takes the value
of `k8s/welearn-api/values.<env>.yaml`, else `values.yaml`, else its default in
`{CONFIG}::Settings`. Its description is the comment above it there.

{tables(settings, by_env)}
"""


def release_section(settings: list[Setting], by_env: dict[str, dict[str, str]]) -> str:
    return f"""{START}
## Settings by environment

{tables(settings, by_env)}
{END}"""


def with_release_section(notes: str, section: str) -> str:
    start, end = notes.find(START), notes.find(END)
    if start != -1 and end > start:
        return notes[:start] + section + notes[end + len(END) :]
    return f"{notes.rstrip()}\n\n{section}\n" if notes.strip() else f"{section}\n"


def main(args: list[str], root: Path = ROOT) -> int:
    settings, by_env = read_all(root)
    if "--release" in args:
        index = args.index("--release")
        notes = Path(args[index + 1]).read_text() if len(args) > index + 1 else ""
        sys.stdout.write(with_release_section(notes, release_section(settings, by_env)))
        return 0
    page = settings_page(settings, by_env)
    path = root / PAGE
    if "--check" in args:
        if not path.exists() or path.read_text() != page:
            print(
                f"{PAGE} is out of date: run `make docs-settings` and commit it",
                file=sys.stderr,
            )
            return 1
        return 0
    path.parent.mkdir(exist_ok=True)
    path.write_text(page)
    print(f"wrote {PAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
