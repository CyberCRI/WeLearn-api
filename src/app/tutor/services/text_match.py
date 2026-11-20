"""
Syllabus v2: comparing texts the way the course and the resources are written. Case,
spaces, line breaks, quote and dash styles are set aside; accents and words are kept.
"""

import re
import unicodedata


def unify_characters(text: str) -> str:
    """
    One way to write each character: lower case, one apostrophe, one quotation mark, one
    dash, "oe" for "œ". A word cut at the end of a line ("struc-\nture") is one word.
    Accents, words and spaces are kept.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[’‘`´]", "'", text)
    text = re.sub(r"[«»“”]", '"', text)
    text = re.sub(r"[‐-―−]", "-", text)
    text = text.replace("…", "...").replace("œ", "oe")
    return re.sub(r"-\s*\n\s*", "", text)


def normalise(text: str) -> str:
    """
    A text as it reads: case, spaces, line breaks, quote and dash styles aside. Accents
    and words are kept. Without spaces, word boundaries are gone too: "the rapist" reads
    as "therapist".
    """
    return re.sub(r"\s+", "", unify_characters(text))


_LIST_MARKER = re.compile(r"^\s*(?:[-–•o▪◦*]|\d+[.)])\s")


def core(text: str) -> str:
    """normalise, without a list marker or a final punctuation mark."""
    return re.sub(r"[.;:,!]+$", "", normalise(_LIST_MARKER.sub("", text, count=1)))


def names_each_other(a: str, b: str, norm=normalise) -> bool:
    """One text contains the other once both are put through `norm` (the same thing,
    worded more or less fully)."""
    x, y = norm(a), norm(b)
    return bool(x) and bool(y) and (x in y or y in x)
