"""Pure parsing helpers for survey spreadsheets.

Survey cells arrive from Excel either as real numbers or as strings that use
European decimal commas ("0,85"), '/'-separated values for multi-band or
multi-mode designs ("7,35/10,65/14,75/11,45"), placeholders ("N/A", "*", "-")
and the odd annotation ("9,9*", "0,41(0,38)", ">25").

Everything here is a pure function so it can be unit-tested on the real
problem values. Nothing is ever silently dropped: a value that cannot be read
comes back with an ``error``, and an annotation that was stripped comes back
with a ``note``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Whole-cell or per-band placeholders meaning "not reported".
MISSING_TOKENS = frozenset({"", "-", "–", "—", "*", "?", "n/a", "na", "nan", "none"})

# Cached results of broken Excel formulas. Not data: read as missing, with an
# error, so the import notes show where the source sheet itself is broken.
# Checked before splitting on "/", because "#DIV/0!" contains the separator.
EXCEL_ERRORS = frozenset({"#value!", "#ref!", "#div/0!", "#n/a", "#name?", "#num!", "#null!", "#spill!", "#calc!"})


def excel_error(text: str) -> str | None:
    """The error message for a cached Excel error value, else None."""
    t = text.strip()
    return f"Excel error {t.upper()} in the source cell" if t.lower() in EXCEL_ERRORS else None

_NUM = r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)"
_TOKEN_RE = re.compile(
    rf"""^
    (?P<qual>[<>≤≥~≈]=?)?\s*      # optional qualifier: > < >= ~
    (?P<num>{_NUM})\s*
    (?P<star>\*+)?\s*                             # trailing asterisk(s)
    (?:\((?P<alt>[^)]*)\))?\s*                    # parenthetical alternative
    (?P<star2>\*+)?
    $""",
    re.VERBOSE,
)
# "N/A" contains the band separator, so neutralise it before splitting.
_NA_RE = re.compile(r"\bn\s*/\s*a\b", re.IGNORECASE)


@dataclass(frozen=True)
class Parsed:
    """One numeric value read from a cell (or from one band of a cell)."""

    value: float | None
    note: str | None = None  # annotation that was stripped, value still usable
    error: str | None = None  # could not be read at all; value is None


def _to_float(num: str) -> float:
    return float(num.replace(",", "."))


def parse_number(raw: object) -> Parsed:
    """Parse a single value. Zero is a real value, never "missing"."""
    if raw is None:
        return Parsed(None)
    if isinstance(raw, bool):  # bool is an int subclass; never valid data here
        return Parsed(None, error=f"unexpected boolean {raw!r}")
    if isinstance(raw, (int, float)):
        return Parsed(float(raw))

    text = str(raw).strip()
    if text.lower() in MISSING_TOKENS:
        return Parsed(None)
    if err := excel_error(text):
        return Parsed(None, error=err)

    m = _TOKEN_RE.match(text)
    if not m:
        return Parsed(None, error=f"unreadable value {text!r}")

    value = _to_float(m["num"])
    notes = []
    if m["qual"]:
        notes.append(f"qualifier '{m['qual']}' dropped")
    if m["star"] or m["star2"]:
        notes.append("asterisk dropped")
    if m["alt"] is not None:
        notes.append(f"parenthetical '({m['alt']})' ignored")
    return Parsed(value, note=f"{text!r} read as {value:g}: " + ", ".join(notes) if notes else None)


def parse_bands(raw: object) -> list[Parsed]:
    """Parse a possibly multi-band cell into one ``Parsed`` per band.

    Returns ``[]`` when the whole cell is empty or a placeholder. A placeholder
    inside a multi-band cell ("-17,89/-/-21,61") keeps its slot as ``None`` so
    the remaining bands stay aligned.
    """
    if raw is None:
        return []
    if not isinstance(raw, str):
        return [parse_number(raw)]

    text = raw.strip()
    if text.lower() in MISSING_TOKENS or _NA_RE.fullmatch(text):
        return []
    if err := excel_error(text):
        return [Parsed(None, error=err)]
    text = _NA_RE.sub("-", text)
    return [parse_number(part) for part in text.split("/")]


def align_bands(
    columns: dict[str, list[Parsed]],
    band_key: str,
) -> tuple[int, dict[str, list[Parsed | None]], list[str]]:
    """Line up multi-band columns into per-band values.

    ``band_key`` names the column that defines the bands (the frequency).
    The band count ``n`` is that column's count; if it has a single value but
    other columns list several values, the design reports several modes at one
    frequency and ``n`` is the largest such count.

    A column with 0 values stays empty, 1 value is broadcast to every band, and
    exactly ``n`` values map one per band. Any other count cannot be aligned:
    that column is emptied and its name is returned in ``mismatched``.
    """
    counts = {k: len(v) for k, v in columns.items()}
    n_freq = counts.get(band_key, 0)
    if n_freq == 0:
        return 0, {}, []
    n = n_freq if n_freq > 1 else max(counts.values())

    aligned: dict[str, list[Parsed | None]] = {}
    mismatched: list[str] = []
    for key, vals in columns.items():
        if len(vals) == 0:
            aligned[key] = [None] * n
        elif len(vals) == 1:
            aligned[key] = [vals[0]] * n
        elif len(vals) == n:
            aligned[key] = list(vals)
        else:
            aligned[key] = [None] * n
            mismatched.append(key)
    return n, aligned, mismatched


_OOXML_ESCAPE_RE = re.compile(r"_x([0-9A-Fa-f]{4})_")
_SPACES_RE = re.compile(r"\s+")


def clean_text(raw: object) -> tuple[str, list[str]]:
    """Clean a text cell. Returns (text, notes describing any repairs).

    Excel stores characters that are illegal in XML as ``_xHHHH_`` escapes; in
    this survey ``Variable_x0002_Gain`` was a hyphen ("Variable-Gain"). A
    decoded control character between two word characters becomes a hyphen,
    anywhere else a space. Whitespace runs collapse to one space.
    """
    if raw is None:
        return "", []
    text = str(raw)
    notes: list[str] = []

    def decode(m: re.Match[str]) -> str:
        ch = chr(int(m[1], 16))
        if unicodedata.category(ch) != "Cc":
            notes.append(f"decoded Excel escape {m[0]} as {ch!r}")
            return ch
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        repl = "-" if before.isalnum() and after.isalnum() else " "
        notes.append(f"replaced Excel escape {m[0]} with {repl!r}")
        return repl

    text = _OOXML_ESCAPE_RE.sub(decode, text)
    # tabs and newlines are ordinary whitespace (collapsed below), not repairs
    stripped = "".join(" " if unicodedata.category(c) == "Cc" and not c.isspace() else c for c in text)
    if stripped != text:
        notes.append("removed control characters")
    return _SPACES_RE.sub(" ", stripped).strip(), notes


def classify(value: str | None, rules: list[tuple[str, str]], default: str) -> str:
    """Return the first group whose regex matches ``value`` (case-insensitive).

    Order matters: "SiGe BiCMOS" contains "CMOS", so SiGe must be tested first.
    """
    if not value:
        return default
    for group, pattern in rules:
        if re.search(pattern, value, re.IGNORECASE):
            return group
    return default
