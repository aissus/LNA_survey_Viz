"""Unit tests for scripts/parsing.py, using the real problem values from the LNA survey."""

import pytest

from parsing import MISSING_TOKENS, Parsed, align_bands, classify, clean_text, parse_bands, parse_number
from survey_sources import LNA


def values(cells):
    return [p.value for p in cells]


# --- numbers -----------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("0,85", 0.85),      # was missing on the old site: parseFloat("0,85") -> 0 -> null
    ("0", 0.0),          # genuine zero must survive (old site: `|| null` dropped it)
    (0, 0.0),
    (0.0, 0.0),
    ("-12", -12.0),      # negative, not the "-" placeholder
    ("-17,89", -17.89),
    ("1,805", 1.805),    # decimal comma, not a thousands separator (LTE band 3)
    (" 3.5 ", 3.5),
    (27, 27.0),
])
def test_parse_number_values(raw, expected):
    p = parse_number(raw)
    assert p.value == pytest.approx(expected)
    assert p.error is None and p.note is None


@pytest.mark.parametrize("raw", [None, "", "  ", "-", "*", "N/A", "n/a", "NA"])
def test_parse_number_missing(raw):
    assert parse_number(raw) == Parsed(None)


@pytest.mark.parametrize("raw, expected, fragment", [
    ("9,9*", 9.9, "asterisk"),
    ("0,41(0,38)", 0.41, "parenthetical"),
    (">25", 25.0, "qualifier"),
])
def test_parse_number_annotations_are_kept_with_a_note(raw, expected, fragment):
    p = parse_number(raw)
    assert p.value == pytest.approx(expected)
    assert fragment in p.note
    assert p.error is None


def test_parse_number_unreadable_is_an_error_not_a_silent_drop():
    p = parse_number("approx. five")
    assert p.value is None
    assert "unreadable" in p.error


def test_parse_number_rejects_booleans():
    assert parse_number(True).error


# --- multi-band cells --------------------------------------------------------

def test_multiband_spot_check_from_claude_md():
    assert values(parse_bands("7,35/10,65/14,75/11,45")) == pytest.approx([7.35, 10.65, 14.75, 11.45])


def test_multiband_with_annotations():
    cells = parse_bands("2,2/0,41(0,38)/1,6/0,19(0,17)")
    assert values(cells) == pytest.approx([2.2, 0.41, 1.6, 0.19])
    assert cells[1].note and cells[3].note and not cells[0].note


def test_multiband_placeholder_keeps_its_slot():
    assert values(parse_bands("-17,89/-/-21,61/-/-24,12")) == [-17.89, None, -21.61, None, -24.12]


@pytest.mark.parametrize("raw", [None, "", "N/A", "n/a", "*", "-"])
def test_whole_cell_missing_is_empty(raw):
    assert parse_bands(raw) == []


def test_na_inside_multiband_is_a_placeholder_not_two_tokens():
    assert values(parse_bands("1,2/N/A/3")) == [1.2, None, 3.0]


def test_single_numeric_cell_is_one_band():
    assert values(parse_bands(20)) == [20.0]


def test_missing_tokens_are_lowercase():
    assert all(t == t.lower() for t in MISSING_TOKENS)


# --- band alignment ----------------------------------------------------------

def P(*vals):
    return [Parsed(v) for v in vals]


def test_align_one_per_band_and_broadcast():
    n, aligned, bad = align_bands({"freq": P(7.35, 10.65, 14.75, 11.45), "nf": P(1.6, 1.6, 1.8, 2.4),
                                   "power": P(20), "iip3": []}, "freq")
    assert n == 4 and bad == []
    assert [p.value for p in aligned["power"]] == [20, 20, 20, 20]
    assert aligned["iip3"] == [None] * 4
    assert [p.value for p in aligned["nf"]] == [1.6, 1.6, 1.8, 2.4]


def test_align_modes_at_one_frequency():
    # Ozdol 2025: fc 3.5 with two operating modes listed in the other columns
    n, aligned, bad = align_bands({"freq": P(3.5), "nf": P(0.85, 0.85), "iip3": P(1, 5)}, "freq")
    assert n == 2 and bad == []
    assert [p.value for p in aligned["freq"]] == [3.5, 3.5]
    assert [p.value for p in aligned["iip3"]] == [1, 5]


def test_align_mismatch_empties_that_column_only():
    # Jeong 2024: 4 frequencies, but 2 bandwidth values
    n, aligned, bad = align_bands({"freq": P(1.84, 1.96, 2.14, 2.66), "bw": P(0.03, 0.02),
                                   "nf": P(1, 2, 3, 4)}, "freq")
    assert n == 4 and bad == ["bw"]
    assert aligned["bw"] == [None] * 4
    assert [p.value for p in aligned["nf"]] == [1, 2, 3, 4]


def test_align_without_frequency():
    assert align_bands({"freq": [], "nf": P(1.0)}, "freq") == (0, {}, [])


# --- text --------------------------------------------------------------------

def test_excel_escape_between_words_becomes_hyphen():
    text, notes = clean_text("Highly Linear Variable_x0002_Gain LNA")
    assert text == "Highly Linear Variable-Gain LNA"
    assert notes and "_x0002_" in notes[0]


def test_excel_escape_of_printable_char_is_decoded():
    assert clean_text("A_x0041_B")[0] == "AAB"


def test_whitespace_collapses_and_plain_text_is_untouched():
    assert clean_text("Concurrent LNA  Using\tInductor ") == ("Concurrent LNA Using Inductor", [])
    assert clean_text("A 22.9–38.2-GHz LNA") == ("A 22.9–38.2-GHz LNA", [])
    assert clean_text(None) == ("", [])


# --- process grouping --------------------------------------------------------

@pytest.mark.parametrize("process, group", [
    ("SiGe BiCMOS", "SiGe"),   # old site: contains "cmos" -> mislabelled "CMOS Bulk"
    ("SiGe BICMOS", "SiGe"),
    ("BiCMOS", "SiGe"),
    ("Si/SiGe BiCMOS", "SiGe"),
    ("BiCMOS SOI", "SiGe"),
    ("CMOS FinFET", "FinFET"),  # old site: "CMOS Bulk"
    ("CMOS FinFet", "FinFET"),
    ("CMOS SOI", "CMOS SOI"),
    ("CMOS SOi", "CMOS SOI"),
    ("RFSOI", "CMOS SOI"),      # old site: "Other"
    ("RFSOI CMOS", "CMOS SOI"),
    ("CMOS ", "CMOS Bulk"),
    ("RF CMOS", "CMOS Bulk"),
    ("GaN HEMT", "III-V"),
    ("Unobtainium", "Other"),
    (None, "Other"),
])
def test_lna_process_groups(process, group):
    assert classify(process, LNA.group_rules, LNA.group_default) == group
