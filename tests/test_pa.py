"""PA survey: the rules in source_pa.py, the reader features it needs, and the
real spreadsheet (ETH PA Survey v11, downloaded by the build and sha256-pinned)."""

import json
import urllib.error
from collections import Counter
from datetime import datetime

import openpyxl
import pytest

from build_data import ROOT, BuildError, build_survey, ensure_source, main, read_rows
from source_pa import PA, efficiency_type, fix_mojibake, pa_type, technology
from survey_sources import LNA, SurveySource

# --- rules, on the real strings of the survey --------------------------------


@pytest.mark.parametrize("process, sheet, expected", [
    ("CMOS_Bulk", "CMOS", "CMOS Bulk"),
    ("Bulk_CMOS", "SiGe", "CMOS Bulk"),  # a CMOS design filed on the SiGe sheet
    ("CMOS", "CMOS", "CMOS Bulk"),  # not labelled bulk or SOI
    ("CMOS_FD_SOI", "CMOS", "CMOS SOI"),
    ("PD_SOI", "CMOS", "CMOS SOI"),
    ("CMOS_SOS", "CMOS", "CMOS SOI"),
    ("CMOS RFSOI", "Multipliers", "CMOS SOI"),
    ("CMOS_Finfet", "CMOS", "FinFET"),
    ("CMOS_Bulk FinFET", "THz Oscillators", "FinFET"),
    ("SiGe BiCMOS", "SiGe", "SiGe"),
    ("BiCMOS_SOI", "CMOS", "SiGe"),  # SiGe HBTs on SOI, not CMOS SOI
    ("Semi SiGe BiCMOS (last stage power combining not on SiGe)", "SiGe", "SiGe"),
    ("GaN on SiC", "Products", "GaN"),
    ("Cree 40010F", "GaN", "GaN"),
    ("AFT20S015N MoSFET", "GaN", "LDMOS"),  # an LDMOS part filed on the GaN sheet
    ("Si MOSFET MRF21030", "Others", "LDMOS"),
    ("InP/GaAs HBT", "GaAs", "InP"),
    ("InGaP/GaAs", "GaAs", "GaAs"),
    (" InGaAs mHEMT", "GaAs", "GaAs"),
    ("pHEMT (ATF-34143)", "GaN", "GaAs"),
    ("SiC MESFET", "Others", "Other"),
    ("BCD", "CMOS", "Other"),
    ("N.A.", "Products", "Other"),
    (None, "GaN", "GaN"),  # empty process: the sheet decides
    (None, "Others", "Other"),
])
def test_technology(process, sheet, expected):
    assert technology(process, sheet) == expected


@pytest.mark.parametrize("note, expected", [
    ("PAE", "PAE"), ("PAE ", "PAE"), ("PAE*", "PAE"), ("TX_PAE", "PAE"), ("Sim PAE", "PAE"),
    ("PAE - Gain at saturation", "PAE"), ("AE", "PAE"),
    ("DE", "DE"), ("DE(SE)", "DE"), ("DE§", "DE"), ("DE (under pulse CW condition)", "DE"),
    ("CE", "CE"), ("SE", "SE"),
    ("Efficiency", "Other"), ("Conversion Efficiency", "Other"), ("DC-THz", "Other"),
    ("NA", None), ("Na", None), ("N/A", None), ("Not Done", None), ("?", None), (None, None),
])
def test_efficiency_type(note, expected):
    assert efficiency_type(note) == expected


@pytest.mark.parametrize("note, expected", [
    ("Analog", "Analog"), ("analog", "Analog"), ("Analog - DAT", "Analog"),
    ("Digital PA", "Digital"), ("Digital Polar Transmitters", "Digital"), ("Diigital", "Digital"),
    ("ET", "Supply-modulated"), ("Dual-path ET", "Supply-modulated"), ("EER", "Supply-modulated"),
    ("DPA (Frequency-varying CCL)", "Not stated"),  # DPA = Doherty PA here, not digital
    ("*Data estimated from graph", "Not stated"), (None, "Not stated"),
])
def test_pa_type(note, expected):
    assert pa_type(note) == expected


@pytest.mark.parametrize("raw, fixed", [
    ("A 26.5â€“29.5 GHz Doherty PA-LNA", "A 26.5–29.5 GHz Doherty PA-LNA"),
    ("with Â±0.5dB Gain Ripple", "with ±0.5dB Gain Ripple"),
    ("A D-Band Ã—8 Frequency Multiplier", "A D-Band ×8 Frequency Multiplier"),
    ("27.7âˆ’dB Gain", "27.7−dB Gain"),
    ("M. OlavsbrÃ¥ten", "M. Olavsbråten"),
    ("E. O‚ÄôMalley", "E. O’Malley"),  # Mac Roman
    ("Built-In AM‚ÄìPM Compensation", "Built-In AM–PM Compensation"),
    ("Using â€œHourglass-Shapedâ€\x9d Transmission", "Using “Hourglass-Shaped” Transmission"),
    ("A 0.12 μm GaN PA – 6–18 GHz", "A 0.12 μm GaN PA – 6–18 GHz"),  # already correct
    ("Plain ASCII", "Plain ASCII"),
])
def test_fix_mojibake(raw, fixed):
    assert fix_mojibake(raw) == fixed


# --- reader features the PA workbook needs (generic, synthetic workbook) ------


def _workbook(path):
    wb = openpyxl.Workbook()
    a = wb.active
    a.title = "A"
    a.append(["Title", "Year", "Freq", "Value", "Freq"])  # second "Freq" is a helper copy
    a.append(["Paper one", 2020, 1.5, 3, "#N/A"])
    a.append([None, None, 2.5, 4, "#N/A"])  # merged with the row above in Title / Year
    a.append(["Paper two", 2021, None, 5, None])  # no frequency: dropped
    a.merge_cells("A2:A3")
    a.merge_cells("B2:B3")
    b = wb.create_sheet(" B ")
    b.append(["Name", "Year", "Freq", "Other value"])
    b.append(["Paper one", 2020, 7, 8])  # same paper, other sheet, other header names
    b.append(["Paper three", 2022, "Feb", 9])  # dropped, same row number as on sheet A
    b.append(["Paper four", 2023, 3, 10])
    b["D4"].value = None
    b.append(["Paper five", 2024, 4, 1])
    b["D5"].value = None
    b["E1"].value = "Range"
    b["E5"].value = datetime(2019, 8, 12)
    wb.save(path)


def _source(**kw):
    return SurveySource(
        id="t",
        source="t.xlsx",
        sheet=("A", " B "),
        header_aliases={" B ": {"Name": "Title", "Other value": "Value"}},
        paper_fields={"title": "Title", "year": "Year"},
        entry_fields={"range": "Range"},
        metrics={"freq": "Freq", "value": "Value"},
        fallbacks={},
        band_key="freq",
        band_metrics=("freq", "value"),
        info_metrics=(),
        id_fields=("year", "title"),
        group_field="range",
        **kw,
    )


def test_reader_first_duplicate_header_wins_and_merged_cells_fill_down(tmp_path):
    _workbook(tmp_path / "t.xlsx")
    rows = read_rows(tmp_path / "t.xlsx", "A")
    assert [r for r, _ in rows] == [2, 3, 4]
    assert rows[0][1]["Freq"] == 1.5  # not the helper column's "#N/A"
    assert rows[1][1]["Title"] == "Paper one" and rows[1][1]["Year"] == 2020
    aliased = read_rows(tmp_path / "t.xlsx", " B ", {"Name": "Title"})
    assert aliased[0][1]["Title"] == "Paper one"


def test_several_sheets_are_one_table_and_rows_know_their_sheet(tmp_path):
    _workbook(tmp_path / "t.xlsx")
    data = build_survey(_source(), root=tmp_path)
    m = data["meta"]
    assert (m["rows_in"], m["rows_dropped"]) == (7, 2)  # row 4 dropped on both sheets
    dropped = {(w["sheet"], w["row"]) for w in data["warnings"] if w["severity"] == "dropped"}
    assert dropped == {("A", 4), ("B", 3)}  # sheet names are recorded without stray spaces
    one = next(p for p in data["papers"] if p["title"] == "Paper one")
    assert [(e["sheet"], e["row"]) for e in one["entries"]] == [("A", 2), ("A", 3), ("B", 2)]
    assert sorted(pt["freq"] for pt in data["points"] if pt["paper"] == one["id"]) == [1.5, 2.5, 7]
    assert all("sheet" in w for w in data["warnings"])


def test_a_date_in_a_text_column_becomes_text_with_a_note(tmp_path):
    _workbook(tmp_path / "t.xlsx")
    data = build_survey(_source(), root=tmp_path)
    five = next(p for p in data["papers"] if p["title"] == "Paper five")
    assert five["entries"][0]["range"] == "2019-08-12"
    assert any("date" in w["issue"] for w in data["warnings"] if w["row"] == 5 and w["sheet"] == "B")
    json.dumps(data)  # serialisable


def test_the_hook_can_fix_paper_fields_of_the_row_that_starts_the_paper(tmp_path):
    _workbook(tmp_path / "t.xlsx")

    def adjust(ctx):
        ctx.fields["year"] = 1999

    data = build_survey(_source(adjust=adjust), root=tmp_path)
    assert {p["year"] for p in data["papers"]} == {1999}


def test_single_sheet_output_has_no_sheet_keys():
    try:
        ensure_source(LNA)
    except (urllib.error.URLError, OSError) as err:
        pytest.skip(f"LNA spreadsheet not available offline: {err}")
    data = build_survey(LNA)
    assert not any("sheet" in e for p in data["papers"] for e in p["entries"])
    assert not any("sheet" in w for w in data["warnings"])


# --- the real PA spreadsheet --------------------------------------------------


@pytest.fixture(scope="module")
def pa():
    try:
        ensure_source(PA)  # downloads when missing; a changed file fails here (BuildError)
    except OSError as exc:  # offline
        pytest.skip(f"PA survey not available: {exc}")
    return build_survey(PA)


def entries(data, **match):
    """(paper, entry, points) for entries whose paper / entry fields match."""
    out = []
    for p in data["papers"]:
        for i, e in enumerate(p["entries"]):
            if all({**p, **e}.get(k) == v for k, v in match.items()):
                pts = [pt for pt in data["points"] if pt["paper"] == p["id"] and pt["entry"] == i]
                out.append((p, e, pts))
    return out


def one(data, **match):
    hits = entries(data, **match)
    assert len(hits) == 1, f"{len(hits)} entries match {match}"
    return hits[0]


def warnings_of(data, sheet, row):
    return [w for w in data["warnings"] if w.get("sheet") == sheet and w["row"] == row]


def test_every_row_is_accounted_for(pa):
    m = pa["meta"]
    n_entries = sum(len(p["entries"]) for p in pa["papers"])
    assert (m["rows_in"], m["rows_dropped"], m["papers_out"], m["points_out"]) == (6196, 122, 4045, 6126)
    assert n_entries + m["rows_dropped"] == m["rows_in"]
    sheets = Counter(e["sheet"] for p in pa["papers"] for e in p["entries"])
    assert set(sheets) == {"CMOS", "SiGe", "GaN", "GaAs", "InP", "LDMOS", "Others", "Products",
                           "Multipliers", "THz Oscillators"}
    # 88 datasheets give their band only in a text note; ranges like "53-59" are not a frequency
    dropped = Counter(w["sheet"] for w in pa["warnings"] if w["severity"] == "dropped")
    assert dropped["Products"] == 97
    assert any(w["raw"] == "53-59" for w in warnings_of(pa, "CMOS", 1306))


def test_multiband_cells_with_decimal_commas(pa):
    _, e, pts = one(pa, first_author="Yao", year=2025, sheet="GaN")
    assert e["raw"]["Frequency (GHz)"] == "1,7/2,4/3,5"
    assert [p["freq"] for p in pts] == pytest.approx([1.7, 2.4, 3.5])
    assert [p["psat"] for p in pts] == pytest.approx([40.2, 40.5, 40.6])
    assert [p["pae"] for p in pts] == pytest.approx([58, 51.7, 43.8])
    # "2.3/2,4": a point and a comma in one cell; a single Psat applies to both bands
    _, _, pts = one(pa, first_author="Shao", year=2025, sheet="GaN")
    assert [p["freq"] for p in pts] == pytest.approx([2.3, 2.4])
    assert [p["psat"] for p in pts] == pytest.approx([42.3, 42.3])


def test_several_values_at_one_frequency_are_modes(pa):
    _, e, pts = one(pa, sheet="CMOS", row=399)
    assert e["raw"]["Gain (dB)"] == "28 / 24.5"
    assert [(p["freq"], p["gain"], p["avg_pae"]) for p in pts] == [(0.9, 28, 42.9), (0.9, 24.5, 38.5)]


def test_band_count_mismatch_is_flagged_not_misaligned(pa):
    _, _, pts = one(pa, sheet="GaN", row=1851)  # 2 frequencies, 5 Psat values
    assert [p["freq"] for p in pts] == [2.6, 3.6] and all(p["psat"] is None for p in pts)
    assert any(w["severity"] == "nulled" and w["column"] == "Psat (dBm)" for w in warnings_of(pa, "GaN", 1851))


def test_helper_columns_with_the_same_header_are_ignored(pa):
    # each sheet repeats "Frequency (GHz)" / "Psat (dBm)" as #N/A-filled plot helpers
    _, _, pts = one(pa, sheet="SiGe", row=2)
    assert pts[0]["freq"] == 0.83 and pts[0]["psat"] is None
    assert not [w for w in pa["warnings"] if "Excel error" in w["issue"]]


def test_merged_cells_keep_follow_up_rows_with_their_paper(pa):
    momeni = [h for h in entries(pa, first_author="Momeni", year=2011) if h[1]["sheet"] == "THz Oscillators"]
    assert [e["row"] for _, e, _ in momeni] == [26, 27, 28, 29, 30]
    assert len({p["id"] for p, _, _ in momeni}) == 1
    assert [pts[0]["freq"] for _, _, pts in momeni] == [256, 104, 121, 482, 482]


def test_papers_can_span_sheets(pa):
    assert sum(len({e["sheet"] for e in p["entries"]}) > 1 for p in pa["papers"]) == 26


def test_header_aliases(pa):
    _, e, _ = one(pa, sheet="GaN", row=130)  # "PA Efficieny Note" on the GaN sheet
    assert e["eff_note"].startswith("DE")
    p, e, pts = one(pa, sheet="Products", row=2)  # datasheet: vendor and product name
    assert (p["first_author"], p["year"], e["circuit"]) == ("Qorvo", None, "PA product")
    assert p["title"].startswith("RF5110G")
    _, e, pts = one(pa, sheet="THz Oscillators", row=21)  # Pout -> psat, RF-to-DC -> dc_rf_eff
    assert (pts[0]["psat"], pts[0]["dc_rf_eff"], e["circuit"]) == (4.6, 2.4, "Oscillator")
    mult = [pts for _, e, pts in entries(pa, sheet="Multipliers")]
    assert sum(pts[0]["psat"] is not None for pts in mult) > 250
    assert sum(pts[0]["conv_gain"] is not None for pts in mult) > 190


def test_text_repairs(pa):
    p, _, _ = one(pa, sheet="CMOS", row=1601)  # year typed as text
    assert p["year"] == 2025
    assert not [p for p in pa["papers"] if isinstance(p["year"], str)]
    titles = [p["title"] or "" for p in pa["papers"]] + [p["first_author"] or "" for p in pa["papers"]]
    assert not [t for t in titles if "â€" in t or "‚Ä" in t or "Ã" in t]
    assert any("26.5–29.5 GHz Doherty PA-LNA" in t for t in titles)
    _, e, _ = one(pa, sheet="GaN", row=875)  # Excel turned the typed range into a date
    assert e["freq_range"] == "2019-08-12"
    assert any("date" in w["issue"] for w in warnings_of(pa, "GaN", 875))


def test_genuine_zeros_survive(pa):
    zeros = [pt for pt in pa["points"] if pt["psat"] == 0]
    assert len(zeros) == 12 and all(isinstance(pt["psat"], float) for pt in zeros)


def test_unreadable_and_implausible_values_are_blanked_with_a_note(pa):
    _, _, pts = one(pa, sheet="CMOS", row=1012)  # PAE "20..5"
    assert pts[0]["pae"] is None
    assert any(w["raw"] == "20..5" and w["severity"] == "nulled" for w in warnings_of(pa, "CMOS", 1012))
    _, _, pts = one(pa, sheet="CMOS", row=644)  # gain "2065"
    assert pts[0]["gain"] is None and pts[0]["psat"] == 13.9
    assert any("not plausible" in w["issue"] for w in warnings_of(pa, "CMOS", 644))
    _, _, pts = one(pa, sheet="THz Oscillators", row=13)  # 80% where Pout / Pdc gives 0.8%
    assert pts[0]["dc_rf_eff"] is None
    # an EVM in dB is negative; 51 cells hold percentages, fractions or a dropped minus sign
    assert not [pt for pt in pa["points"] if pt["evm"] is not None and pt["evm"] >= 0]
    assert sum("EVM in dB must be negative" in w["issue"] for w in pa["warnings"]) == 51
    assert not [pt for pt in pa["points"] if pt["pae_1db"] is not None and pt["pae_1db"] < 0]


def test_groups_and_categories(pa):
    E = [e for p in pa["papers"] for e in p["entries"]]
    assert Counter(e["circuit"] for e in E) == {"PA": 5539, "Multiplier": 264, "PA product": 168, "Oscillator": 103}
    assert {e["group"] for e in E} == {"CMOS", "SiGe", "III-V", "LDMOS", "Other"}
    assert {e["group"] for e in E if e["technology"] in ("GaN", "GaAs", "InP")} == {"III-V"}
    assert {e["group"] for e in E if e["technology"] in ("CMOS Bulk", "CMOS SOI", "FinFET")} == {"CMOS"}
    _, e, _ = one(pa, sheet="SiGe", row=459)  # "Bulk_CMOS" on the SiGe sheet
    assert (e["technology"], e["group"]) == ("CMOS Bulk", "CMOS")
    assert one(pa, sheet="GaN", row=451)[1]["technology"] == "LDMOS"  # "AFT20S015N MoSFET"
    assert all(e["pa_type"] == "Not stated" for e in E if e["circuit"] != "PA")
    assert not [e for e in E if None in e.values()]  # empty text fields are left out


def test_efficiency_that_is_not_pae_is_flagged_per_metric(pa):
    _, e, _ = one(pa, sheet="LDMOS", row=5)  # PAEmax "39(DE)", note "DE"
    assert set(e["metric_flags"]["not_pae"]) == {"pae", "avg_pae"}
    _, e, _ = one(pa, sheet="CMOS", row=899)  # efficiency note "SE"
    assert set(e["metric_flags"]["not_pae"]) == {"pae", "pae_1db"}
    flagged = Counter(k for p in pa["papers"] for e in p["entries"] for k in e.get("metric_flags", {}).get("not_pae", []))
    assert flagged == {"pae": 962, "avg_pae": 539, "pae_1db": 161}


def test_caveats_from_notes_and_annotations(pa):
    assert pa["meta"]["flag_counts"] == {"system": 22, "offchip": 12, "estimated": 12, "pulsed": 9}
    assert pa["meta"]["metric_flag_counts"] == {"not_pae": 1063, "simulated": 1, "estimated": 3}
    assert one(pa, sheet="CMOS", row=1119)[1]["metric_flags"]["simulated"] == ["pae"]  # "Sim PAE"
    assert one(pa, sheet="CMOS", row=899)[2][0]["flags"] == ["system"]  # "21.4(TX)"
    assert one(pa, sheet="CMOS", row=22)[2][0]["flags"] == ["system"]  # "with T/R switch"
    assert one(pa, sheet="SiGe", row=389)[2][0]["flags"] == ["offchip"]  # combining not on SiGe
    assert one(pa, sheet="GaN", row=130)[2][0]["flags"] == ["pulsed"]  # "pulsed CW condition"
    assert one(pa, sheet="CMOS", row=398)[2][0]["flags"] == ["estimated"]  # "*Data estimated from graph"


def test_cli_writes_pa_json(pa, tmp_path, capsys):
    assert main(["pa", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "pa.json").read_text(encoding="utf-8"))
    assert data["meta"]["survey"] == "pa"
    out = capsys.readouterr().out
    assert "[pa] 6196 rows" in out and "CMOS row 1306 [dropped]" in out


def test_a_changed_spreadsheet_fails_the_build(tmp_path):
    (tmp_path / "sources" / "pa").mkdir(parents=True)
    (tmp_path / PA.source).write_bytes(b"not the pinned file")
    with pytest.raises(BuildError, match="sha256"):
        ensure_source(PA, root=tmp_path)
