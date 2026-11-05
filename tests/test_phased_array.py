"""Checks of the phased-array survey import against the real spreadsheet (v3.0).

The spreadsheet is not committed: the fixture downloads it like the build does
(and checks the pinned sha256). Without network access the tests are skipped;
a hash mismatch fails them.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from build_data import ROOT, build_survey, ensure_source, read_rows
from source_phased_array import (
    CONVERSION_FE_ONLY,
    CONVERSION_INTEGRATED,
    CONVERSION_NA,
    PHASED_ARRAY,
    split_type,
)


@pytest.fixture(scope="module")
def sheet():
    """{excel row: {header: cell}} straight from the spreadsheet, for cross-checks."""
    try:
        path = ensure_source(PHASED_ARRAY)
    except OSError as e:  # no network (URLError, timeout); a hash mismatch raises BuildError and fails
        pytest.skip(f"phased-array spreadsheet not available: {e}")
    return dict(read_rows(path, PHASED_ARRAY.sheet))


@pytest.fixture(scope="module")
def pa(sheet):
    return build_survey(PHASED_ARRAY)


@pytest.fixture(scope="module")
def by_row(pa):
    """{excel row: (paper, entry, point)}; every entry here has exactly one point."""
    out = {}
    for pt in pa["points"]:
        paper = next(p for p in pa["papers"] if p["id"] == pt["paper"])
        entry = paper["entries"][pt["entry"]]
        out[entry["row"]] = (paper, entry, pt)
    return out


def rows_where(by_row, **match):
    return {r: v for r, v in by_row.items() if all(v[0].get(k, v[1].get(k)) == x for k, x in match.items())}


def notes_of(pa, row):
    return [w["issue"] for w in pa["warnings"] if w["row"] == row and w["severity"] == "note"]


# --- source and bookkeeping ----------------------------------------------------

def test_source_is_pinned_and_downloaded_not_committed(pa):
    assert PHASED_ARRAY.url and PHASED_ARRAY.url.endswith("phased_array_survey_v3.xlsx")
    assert pa["meta"]["source_sha256"] == PHASED_ARRAY.sha256
    assert "sources/phased_array/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_every_row_is_one_entry_and_one_point(pa, by_row):
    m = pa["meta"]
    assert (m["rows_in"], m["rows_dropped"], m["papers_out"], m["points_out"]) == (151, 0, 121, 151)
    assert len(by_row) == 151  # one band per row
    multi = [p for p in pa["papers"] if len(p["entries"]) > 1]
    assert len(multi) == 27  # TX / RX modes, two bands, 64 vs 256 elements
    zihir = next(p for p in multi if p["first_author"] == "Zihir")
    assert sorted(by_row[e["row"]][2]["n_elements"] for e in zihir["entries"]) == [64, 256]


def test_every_paper_has_a_publication_month(pa):
    assert all(isinstance(p["month"], int) and 1 <= p["month"] <= 12 for p in pa["papers"])


def test_author_names_are_trimmed(pa):
    assert not [p["first_author"] for p in pa["papers"] if p["first_author"] != p["first_author"].strip()]
    assert any(p["first_author"] == "Kibaroglu" for p in pa["papers"])  # " Kibaroglu" in the sheet


# --- type, process, antenna ----------------------------------------------------

@pytest.mark.parametrize("label, expected", [
    ("TX", ("TX", CONVERSION_INTEGRATED)),
    ("TRX FE", ("TRX", CONVERSION_FE_ONLY)),
    (" RX FE", ("RX", CONVERSION_FE_ONLY)),  # leading space, row 147
    ("Oscillator", ("Oscillator", CONVERSION_NA)),
    ("Relay", ("Relay", CONVERSION_NA)),
    ("PA", None),
])
def test_split_type(label, expected):
    assert split_type(label) == expected


def test_type_and_frontend_categories(by_row):
    entries = [e for _, e, _ in by_row.values()]
    assert Counter(e["type"] for e in entries) == {"TRX": 47, "RX": 37, "TX": 34, "Oscillator": 18, "Relay": 15}
    assert Counter(e["conversion"] for e in entries) == {
        CONVERSION_INTEGRATED: 83, CONVERSION_FE_ONLY: 35, CONVERSION_NA: 33}
    assert Counter(e["type_label"] for e in entries if e["conversion"] == CONVERSION_FE_ONLY) == {
        "TX FE": 16, "RX FE": 11, "TRX FE": 8}
    assert by_row[147][1]["type_label"] == "RX FE"


def test_process_groups(by_row):
    groups = Counter(e["group"] for _, e, _ in by_row.values())
    assert groups == {"CMOS Bulk": 63, "SiGe": 52, "CMOS SOI": 21, "FinFET": 5, "Not reported": 10}
    blank = {e["row"] for _, e, _ in by_row.values() if e["group"] == "Not reported"}
    # 8 passive reflectarrays + 2 SATCOM arrays (Wang 2022, Raeesi 2022)
    assert {by_row[r][1]["type"] for r in blank} == {"Relay", "TX"}


def test_antenna_categories(by_row):
    antenna = Counter(e["antenna"] for _, e, _ in by_row.values())
    assert antenna == {"PCB": 55, "None": 37, "Chip": 30, "AiP": 26, "Not reported": 3}


def test_text_placeholders_are_missing(by_row):
    entries = [e for _, e, _ in by_row.values()]
    for key in ("dimension", "co_aperture", "antenna_type"):
        assert "N/A" not in {e[key] for e in entries}
    assert Counter(e["co_aperture"] for e in entries) == {None: 97, "Yes": 34, "No": 20}


def test_implausible_finfet_node_is_left_out(pa, by_row):
    _, monroe, _ = by_row[99]
    assert monroe["group"] == "FinFET" and monroe["node"] is None
    assert any("not a FinFET node" in n for n in notes_of(pa, 99))
    finfet_nodes = {e["node"] for _, e, _ in by_row.values() if e["group"] == "FinFET"} - {None}
    assert finfet_nodes and max(finfet_nodes) <= 22


# --- RX Pdc = 0 (Readme 6.4) ----------------------------------------------------

def test_rx_pdc_zero_means_included_in_tx_pdc(pa, sheet, by_row):
    zero_rows = {r for r, cells in sheet.items() if cells["RX Pdc (W)"] == 0}
    assert len(zero_rows) == 10
    for r in zero_rows:
        _, entry, pt = by_row[r]
        assert pt["rx_pdc"] is None and pt["rx_pdc_el"] is None
        assert pt["tx_pdc"] is not None  # always reported next to a TX Pdc
        if pt["n_rx"]:
            assert entry["metric_flags"]["pdc_shared"] == ["tx_pdc", "tx_pdc_el"]
        else:  # Zihir 2016: transmit-only, the zero just means "no receiver"
            assert "pdc_shared" not in entry.get("metric_flags", {})
    shared = {r for r, (_, e, _) in by_row.items() if "pdc_shared" in e.get("metric_flags", {})}
    assert len(shared) == 8
    assert {by_row[r][0]["first_author"] for r in zero_rows - shared} == {"Zihir"}
    # nothing anywhere is a 0 W / 0 mW receiver any more
    assert not [r for r, (_, _, pt) in by_row.items() if pt["rx_pdc"] == 0 or pt["rx_pdc_el"] == 0]


# --- relays: antenna gain in dBi (Readme 6.8), elements ------------------------

def test_passive_relay_gain_is_antenna_gain(pa, by_row):
    passive = {r for r, (_, _, pt) in by_row.items() if "passive" in pt.get("flags", [])}
    assert len(passive) == 10
    assert {by_row[r][1]["type"] for r in passive} == {"Relay"}
    gains = {by_row[r][0]["first_author"]: by_row[r][2]["ant_gain"] for r in passive if by_row[r][2]["ant_gain"] is not None}
    assert gains == {"Carrasco": 25.6, "Wang": 34.3, "Xia": 31.9, "Hu": 25.9, "Gros": 22.5}
    effs = {by_row[r][0]["first_author"]: by_row[r][2]["ap_eff"] for r in passive if by_row[r][2]["ap_eff"] is not None}
    assert effs == {"Wang": 48.6, "Xia": 49.2}
    # antenna gain / aperture efficiency exist only on passive arrays
    assert not [r for r, (_, _, pt) in by_row.items() if r not in passive and (pt["ant_gain"] is not None or pt["ap_eff"] is not None)]


def test_no_relay_gain_counts_as_conversion_gain(pa, by_row):
    relays = rows_where(by_row, type="Relay")
    assert len(relays) == 15
    assert all(pt["tx_gain"] is None and pt["tx_eff"] is None for _, _, pt in relays.values())
    # Nooshabadi 2023, an active relay: 5.6 with no unit; kept out of both gain metrics
    _, _, nooshabadi = by_row[144]
    assert nooshabadi["ant_gain"] is None
    assert any("active relay" in n for n in notes_of(pa, 144))
    # every non-relay conversion gain is kept
    tx_gains = [pt["tx_gain"] for _, e, pt in by_row.values() if e["type"] != "Relay" and pt["tx_gain"] is not None]
    assert len(tx_gains) == 33


def test_relay_elements_are_unit_cells(pa, sheet, by_row):
    # Xia 2016 lists 0 elements; passive relays have no TX/RX chains
    assert sheet[18]["Total Number of Elements (TX + RX)"] == 0
    assert by_row[18][2]["n_elements"] == 1296  # 1 tile x 1296 unit cells
    filled = {r for r in by_row if any("unit cells" in n for n in notes_of(pa, r))}
    assert {by_row[r][0]["first_author"] for r in filled} == {"Guclu", "Carrasco", "Wang", "Xia", "Tan", "Gros"}
    for r in filled:
        cells = sheet[r]
        assert by_row[r][2]["n_elements"] == cells["# of Array Tile"] * cells["# of Antennas per Tile"]
    assert by_row[38][2]["n_elements"] is None  # Hu 2018 gives no unit-cell count
    # active relays keep the survey's TX + RX count (Monroe: 98 x 98 units, TX and RX each)
    assert by_row[99][2]["n_elements"] == 19208


def test_elements_are_tx_plus_rx(pa, by_row):
    assert not [r for r, (_, _, pt) in by_row.items() if pt["n_elements"] == 0]
    for r, (_, e, pt) in by_row.items():
        if pt["n_elements"] is not None and r not in {7, 8, 9, 13, 18, 34, 73}:  # unit-cell rows
            assert pt["n_elements"] == pt["n_tx"] + pt["n_rx"], r
    # Wang 2022 (28/39 GHz): blank in the sheet, filled with TX + RX like every other row
    assert by_row[105][2]["n_elements"] == by_row[106][2]["n_elements"] == 64
    assert any("filled with TX + RX" in n for n in notes_of(pa, 105))


# --- caveats from the Note column ----------------------------------------------

def test_caveat_counts(pa):
    assert pa["meta"]["flag_counts"] == {"passive": 10, "commercial": 7, "non_silicon": 1, "cryogenic": 1}
    assert pa["meta"]["metric_flag_counts"] == {"pdc_scope": 3, "external": 2, "pdc_shared": 8}


def test_commercial_beamformers(by_row):
    rows = {r for r, (_, _, pt) in by_row.items() if "commercial" in pt.get("flags", [])}
    assert Counter(by_row[r][0]["first_author"] for r in rows) == {"Aljuhani": 2, "Gültepe": 3, "Wang": 2}
    assert all("ommer" in by_row[r][1]["note"] for r in rows)


def test_dc_power_beyond_the_array_flags_power_only(by_row):
    rows = {r: e for r, (_, e, _) in by_row.items() if "pdc_scope" in e.get("metric_flags", {})}
    assert {by_row[r][0]["first_author"] for r in rows} == {"Sowlati", "Rio"}
    assert all(e["metric_flags"]["pdc_scope"] == ["rx_pdc", "rx_pdc_el", "tx_pdc", "tx_pdc_el"] for e in rows.values())
    assert not any(by_row[r][2].get("flags") for r in rows)  # EIRP, NF, ... stay comparable


def test_array_results_from_another_paper(by_row):
    rows = {r: e for r, (_, e, _) in by_row.items() if "external" in e.get("metric_flags", {})}
    assert sorted(rows) == [36, 37]  # Lee 2018 TX and RX: array results from Gu et al., ECTC 2018
    assert all(e["metric_flags"]["external"] == ["ap_x", "ap_y", "eirp", "n_eirp"] for e in rows.values())


def test_non_silicon_and_cryogenic(by_row):
    flagged = lambda f: {by_row[r][0]["first_author"] for r, (_, _, pt) in by_row.items() if f in pt.get("flags", [])}
    assert flagged("non_silicon") == {"Zhao"}  # WLCSP of CMOS RX and GaAs LNA
    assert flagged("cryogenic") == {"Wang"}  # THz Cryo-CMOS backscatter transceiver, 4 K
    assert by_row[140][0]["year"] == 2023


# --- formula columns: Excel errors, and the formulas the viewer recomputes ------

def test_excel_errors_become_missing_with_a_warning(pa, sheet, by_row):
    errors = {(r, h) for r, cells in sheet.items() for h, v in cells.items()
              if isinstance(v, str) and v.startswith("#") and h in PHASED_ARRAY.metrics.values()}
    warned = {(w["row"], w["column"]) for w in pa["warnings"] if w["severity"] == "nulled" and "Excel error" in w["issue"]}
    assert errors and errors == warned
    assert {w["severity"] for w in pa["warnings"]} == {"nulled", "note"}  # nothing else goes missing
    assert by_row[47][2]["ap_x"] is None  # Shahramian 2019: broken #REF! formula in the source
    assert all(not (isinstance(v, float) and math.isnan(v)) for _, _, pt in by_row.values() for v in pt.values())


def _max_pdc(pt):
    vs = [v for v in (pt["tx_pdc"], pt["rx_pdc"]) if v is not None]
    return max(vs) if vs else None


@pytest.mark.parametrize("header, compute", [
    ("Array Aperture Size (cm2)",
     lambda pt: pt["ap_x"] * pt["ap_y"] if pt["ap_x"] and pt["ap_y"] else None),
    ("Array Aperture Area / Element (mm2)",
     lambda pt: pt["ap_x"] * pt["ap_y"] * 100 / (pt["tiles"] * pt["ant_per_tile"])
     if pt["ap_x"] and pt["ap_y"] and pt["tiles"] and pt["ant_per_tile"] else None),
    ("Max Package Thermal Density (W/cm2)",
     lambda pt: _max_pdc(pt) / (pt["ap_x"] * pt["ap_y"]) if _max_pdc(pt) is not None and pt["ap_x"] and pt["ap_y"] else None),
    ("Max Chip Thermal Density (W/cm2)",
     lambda pt: _max_pdc(pt) / (pt["tiles"] * pt["ic_per_tile"]) / (pt["chip_area"] / 100)
     if _max_pdc(pt) is not None and pt["tiles"] and pt["ic_per_tile"] and pt["chip_area"] else None),
])
def test_recomputed_formulas_match_the_sheet(sheet, by_row, header, compute):
    """site/surveys/phased_array.js recomputes these columns from the exported inputs; same formulas here."""
    compared = 0
    for r, (_, _, pt) in by_row.items():
        cached = sheet[r][header]
        ours = compute(pt)
        if not isinstance(cached, (int, float)):
            assert ours is None, (r, header, cached, ours)
            continue
        assert ours == pytest.approx(cached, rel=1e-9), (r, header)
        compared += 1
    assert compared > 50
