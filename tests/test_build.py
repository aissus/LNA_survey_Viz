"""End-to-end checks of build_data.py against the real LNA spreadsheet.

The spreadsheet is downloaded from ETH when it is missing (as in CI). Without
a network the tests are skipped; a file that does not match the pinned sha256
fails them, because the counts below belong to that exact file.
"""

import json
import urllib.error
from collections import Counter

import pytest

from build_data import build_survey, ensure_source, main
from survey_sources import LNA


@pytest.fixture(scope="module")
def lna_source():
    try:
        return ensure_source(LNA)
    except (urllib.error.URLError, OSError) as err:
        pytest.skip(f"LNA spreadsheet not available offline: {err}")


@pytest.fixture(scope="module")
def lna(lna_source):
    return build_survey(LNA)


def points_of(data, **paper_match):
    paper = next(p for p in data["papers"] if all(p[k] == v for k, v in paper_match.items()))
    return paper, [pt for pt in data["points"] if pt["paper"] == paper["id"]]


def test_every_row_is_accounted_for(lna):
    m = lna["meta"]
    entries = sum(len(p["entries"]) for p in lna["papers"])
    assert m["rows_in"] == 561  # 472 of release 3.0 + 89 rows ETH appended in 2026
    assert entries + m["rows_dropped"] == m["rows_in"]
    dropped = {w["row"] for w in lna["warnings"] if w["severity"] == "dropped"}
    assert len(dropped) == m["rows_dropped"]


def test_spot_check_multiband_paper(lna):
    paper, pts = points_of(lna, first_author="Afifi", year=2025)
    assert [p["freq"] for p in pts] == pytest.approx([7.35, 10.65, 14.75, 11.45])
    assert [p["nf"] for p in pts] == pytest.approx([1.6, 1.6, 1.8, 2.4])
    assert [p["power"] for p in pts] == [20, 20, 20, 20]  # single value broadcast
    assert [p["band"] for p in pts] == [0, 1, 2, 3]
    assert paper["entries"][0]["sources"] == {"freq": "fc", "nf": "nf_spec"}


def test_rows_of_the_same_paper_are_one_paper(lna):
    # Seguchi 2022 lists its triple-band LNA as three separate rows
    paper, pts = points_of(lna, first_author="Seguchi", year=2022)
    assert len(paper["entries"]) == 3
    assert sorted(p["freq"] for p in pts) == pytest.approx([1.4, 2.42, 4.21])


def test_genuine_zero_iip3_survives(lna):
    zeros = [pt for pt in lna["points"] if pt["iip3"] == 0]
    assert len(zeros) == 2
    assert all(isinstance(pt["iip3"], float) for pt in zeros)


def test_frequency_falls_back_to_f_at_nfmin(lna):
    sources = Counter(e["sources"].get("freq") for p in lna["papers"] for e in p["entries"])
    assert sources["f_nfmin"] > 300 and sources["fc"] > 100
    assert all(pt["freq"] is not None for pt in lna["points"])


def test_sige_bicmos_is_not_cmos_bulk(lna):
    groups = Counter(e["group"] for p in lna["papers"] for e in p["entries"] if "sige" in (e["process"] or "").lower())
    assert set(groups) == {"SiGe"}


def test_titles_have_no_excel_escapes(lna):
    assert not [p["title"] for p in lna["papers"] if "_x00" in p["title"]]
    assert any("Variable-Gain" in p["title"] for p in lna["papers"])


def test_paper_ids_are_unique_and_stable(lna):
    ids = [p["id"] for p in lna["papers"]]
    assert len(ids) == len(set(ids))
    assert build_survey(LNA)["papers"][0]["id"] == ids[0]


def test_mismatched_column_is_flagged_not_silently_dropped(lna):
    flagged = [w for w in lna["warnings"] if w["severity"] == "nulled"]
    assert any(w["raw"] == "0,03/0,02" for w in flagged)


def test_caveats_reach_the_points(lna):
    counts = lna["meta"]["flag_counts"]
    assert set(counts) == {"simulated", "system", "cryogenic"}
    # Puyal 2024: alternating 300 K / 4.2 K bands, only the cold ones are cryogenic
    paper, pts = points_of(lna, first_author="Puyal", year=2024)
    assert [p.get("temp_k") for p in pts] == [300, 4.2] * 4
    assert [p.get("flags", []) for p in pts] == [[], ["cryogenic"]] * 4
    # individually simulated values are listed on the entry, not as a point flag
    sim_entries = [e for p in lna["papers"] for e in p["entries"] if e.get("metric_flags")]
    assert len(sim_entries) == 17
    assert all(set(e["metric_flags"]) == {"simulated"} for e in sim_entries)
    assert set().union(*(e["metric_flags"]["simulated"] for e in sim_entries)) == {"nf", "nfmin", "gain", "iip3", "p1db"}


def test_problems_in_the_rows_appended_in_2026_are_reported(lna):
    issues = {(w["row"], w["severity"]): w for w in lna["warnings"] if w["row"] > 473}
    assert issues[(536, "nulled")]["raw"] == "70-88"  # a power range, not a value
    assert "--17" in issues[(541, "nulled")]["raw"]  # a doubled minus sign in band 4 of P1dB
    assert issues[(546, "dropped")]["issue"].startswith("no usable frequency")
    assert lna["meta"]["rows_dropped"] == 1


def test_cli_writes_json(lna_source, tmp_path, capsys):
    assert main(["lna", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "lna.json").read_text(encoding="utf-8"))
    assert data["meta"]["survey"] == "lna"
    assert "[lna] 561 rows" in capsys.readouterr().out


def test_cli_rejects_unknown_survey():
    with pytest.raises(SystemExit):
        main(["nope"])
