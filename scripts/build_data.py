#!/usr/bin/env python3
"""Build ``site/data/<survey>.json`` from each survey spreadsheet.

    uv run scripts/build_data.py            # every survey
    uv run scripts/build_data.py lna -v     # one survey, list every note too

Data model
  paper  one publication (rows sharing the survey's id_fields, e.g. year +
         first author + title). Older survey versions list a multi-band paper
         as one row per band; newer ones put "a/b/c" in a single row.
  entry  one spreadsheet row of a paper (process, comment, raw cell text).
  point  one band (or operating mode) of an entry; this is what gets plotted.

Every row is accounted for: it is either emitted as an entry or listed in
``warnings`` with severity "dropped". The build fails otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl

from caveats import Caveats, parse_caveats
from parsing import Parsed, align_bands, classify, clean_text, parse_bands
from survey_sources import SURVEYS, SurveySource

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "site" / "data"


class BuildError(RuntimeError):
    pass


def read_rows(path: Path, sheet: str) -> list[tuple[int, dict[str, Any]]]:
    """Return (excel_row_number, {header: value}) for every non-empty data row."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb[sheet].iter_rows(values_only=True)
        header = [str(h).strip() if h is not None else "" for h in next(rows)]
        out = []
        for excel_row, values in enumerate(rows, start=2):
            if all(v is None or (isinstance(v, str) and not v.strip()) for v in values):
                continue
            out.append((excel_row, dict(zip(header, values))))
        return out
    finally:
        wb.close()


def _clean_field(value: Any) -> tuple[Any, list[str]]:
    if isinstance(value, str):
        text, notes = clean_text(value)
        return (text or None), notes
    if isinstance(value, float) and value.is_integer():
        return int(value), []
    return value, []


def _paper_key(paper: dict[str, Any], keys: tuple[str, ...]) -> str:
    return "|".join(re.sub(r"[^a-z0-9]+", "", str(paper.get(k) or "").lower()) for k in keys)


def _paper_id(key: str) -> str:
    """Stable short id, so shared links survive re-sorting and new survey versions."""
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def _warn(pid: str, row: int, column: str, raw: Any, issue: str, severity: str) -> dict[str, Any]:
    return {
        "paper": pid,
        "row": row,
        "column": column,
        "raw": None if raw is None else str(raw),
        "issue": issue,
        "severity": severity,
    }


def _parse_entry(src: SurveySource, row: dict[str, Any], warn) -> tuple[int, dict, dict, list]:
    """Parse the numeric cells of one row.

    Returns (n_bands, aligned values per metric, sources used for fallbacks,
    per-band frequency values). n_bands == 0 means the row cannot be plotted.
    """
    parsed = {k: parse_bands(row.get(h)) for k, h in src.metrics.items()}
    header_of = dict(src.metrics)

    # Resolve fallbacks (e.g. freq = fc, else f@NFmin) and record the source used.
    resolved: dict[str, list[Parsed]] = dict(parsed)
    sources: dict[str, str] = {}
    for out_key, (primary, fallback) in src.fallbacks.items():
        chosen = next(
            (k for k in (primary, fallback) if any(p.value is not None for p in parsed[k])),
            None,
        )
        resolved[out_key] = parsed[chosen] if chosen else []
        header_of[out_key] = src.metrics[chosen or primary]
        if chosen:
            sources[out_key] = chosen

    # Report unreadable tokens and stripped annotations, once per cell.
    seen: set[tuple[str, str]] = set()
    for key in (*src.band_metrics, *src.info_metrics):
        header = header_of[key]
        for i, p in enumerate(resolved[key]):
            for msg, sev in ((p.error, "nulled"), (p.note, "note")):
                if msg and (header, msg) not in seen:
                    seen.add((header, msg))
                    band = f"band {i + 1}: " if len(resolved[key]) > 1 else ""
                    warn(header, row.get(header), band + msg, sev)

    n, aligned, mismatched = align_bands({k: resolved[k] for k in src.band_metrics}, src.band_key)
    for key in mismatched:
        warn(header_of[key], row.get(header_of[key]),
             f"{len(resolved[key])} values for {n} bands; not plotted for this paper", "nulled")

    freq = [p.value if p else None for p in aligned.get(src.band_key, [])]
    if n == 0 or all(v is None for v in freq):
        header = header_of[src.band_key]
        warn(header, row.get(header), "no usable frequency; row not plotted", "dropped")
        return 0, {}, sources, []

    for key in src.info_metrics:
        vals = resolved[key]
        if len(vals) in (0, 1, n):
            aligned[key] = [vals[0]] * n if len(vals) == 1 else (list(vals) or [None] * n)
        else:
            aligned[key] = [None] * n
            warn(header_of[key], row.get(header_of[key]),
                 f"{len(vals)} values for {n} bands; shown as raw text only", "note")
    for i, v in enumerate(freq):
        if v is None:
            header = header_of[src.band_key]
            warn(header, row.get(header), f"band {i + 1} has no frequency; band not plotted", "nulled")
    return n, aligned, sources, freq


def build_survey(src: SurveySource, root: Path = ROOT) -> dict[str, Any]:
    """Read one survey spreadsheet and return the JSON-ready dataset."""
    path = root / src.source
    rows = read_rows(path, src.sheet)

    papers: dict[str, dict[str, Any]] = {}  # insertion-ordered by first appearance
    points: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    dropped_rows: set[int] = set()
    entries_out = 0

    for excel_row, row in rows:
        fields: dict[str, Any] = {}
        text_notes: list[tuple[str, str]] = []
        for key, header in (*src.paper_fields.items(), *src.entry_fields.items()):
            fields[key], notes = _clean_field(row.get(header))
            text_notes += [(header, n) for n in notes]

        pid = _paper_id(_paper_key(fields, src.id_fields))

        def warn(column: str, raw: Any, issue: str, severity: str) -> None:
            warnings.append(_warn(pid, excel_row, column, raw, issue, severity))

        for header, note in text_notes:
            warn(header, row.get(header), note, "note")

        n, aligned, sources, freq = _parse_entry(src, row, warn)
        if n == 0:
            dropped_rows.add(excel_row)
            continue

        paper = papers.get(pid)
        if paper is None:
            paper = {"id": pid, **{k: fields[k] for k in src.paper_fields}, "entries": []}
            papers[pid] = paper
        else:
            for key, header in src.paper_fields.items():
                if str(fields[key]).lower() != str(paper[key]).lower():
                    warn(header, row.get(header),
                         f"differs from this paper's first row ({paper[key]!r}); first row kept", "note")

        raw = {h: str(row[h]) for h in src.metrics.values() if row.get(h) is not None}
        for prev in paper["entries"]:
            if prev["raw"] == raw:
                warn("", None, f"same values as row {prev['row']} of this paper (duplicate row?)", "note")
                break

        cav = Caveats()
        if src.caveats:
            rules = src.caveats
            cav = parse_caveats(fields.get(rules.text_field), fields.get(rules.title_field or ""), n, rules)
            for note in cav.notes:
                warn(src.entry_fields.get(rules.text_field, rules.text_field), fields.get(rules.text_field), note, "note")

        entry_idx = len(paper["entries"])
        entry: dict[str, Any] = {
            "row": excel_row,
            **{k: fields[k] for k in src.entry_fields},
            "group": classify(fields.get(src.group_field), src.group_rules, src.group_default),
            "n_bands": n,
            "sources": sources,
            "raw": raw,
        }
        if cav.sim_metrics:
            entry["sim_metrics"] = sorted(cav.sim_metrics)
        paper["entries"].append(entry)
        entries_out += 1

        for i in range(n):
            if freq[i] is None:
                continue
            point: dict[str, Any] = {"paper": pid, "entry": entry_idx, "band": i}
            for key in (*src.band_metrics, *src.info_metrics):
                p = aligned[key][i]
                point[key] = p.value if p else None
            if cav.temps_k and cav.temps_k[i] is not None:
                point["temp_k"] = cav.temps_k[i]
            flags = cav.band_flags(i)
            if flags:
                point["flags"] = flags
            points.append(point)

    rows_in = len(rows)
    if rows_in != entries_out + len(dropped_rows):
        raise BuildError(f"{src.id}: {rows_in} rows in, {entries_out} emitted + {len(dropped_rows)} dropped")
    flagged = {w["row"] for w in warnings if w["severity"] == "dropped"}
    if flagged != dropped_rows:
        raise BuildError(f"{src.id}: dropped rows without a warning: {sorted(dropped_rows - flagged)}")

    return {
        "meta": {
            "survey": src.id,
            "source_file": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows_in": rows_in,
            "rows_dropped": len(dropped_rows),
            "papers_out": len(papers),
            "points_out": len(points),
            "warning_counts": dict(Counter(w["severity"] for w in warnings)),
            "flag_counts": dict(Counter(f for pt in points for f in pt.get("flags", []))),
        },
        "papers": list(papers.values()),
        "points": points,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("surveys", nargs="*", help=f"survey ids (default: all of {', '.join(SURVEYS)})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    ap.add_argument("-v", "--verbose", action="store_true", help="also list 'note' warnings")
    args = ap.parse_args(argv)

    unknown = [s for s in args.surveys if s not in SURVEYS]
    if unknown:
        ap.error(f"unknown survey(s): {', '.join(unknown)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for sid in args.surveys or list(SURVEYS):
        data = build_survey(SURVEYS[sid])
        out = args.out / f"{sid}.json"
        out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        m = data["meta"]
        print(
            f"[{sid}] {m['rows_in']} rows -> {m['papers_out']} papers, {m['points_out']} points, "
            f"{m['rows_dropped']} rows dropped; warnings {m['warning_counts']} -> {out}"
        )
        for w in data["warnings"]:
            if w["severity"] != "note" or args.verbose:
                print(f"  row {w['row']:>4} [{w['severity']}] {w['column']}: {w['raw']!r} -> {w['issue']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
