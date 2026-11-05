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
import urllib.request
import warnings as pywarnings  # "warnings" is the import-notes list below
from collections import Counter
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import openpyxl

from caveats import Caveats, parse_caveats
from parsing import Parsed, align_bands, classify, clean_text, parse_bands
from registry import SURVEYS
from survey_sources import RowContext, SurveySource

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "site" / "data"


class BuildError(RuntimeError):
    pass


def ensure_source(src: SurveySource, root: Path = ROOT) -> Path:
    """Path of the survey spreadsheet; downloads it first when it is fetched at build time."""
    path = root / src.source
    if not path.exists() and src.url:
        print(f"[{src.id}] downloading {src.url}", file=sys.stderr)
        req = urllib.request.Request(src.url, headers={"User-Agent": "survey-viz-build"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        if src.sha256 and hashlib.sha256(data).hexdigest() != src.sha256:
            raise BuildError(f"{src.id}: downloaded file does not match the pinned sha256; "
                             "the survey was probably updated, check it and update the pin")
        path.write_bytes(data)
    if not path.exists():
        raise BuildError(f"{src.id}: source spreadsheet not found: {path}")
    if src.sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != src.sha256:
        raise BuildError(f"{src.id}: {path.name} does not match the pinned sha256")
    return path


def _load_workbook(path: Path) -> Any:
    # not read_only: merged ranges are only available on a fully loaded sheet
    with pywarnings.catch_warnings():  # charts and shapes are not needed
        pywarnings.filterwarnings("ignore", message="DrawingML support is incomplete")
        return openpyxl.load_workbook(path, data_only=True)


def _sheet_rows(ws: Any, aliases: dict[str, str] | None = None) -> list[tuple[int, dict[str, Any]]]:
    """Return (excel_row_number, {header: value}) for every non-empty data row.

    ``aliases`` renames this sheet's headers. When a header repeats, the first
    column wins (later copies are typically formula helper columns). A cell
    merged over several rows applies to each of them, as it does on screen.
    """
    merged: dict[tuple[int, int], Any] = {}  # (row, col) -> value of the merged range
    for rng in ws.merged_cells.ranges:
        top = ws.cell(rng.min_row, rng.min_col).value
        for r in range(rng.min_row + 1, rng.max_row + 1):
            merged[(r, rng.min_col)] = top
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    header = [(aliases or {}).get(h, h) for h in header]
    out = []
    for excel_row, values in enumerate(rows, start=2):
        values = [merged.get((excel_row, c), v) for c, v in enumerate(values, start=1)]
        if all(v is None or (isinstance(v, str) and not v.strip()) for v in values):
            continue
        row: dict[str, Any] = {}
        for h, v in zip(header, values):
            row.setdefault(h, v)
        out.append((excel_row, row))
    return out


def read_rows(path: Path, sheet: str, aliases: dict[str, str] | None = None) -> list[tuple[int, dict[str, Any]]]:
    """Rows of one sheet; see ``_sheet_rows``."""
    wb = _load_workbook(path)
    try:
        return _sheet_rows(wb[sheet], aliases)
    finally:
        wb.close()


def _clean_field(value: Any) -> tuple[Any, list[str]]:
    if isinstance(value, str):
        text, notes = clean_text(value)
        return (text or None), notes
    if isinstance(value, float) and value.is_integer():
        return int(value), []
    if isinstance(value, (date, time)):  # Excel read a typed range such as "8-12" as a date
        text = value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
        return text, [f"Excel stored this cell as the date/time {text}; what was typed is lost"]
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
    path = ensure_source(src, root)
    sheets = (src.sheet,) if isinstance(src.sheet, str) else tuple(src.sheet)
    multi = len(sheets) > 1  # then entries and warnings record their sheet
    wb = _load_workbook(path)
    try:
        rows = [
            (sheet.strip(), excel_row, row)  # recorded without stray spaces (" Multipliers")
            for sheet in sheets
            for excel_row, row in _sheet_rows(wb[sheet], src.header_aliases.get(sheet))
        ]
    finally:
        wb.close()

    papers: dict[str, dict[str, Any]] = {}  # insertion-ordered by first appearance
    points: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    dropped_rows: set[tuple[str | None, int]] = set()  # (sheet if multi, row)
    entries_out = 0

    for sheet, excel_row, row in rows:
        fields: dict[str, Any] = {}
        text_notes: list[tuple[str, str]] = []
        for key, header in (*src.paper_fields.items(), *src.entry_fields.items()):
            fields[key], notes = _clean_field(row.get(header))
            text_notes += [(header, n) for n in notes]
        if multi:
            fields["sheet"] = sheet

        pid = _paper_id(_paper_key(fields, src.id_fields))

        def warn(column: str, raw: Any, issue: str, severity: str) -> None:
            w = _warn(pid, excel_row, column, raw, issue, severity)
            if multi:
                w["sheet"] = sheet
            warnings.append(w)

        for header, note in text_notes:
            warn(header, row.get(header), note, "note")

        n, aligned, sources, freq = _parse_entry(src, row, warn)
        if n == 0:
            dropped_rows.add((sheet if multi else None, excel_row))
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

        ctx = RowContext(
            row=excel_row,
            raw=row,
            fields=fields,
            values={k: aligned[k] for k in (*src.band_metrics, *src.info_metrics)},
            n_bands=n,
            metric_flags={"simulated": set(cav.sim_metrics)} if cav.sim_metrics else {},
        )
        if src.adjust:
            src.adjust(ctx)
            for column, note in ctx.notes:
                warn(column, row.get(column), note, "note")
            if not paper["entries"]:  # this row starts the paper: keep the hook's fixes
                paper.update({k: ctx.fields[k] for k in src.paper_fields})

        entry_idx = len(paper["entries"])
        entry: dict[str, Any] = {
            "row": excel_row,
            **{k: v for k, v in ctx.fields.items() if k not in src.paper_fields},
            "group": classify(ctx.fields.get(src.group_field), src.group_rules, src.group_default),
            "n_bands": n,
            "sources": sources,
            "raw": raw,
        }
        metric_flags = {f: sorted(ms) for f, ms in sorted(ctx.metric_flags.items()) if ms}
        if metric_flags:
            entry["metric_flags"] = metric_flags
        paper["entries"].append(entry)
        entries_out += 1

        freq = [p.value if p else None for p in ctx.values[src.band_key]]
        for i in range(n):
            if freq[i] is None:
                continue
            point: dict[str, Any] = {"paper": pid, "entry": entry_idx, "band": i}
            for key in (*src.band_metrics, *src.info_metrics):
                p = ctx.values[key][i]
                point[key] = p.value if p else None
            if cav.temps_k and cav.temps_k[i] is not None:
                point["temp_k"] = cav.temps_k[i]
            flags = sorted(set(cav.band_flags(i)) | ctx.flags)
            if flags:
                point["flags"] = flags
            points.append(point)

    rows_in = len(rows)
    if rows_in != entries_out + len(dropped_rows):
        raise BuildError(f"{src.id}: {rows_in} rows in, {entries_out} emitted + {len(dropped_rows)} dropped")
    flagged = {(w.get("sheet"), w["row"]) for w in warnings if w["severity"] == "dropped"}
    if flagged != dropped_rows:
        raise BuildError(f"{src.id}: dropped rows without a warning: {sorted(dropped_rows - flagged, key=str)}")

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
            # points whose entry has a caveat on some of its values only
            "metric_flag_counts": dict(Counter(
                f for pt in points
                for f in papers[pt["paper"]]["entries"][pt["entry"]].get("metric_flags", {})
            )),
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
                where = f"{w['sheet']} row {w['row']}" if "sheet" in w else f"row {w['row']:>4}"
                print(f"  {where} [{w['severity']}] {w['column']}: {w['raw']!r} -> {w['issue']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
