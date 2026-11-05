"""ETH IDEAS phased-array survey (v3.0, 2024-01-17): how to read the spreadsheet.

The workbook is not committed; the build downloads it (``url``) and checks its
sha256. One row is one array configuration of a paper; a paper spans several
rows when it reports TX and RX modes, two bands or two array sizes. Every row
has a single carrier frequency, so every entry is one point.

Survey-specific rules live in ``adjust`` (all of them verified on v3.0, see
tests/test_phased_array.py). The Readme sheet's notes 6.1-6.9 define the
metrics; the numbers below refer to them.

* Type: "TX FE" / "RX FE" / "TRX FE" (6.2: the frequency conversion is on a
  separate chip) become type TX / RX / TRX plus ``conversion`` = "Front-end
  only", so type can drive the marker shape and front-end-only designs can be
  filtered on their own.
* RX Pdc = 0 next to a TX Pdc means "reported together in TX Pdc" (6.4), not
  0 W: the zero is left out and TX Pdc is marked ``pdc_shared``. On an array
  without RX elements the zero just means "no receiver" and is left out.
* Passive reflectarrays / RIS (Note "Passive", "RIS ...", "Passive
  metasurface"): "TX Conversion Gain" is the antenna gain in dBi (6.8) and
  "TX Efficiency" the aperture efficiency, so both move to their own metrics
  (``ant_gain``, ``ap_eff``). An active relay's gain has no stated unit and is
  kept out of the conversion gain.
* Elements of a relay: a passive relay has no TX/RX chains (the survey writes
  0 or leaves it blank), so its element count is its unit cells, tiles x
  antennas per tile. Everywhere else elements = TX + RX elements, as in the
  sheet's own formula.
* Caveats from the Note column (and one title): passive, commercial
  beamformer ICs, non-silicon parts, cryogenic operation (point flags); DC
  power that covers more than the array, TX Pdc that includes RX, array
  results taken from another paper (flags on the affected metrics only).
"""

from __future__ import annotations

import re

from parsing import Parsed
from survey_sources import RowContext, SurveySource

TYPE_HEADER = "Type (TX, RX, TRX, Oscillator)"
NOTE_HEADER = "Note"

# Base types (Readme 6.2) and the canonical spelling of each.
TYPES = {"tx": "TX", "rx": "RX", "trx": "TRX", "oscillator": "Oscillator", "relay": "Relay"}
_TYPE_RE = re.compile(r"(?P<base>TRX|TX|RX|Oscillator|Relay)(?:\s+(?P<fe>FE))?", re.IGNORECASE)

# Values of the "conversion" category.
CONVERSION_INTEGRATED = "Integrated"
CONVERSION_FE_ONLY = "Front-end only"
CONVERSION_NA = "Not applicable"  # oscillator arrays and relays have no up/down-conversion

ANTENNA_NOT_REPORTED = "Not reported"

# Text placeholders in descriptive columns; numeric columns go through parsing.MISSING_TOKENS.
_TEXT_PLACEHOLDERS = {"n/a", "na", "-", ""}

# Caveats written in the Note column (case-insensitive). Typos are the survey's own:
# "Commerical", "Commericial", "Intgeration".
PASSIVE_RE = re.compile(r"passive|\bRIS\b", re.IGNORECASE)
COMMERCIAL_RE = re.compile(r"\bcommer\w*al\b", re.IGNORECASE)
NON_SILICON_RE = re.compile(r"\b(?:GaAs|InP|GaN)\b", re.IGNORECASE)
# "Power Consumption of full radio including ...", "DC power of entire ... package of ..."
PDC_SCOPE_RE = re.compile(r"\b(?:power consumption|dc power) of (?:the )?(?:full|entire)\b", re.IGNORECASE)
# "Array Results from Gu et al. ..."
EXTERNAL_RE = re.compile(r"\bresults from\b", re.IGNORECASE)
CRYO_TITLE_RE = re.compile(r"\bcryo", re.IGNORECASE)

# Metrics a caveat applies to, when it applies to some values of an entry only.
PDC_METRICS = ("tx_pdc", "rx_pdc", "tx_pdc_el", "rx_pdc_el")
TX_PDC_METRICS = ("tx_pdc", "tx_pdc_el")
# Array-level (antenna / package) results; Lee 2018 takes these from Gu et al., ECTC 2018.
ARRAY_METRICS = ("eirp", "n_eirp", "ap_x", "ap_y")

# FinFET nodes are 22 nm and below; Monroe 2022 lists 265 (its carrier frequency).
FINFET_MAX_NODE_NM = 30


def _get(ctx: RowContext, key: str) -> float | None:
    vals = ctx.values.get(key) or []
    p = vals[0] if vals else None
    return p.value if p else None


def _set(ctx: RowContext, key: str, value: float | None) -> None:
    ctx.values[key] = [None if value is None else Parsed(float(value))] * ctx.n_bands


def _flag_metrics(ctx: RowContext, caveat: str, metrics: tuple[str, ...]) -> None:
    ctx.metric_flags.setdefault(caveat, set()).update(metrics)


def split_type(label: str | None) -> tuple[str, str] | None:
    """("TX FE") -> ("TX", "Front-end only"); None when the label is not a known type."""
    m = _TYPE_RE.fullmatch((label or "").strip())
    if not m:
        return None
    base = TYPES[m["base"].lower()]
    if m["fe"]:
        return base, CONVERSION_FE_ONLY
    return base, CONVERSION_INTEGRATED if base in ("TX", "RX", "TRX") else CONVERSION_NA


def _clean_text_fields(ctx: RowContext) -> None:
    for key in ("dimension", "co_aperture", "antenna_type", "antenna", "process"):
        v = ctx.fields.get(key)
        if isinstance(v, str) and v.strip().lower() in _TEXT_PLACEHOLDERS:
            ctx.fields[key] = None


def _type(ctx: RowContext) -> str:
    label = ctx.fields.get("type")
    parts = split_type(label)
    ctx.fields["type_label"] = label
    if parts is None:
        ctx.notes.append((TYPE_HEADER, f"unknown type {label!r}; shown as 'Other'"))
        ctx.fields["type"], ctx.fields["conversion"] = "Other", CONVERSION_NA
        return "Other"
    ctx.fields["type"], ctx.fields["conversion"] = parts
    return parts[0]


def _node(ctx: RowContext) -> None:
    node, process = ctx.fields.get("node"), ctx.fields.get("process") or ""
    if isinstance(node, (int, float)) and "finfet" in process.lower() and node > FINFET_MAX_NODE_NM:
        ctx.notes.append(("Tech Node (nm)", f"{node} nm is not a FinFET node; left out"))
        ctx.fields["node"] = None


def _rx_pdc_zero(ctx: RowContext) -> None:
    """Readme 6.4: TX and RX power reported together go under TX Pdc, with RX Pdc = 0."""
    zero = [k for k in ("rx_pdc", "rx_pdc_el") if _get(ctx, k) == 0]
    if not zero:
        return
    for k in zero:
        _set(ctx, k, None)
    header = "RX Pdc (W)" if "rx_pdc" in zero else "RX Pdc (mW)"
    if not _get(ctx, "n_rx"):
        ctx.notes.append((header, "RX Pdc = 0 on an array without RX elements: not applicable, left out"))
    elif _get(ctx, "tx_pdc") is None and _get(ctx, "tx_pdc_el") is None:
        ctx.notes.append((header, "RX Pdc = 0 with no TX Pdc reported; left out"))
    else:
        ctx.notes.append((header, "RX Pdc = 0: reported together with TX Pdc (Readme 6.4); left out, TX Pdc marked"))
        _flag_metrics(ctx, "pdc_shared", TX_PDC_METRICS)


def _relay_gain(ctx: RowContext, base: str, passive: bool) -> None:
    """Keep antenna gain (dBi) and aperture efficiency of passive arrays out of the TX metrics.

    ``ant_gain`` / ``ap_eff`` read the same columns as ``tx_gain`` / ``tx_eff``;
    exactly one of each pair keeps the value.
    """
    if passive:
        for tx_key, own_key, header, what in (
            ("tx_gain", "ant_gain", "TX Conversion Gain (dB)", "antenna gain in dBi (Readme 6.8)"),
            ("tx_eff", "ap_eff", "TX Efficiency (%)", "aperture efficiency"),
        ):
            if _get(ctx, tx_key) is not None:
                ctx.notes.append((header, f"passive array: this is its {what}, not a TX value; shown as {own_key}"))
            _set(ctx, tx_key, None)
        return
    _set(ctx, "ant_gain", None)
    _set(ctx, "ap_eff", None)
    if base == "Relay" and _get(ctx, "tx_gain") is not None:
        ctx.notes.append(("TX Conversion Gain (dB)",
                          "gain of an active relay, unit not stated (dB or dBi); not used as TX conversion gain"))
        _set(ctx, "tx_gain", None)


def _elements(ctx: RowContext, base: str, passive: bool) -> None:
    header = "Total Number of Elements (TX + RX)"
    n, n_tx, n_rx = _get(ctx, "n_elements"), _get(ctx, "n_tx"), _get(ctx, "n_rx")
    if n:
        return
    if base == "Relay" and passive and not n_tx and not n_rx:
        tiles, per_tile = _get(ctx, "tiles"), _get(ctx, "ant_per_tile")
        if tiles and per_tile:
            cells = tiles * per_tile
            _set(ctx, "n_elements", cells)
            ctx.notes.append((header, f"passive relay without TX/RX chains: elements = unit cells "
                                      f"(tiles × antennas per tile) = {cells:g}"))
        else:
            _set(ctx, "n_elements", None)
            ctx.notes.append((header, "passive relay without TX/RX chains or a unit-cell count; no element count"))
    elif n is None and n_tx is not None and n_rx is not None and n_tx + n_rx > 0:
        _set(ctx, "n_elements", n_tx + n_rx)
        ctx.notes.append((header, f"blank; filled with TX + RX elements = {n_tx + n_rx:g} (the column's formula)"))
    elif n == 0:
        _set(ctx, "n_elements", None)
        ctx.notes.append((header, "0 elements; left out"))


def _caveats(ctx: RowContext, note: str, passive: bool) -> None:
    if passive:
        ctx.flags.add("passive")
    if COMMERCIAL_RE.search(note):
        ctx.flags.add("commercial")
    if NON_SILICON_RE.search(note):
        ctx.flags.add("non_silicon")
    if CRYO_TITLE_RE.search(ctx.fields.get("title") or ""):
        ctx.flags.add("cryogenic")
    if PDC_SCOPE_RE.search(note):
        _flag_metrics(ctx, "pdc_scope", PDC_METRICS)
    if EXTERNAL_RE.search(note):
        _flag_metrics(ctx, "external", ARRAY_METRICS)


def adjust(ctx: RowContext) -> None:
    """Survey-specific rules for one row of the phased-array survey (see the module docstring)."""
    _clean_text_fields(ctx)
    if ctx.fields.get("antenna") is None:
        ctx.fields["antenna"] = ANTENNA_NOT_REPORTED
    base = _type(ctx)
    note = ctx.fields.get("note") or ""
    passive = bool(PASSIVE_RE.search(note))
    _node(ctx)
    _rx_pdc_zero(ctx)
    _relay_gain(ctx, base, passive)
    _elements(ctx, base, passive)
    _caveats(ctx, note, passive)


PHASED_ARRAY = SurveySource(
    id="phased_array",
    source="sources/phased_array/phased_array_survey_v3.xlsx",
    url="https://ethz.ch/content/dam/ethz/special-interest/itet/ideas-dam/documents/phased_array_survey_v3.xlsx",
    sha256="715284d6317149b8ba1637001219eb7a3ebe675cf928c0dfa92b37f6497f7478",
    sheet="Data",
    paper_fields={
        "venue": "Publication",
        "year": "Year",
        "month": "Month",
        "first_author": "Last Name (1st Author)",
        "title": "Paper Title",
    },
    entry_fields={
        "type": TYPE_HEADER,
        "process": "Process (CMOS_Bulk, CMOS_SOI, CMOS_FinFET, SiGe)",
        "node": "Tech Node (nm)",
        "dimension": "Dimension (1D, 2D)",
        "tx_pol": "TX # of Polarization",
        "rx_pol": "RX # of Polarization",
        "antenna": "Antenna on (None, Chip, AiP, PCB)",
        "antenna_type": "Antenna Integration Type",
        "co_aperture": "TX RX Co-Aperture (YES/NO)",
        "note": NOTE_HEADER,
    },
    # Formula columns that are recomputed in the viewer from these inputs instead
    # (site/surveys/phased_array.js): aperture size and area per element, package
    # and chip thermal density. Kept as in the sheet: total Pout/Prad, TX/RX Pdc (W),
    # chip size and chip core area per element (some rows are typed, not formulas).
    metrics={
        "freq": "Frequency (GHz)",
        "n_elements": "Total Number of Elements (TX + RX)",
        "n_tx": "Total # of TX Elements",
        "n_rx": "Total # of RX Elements",
        "eirp": "TX EIRP (dBm)",
        "n_eirp": "# of Elements @ EIRP Measurement",
        "pout": "Total Pout/Prad (dBm)",
        "tx_pdc": "TX Pdc (W)",
        "rx_pdc": "RX Pdc (W)",
        "tiles": "# of Array Tile",
        "ant_per_tile": "# of Antennas per Tile",
        "ic_per_tile": "# of IC per Tile",
        "tx_per_ic": "# of TX Element per IC",
        "rx_per_ic": "# of RX Element per IC",
        "ap_x": "Array Aperture Dimension -X (cm)",
        "ap_y": "Array Aperture Dimension -Y (cm)",
        "chip_area": "Chip Size (mm2)",
        "core_area_el": "Chip Core Area / Element (mm2)",
        # single-element performance
        "psat": "TX Psat/Prad (dBm)",
        "tx_eff": "TX Efficiency (%)",
        "tx_gain": "TX Conversion Gain (dB)",
        "ant_gain": "TX Conversion Gain (dB)",  # passive arrays only (adjust); antenna gain, dBi
        "ap_eff": "TX Efficiency (%)",  # passive arrays only (adjust); aperture efficiency
        "nf": "RX NF (dB)",
        "rx_gain": "RX Conversion Gain (dB)",
        "tx_pdc_el": "TX Pdc (mW)",
        "rx_pdc_el": "RX Pdc (mW)",
    },
    fallbacks={},
    band_key="freq",
    band_metrics=(
        "freq", "n_elements", "n_tx", "n_rx", "eirp", "n_eirp", "pout", "tx_pdc", "rx_pdc",
        "tiles", "ant_per_tile", "ic_per_tile", "tx_per_ic", "rx_per_ic", "ap_x", "ap_y",
        "chip_area", "core_area_el", "psat", "tx_eff", "tx_gain", "ant_gain", "ap_eff",
        "nf", "rx_gain", "tx_pdc_el", "rx_pdc_el",
    ),
    info_metrics=(),
    id_fields=("year", "first_author", "title"),
    group_field="process",
    group_rules=[
        ("FinFET", r"finfet"),
        ("SiGe", r"sige"),
        ("CMOS SOI", r"soi"),
        ("CMOS Bulk", r"cmos"),
    ],
    group_default="Not reported",  # blank: passive reflectarrays and two arrays of commercial ICs
    adjust=adjust,
)
