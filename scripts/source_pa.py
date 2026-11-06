"""ETH PA survey (Power Amplifiers Performance Survey 2000-Present, v11).

The workbook keeps one sheet per technology (CMOS, SiGe, GaN, GaAs, InP,
LDMOS, Others) plus product datasheets, frequency multipliers and THz
oscillators. They are read as one table; each entry records its sheet and a
"circuit" category (PA, PA product, Multiplier, Oscillator).

Survey-specific rules live in ``adjust``:
  - technology from the process string (sheet as fallback); the colour group
    is its family (GaN, GaAs and InP are III-V);
  - the "PAE Note" column and "(DE)"-style annotations: efficiencies that are
    drain / collector / system efficiency, not PAE, are flagged per metric;
  - caveats written in the notes columns (simulated, pulsed, system-level,
    off-chip matching / combining, estimated from a figure);
  - a year typed as text becomes a number; UTF-8 text that was decoded as
    Windows-1252 or Mac Roman ("26.5â€“29.5", "O‚ÄôMalley") is repaired;
  - empty text fields are left out of the entry (most columns exist on one
    sheet only, and the JSON is ~6k entries).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from parsing import clean_text
from survey_sources import RowContext, SurveySource

PA_SHEETS = ("CMOS", "SiGe", "GaN", "GaAs", "InP", "LDMOS", "Others")
CIRCUIT_BY_SHEET = {
    **{s: "PA" for s in PA_SHEETS},
    "Products": "PA product",
    "Multipliers": "Multiplier",
    "THz Oscillators": "Oscillator",
}
# technology when the process cell is empty
SHEET_TECHNOLOGY = {"SiGe": "SiGe", "GaN": "GaN", "GaAs": "GaAs", "InP": "InP", "LDMOS": "LDMOS"}

# (technology, regex on the process string); first match wins, so the order matters:
# "SiGe BiCMOS" and "BiCMOS_SOI" are SiGe, "CMOS_FD_SOI" is SOI before plain CMOS,
# "GaN on SiC" is GaN before the SiC rule, "InP/GaAs HBT" (InP/GaAsSb DHBT) is InP.
TECHNOLOGY_RULES: list[tuple[str, str]] = [
    ("FinFET", r"fin\s*fet"),
    ("SiGe", r"sige|bicmos"),
    ("CMOS SOI", r"soi|sos"),  # FD/PD-SOI, RFSOI, silicon-on-sapphire
    ("CMOS Bulk", r"cmos"),  # includes plain "CMOS" (not labelled bulk or SOI)
    # the two discrete "MOSFET"s (AFT20S015N, MRF21030) are LDMOS parts
    ("LDMOS", r"ldmos|mosfet"),
    ("GaN", r"gan|cgh\d|cg2h|cree"),
    ("InP", r"\binp\b"),
    ("Other", r"\bsic\b"),  # SiC MESFET
    ("GaAs", r"gaas|ingap|phemt|mhemt|ingaas|hfet|hjfet|bihemt|mesfet"),
]
# technology -> colour group (family); anything else is "Other"
# Colour = technology family; the exact technology (bulk / SOI / FinFET, GaN /
# GaAs / InP) is the "technology" category, which picks the marker shape.
# Only three hues stay distinct for colour-blind readers in a scatter, and
# CMOS, SiGe and III-V together are 89% of the points.
GROUP_RULES: list[tuple[str, str]] = [
    ("CMOS", r"^(CMOS Bulk|CMOS SOI|FinFET)$"),
    ("SiGe", r"^SiGe$"),
    ("III-V", r"^(GaN|GaAs|InP)$"),
    ("LDMOS", r"^LDMOS$"),
]

# UTF-8 decoded as Windows-1252 ("â€“", "Ã—", "Â±") or Mac Roman ("‚Äì", "‚Äô")
_MOJIBAKE = re.compile(r"â€|âˆ|Ã|Â|‚Ä")

# efficiency-type note (column "PAE Note") -> normalised type
_EFF_TYPES: list[tuple[str, str]] = [
    ("DE", r"^de\b|drain"),
    ("CE", r"^ce\b|collector"),
    ("SE", r"^se\b"),
    ("PAE", r"pae|^ae$"),  # "PAE ", "PAE*", "TX_PAE", "Sim PAE", "AE" (a typo)
    ("Other", r"efficiency|dc-thz"),  # "Efficiency", "Conversion Efficiency", "DC-THz"
]
EFF_METRICS = ("pae", "pae_1db", "avg_pae")
# an efficiency cell or modulation note saying the value is not PAE: "70(CE)", "41* (DE)",
# "37 (overall system efficiency)", "DE (with digital predistortion)", "64 QAM PAE(DE)"
_NOT_PAE_CELL = re.compile(r"\((?:DE|CE|SE)\)|^\s*DE\b|system efficiency", re.IGNORECASE)

NOTE_FIELDS = ("note1", "note2", "pa_note", "eff_note", "mod_note", "process")
_SIM = re.compile(r"\bsimulat", re.IGNORECASE)
_PULSED = re.compile(r"\bpulsed?\b(?![- ]?width)", re.IGNORECASE)  # not "pulse-width modulation"
# measured with more than the PA: a T/R switch, an antenna, a TX chain
_SYSTEM = re.compile(r"t/r\b|front-?end|antenna|\bbfic\b|\btx path|^tx$|tx_pae", re.IGNORECASE)
_OFFCHIP = re.compile(r"off-?chip|ltcc|\bsip\b|spatial power|combining module|combining not on", re.IGNORECASE)
_ESTIMATED = re.compile(r"estimat|from graph|from plot", re.IGNORECASE)

_PA_TYPES: list[tuple[str, re.Pattern[str]]] = [
    ("Analog", re.compile(r"^analog", re.IGNORECASE)),
    ("Digital", re.compile(r"di+gital", re.IGNORECASE)),  # "Diigital" once
    # case-sensitive: envelope tracking (ET) / envelope elimination and restoration (EER)
    ("Supply-modulated", re.compile(r"\bET\b|\bEER\b|[Ee]nvelope")),
]

# output key -> canonical header (after header_aliases)
PAPER_FIELDS = {
    "venue": "Publication",
    "year": "Year",
    "month": "Month",
    "first_author": "Last Name (1st Author)",  # vendor, for datasheets
    "title": "Paper Title",  # product name, for datasheets
}
ENTRY_FIELDS = {
    "process": "Process (CMOS_Bulk, CMOS_SOI, SiGe)",
    "node": "Process node",
    "eff_note": "PAE Note",
    "cw_band": "CW Performance",
    "modulation": "RF PA Modulations",
    "mod_rate": "Modulation Speed (Msym/s)",  # mixed units ("54 Mbps", "20 MHz"): text only
    "mod_note": "RF PA Note (Modulation Type)",
    "pa_note": "RF PA Note (Analog PA or Digital PA)",
    "pcb": "PCB",
    "note1": "General 1",
    "note2": "General 2",
    "freq_range": "General 3 (frequency range)",
    "mult_factor": "Multiplication factor",
    "out_bw": "Output bandwidth (GHz)",
    "osc_type": "Oscillator type",
    "tuning": "Tuning range (%)",
    "phase_noise": "Phase noise (dBc/Hz)",
}
METRICS = {
    "freq": "Frequency (GHz)",
    "psat": "Psat (dBm)",
    "pae": "PAEmax (%)",
    "p1db": "P1dB (dBm)",
    "pae_1db": "PAE_1dB (%)",
    "gain": "Gain (dB)",
    "avg_pout": "Average Pout (dBm)",
    "avg_pae": "Average PAE (%)",
    "evm": "EVM (dB)",
    "core_area": "Core_area",
    "chip_area": "Chip_area",
    "pdc": "Pdc (mW)",
    "dc_rf_eff": "DC-to-RF efficiency (%)",
    "conv_gain": "Conversion gain (dB)",
}


def technology(process: str | None, sheet: str | None) -> str:
    """Technology of one entry: from the process string, else from its sheet."""
    if process:
        for tech, pattern in TECHNOLOGY_RULES:
            if re.search(pattern, process, re.IGNORECASE):
                return tech
        return "Other"
    return SHEET_TECHNOLOGY.get(sheet or "", "Other")


def efficiency_type(note: str | None) -> str | None:
    """Normalised "PAE Note": PAE, DE, CE, SE, Other, or None when not stated."""
    if not note:
        return None
    text = note.strip()
    for kind, pattern in _EFF_TYPES:
        if re.search(pattern, text, re.IGNORECASE):
            return kind
    return None  # "NA", "Not Done", "?", "Tx"


def pa_type(note: str | None) -> str:
    """Analog / Digital / Supply-modulated PA, from "RF PA Note (Analog PA or Digital PA)"."""
    for kind, pattern in _PA_TYPES:
        if note and pattern.search(note.strip()):
            return kind
    return "Not stated"


def fix_mojibake(text: str) -> str:
    """Undo UTF-8 text that was decoded as Windows-1252 or Mac Roman; else unchanged."""
    if not _MOJIBAKE.search(text):
        return text
    for encoding in ("cp1252", "mac_roman"):
        data = bytearray()
        try:
            for ch in text:
                try:
                    data += ch.encode(encoding)
                except UnicodeEncodeError:
                    if ord(ch) > 0xFF:
                        raise
                    data.append(ord(ch))  # bytes cp1252 leaves undefined, e.g. 0x9D
            return data.decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def _has(ctx: RowContext, key: str) -> bool:
    return any(p is not None and p.value is not None for p in ctx.values.get(key, []))


def _flag_metrics(ctx: RowContext, caveat: str, keys) -> None:
    keys = {k for k in keys if _has(ctx, k)}
    if keys:
        ctx.metric_flags.setdefault(caveat, set()).update(keys)


# Values that cannot be what the column says; each is blanked with an import note.
# (metric, test, why) — `why` is formatted with the value v.
IMPLAUSIBLE: list[tuple[str, Callable[[float], bool], str]] = [
    # 51 cells: "2.9", "0.021", "23.5", "32" ... (percent, fraction, dropped minus sign or ACPR)
    ("evm", lambda v: v >= 0, "EVM in dB must be negative; {v:g} is probably a percentage, a fraction, "
                              "a dropped minus sign or an ACPR value"),
    # "2065" (CMOS) and "100" (GaN); the multi-stage GaN modules at 65-73 dB stay
    ("gain", lambda v: v > 80, "a gain of {v:g} dB is not plausible for a PA (typo?)"),
    *((k, lambda v: not 0 <= v <= 100, "an efficiency of {v:g}% is outside 0-100%")
      for k in ("pae", "pae_1db", "avg_pae", "dc_rf_eff")),
]


def _null_implausible(ctx: RowContext) -> None:
    for key, test, why in IMPLAUSIBLE:
        vals = ctx.values.get(key, [])
        noted = set()  # a single value broadcast to several bands is noted once
        for i, p in enumerate(vals):
            if p is not None and p.value is not None and test(p.value):
                if p.value not in noted:
                    noted.add(p.value)
                    ctx.notes.append((METRICS[key], f"{why.format(v=p.value)}; not plotted"))
                vals[i] = None
    # DC-to-RF efficiency 100x off from Pout / Pdc ("80" for 0.8%)
    eff, pout, pdc = (ctx.values.get(k, []) for k in ("dc_rf_eff", "psat", "pdc"))
    for i, (e, po, pd) in enumerate(zip(eff, pout, pdc)):
        if e and po and pd and None not in (e.value, po.value, pd.value) and pd.value > 0:
            calc = 100 * 10 ** (po.value / 10) / pd.value
            if not calc / 10 <= e.value <= calc * 10:
                ctx.notes.append((METRICS["dc_rf_eff"], f"{e.value:g}% but Pout / Pdc gives {calc:.2g}%; not plotted"))
                eff[i] = None


def adjust(ctx: RowContext) -> None:
    f = ctx.fields
    for key, header in (*PAPER_FIELDS.items(), *ENTRY_FIELDS.items()):
        raw = ctx.raw.get(header)
        if isinstance(raw, str) and (fixed := fix_mojibake(raw)) != raw:
            f[key] = clean_text(fixed)[0] or None
            ctx.notes.append((header, "repaired text that was UTF-8 decoded as Windows-1252 / Mac Roman"))
    if isinstance(f.get("year"), str) and f["year"].strip().isdigit():
        ctx.notes.append(("Year", f"year written as text {f['year']!r}; read as a number"))
        f["year"] = int(f["year"])

    _null_implausible(ctx)

    sheet = f.get("sheet")
    f["circuit"] = CIRCUIT_BY_SHEET.get(sheet, "PA")
    f["technology"] = technology(f.get("process"), sheet)
    f["eff_type"] = efficiency_type(f.get("eff_note"))
    f["pa_type"] = pa_type(f.get("pa_note")) if f["circuit"] == "PA" else "Not stated"

    # Efficiency that is not PAE (the survey plots it as PAE; Readme 6.4).
    eff = f["eff_type"]
    if eff and eff != "PAE":
        _flag_metrics(ctx, "not_pae", EFF_METRICS)
    for key, header in (("pae", "PAEmax (%)"), ("pae_1db", "PAE_1dB (%)"), ("avg_pae", "Average PAE (%)")):
        cell = ctx.raw.get(header)
        if isinstance(cell, str) and _NOT_PAE_CELL.search(cell):
            _flag_metrics(ctx, "not_pae", (key,))
    mod_note = f.get("mod_note")
    if isinstance(mod_note, str) and re.search(r"^\s*DE\b|PAE\s*\(DE\)", mod_note):
        _flag_metrics(ctx, "not_pae", ("avg_pae",))
    if isinstance(f.get("eff_note"), str) and re.search(r"\bsim\b", f["eff_note"], re.IGNORECASE):
        _flag_metrics(ctx, "simulated", ("pae",))  # "Sim PAE"

    notes = [str(f[k]) for k in NOTE_FIELDS if f.get(k) is not None]
    for text in notes:
        if _SIM.search(text):
            ctx.flags.add("simulated")
        if _PULSED.search(text):
            ctx.flags.add("pulsed")
        if _SYSTEM.search(text):
            ctx.flags.add("system")
        if _OFFCHIP.search(text):
            ctx.flags.add("offchip")
        if _ESTIMATED.search(text):
            # "The PAE @P1dB is estimated from reported figures", "worst case (estimated) ... EVM"
            if re.search(r"p1db", text, re.IGNORECASE):
                _flag_metrics(ctx, "estimated", ("pae_1db",))
            elif re.search(r"\bevm\b", text, re.IGNORECASE):
                _flag_metrics(ctx, "estimated", ("evm",))
            else:
                ctx.flags.add("estimated")
    # "21.4(TX)": the value was measured on the transmitter, not the PA alone
    if any(isinstance(ctx.raw.get(h), str) and "(tx)" in ctx.raw[h].lower() for h in METRICS.values()):
        ctx.flags.add("system")

    for key in [k for k, v in f.items() if v is None and k not in PAPER_FIELDS]:
        del f[key]


PA = SurveySource(
    id="pa",
    source="sources/pa/PA-Survey-v11.xlsx",
    url="https://ethz.ch/content/dam/ethz/special-interest/itet/ideas-dam/documents/PA-Survey-v11.xlsx",
    sha256="f52000371ae61146d5e012e8efa5f3871cd8446a275345f1f4326a22520a2579",
    # "THz Oscillator Arrays" is left out: it reports total radiated power / EIRP of
    # whole arrays, which the survey itself does not compare with single PAs.
    sheet=(*PA_SHEETS, "Products", " Multipliers", "THz Oscillators"),
    header_aliases={
        "GaN": {"PA Efficieny Note": "PAE Note"},
        "InP": {"PA Efficieny Note": "PAE Note"},
        "GaAs": {"PA Efficiency Note": "PAE Note"},
        "LDMOS": {"PA Efficiency Note": "PAE Note"},
        "Products": {
            "Name": "Last Name (1st Author)",  # vendor
            "Product": "Paper Title",
            "Process node": "Process (CMOS_Bulk, CMOS_SOI, SiGe)",
        },
        " Multipliers": {
            "Pout Max (dBm)": "Psat (dBm)",  # the survey plots it with Psat
            "Gain (dB)": "Conversion gain (dB)",
            "Multiplier Type (Doubler, Tripler…)": "Multiplication factor",
            "Bandwidth (GHz)": "Output bandwidth (GHz)",
            "Note": "General 1",
        },
        "THz Oscillators": {
            "Pout (dBm)": "Psat (dBm)",  # the survey plots it with Psat
            "RF-to-DC efficiency (%)": "DC-to-RF efficiency (%)",
            "Oscillator Type (fundamental, 2nd harmonic...)": "Oscillator type",
            "Phase Noise(dBc/Hz)": "Phase noise (dBc/Hz)",
            "BW (%)": "Tuning range (%)",
        },
    },
    paper_fields=PAPER_FIELDS,
    entry_fields=ENTRY_FIELDS,
    metrics=METRICS,
    fallbacks={},
    band_key="freq",
    # several values at one frequency ("28 / 24.5") are operating modes: one point each
    band_metrics=("freq", "psat", "pae", "p1db", "pae_1db", "gain", "avg_pout", "avg_pae", "evm"),
    info_metrics=("core_area", "chip_area", "pdc", "dc_rf_eff", "conv_gain"),
    id_fields=("year", "first_author", "title"),
    group_field="technology",  # set by adjust
    group_rules=GROUP_RULES,
    group_default="Other",
    adjust=adjust,
)
