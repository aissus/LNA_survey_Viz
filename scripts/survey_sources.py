"""Build-time description of each survey source spreadsheet.

Adding a survey = a ``SurveySource`` (how to read the sheet), usually in its own
``scripts/source_<id>.py``, one line in ``scripts/registry.py``, and one
``site/surveys/<id>.js`` (how to display it). The LNA source lives here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from caveats import CaveatRules
from parsing import Parsed


@dataclass
class RowContext:
    """One parsed spreadsheet row, handed to a survey's ``adjust`` hook.

    The hook may change everything here in place, for survey-specific rules
    that the generic parsing cannot know (a unit that differs on some rows, a
    zero that means "included elsewhere", a caveat written in a notes column).
    """

    row: int  # Excel row number
    raw: dict[str, Any]  # header -> cell value, untouched
    # cleaned paper + entry fields; keys the hook adds become entry fields too
    # (e.g. a derived category such as "type_family")
    fields: dict[str, Any]
    # metric key -> one value per band (None = missing); band_metrics + info_metrics
    values: dict[str, list[Parsed | None]]
    n_bands: int
    flags: set[str] = field(default_factory=set)  # caveats for the whole entry (every band)
    # caveat -> metric keys it applies to, e.g. {"simulated": {"iip3"}}
    metric_flags: dict[str, set[str]] = field(default_factory=dict)
    # (column header, message) recorded as "note" warnings in the import notes
    notes: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class SurveySource:
    id: str
    source: str  # path relative to the repo root
    sheet: str
    # output key -> spreadsheet header. A paper may span several rows (older survey
    # versions list one row per band); rows with the same id_fields are one paper.
    paper_fields: dict[str, str]  # bibliographic, taken from the paper's first row
    entry_fields: dict[str, str]  # per row (e.g. process, comment)
    # output key -> spreadsheet header, for numeric columns that may be multi-band
    metrics: dict[str, str]
    # output key -> (primary metric key, fallback metric key); each entry records
    # which one it used in "sources"
    fallbacks: dict[str, tuple[str, str]]
    # metric (after fallbacks) that defines the bands; must have a value to plot
    band_key: str
    # metrics plotted per band; a column that cannot be aligned is emptied and flagged
    band_metrics: tuple[str, ...]
    # paper fields that identify a paper; also hashed into its stable id
    id_fields: tuple[str, ...]
    # informational metrics: aligned to the bands when the counts allow it
    info_metrics: tuple[str, ...]
    # (group, regex) rules applied to the entry field group_field; first match wins
    group_field: str
    group_rules: list[tuple[str, str]] = field(default_factory=list)
    group_default: str = "Other"
    # how simulated / system-level / cryogenic entries are written (see caveats.py)
    caveats: CaveatRules | None = None
    # When set, the spreadsheet is not committed: the build downloads it from
    # url into `source` if it is missing, and always checks its sha256.
    url: str | None = None
    sha256: str | None = None
    # Survey-specific rules, run on every row after parsing (see RowContext).
    adjust: Callable[[RowContext], None] | None = None


LNA = SurveySource(
    id="lna",
    source="sources/lna/LNA_Survey_v3.0.xlsx",
    # Downloaded from ETH at build time, not committed. ETH updates this file in
    # place (same name): when the pin no longer matches, check the new rows and
    # update sha256 and the tests' counts.
    url="https://ethz.ch/content/dam/ethz/special-interest/itet/ideas-dam/documents/LNA_Survey_v3.0.xlsx",
    sha256="84cf7fbd887e9cec24291af82bbdcb2a61f395bcc26c4455c09809a564a7f9bd",
    sheet="All",
    paper_fields={
        "venue": "Publication",
        "year": "Year",
        "month": "Month",
        "first_author": "Last Name (1st Author)",
        "title": "Paper Title",
    },
    entry_fields={
        "process": "Process",
        "node": "Node (nm)",
        "p1db_ref": "I/O P1dB",
        "comment": "Comment",
    },
    metrics={
        "fc": "fc",
        "f_nfmin": "f@NFmin (GHz)",
        "nf_spec": "NF",
        "nfmin": "NFmin",
        "gain": "Gain (dB)",
        "p1db": "P1dB (dBm)",
        "iip3": "IIP3 (dBm)",
        "bw3db": "3dB BW (GHz)",
        "power": "Power(mW)",
    },
    # Plot frequency: fc when reported, else the frequency of minimum NF.
    # Plot NF: NF at fc when reported, else NFmin. Each entry records the source.
    fallbacks={
        "freq": ("fc", "f_nfmin"),
        "nf": ("nf_spec", "nfmin"),
    },
    band_key="freq",
    band_metrics=("freq", "nf", "gain", "p1db", "iip3", "bw3db", "power"),
    info_metrics=("f_nfmin", "nfmin"),
    id_fields=("year", "first_author", "title"),
    group_field="process",
    group_rules=[
        ("FinFET", r"finfet"),
        ("SiGe", r"sige|bicmos"),  # before CMOS: "SiGe BiCMOS" contains "CMOS"
        ("CMOS SOI", r"soi"),  # includes "RFSOI"
        ("CMOS Bulk", r"cmos"),
        ("III-V", r"gaas|inp|gan"),
    ],
    caveats=CaveatRules(
        text_field="comment",
        title_field="title",
        # "IIP3 simulated", "Simulated NF", "Gain and IIP3 simulated", "NF* simulated", ...
        metric_tokens=(
            (r"\bNF\b", ("nf", "nfmin")),
            (r"\bgain\b", ("gain",)),
            (r"\bIIP3\b", ("iip3",)),
            (r"\b[IO]?P1dB\b", ("p1db",)),
        ),
        # "Receiver Frontend", "System", "Rx consumption", "2x2 Array", "Beamformer", ...
        system_pattern=r"receiver|front-?end|\bsystem\b|\brx\b|\btrx\b|\barray\b|beamformer",
        # five cryogenic papers have no "T:" note, only the title says so
        cryo_title_pattern=r"cryo|quantum|qubit",
    ),
)
