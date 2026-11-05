"""Caveats read from the free-text fields of a survey row.

Rows carry notes such as "Simulated", "IIP3 simulated", "Receiver Frontend"
or "T:300K/4K". They decide whether a value can be compared with a measured,
standalone, room-temperature design, so the viewer marks and can filter them:

  simulated  the whole entry is simulated (individual simulated values are
             listed in ``sim_metrics`` instead)
  system     measured as part of a receiver / system, so gain, NF and power
             include more than the LNA
  cryogenic  measured at cryogenic temperature (per band when the note gives
             one temperature per band, e.g. "T:300K/4K")

Everything here is pure and unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CRYOGENIC_BELOW_K = 100.0

# "T:300K/4,2K(4Κ)": the survey uses decimal commas and once a Greek capital
# kappa (U+039A) instead of K, so the clause is cut out before splitting on ",".
_TEMP_CLAUSE_RE = re.compile(r"\bT\s*[:=]\s*(?P<temps>[0-9.,/\s()KΚ]+)", re.IGNORECASE)
_TEMP_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[KΚ]", re.IGNORECASE)
_SIM_RE = re.compile(r"simulat", re.IGNORECASE)


@dataclass(frozen=True)
class CaveatRules:
    """How one survey writes its caveats."""

    text_field: str  # entry field holding the notes, e.g. "comment"
    title_field: str | None  # paper field searched for cryo_title_pattern
    # (regex, metric keys) pairs: which metric a "<token> simulated" note refers to
    metric_tokens: tuple[tuple[str, tuple[str, ...]], ...]
    system_pattern: str
    cryo_title_pattern: str | None = None


@dataclass
class Caveats:
    flags: set[str] = field(default_factory=set)  # entry-wide flags
    sim_metrics: set[str] = field(default_factory=set)  # individually simulated metrics
    temps_k: list[float | None] = field(default_factory=list)  # one per band, or empty
    notes: list[str] = field(default_factory=list)  # things a maintainer should check

    def band_flags(self, band: int) -> list[str]:
        """Flags of one band: the entry flags plus "cryogenic" for a cold band."""
        flags = set(self.flags)
        if self.temps_k:
            t = self.temps_k[band]
            if t is not None and t < CRYOGENIC_BELOW_K:
                flags.add("cryogenic")
        return sorted(flags)


def parse_temperatures(text: str) -> list[float]:
    """Temperatures in kelvin from a "T:300K/4,2K(4Κ)" clause, one per '/' part."""
    temps = []
    for part in text.split("/"):
        m = _TEMP_RE.search(part)
        if m:
            temps.append(float(m[1].replace(",", ".")))
    return temps


def parse_caveats(text: str | None, title: str | None, n_bands: int, rules: CaveatRules) -> Caveats:
    cav = Caveats()
    text = text or ""

    m = _TEMP_CLAUSE_RE.search(text)
    if m:
        temps = parse_temperatures(m["temps"])
        text = text[: m.start()] + text[m.end():]
        if len(temps) == n_bands:
            cav.temps_k = list(temps)
        elif len(temps) == 1:
            cav.temps_k = temps * n_bands
        elif temps and n_bands % len(temps) == 0:
            # "T:300K/4,2K" with 8 bands: the values alternate 300 K / 4.2 K
            cav.temps_k = temps * (n_bands // len(temps))
            cav.notes.append(f"{len(temps)} temperatures for {n_bands} bands; assumed to repeat in that order")
        elif temps:
            cav.notes.append(f"{len(temps)} temperatures for {n_bands} bands; whole entry marked cryogenic")
            if min(temps) < CRYOGENIC_BELOW_K:
                cav.flags.add("cryogenic")

    for clause in (c.strip() for c in re.split(r"[,;]", text)):
        if not clause:
            continue
        if _SIM_RE.search(clause):
            metrics = [keys for pattern, keys in rules.metric_tokens if re.search(pattern, clause, re.IGNORECASE)]
            if metrics:
                cav.sim_metrics.update(k for keys in metrics for k in keys)
            else:
                cav.flags.add("simulated")
        if re.search(rules.system_pattern, clause, re.IGNORECASE):
            cav.flags.add("system")

    if (
        not cav.temps_k
        and rules.cryo_title_pattern
        and title
        and re.search(rules.cryo_title_pattern, title, re.IGNORECASE)
    ):
        cav.flags.add("cryogenic")
    return cav
