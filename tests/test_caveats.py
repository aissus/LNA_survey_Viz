"""Caveat parsing, on the real comment strings of the LNA survey."""

import pytest

from caveats import parse_caveats, parse_temperatures
from survey_sources import LNA

RULES = LNA.caveats


def cav(comment, title="An LNA", n=1):
    return parse_caveats(comment, title, n, RULES)


@pytest.mark.parametrize("comment", [
    "Simulated",
    "Simulated, Consumption at max data rate",
])
def test_whole_entry_simulated(comment):
    c = cav(comment)
    assert c.flags == {"simulated"} and not c.sim_metrics


@pytest.mark.parametrize("comment, metrics", [
    ("IIP3 simulated", {"iip3"}),
    ("IP1dB simulated", {"p1db"}),
    ("Simulated NF", {"nf", "nfmin"}),
    ("NF Simulation", {"nf", "nfmin"}),
    ("NF and IP1dB simulated", {"nf", "nfmin", "p1db"}),
    ("IP1dB simulated, NF* simulated", {"nf", "nfmin", "p1db"}),
])
def test_individual_values_simulated(comment, metrics):
    c = cav(comment)
    assert c.sim_metrics == metrics
    assert "simulated" not in c.flags


@pytest.mark.parametrize("comment, flags, metrics", [
    ("Receiver Frontend", {"system"}, set()),
    ("System, Power per channel", {"system"}, set()),
    ("Rx consumption", {"system"}, set()),
    ("16 Elements Array, Antenna to Digital", {"system"}, set()),
    ("Beamformer", {"system"}, set()),
    ("Simulated, Receiver frontend", {"simulated", "system"}, set()),
    ("Simulation, Receiver Frontend", {"simulated", "system"}, set()),
    ("Receiver Frontend, Gain and IIP3 simulated", {"system"}, {"gain", "iip3"}),
    ("Receiver frontend, NF simulated, Consumption per channel", {"system"}, {"nf", "nfmin"}),
])
def test_system_level_and_mixed(comment, flags, metrics):
    c = cav(comment)
    assert c.flags == flags and c.sim_metrics == metrics


@pytest.mark.parametrize("comment", [
    "Bidirectional",
    "IIP3 @ 39GHz",
    "Linearity @ 10.7GHz",
    "NF available only at f>=140GHz",
    None,
])
def test_plain_notes_are_not_caveats(comment):
    c = cav(comment)
    assert not c.flags and not c.sim_metrics and not c.temps_k


def test_temperatures_parse_decimal_commas_and_greek_kappa():
    assert parse_temperatures("300K/4,2K(4Κ)") == [300.0, 4.2]
    assert parse_temperatures("3,6K") == [3.6]


def test_one_temperature_per_band():
    c = cav("T:300K/4K/300K/4K", n=4)
    assert c.temps_k == [300, 4, 300, 4]
    assert [c.band_flags(i) for i in range(4)] == [[], ["cryogenic"], [], ["cryogenic"]]


def test_single_temperature_applies_to_every_band_and_rest_of_note_still_parsed():
    c = cav("T:3,6K, *OIP3=4dBm", n=2)
    assert c.temps_k == [3.6, 3.6]
    assert c.band_flags(1) == ["cryogenic"]
    c = cav("T:300K/12K, Linearity @ 0,3GHz", n=2)
    assert c.temps_k == [300, 12] and not c.flags


def test_temperatures_repeat_when_bands_are_a_multiple():
    # Puyal 2024: 8 NF values alternating 300 K / 4.2 K
    c = cav("T:300K/4,2K(4Κ)", n=8)
    assert c.temps_k == [300, 4.2] * 4
    assert c.notes


def test_unalignable_temperatures_mark_the_whole_entry():
    c = cav("T:300K/4K/77K", n=2)
    assert c.flags == {"cryogenic"} and not c.temps_k and c.notes


@pytest.mark.parametrize("title", [
    "A Cryogenic Broadband Sub-1-dB NF CMOS Low Noise Amplifier for Quantum Applications",
    "Sub-10-GHz Cryo-CMOS LNAs Achieving Up to 0.07-dB Average NF",
    "A 7.3-9.5-GHz LNA Based on Passive Noise Cancellation for Qubit State Readout",
])
def test_cryogenic_from_title_when_no_temperature_note(title):
    assert cav(None, title, n=2).band_flags(0) == ["cryogenic"]


def test_title_does_not_override_explicit_temperatures():
    c = cav("T:300K/4K", "A Cryogenic LNA", n=2)
    assert c.band_flags(0) == [] and c.band_flags(1) == ["cryogenic"]


def test_room_temperature_titles_are_not_cryogenic():
    title = "A Tri-Band Dual-Concurrent Wi-Fi 802.11be Transceiver Achieving -46 dB TX/RX EVM Floor at 7.1 GHz"
    assert cav("Receiver Frontend, OIP3 > 31", title).flags == {"system"}
