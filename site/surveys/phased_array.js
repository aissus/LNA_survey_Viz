// @ts-check
/**
 * Phased-array survey (ETH IDEAS, v3.0). Everything specific to this survey's
 * display lives here; how the spreadsheet is read lives in
 * scripts/source_phased_array.py.
 *
 * Several survey columns are spreadsheet formulas. Where the formula is clear,
 * it is recomputed here from its raw inputs (aperture area and area per
 * element, package and chip thermal density), together with the efficiencies
 * the survey's Readme 6.9 defines and λ/2 = 150/f mm. Total Pout/Prad, TX/RX
 * Pdc (W), chip size and chip core area per element are taken from the sheet,
 * because some of their cells are typed values rather than formulas.
 */

import { NEUTRAL, PROCESS_COLORS } from './palette.js';

const FREQ_TICKS = [2, 5, 10, 20, 50, 100, 200, 500, 1000];
const COUNT_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

/** @param {unknown} v @returns {v is number} */
const num = (v) => typeof v === 'number' && Number.isFinite(v);
/** @param {...unknown} vs */
const positive = (...vs) => vs.every((v) => num(v) && v > 0);

/** dBm -> W */
const dbmToW = (/** @type {number} */ dbm) => 10 ** (dbm / 10) / 1000;

/**
 * λ/2 in air at f GHz, in mm: c / (2f) = 150 / f.
 * @param {unknown} f
 * @returns {number | null}
 */
export function halfLambdaMm(f) {
  return positive(f) ? 150 / /** @type {number} */ (f) : null;
}

/**
 * Larger of the TX and RX DC power in W, as the survey's thermal densities use
 * it (Readme 6.5: for a TRX, the mode that draws more).
 * @param {Record<string, any>} p
 * @returns {number | null}
 */
export function maxPdcW({ tx_pdc, rx_pdc }) {
  const vs = [tx_pdc, rx_pdc].filter(num);
  return vs.length ? Math.max(...vs) : null;
}

/** Array aperture area in cm² (sheet: X · Y). @param {Record<string, any>} p */
export function apertureArea({ ap_x, ap_y }) {
  return positive(ap_x, ap_y) ? ap_x * ap_y : null;
}

/** Aperture area per antenna in mm² (sheet: area · 100 / (tiles · antennas per tile)). @param {Record<string, any>} p */
export function apertureAreaPerElement(p) {
  const a = apertureArea(p);
  return a !== null && positive(p.tiles, p.ant_per_tile) ? (a * 100) / (p.tiles * p.ant_per_tile) : null;
}

/** Package thermal density in W/cm²: max(TX, RX Pdc) / aperture area. @param {Record<string, any>} p */
export function packageThermalDensity(p) {
  const pdc = maxPdcW(p);
  const a = apertureArea(p);
  return pdc !== null && a !== null ? pdc / a : null;
}

/**
 * Chip thermal density in W/cm²: max(TX, RX Pdc) per IC over the chip size
 * (sheet: Pdc / (tiles · ICs per tile) / (chip size in mm² · 0.01)).
 * @param {Record<string, any>} p
 */
export function chipThermalDensity(p) {
  const pdc = maxPdcW(p);
  return pdc !== null && positive(p.tiles, p.ic_per_tile, p.chip_area)
    ? pdc / (p.tiles * p.ic_per_tile) / (p.chip_area / 100)
    : null;
}

/** EIRP / TX Pdc in % (Readme 6.9). Above 100 % is normal: EIRP includes the array gain. @param {Record<string, any>} p */
export function eirpEfficiency({ eirp, tx_pdc }) {
  return num(eirp) && positive(tx_pdc) ? (dbmToW(eirp) / tx_pdc) * 100 : null;
}

/** Total Pout (or Prad) / TX Pdc in % (Readme 6.9). @param {Record<string, any>} p */
export function pradEfficiency({ pout, tx_pdc }) {
  return num(pout) && positive(tx_pdc) ? (dbmToW(pout) / tx_pdc) * 100 : null;
}

/** Chip core area per element over the λ/2 × λ/2 grid cell; below 1 fits the grid (Readme 6.1). @param {Record<string, any>} p */
export function areaToGrid({ core_area_el, freq }) {
  const hl = halfLambdaMm(freq);
  return positive(core_area_el) && hl !== null ? core_area_el / (hl * hl) : null;
}

/** Publication date as a decimal year, mid-month. @param {Record<string, any>} p */
export function pubDate({ year, month }) {
  if (!num(year)) return null;
  return num(month) && month >= 1 && month <= 12 ? year + (month - 0.5) / 12 : year + 0.5;
}

/**
 * EIRP of N identical elements, each radiating P·G (dBm): P·G + 20·log10(N)
 * (N² = N times the power, times N in array gain).
 * @param {number} perElementDbm
 * @returns {import('../js/types.js').RefLine['y']}
 */
const eirpScaling = (perElementDbm) => (n) => (positive(n) ? perElementDbm + 20 * Math.log10(n) : null);

/** @param {string} title */
const scholar = (title) => `https://scholar.google.com/scholar?q=${encodeURIComponent(title)}`;

/** @type {import('../js/types.js').SurveyConfig} */
const phasedArray = {
  id: 'phased_array',
  title: 'Phased-Array Survey Visualizer',
  subtitle:
    'Published silicon phased arrays since 2010: EIRP, element count, chip area per element, noise figure and thermal density across frequency.',
  source: {
    label: 'ETH Zürich IDEAS phased-array survey',
    url: 'https://ideas.ethz.ch/Surveys/phased-array-survey.html',
    citation:
      'Hua Wang, Yuqi Liu, Naga Sasikanth Mannem, "Phased Array Survey 2010-Present," [Online]. Available: https://ideas.ethz.ch/Surveys/phased-array-survey.html',
  },

  // Order matters for derived metrics: a metric's dependsOn inputs are listed before it.
  metrics: {
    freq: { label: 'Frequency', short: 'f', unit: 'GHz', scale: 'log', digits: 1, ticks: FREQ_TICKS },
    n_elements: {
      label: 'Elements (TX + RX)',
      short: 'N',
      scale: 'log',
      digits: 0,
      ticks: COUNT_TICKS,
      description:
        'TX + RX elements as the survey counts them: signal chains, so a transceiver whose TX and RX share an antenna counts it twice. A passive relay has no chains; its unit cells (tiles · antennas per tile) are counted instead.',
    },
    n_tx: { label: 'TX elements', short: 'N TX', scale: 'log', digits: 0, ticks: COUNT_TICKS },
    n_rx: { label: 'RX elements', short: 'N RX', scale: 'log', digits: 0, ticks: COUNT_TICKS },
    eirp: { label: 'EIRP', unit: 'dBm', digits: 1, description: 'TX EIRP of the array (with the elements counted in "Elements at EIRP").' },
    n_eirp: {
      label: 'Elements at EIRP',
      short: 'N EIRP',
      scale: 'log',
      digits: 0,
      ticks: COUNT_TICKS,
      description: 'Active antenna elements of the EIRP measurement (Readme 6.9).',
    },
    pout: {
      label: 'Total Pout / Prad',
      short: 'Pout',
      unit: 'dBm',
      digits: 1,
      description: 'Total TX output (or radiated) power: per-element Psat/Prad + 10·log10(TX elements), or as reported.',
    },
    tx_pdc: { label: 'TX DC power', short: 'Pdc TX', unit: 'W', scale: 'log', digits: 3, description: 'Whole array, TX mode.' },
    rx_pdc: { label: 'RX DC power', short: 'Pdc RX', unit: 'W', scale: 'log', digits: 3, description: 'Whole array, RX mode.' },
    tiles: { label: 'Array tiles', scale: 'log', digits: 0, description: '1 for a single chip not built into a tileable module (Readme 6.6).' },
    ant_per_tile: { label: 'Antennas per tile', scale: 'log', digits: 0, ticks: COUNT_TICKS },
    ic_per_tile: { label: 'ICs per tile', scale: 'log', digits: 0, ticks: COUNT_TICKS },
    tx_per_ic: { label: 'TX elements per IC', scale: 'log', digits: 0, ticks: COUNT_TICKS },
    rx_per_ic: { label: 'RX elements per IC', scale: 'log', digits: 0, ticks: COUNT_TICKS },
    ap_x: { label: 'Aperture X', unit: 'cm', digits: 2 },
    ap_y: { label: 'Aperture Y', unit: 'cm', digits: 2 },
    chip_area: { label: 'Chip size', unit: 'mm²', scale: 'log', digits: 2, description: 'Die area of one IC (X · Y).' },
    core_area_el: {
      label: 'Chip core area / element',
      short: 'A core/el',
      unit: 'mm²',
      scale: 'log',
      digits: 3,
      description:
        'Core area of the RF and LO paths (no pads, supply or bias), per element, as the survey computes it: 2 · core area / (TX + RX elements per IC). Compare with the λ/2 × λ/2 grid cell (Readme 6.1).',
    },
    psat: { label: 'TX Psat / Prad per element', short: 'Psat', unit: 'dBm', digits: 1 },
    tx_eff: { label: 'TX efficiency per element', short: 'η TX', unit: '%', digits: 1 },
    tx_gain: { label: 'TX conversion gain', short: 'G TX', unit: 'dB', digits: 1 },
    ant_gain: {
      label: 'Antenna gain (passive arrays)',
      short: 'G ant',
      unit: 'dBi',
      digits: 1,
      description: 'Passive reflectarrays and RIS: the survey lists their antenna gain under TX conversion gain (Readme 6.8).',
    },
    ap_eff: {
      label: 'Aperture efficiency (passive arrays)',
      short: 'η ap',
      unit: '%',
      digits: 1,
      description: 'Passive reflectarrays: the survey lists their aperture efficiency under TX efficiency.',
    },
    nf: { label: 'RX noise figure', short: 'NF', unit: 'dB', digits: 1 },
    rx_gain: { label: 'RX conversion gain', short: 'G RX', unit: 'dB', digits: 1 },
    tx_pdc_el: { label: 'TX DC power per element', short: 'Pdc TX/el', unit: 'mW', scale: 'log', digits: 1 },
    rx_pdc_el: { label: 'RX DC power per element', short: 'Pdc RX/el', unit: 'mW', scale: 'log', digits: 1 },
    // derived (see `derived` below)
    ap_area: {
      label: 'Aperture area',
      unit: 'cm²',
      scale: 'log',
      digits: 2,
      description: 'Array aperture X · Y (recomputed from the sheet inputs).',
      dependsOn: ['ap_x', 'ap_y'],
    },
    ap_area_el: {
      label: 'Aperture area / antenna',
      unit: 'mm²',
      scale: 'log',
      digits: 2,
      description: 'Aperture area / (tiles · antennas per tile) (recomputed from the sheet inputs).',
      dependsOn: ['ap_area', 'tiles', 'ant_per_tile'],
    },
    pkg_td: {
      label: 'Package thermal density',
      unit: 'W/cm²',
      scale: 'log',
      digits: 2,
      description: 'max(TX, RX DC power) / aperture area: the aperture stands in for the package (Readme 6.5).',
      dependsOn: ['tx_pdc', 'rx_pdc', 'ap_area'],
    },
    chip_td: {
      label: 'Chip thermal density',
      unit: 'W/cm²',
      scale: 'log',
      digits: 2,
      description: 'max(TX, RX DC power) per IC / chip size (Readme 6.5).',
      dependsOn: ['tx_pdc', 'rx_pdc', 'tiles', 'ic_per_tile', 'chip_area'],
    },
    eirp_eff: {
      label: 'EIRP / Pdc',
      unit: '%',
      scale: 'log',
      digits: 2,
      description: 'EIRP (in W) / TX DC power (Readme 6.9). It includes the array gain, so it exceeds 100 % for large arrays.',
      dependsOn: ['eirp', 'tx_pdc'],
    },
    prad_eff: {
      label: 'Pout / Pdc',
      unit: '%',
      scale: 'log',
      digits: 2,
      description: 'Total Pout/Prad (in W) / TX DC power (Readme 6.9).',
      dependsOn: ['pout', 'tx_pdc'],
    },
    area_ratio: {
      label: 'Core area / (λ/2)²',
      short: 'A/(λ/2)²',
      scale: 'log',
      digits: 2,
      description: 'Chip core area per element over the λ/2 × λ/2 grid cell at the carrier frequency (λ/2 = 150/f mm). Below 1 fits the grid.',
      dependsOn: ['core_area_el', 'freq'],
    },
    date: { label: 'Publication date', digits: 2, description: 'Year + (month − 0.5) / 12: mid-month, as a decimal year.' },
    year: { label: 'Year', digits: 0 },
  },
  paperFields: ['year', 'month'],
  // Evaluated in this order on every point.
  derived: {
    ap_area: apertureArea,
    ap_area_el: apertureAreaPerElement,
    pkg_td: packageThermalDensity,
    chip_td: chipThermalDensity,
    eirp_eff: eirpEfficiency,
    prad_eff: pradEfficiency,
    area_ratio: areaToGrid,
    date: pubDate,
  },

  groups: {
    label: 'Process',
    order: ['CMOS Bulk', 'SiGe', 'CMOS SOI', 'FinFET', 'Not reported'],
    colors: PROCESS_COLORS,
    // Marker shape comes from the type category; these apply only where it does not.
    symbols: {
      'CMOS Bulk': 'circle',
      SiGe: 'diamond',
      'CMOS SOI': 'square',
      FinFET: 'triangle-up',
      'Not reported': 'x',
    },
    fallbackColor: NEUTRAL,
  },

  categories: {
    type: {
      label: 'Type',
      field: 'type',
      order: ['TX', 'RX', 'TRX', 'Oscillator', 'Relay'],
      symbols: {
        TX: 'triangle-up',
        RX: 'triangle-down',
        TRX: 'diamond',
        Oscillator: 'pentagon',
        Relay: 'square',
      },
      descriptions: {
        TX: 'Transmit array with baseband/IF (Readme 6.2).',
        RX: 'Receive array with baseband/IF.',
        TRX: 'Transceiver array.',
        Oscillator: 'Oscillator array: generates power like a TX, without an input signal path.',
        Relay: 'Reflectarray, reconfigurable intelligent surface or relay array (passive or active).',
      },
    },
    conversion: {
      label: 'Frequency conversion',
      field: 'conversion',
      order: ['Integrated', 'Front-end only', 'Not applicable'],
      descriptions: {
        Integrated: 'Up/down-conversion is part of the design.',
        'Front-end only': 'RF front-end only ("FE" in the survey): a separate chip does the up/down-conversion (Readme 6.2).',
        'Not applicable': 'Oscillator arrays and relays.',
      },
    },
    antenna: {
      label: 'Antenna on',
      field: 'antenna',
      order: ['Chip', 'AiP', 'PCB', 'None', 'Not reported'],
      descriptions: {
        Chip: 'On-chip antennas (some with a lens).',
        AiP: 'Antenna in package on a packaging substrate such as LTCC, glass or organic (Readme 6.3).',
        PCB: 'Antennas on a PCB.',
        None: 'No antennas: measured through probes or connectors.',
        'Not reported': 'Not given in the survey (passive reflectarrays).',
      },
    },
  },

  filters: ['freq', 'year', 'n_elements', 'eirp', 'nf'],
  panels: [
    { x: 'freq', y: 'eirp', title: 'EIRP vs frequency' },
    {
      x: 'freq',
      y: 'core_area_el',
      title: 'Chip core area per element vs frequency',
      refs: [{ label: '(λ/2)² grid cell', y: (f) => (positive(f) ? (150 / f) ** 2 : null), dash: 'dash' }],
    },
    { x: 'freq', y: 'nf', title: 'RX noise figure vs frequency' },
    { x: 'freq', y: 'psat', title: 'TX Psat per element vs frequency' },
    { x: 'freq', y: 'chip_td', title: 'Chip thermal density vs frequency' },
    {
      x: 'n_eirp',
      y: 'eirp',
      title: 'EIRP vs elements',
      refs: [
        { label: '20·log₁₀N + 0 dBm', y: eirpScaling(0), dash: 'dot' },
        { label: '20·log₁₀N + 10 dBm', y: eirpScaling(10), dash: 'dash' },
        { label: '20·log₁₀N + 20 dBm', y: eirpScaling(20), dash: 'longdash' },
      ],
    },
    { x: 'date', y: 'n_elements', title: 'Elements vs publication date' },
  ],
  explorer: {
    metrics: [
      'freq', 'n_elements', 'n_tx', 'n_rx', 'eirp', 'n_eirp', 'pout', 'eirp_eff', 'prad_eff',
      'tx_pdc', 'rx_pdc', 'tx_pdc_el', 'rx_pdc_el', 'psat', 'tx_eff', 'tx_gain', 'nf', 'rx_gain',
      'ant_gain', 'ap_eff', 'chip_area', 'core_area_el', 'area_ratio', 'ap_area', 'ap_area_el',
      'pkg_td', 'chip_td', 'tiles', 'ic_per_tile', 'date', 'year',
    ],
    // the survey's own "DC-to-RF efficiency vs # of elements" plot (Readme 6.9)
    x: 'n_eirp',
    y: 'eirp_eff',
    color: 'group',
  },
  hover: ['freq', 'n_elements', 'eirp', 'psat', 'nf'],
  table: ['freq', 'n_elements', 'eirp', 'psat', 'nf', 'tx_pdc', 'core_area_el', 'chip_td'],
  info: [
    'freq', 'n_elements', 'n_tx', 'n_rx', 'eirp', 'n_eirp', 'pout', 'eirp_eff', 'prad_eff',
    'tx_pdc', 'rx_pdc', 'psat', 'tx_eff', 'tx_gain', 'tx_pdc_el', 'nf', 'rx_gain', 'rx_pdc_el',
    'ant_gain', 'ap_eff', 'tiles', 'ant_per_tile', 'ic_per_tile', 'tx_per_ic', 'rx_per_ic',
    'chip_area', 'core_area_el', 'area_ratio', 'ap_x', 'ap_y', 'ap_area', 'ap_area_el', 'pkg_td', 'chip_td',
  ],

  paper: {
    citation: (p) => [p.first_author, [p.venue, p.year].filter(Boolean).join(' ')].filter(Boolean).join(', '),
    link: (p) => (p.title ? scholar(p.title) : null),
    entryDetails: (e) => {
      /** @type {Array<[string, string]>} */
      const rows = [];
      if (e.type_label) {
        const fe = e.conversion === 'Front-end only' ? ' (front-end only: conversion on a separate chip)' : '';
        rows.push(['Type', `${e.type_label}${fe}`]);
      }
      if (e.process || e.node != null) {
        const process = e.process ? String(e.process).replaceAll('_', ' ') : e.group;
        rows.push(['Process', e.node != null ? `${process}, ${e.node} nm` : process]);
      }
      if (e.antenna) {
        const sub = e.antenna_type && e.antenna_type !== e.antenna ? ` (${e.antenna_type})` : '';
        rows.push(['Antenna on', `${e.antenna}${sub}`]);
      }
      if (e.dimension != null) rows.push(['Array', `${e.dimension}D`]);
      const pol = [e.tx_pol != null ? `TX ${e.tx_pol}` : null, e.rx_pol != null ? `RX ${e.rx_pol}` : null].filter(Boolean);
      if (pol.length) rows.push(['Polarizations', pol.join(', ')]);
      if (e.co_aperture) rows.push(['TX/RX co-aperture', e.co_aperture]);
      if (e.note) rows.push(['Note', e.note]);
      return rows;
    },
  },

  // Set by scripts/source_phased_array.py from the Note column (and one title).
  caveats: {
    passive: {
      label: 'Passive array',
      short: 'passive',
      description:
        'Passive reflectarray, RIS or metasurface: no amplification in the signal path. Its gain is an antenna gain in dBi and its efficiency an aperture efficiency, shown as their own metrics.',
    },
    commercial: {
      label: 'Commercial ICs',
      short: 'COTS',
      description:
        'Array module built from commercial beamformer ICs (some with external LNAs): array-level results, not a new IC. Process and chip area are often not reported.',
    },
    non_silicon: {
      label: 'Non-silicon part',
      short: 'non-Si',
      description: 'Includes a non-silicon component (a GaAs LNA), so receiver performance and power are not those of the silicon IC alone.',
    },
    cryogenic: {
      label: 'Cryogenic',
      short: 'cryo',
      description: 'Operated at cryogenic temperature (4 K).',
    },
    pdc_scope: {
      label: 'DC power beyond the array',
      short: 'Pdc+',
      description:
        'The DC power covers more than the array front-end (the full radio including its SoC, or the whole package including up-converter and frequency multiplier). DC power, and the efficiencies and thermal densities computed from it, are not comparable.',
    },
    pdc_shared: {
      label: 'TX Pdc includes RX',
      short: 'Pdc TX+RX',
      description:
        'TX and RX DC power are reported together under TX Pdc (survey Readme 6.4; its RX Pdc of 0 is left out). TX Pdc, and the efficiencies and thermal densities computed from it, include the receiver.',
    },
    external: {
      label: 'Array results from another paper',
      short: 'ext. array',
      description: 'Array-level results (EIRP, aperture) come from a companion paper named in the Note, not from this one.',
    },
  },

  manual: {
    fields: [
      { metric: 'freq', required: true, placeholder: '28' },
      { metric: 'n_elements', placeholder: '64' },
      { metric: 'eirp', placeholder: '50' },
      { metric: 'n_eirp', placeholder: '64' },
      { metric: 'tx_pdc', placeholder: '5' },
      { metric: 'psat', placeholder: '15' },
      { metric: 'nf', placeholder: '5' },
      { metric: 'core_area_el', placeholder: '2' },
    ],
  },
};

export default phasedArray;
