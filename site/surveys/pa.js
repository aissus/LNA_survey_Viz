// @ts-check
/**
 * PA survey (ETH PA Survey, v11). Everything PA-specific about the viewer lives
 * in this file; how the spreadsheet is read lives in scripts/source_pa.py.
 */

import { NEUTRAL, PROCESS_COLORS } from './palette.js';

const FREQ_TICKS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000];

/** @param {number} dbm @returns {number} watts */
const watts = (dbm) => 10 ** (dbm / 10) / 1000;

/**
 * ITRS PA figure of merit: Psat · G · PAE · f², with Psat in W, gain G linear,
 * PAE as a fraction and f in GHz (W·GHz²). Null when an input is missing or invalid.
 * @param {Record<string, any>} p
 * @returns {number | null}
 */
export function itrsFom({ psat, gain, pae, freq }) {
  if (![psat, gain, pae, freq].every(Number.isFinite) || pae <= 0 || freq <= 0) return null;
  return watts(psat) * 10 ** (gain / 10) * (pae / 100) * freq ** 2;
}

/**
 * Power density as the survey defines it: Psat (W) / PA core area (mm²).
 * @param {Record<string, any>} p
 * @returns {number | null}
 */
export function powerDensity({ psat, core_area }) {
  if (!Number.isFinite(psat) || !Number.isFinite(core_area) || core_area <= 0) return null;
  return watts(psat) / core_area;
}

/** @param {string} title */
const scholar = (title) => `https://scholar.google.com/scholar?q=${encodeURIComponent(title)}`;

/** A product datasheet rather than a paper. @param {any} paper */
const isProduct = (paper) => (paper.entries ?? []).some((/** @type {any} */ e) => e.circuit === 'PA product');

/** @param {unknown} v @param {string} [unit] */
const withUnit = (v, unit) => (unit && typeof v === 'number' ? `${v} ${unit}` : String(v));

/** @type {import('../js/types.js').SurveyConfig} */
const pa = {
  id: 'pa',
  title: 'PA Survey Visualizer',
  subtitle: 'Published power amplifiers, transmitters and THz sources, from below 1 GHz to above 1 THz, in CMOS, SiGe, III-V and LDMOS. Filter by technology and spec, compare with your own design, and open the papers.',
  source: {
    label: 'ETH Zürich IDEAS PA survey (ETH PA Survey)',
    url: 'https://ideas.ethz.ch/Surveys/pa-survey.html',
    citation:
      'Hua Wang, Filippo Svelto, Basem Abdelaziz, Yuqi Liu, Chenhao Chu, Jinglong Xu, Niccolò Villaggi, Abdelrahman Habib, Islam Tolaib, Argyro Kamperi, Apostolos Christou, Yizhou Xu, Mohsen Ghorbanpoor, Andrea Ruffino, Mohamed Eleraky, Bryan Lin, Edward Liu, Kyungsik Choi, Tzu-Yuan Huang, Hossein Jalili, Naga Sasikanth Mannem, Jeongsoo Park, Jeongseok Lee, David Munzer, Sensen Li, Fei Wang, Amr S. Ahmed, Christopher Snyder, Huy Thong Nguyen, and Michael Edward Duffy Smith, "Power Amplifiers Performance Survey 2000-Present," [Online]. Available: https://ideas.ethz.ch/surveys/pa-survey.html',
  },

  metrics: {
    freq: { label: 'Frequency', short: 'f', unit: 'GHz', scale: 'log', digits: 2, ticks: FREQ_TICKS },
    psat: {
      label: 'Saturated output power',
      short: 'Psat',
      unit: 'dBm',
      digits: 1,
      description: 'Psat. For frequency multipliers the maximum output power, for oscillators the output power (the survey plots both with Psat).',
    },
    pae: {
      label: 'Peak PAE',
      short: 'PAE',
      unit: '%',
      digits: 1,
      description: 'Peak power-added efficiency. Designs that report only drain / collector efficiency are marked "Not PAE".',
    },
    p1db: { label: 'P1dB', unit: 'dBm', digits: 1, description: 'Output power at 1-dB gain compression.' },
    pae_1db: { label: 'PAE at P1dB', short: 'PAE₁dB', unit: '%', digits: 1 },
    gain: { label: 'Gain', unit: 'dB', digits: 1 },
    avg_pout: {
      label: 'Average output power',
      short: 'Pavg',
      unit: 'dBm',
      digits: 1,
      description: 'Average output power with a modulated signal (see the modulation in the paper card).',
    },
    avg_pae: {
      label: 'Average PAE',
      short: 'PAEavg',
      unit: '%',
      digits: 1,
      description: 'Average efficiency with a modulated signal, i.e. at back-off.',
    },
    evm: { label: 'EVM', unit: 'dB', digits: 1, description: 'Error vector magnitude of the modulated test.' },
    core_area: {
      label: 'Core area',
      unit: 'mm²',
      scale: 'log',
      digits: 3,
      description: 'PA core area as the survey estimates it: active and passive parts in the RF path; pads, supply, bias and bypass networks excluded.',
    },
    chip_area: { label: 'Chip area', unit: 'mm²', scale: 'log', digits: 3 },
    density: {
      label: 'Power density',
      short: 'Psat/area',
      unit: 'W/mm²',
      scale: 'log',
      digits: 3,
      description: 'Psat (W) / core area (mm²), the survey\'s power-density metric.',
      dependsOn: ['psat', 'core_area'],
    },
    fom: {
      label: 'ITRS FOM',
      unit: 'W·GHz²',
      // log: values span ~10 decades
      scale: 'log',
      digits: 1,
      description: 'Psat · G · PAE · f²: Psat in W, gain G linear, PAE as a fraction (not %), f in GHz.',
      dependsOn: ['psat', 'gain', 'pae', 'freq'],
    },
    pdc: { label: 'DC power', short: 'Pdc', unit: 'mW', scale: 'log', digits: 1, description: 'Frequency multipliers and oscillators.' },
    dc_rf_eff: { label: 'DC-to-RF efficiency', short: 'ηDC-RF', unit: '%', digits: 2, description: 'Oscillators only.' },
    conv_gain: { label: 'Conversion gain', unit: 'dB', digits: 1, description: 'Frequency multipliers only.' },
    year: { label: 'Year', digits: 0 },
  },
  paperFields: ['year'],
  derived: { fom: itrsFom, density: powerDensity },

  // Colour by technology family; the exact process (GaN / GaAs / InP within
  // III-V) is told apart by marker shape, through the "technology" category.
  groups: {
    label: 'Technology',
    order: ['CMOS Bulk', 'CMOS SOI', 'FinFET', 'SiGe', 'III-V', 'LDMOS', 'Other'],
    colors: PROCESS_COLORS,
    symbols: {
      'CMOS Bulk': 'circle',
      SiGe: 'diamond',
      'CMOS SOI': 'square',
      FinFET: 'triangle-up',
      'III-V': 'cross',
      LDMOS: 'hexagon',
      Other: 'x',
    },
    fallbackColor: NEUTRAL,
  },
  categories: {
    technology: {
      label: 'Process',
      field: 'technology',
      order: ['CMOS Bulk', 'CMOS SOI', 'FinFET', 'SiGe', 'GaN', 'GaAs', 'InP', 'LDMOS', 'Other'],
      // same shapes as the LNA survey for the silicon groups
      symbols: {
        'CMOS Bulk': 'circle',
        'CMOS SOI': 'square',
        FinFET: 'triangle-up',
        SiGe: 'diamond',
        GaN: 'triangle-down',
        GaAs: 'pentagon',
        InP: 'star',
        LDMOS: 'hexagon',
        Other: 'x',
      },
      descriptions: {
        'CMOS Bulk': 'Bulk CMOS, including entries labelled only "CMOS".',
        'CMOS SOI': 'FD-SOI, PD-SOI, RF-SOI and silicon-on-sapphire CMOS.',
        GaAs: 'GaAs pHEMT / mHEMT / HBT, including InGaP/GaAs HBTs and InGaAs mHEMTs.',
        InP: 'InP HBT / DHBT / HEMT.',
        Other: 'Silicon bipolar, SiC, BCD, or not stated.',
      },
    },
    circuit: {
      label: 'Circuit',
      field: 'circuit',
      order: ['PA', 'PA product', 'Multiplier', 'Oscillator'],
      descriptions: {
        PA: 'Power amplifiers and transmitters from papers.',
        'PA product': 'Commercial PAs, from datasheets (Qorvo, Skyworks); no year.',
        Multiplier: 'Frequency multipliers (Psat is their maximum output power).',
        Oscillator: 'THz oscillators (Psat is their output power).',
      },
    },
    pa_type: {
      label: 'PA type',
      field: 'pa_type',
      order: ['Analog', 'Digital', 'Supply-modulated', 'Not stated'],
      descriptions: {
        Digital: 'Digital / switched-capacitor PAs and digital transmitters.',
        'Supply-modulated': 'Envelope tracking (ET) or envelope elimination and restoration (EER).',
        'Not stated': 'The survey does not say (most entries), or not a PA.',
      },
    },
  },

  filters: ['freq', 'year', 'psat', 'pae', 'gain'],
  panels: [
    { x: 'freq', y: 'psat', title: 'Saturated output power vs frequency' },
    { x: 'freq', y: 'pae', title: 'Peak PAE vs frequency' },
    { x: 'freq', y: 'gain', title: 'Gain vs frequency' },
    { x: 'freq', y: 'fom', title: 'ITRS figure of merit vs frequency' },
    { x: 'freq', y: 'density', title: 'Power density vs frequency' },
    { x: 'psat', y: 'pae', title: 'Peak PAE vs Psat' },
    { x: 'avg_pout', y: 'avg_pae', title: 'Modulated signals: average PAE vs average output power' },
  ],
  explorer: {
    metrics: ['freq', 'psat', 'pae', 'p1db', 'pae_1db', 'gain', 'avg_pout', 'avg_pae', 'evm', 'fom', 'density',
      'core_area', 'chip_area', 'pdc', 'dc_rf_eff', 'conv_gain', 'year'],
    x: 'psat',
    y: 'pae',
    color: 'group',
  },
  hover: ['freq', 'psat', 'pae', 'gain'],
  table: ['freq', 'psat', 'pae', 'gain', 'p1db', 'fom'],
  info: ['freq', 'psat', 'pae', 'p1db', 'pae_1db', 'gain', 'avg_pout', 'avg_pae', 'evm', 'fom', 'core_area', 'chip_area',
    'density', 'pdc', 'dc_rf_eff', 'conv_gain'],

  paper: {
    citation: (p) => {
      const where = isProduct(p) ? 'datasheet' : [p.venue, p.year].filter(Boolean).join(' ');
      return [p.first_author, where].filter(Boolean).join(', ');
    },
    // Google Scholar does not index datasheets
    link: (p) => (p.title && !isProduct(p) ? scholar(p.title) : null),
    entryDetails: (e) => {
      /** @type {Array<[string, string]>} */
      const rows = [];
      /** @param {string} label @param {unknown} v @param {string} [unit] */
      const add = (label, v, unit) => {
        if (v != null && v !== '') rows.push([label, withUnit(v, unit)]);
      };
      if (e.sheet) rows.push(['Survey sheet', `${e.sheet}, row ${e.row}`]);
      if (e.process) rows.push(['Process', e.process === e.technology ? e.process : `${e.process} (${e.technology})`]);
      add('Node', e.node, 'nm');
      add('Efficiency reported as', e.eff_note);
      add('PA type', e.pa_note);
      add('CW band', e.cw_band, 'GHz');
      add('Frequency range', e.freq_range, 'GHz');
      add('Modulation', [e.modulation, e.mod_note].filter((v) => v != null).join('; '));
      add('Symbol rate', e.mod_rate, 'Msym/s');
      add('PCB', e.pcb);
      add('Multiplication factor', e.mult_factor == null ? null : `×${e.mult_factor}`);
      add('Output bandwidth', e.out_bw, 'GHz');
      add('Oscillator type', e.osc_type);
      add('Tuning range', e.tuning, '%');
      add('Phase noise', e.phase_noise, 'dBc/Hz');
      add('Notes', [e.note1, e.note2].filter((v) => v != null).join('; '));
      return rows;
    },
  },

  // Set by scripts/source_pa.py from the survey's notes and annotations.
  caveats: {
    not_pae: {
      label: 'Not PAE',
      short: 'DE',
      description: 'Efficiency reported as drain, collector or system efficiency (DE / CE / SE), not PAE. The survey plots it with PAE; DE is higher than PAE, the more so the lower the gain. A FOM computed from it carries the same caveat.',
    },
    simulated: {
      label: 'Simulated',
      short: 'sim',
      description: 'Simulation results, not measurements (the values marked).',
    },
    system: {
      label: 'System-level',
      short: 'system',
      description: 'Measured with more than the PA: through a T/R switch, with an integrated antenna, or on the whole transmitter.',
    },
    offchip: {
      label: 'Off-chip output network',
      short: 'off-chip',
      description: 'Part of the output matching or power combining is off chip (LTCC, module, spatial combiner), so Psat and efficiency are not those of a fully integrated PA.',
    },
    pulsed: {
      label: 'Pulsed',
      short: 'pulsed',
      description: 'Measured with a pulsed signal, which limits self-heating; CW results are usually lower.',
    },
    estimated: {
      label: 'Estimated',
      short: 'est.',
      description: 'Read off a figure in the paper, or estimated, rather than quoted (as the survey notes it).',
    },
  },

  manual: {
    fields: [
      { metric: 'freq', required: true, placeholder: '28' },
      { metric: 'psat', required: true, placeholder: '20' },
      { metric: 'pae', required: true, placeholder: '30' },
      { metric: 'gain', placeholder: '15' },
      { metric: 'core_area', placeholder: '0.3' },
    ],
  },
};

export default pa;
