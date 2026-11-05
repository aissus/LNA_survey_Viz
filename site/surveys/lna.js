// @ts-check
/**
 * LNA survey. Everything LNA-specific about the viewer lives in this file;
 * how the spreadsheet is read lives in scripts/survey_sources.py.
 */

const FREQ_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];

/**
 * LNA figure of merit: G·f / (P·(F − 1)), with gain G and noise factor F
 * linear, f in GHz and P in mW. Null when an input is missing or invalid.
 * @param {Record<string, any>} p
 * @returns {number | null}
 */
export function lnaFom({ gain, freq, power, nf }) {
  if (![gain, freq, power, nf].every(Number.isFinite) || freq <= 0 || power <= 0 || nf <= 0) return null;
  const G = 10 ** (gain / 10);
  const F = 10 ** (nf / 10);
  return (G * freq) / (power * (F - 1));
}

/** @param {string} title */
const scholar = (title) => `https://scholar.google.com/scholar?q=${encodeURIComponent(title)}`;

const P1DB_REF = { I: 'input-referred', O: 'output-referred' };

/** @type {import('../js/types.js').SurveyConfig} */
const lna = {
  id: 'lna',
  title: 'LNA Survey Visualizer',
  subtitle: 'Published low-noise amplifiers. Filter by spec, compare with your own design, and open the papers.',
  source: {
    label: 'ETH Zürich IDEAS LNA survey',
    url: 'https://ideas.ethz.ch/Surveys/lna-survey.html',
    citation:
      'Hua Wang, Argyro Kamperi, Boce Lin, Basem Abdelaziz Abdelmagid Ali, "Low Noise Amplifier Noise Figure Survey 2018-present," [Online]. Available: https://ideas.ethz.ch/Surveys/lna-survey.html',
  },

  metrics: {
    freq: { label: 'Frequency', short: 'f', unit: 'GHz', scale: 'log', digits: 2, ticks: FREQ_TICKS },
    nf: { label: 'Noise figure', short: 'NF', unit: 'dB', digits: 2 },
    gain: { label: 'Gain', unit: 'dB', digits: 1 },
    power: { label: 'Power', short: 'P', unit: 'mW', digits: 1 },
    iip3: { label: 'IIP3', unit: 'dBm', digits: 1 },
    p1db: { label: 'P1dB', unit: 'dBm', digits: 1 },
    bw3db: { label: '3-dB bandwidth', short: 'BW', unit: 'GHz', digits: 2 },
    nfmin: { label: 'NFmin', unit: 'dB', digits: 2 },
    f_nfmin: { label: 'f at NFmin', unit: 'GHz', scale: 'log', digits: 2, ticks: FREQ_TICKS },
    fom: {
      label: 'FOM',
      // log: values span ~6 decades (cryogenic designs reach 1e5+); linear hides 95% of points
      scale: 'log',
      digits: 1,
      description: 'G·f / (P·(F − 1)): gain G and noise factor F linear, f in GHz, P in mW',
      dependsOn: ['gain', 'freq', 'power', 'nf'],
    },
    year: { label: 'Year', digits: 0 },
  },
  paperFields: ['year'],
  derived: { fom: lnaFom },

  // Three hues validate for a scatter (all pairs, colour-blind safe); the
  // smaller groups share a neutral grey and are told apart by marker shape.
  groups: {
    label: 'Process',
    order: ['CMOS Bulk', 'SiGe', 'CMOS SOI', 'FinFET', 'III-V', 'Other'],
    colors: { 'CMOS Bulk': '#2a78d6', SiGe: '#eb6834', 'CMOS SOI': '#1baf7a' },
    symbols: {
      'CMOS Bulk': 'circle',
      SiGe: 'diamond',
      'CMOS SOI': 'square',
      FinFET: 'triangle-up',
      'III-V': 'cross',
      Other: 'x',
    },
    fallbackColor: '#6f6e69',
  },

  filters: ['freq', 'year', 'nf', 'gain', 'power'],
  panels: [
    { x: 'freq', y: 'nf', title: 'Noise figure vs frequency' },
    { x: 'freq', y: 'gain', title: 'Gain vs frequency' },
    { x: 'freq', y: 'power', title: 'Power vs frequency' },
    { x: 'freq', y: 'iip3', title: 'IIP3 vs frequency' },
    { x: 'freq', y: 'fom', title: 'Figure of merit vs frequency' },
    { x: 'nf', y: 'gain', title: 'Gain vs noise figure' },
  ],
  explorer: {
    metrics: ['freq', 'nf', 'gain', 'power', 'iip3', 'p1db', 'bw3db', 'fom', 'nfmin', 'f_nfmin', 'year'],
    x: 'power',
    y: 'gain',
    color: 'group',
  },
  hover: ['freq', 'nf', 'gain', 'power', 'iip3'],
  table: ['freq', 'nf', 'gain', 'power', 'iip3', 'fom'],
  info: ['freq', 'nf', 'gain', 'power', 'iip3', 'p1db', 'bw3db', 'fom', 'f_nfmin', 'nfmin'],

  paper: {
    citation: (p) => [p.first_author, [p.venue, p.year].filter(Boolean).join(' ')].filter(Boolean).join(', '),
    link: (p) => (p.title ? scholar(p.title) : null),
    entryDetails: (e) => {
      /** @type {Array<[string, string]>} */
      const rows = [];
      if (e.process) rows.push(['Process', e.process === e.group ? e.process : `${e.process} (${e.group})`]);
      if (e.node != null) rows.push(['Node', typeof e.node === 'number' ? `${e.node} nm` : String(e.node)]);
      const ref = P1DB_REF[/** @type {'I'|'O'} */ (e.p1db_ref)];
      if (ref) rows.push(['P1dB', ref]);
      if (e.comment) rows.push(['Comment', e.comment]);
      return rows;
    },
  },
  sourceNotes: {
    freq: { f_nfmin: 'Frequency is f at NFmin (fc not reported).' },
    nf: { nfmin: 'NF is NFmin (NF at fc not reported).' },
  },

  // Set by scripts/caveats.py from the survey's Comment column (and titles).
  caveats: {
    simulated: {
      label: 'Simulated',
      short: 'sim',
      description: 'Simulation results, not measurements (the whole entry, or the values marked).',
    },
    system: {
      label: 'System-level',
      short: 'system',
      description: 'Measured as part of a receiver front-end, array or system, so gain, NF and power include more than the LNA.',
    },
    cryogenic: {
      label: 'Cryogenic',
      short: 'cryo',
      description: 'Measured at cryogenic temperature (e.g. 4 K). NF and FOM are not comparable with room-temperature designs.',
    },
  },

  manual: {
    fields: [
      { metric: 'freq', required: true, placeholder: '28' },
      { metric: 'nf', required: true, placeholder: '2.5' },
      { metric: 'gain', required: true, placeholder: '15' },
      { metric: 'power', placeholder: '20' },
      { metric: 'iip3', placeholder: '-5' },
    ],
  },
};

export default lna;
