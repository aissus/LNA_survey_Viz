import { test } from 'node:test';
import assert from 'node:assert/strict';

import { decodeState, encodeState, surveyFromSearch } from '../../site/js/url.js';

const defaults = {
  survey: 'lna',
  groups: null,
  exclude: [],
  ranges: {},
  selected: [],
  focus: null,
  minorGrid: false,
  explorer: { x: 'power', y: 'gain', color: 'group', logX: null, logY: null },
  sort: { key: 'year', dir: -1 },
  onlySelected: false,
  zoomRev: 0,
  manualRev: 0,
};
const known = {
  metrics: new Set(['freq', 'nf', 'gain', 'power', 'fom', 'year']),
  sortKeys: new Set(['title', 'year', 'fom']),
  colorKeys: new Set(['group', 'year', 'freq']),
  caveats: new Set(['simulated', 'system', 'cryogenic']),
};

test('default state encodes to an empty query', () => {
  assert.equal(encodeState(defaults, defaults), '');
});

test('round trip of every shareable field', () => {
  const s = {
    ...defaults,
    groups: ['SiGe', 'CMOS SOI'],
    exclude: ['simulated', 'cryogenic'],
    ranges: { freq: [5.338537, 105.4361], year: [2019, 2025] },
    selected: ['458217fb', '0290225f'],
    focus: '458217fb',
    minorGrid: true,
    onlySelected: true,
    explorer: { x: 'freq', y: 'fom', color: 'year', logX: null, logY: true },
    sort: { key: 'fom', dir: -1 },
  };
  const back = decodeState(`?${encodeState(s, defaults)}`, defaults, known);
  assert.deepEqual(back.groups, s.groups);
  assert.deepEqual(back.exclude, s.exclude);
  assert.deepEqual(back.ranges.year, [2019, 2025]);
  assert.ok(Math.abs(back.ranges.freq[0] - 5.338537) < 1e-4);
  assert.deepEqual(back.selected, s.selected);
  assert.equal(back.focus, s.focus);
  assert.equal(back.minorGrid, true);
  assert.equal(back.onlySelected, true);
  assert.deepEqual(back.explorer, s.explorer);
  assert.deepEqual(back.sort, s.sort);
});

test('an empty group list means "none", which differs from "all"', () => {
  const none = { ...defaults, groups: [] };
  assert.deepEqual(decodeState(`?${encodeState(none, defaults)}`, defaults, known).groups, []);
  assert.equal(decodeState('?', defaults, known).groups, null);
});

test('malformed or unknown values are ignored, never thrown', () => {
  const s = decodeState('?r.freq=abc~2&r.nope=1~2&r.nf=5~1&sel=XYZ,458217fb&f=<script>&ex=bogus,nf,evil&exlog=1x&sort=-hack&x=system,nope', defaults, known);
  assert.deepEqual(s.ranges, {});
  assert.deepEqual(s.exclude, ['system']);
  assert.deepEqual(s.selected, ['458217fb']);
  assert.equal(s.focus, null);
  assert.deepEqual(s.explorer, { x: 'power', y: 'nf', color: 'group', logX: true, logY: null });
  assert.deepEqual(s.sort, defaults.sort);
});

test('decode does not mutate the defaults', () => {
  decodeState('?r.freq=1~2&g=SiGe', defaults, known);
  assert.deepEqual(defaults.ranges, {});
  assert.equal(defaults.groups, null);
});

test('survey id is read before anything else', () => {
  assert.equal(surveyFromSearch('?s=pa&r.freq=1~2'), 'pa');
  assert.equal(surveyFromSearch(''), null);
});
