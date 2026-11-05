import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  applyDerived, excludeCaveats, extent, filterByGroups, filterByRanges, groupsInData, hasCaveat, joinData, sortBy, withDependents,
} from '../../site/js/model.js';

const cfg = {
  paperFields: ['year'],
  // derived functions must guard their inputs: null / 0 would otherwise coerce to 0
  derived: { ratio: (p) => (Number.isFinite(p.gain) && Number.isFinite(p.nf) ? p.gain / p.nf : null) },
};

const json = {
  meta: { survey: 't' },
  papers: [
    { id: 'aaaaaaaa', year: 2024, entries: [{ group: 'SiGe', n_bands: 2, sources: { freq: 'fc' } }] },
    { id: 'bbbbbbbb', year: 2020, entries: [{ group: 'CMOS Bulk', n_bands: 1, sources: {} }, { group: 'CMOS Bulk', n_bands: 1, sources: {} }] },
  ],
  points: [
    { paper: 'aaaaaaaa', entry: 0, band: 0, freq: 7.35, nf: 1.6, gain: 23 },
    { paper: 'aaaaaaaa', entry: 0, band: 1, freq: 10.65, nf: 1.6, gain: null },
    { paper: 'bbbbbbbb', entry: 0, band: 0, freq: 2.4, nf: 0, gain: 10 },
    { paper: 'bbbbbbbb', entry: 1, band: 0, freq: 5.2, nf: 3.2, gain: 12 },
  ],
  warnings: [{ paper: 'aaaaaaaa', row: 3, severity: 'note', issue: 'x' }],
};

test('joinData flattens papers, entries and bands into points', () => {
  const m = joinData(json, cfg);
  assert.equal(m.points.length, 4);
  assert.deepEqual(m.points.map((p) => p.id), ['aaaaaaaa.0.0', 'aaaaaaaa.0.1', 'bbbbbbbb.0.0', 'bbbbbbbb.1.0']);
  assert.equal(m.points[0].group, 'SiGe');
  assert.equal(m.points[0].year, 2024);
  assert.equal(m.points[0].nBands, 2);
  assert.equal(m.byPaper.get('bbbbbbbb').length, 2);
  assert.equal(m.warnings.get('aaaaaaaa').length, 1);
});

test('derived metrics: non-finite results become null', () => {
  const m = joinData(json, cfg);
  assert.equal(m.points[0].ratio, 23 / 1.6);
  assert.equal(m.points[1].ratio, null); // gain missing
  assert.equal(m.points[2].ratio, null); // division by zero -> Infinity
  assert.equal(applyDerived({ gain: 1, nf: 2 }, { derived: {} }).ratio, undefined);
});

test('range filters keep points that did not report the metric, and zeros are values', () => {
  const m = joinData(json, cfg);
  const kept = filterByRanges(m.points, { gain: [11, 30] });
  assert.deepEqual(kept.map((p) => p.id), ['aaaaaaaa.0.0', 'aaaaaaaa.0.1', 'bbbbbbbb.1.0']);
  const nf = filterByRanges(m.points, { nf: [0.5, 5] });
  assert.ok(!nf.some((p) => p.id === 'bbbbbbbb.0.0'), 'NF = 0 is a value, filtered like any other');
  assert.equal(filterByRanges(m.points, {}), m.points);
});

test('group filter and group order', () => {
  const m = joinData(json, cfg);
  assert.equal(filterByGroups(m.points, null).length, 4);
  assert.equal(filterByGroups(m.points, ['SiGe']).length, 2);
  assert.deepEqual(groupsInData(m.points, ['CMOS Bulk', 'SiGe', 'FinFET']), ['CMOS Bulk', 'SiGe']);
  assert.deepEqual(groupsInData([{ group: 'Zeta' }, { group: 'SiGe' }], ['SiGe']), ['SiGe', 'Zeta']);
});

test('extent ignores missing values, and non-positive ones for log scales', () => {
  const pts = [{ v: -1 }, { v: 0 }, { v: 3 }, { v: null }, { v: Number.NaN }];
  assert.deepEqual(extent(pts, 'v'), [-1, 3]);
  assert.deepEqual(extent(pts, 'v', true), [3, 3]);
  assert.equal(extent([{ v: null }], 'v'), null);
});

test('sortBy keeps missing values last in both directions and is stable', () => {
  const rows = [{ k: 2 }, { k: null }, { k: 1 }, { k: 2, tie: true }];
  assert.deepEqual(sortBy(rows, (r) => r.k, 1).map((r) => r.k), [1, 2, 2, null]);
  assert.deepEqual(sortBy(rows, (r) => r.k, -1).map((r) => r.k), [2, 2, 1, null]);
  assert.equal(sortBy(rows, (r) => r.k, 1)[2].tie, true);
  assert.deepEqual(sortBy([{ k: 'b' }, { k: 'a' }], (r) => r.k, 1).map((r) => r.k), ['a', 'b']);
});

// --- caveats -----------------------------------------------------------------

const cavCfg = {
  paperFields: [],
  metrics: { gain: {}, iip3: {}, nf: {}, fom: { dependsOn: ['gain', 'nf'] } },
  derived: {},
};
const cavJson = {
  papers: [
    { id: 'aaaaaaaa', entries: [{ group: 'G', n_bands: 2, metric_flags: { simulated: ['iip3'] } }] },
    { id: 'bbbbbbbb', entries: [{ group: 'G', n_bands: 1, metric_flags: { simulated: ['gain'] } }] },
    { id: 'cccccccc', entries: [{ group: 'G', n_bands: 2 }] },
  ],
  points: [
    { paper: 'aaaaaaaa', entry: 0, band: 0, iip3: -5, gain: 20 },
    { paper: 'aaaaaaaa', entry: 0, band: 1, iip3: -4, gain: 21, flags: ['system'] },
    { paper: 'bbbbbbbb', entry: 0, band: 0, iip3: 1, gain: 15, fom: 99 },
    { paper: 'cccccccc', entry: 0, band: 0, gain: 10, temp_k: 300 },
    { paper: 'cccccccc', entry: 0, band: 1, gain: 30, temp_k: 4, flags: ['cryogenic'] },
  ],
};

test('a simulated input makes derived metrics simulated too', () => {
  assert.deepEqual(withDependents(['gain'], cavCfg).sort(), ['fom', 'gain']);
  assert.deepEqual(withDependents(['iip3'], cavCfg), ['iip3']);
  assert.deepEqual(withDependents([], cavCfg), []);
});

test('hasCaveat: flags always, simulated values only on the plots that show them', () => {
  const { points } = joinData(cavJson, cavCfg);
  assert.equal(points[3].tempK, 300);
  assert.equal(hasCaveat(points[0], ['freq', 'iip3']), true);
  assert.equal(hasCaveat(points[0], ['freq', 'gain']), false);
  assert.equal(hasCaveat(points[1], ['freq', 'gain']), true);
  assert.equal(hasCaveat(points[2], ['freq', 'fom']), true, 'fom derives from a simulated gain');
});

test('excludeCaveats drops flagged points and blanks individually simulated values', () => {
  const { points } = joinData(cavJson, cavCfg);
  assert.equal(excludeCaveats(points, []), points);
  const noSystem = excludeCaveats(points, ['system']);
  assert.deepEqual(noSystem.map((p) => p.id), ['aaaaaaaa.0.0', 'bbbbbbbb.0.0', 'cccccccc.0.0', 'cccccccc.0.1']);
  const measured = excludeCaveats(points, ['simulated', 'cryogenic']);
  assert.deepEqual(measured.map((p) => p.id), ['aaaaaaaa.0.0', 'aaaaaaaa.0.1', 'bbbbbbbb.0.0', 'cccccccc.0.0']);
  assert.equal(measured[0].iip3, null);
  assert.equal(measured[0].gain, 20);
  assert.equal(measured[2].gain, null);
  assert.equal(measured[2].fom, null);
  assert.deepEqual(measured[0].maskedSim, ['iip3']);
  assert.equal(points[0].iip3, -5, 'the source points are not mutated');
});
