import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';

import { joinData } from '../../site/js/model.js';
import cfg, {
  apertureArea,
  apertureAreaPerElement,
  areaToGrid,
  chipThermalDensity,
  eirpEfficiency,
  halfLambdaMm,
  packageThermalDensity,
  pradEfficiency,
  pubDate,
} from '../../site/surveys/phased_array.js';

const close = (a, b, rel = 1e-9) => assert.ok(Math.abs(a - b) <= rel * Math.abs(b), `${a} != ${b}`);

// Valdes-Garcia 2010 (row 2 of the survey), with the sheet's cached formula results
const row2 = {
  freq: 60, ap_x: 2.8, ap_y: 2.8, tiles: 1, ant_per_tile: 16, ic_per_tile: 1, chip_area: 43.875,
  tx_pdc: 3.8, rx_pdc: null, eirp: 42, pout: 21.34119982655925, core_area_el: 3.017025, year: 2010, month: 12,
};

test('recomputed formula columns match the survey sheet', () => {
  close(apertureArea(row2), 7.839999999999999);
  close(apertureAreaPerElement(row2), 48.99999999999999);
  close(packageThermalDensity(row2), 0.48469387755102045);
  close(chipThermalDensity(row2), 8.66096866096866);
});

test('efficiencies as the survey Readme 6.9 defines them', () => {
  close(eirpEfficiency(row2), ((10 ** 4.2 / 1000) / 3.8) * 100); // 417 %: EIRP includes the array gain
  close(pradEfficiency(row2), 3.584, 1e-3);
});

test('λ/2 grid and publication date', () => {
  assert.equal(halfLambdaMm(150), 1);
  close(areaToGrid(row2), 3.017025 / 2.5 ** 2);
  close(pubDate(row2), 2010 + 11.5 / 12);
  assert.equal(pubDate({ year: 2020 }), 2020.5);
  const grid = cfg.panels.find((p) => p.y === 'core_area_el')?.refs?.[0];
  assert.equal(grid?.y(150), 1);
  for (const ref of cfg.panels.find((p) => p.x === 'n_eirp')?.refs ?? []) {
    close(/** @type {number} */ (ref.y(100)) - /** @type {number} */ (ref.y(10)), 20); // 20 dB per decade of N
  }
});

test('derived metrics are null for missing or invalid inputs', () => {
  assert.equal(apertureArea({ ap_x: null, ap_y: 2 }), null);
  assert.equal(apertureAreaPerElement({ ...row2, ant_per_tile: null }), null);
  assert.equal(packageThermalDensity({ ...row2, tx_pdc: null }), null);
  assert.equal(chipThermalDensity({ ...row2, chip_area: 0 }), null); // passive relays: no chip
  assert.equal(eirpEfficiency({ eirp: 30, tx_pdc: 0 }), null);
  assert.equal(areaToGrid({ core_area_el: 1, freq: null }), null);
  assert.equal(halfLambdaMm(0), null);
  assert.equal(pubDate({ year: null, month: 3 }), null);
  // the RX power alone sets the thermal density of a receive array
  close(packageThermalDensity({ ap_x: 2, ap_y: 2, tx_pdc: null, rx_pdc: 8 }), 2);
});

const DATA = new URL('../../site/data/phased_array.json', import.meta.url);

test('built data matches the config (metrics, caveats, categories)', { skip: !existsSync(DATA) && 'run scripts/build_data.py first' }, () => {
  const json = JSON.parse(readFileSync(DATA, 'utf8'));
  const { points } = joinData(json, cfg);
  assert.equal(points.length, json.meta.points_out);
  const meta = new Set(['paper', 'entry', 'band', 'flags', 'temp_k']);
  for (const key of Object.keys(json.points[0])) if (!meta.has(key)) assert.ok(cfg.metrics[key], `data key ${key} has no metric`);
  for (const f of Object.keys(json.meta.flag_counts)) assert.ok(cfg.caveats?.[f], `point flag ${f} has no caveat config`);
  for (const paper of json.papers) {
    for (const e of paper.entries) {
      for (const [f, keys] of Object.entries(e.metric_flags ?? {})) {
        assert.ok(cfg.caveats?.[f], `metric flag ${f} has no caveat config`);
        for (const k of keys) assert.ok(cfg.metrics[k], `metric flag ${f} names unknown metric ${k}`);
      }
      for (const [k, c] of Object.entries(cfg.categories ?? {})) {
        assert.ok(c.order.includes(e[c.field]), `row ${e.row}: ${k} value ${e[c.field]} not in the category order`);
      }
      assert.ok(cfg.groups.order.includes(e.group), `row ${e.row}: unknown group ${e.group}`);
    }
  }
  // a caveat on TX Pdc reaches what is computed from it
  const shared = points.find((p) => p.metricFlags.pdc_shared);
  assert.ok(shared?.metricFlags.pdc_shared.includes('eirp_eff') && shared.metricFlags.pdc_shared.includes('chip_td'));
  // every point has a publication date for the elements-vs-date panel
  assert.ok(points.every((p) => Number.isFinite(p.date)));
});
