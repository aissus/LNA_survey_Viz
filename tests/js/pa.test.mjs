import { test } from 'node:test';
import assert from 'node:assert/strict';

import { itrsFom, powerDensity } from '../../site/surveys/pa.js';

test('PA ITRS FOM matches a hand calculation', () => {
  // 20 dBm = 0.1 W, 20 dB = 100x, 40 % PAE, 28 GHz: 0.1 * 100 * 0.4 * 784
  assert.ok(Math.abs(itrsFom({ psat: 20, gain: 20, pae: 40, freq: 28 }) - 3136) < 1e-9);
});

test('PA ITRS FOM is null when an input is missing or invalid', () => {
  const ok = { psat: 20, gain: 20, pae: 40, freq: 28 };
  for (const key of Object.keys(ok)) assert.equal(itrsFom({ ...ok, [key]: null }), null, `${key} missing`);
  assert.equal(itrsFom({ ...ok, pae: 0 }), null);
  assert.equal(itrsFom({ ...ok, freq: 0 }), null);
  assert.ok(itrsFom({ ...ok, psat: -3, gain: 0 }) > 0, 'negative dBm and 0 dB gain are valid');
});

test('PA power density is Psat (W) per core area (mm²), as the survey computes it', () => {
  // 30 dBm = 1 W over 0.5 mm²
  assert.ok(Math.abs(powerDensity({ psat: 30, core_area: 0.5 }) - 2) < 1e-12);
  assert.equal(powerDensity({ psat: 30, core_area: 0 }), null);
  assert.equal(powerDensity({ psat: 30, core_area: null }), null);
  assert.equal(powerDensity({ psat: null, core_area: 1 }), null);
});
