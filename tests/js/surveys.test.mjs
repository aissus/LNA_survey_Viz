import { test } from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_SURVEY, SURVEYS } from '../../site/surveys/index.js';
import { lnaFom } from '../../site/surveys/lna.js';

// Every survey config must only reference metrics it defines. This is the
// check that catches a typo in a new survey (e.g. pa.js) before it ships.
for (const [id, entry] of Object.entries(SURVEYS)) {
  test(`survey config "${id}" is self-consistent`, async () => {
    const cfg = (await entry.load()).default;
    assert.equal(cfg.id, id);
    const metrics = new Set(Object.keys(cfg.metrics));
    const used = [
      ...cfg.filters,
      ...cfg.panels.flatMap((p) => [p.x, p.y]),
      ...cfg.explorer.metrics,
      cfg.explorer.x,
      cfg.explorer.y,
      ...cfg.hover,
      ...cfg.table,
      ...cfg.info,
      ...cfg.manual.fields.map((f) => f.metric),
      ...Object.keys(cfg.derived ?? {}),
      ...Object.keys(cfg.sourceNotes ?? {}),
    ];
    for (const key of used) assert.ok(metrics.has(key), `${id}: "${key}" is used but not defined in metrics`);
    assert.ok(cfg.explorer.color === 'group' || metrics.has(cfg.explorer.color));
    for (const g of Object.keys(cfg.groups.colors)) assert.ok(cfg.groups.order.includes(g), `${id}: colour for unknown group ${g}`);
    for (const [k, m] of Object.entries(cfg.metrics)) {
      if (m.ticks) assert.equal(m.scale, 'log', `${id}: ${k} has ticks but is not log`);
      for (const d of m.dependsOn ?? []) assert.ok(metrics.has(d), `${id}: ${k} depends on unknown metric ${d}`);
    }
    for (const [k, c] of Object.entries(cfg.caveats ?? {})) {
      assert.ok(c.label && c.short && c.description, `${id}: caveat ${k} needs label, short and description`);
    }
    assert.equal(typeof cfg.paper.citation({ first_author: 'A', venue: 'B', year: 2020 }), 'string');
  });
}

test('default survey exists', () => {
  assert.ok(Object.hasOwn(SURVEYS, DEFAULT_SURVEY));
});

test('LNA FOM matches a hand calculation', () => {
  // Afifi 2025, band 1: 7.35 GHz, NF 1.6 dB, gain 23 dB, 20 mW
  const G = 10 ** 2.3;
  const F = 10 ** 0.16;
  const expected = (G * 7.35) / (20 * (F - 1));
  assert.ok(Math.abs(lnaFom({ gain: 23, freq: 7.35, power: 20, nf: 1.6 }) - expected) < 1e-9);
  assert.ok(Math.abs(expected - 164.6) < 0.1);
});

test('LNA FOM is null when an input is missing or physically invalid', () => {
  const ok = { gain: 15, freq: 28, power: 20, nf: 2.5 };
  for (const key of Object.keys(ok)) {
    assert.equal(lnaFom({ ...ok, [key]: null }), null, `${key} missing`);
  }
  assert.equal(lnaFom({ ...ok, power: 0 }), null);
  assert.equal(lnaFom({ ...ok, nf: 0 }), null);
  assert.ok(lnaFom({ ...ok, gain: 0 }) > 0, '0 dB gain is a valid value');
});
