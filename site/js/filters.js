// @ts-check
/** Sidebar filters: group checkboxes and one dual-range slider per configured metric. */

import { fmt, h, metricLabel } from './format.js';
import { extent } from './model.js';

/** @typedef {import('./types.js').SurveyConfig} SurveyConfig */
/** @typedef {import('./types.js').AppState} AppState */
/** @typedef {import('./types.js').Point} Point */
/** @typedef {import('./state.js').Store} Store */

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * A small marker icon matching the Plotly symbol, so group identity is never
 * colour alone.
 * @param {string} symbol @param {string} color
 */
export function symbolIcon(symbol, color) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 12 12');
  svg.setAttribute('width', '12');
  svg.setAttribute('height', '12');
  svg.setAttribute('aria-hidden', 'true');
  svg.classList.add('symbol');
  /** @type {Record<string, string>} */
  const shapes = {
    circle: '<circle cx="6" cy="6" r="5"/>',
    'circle-open': '<circle cx="6" cy="6" r="4.2" fill="none" stroke-width="1.6" stroke="currentColor"/>',
    square: '<rect x="1.5" y="1.5" width="9" height="9"/>',
    diamond: '<path d="M6 0.5 11.5 6 6 11.5 0.5 6z"/>',
    'triangle-up': '<path d="M6 1 11.5 11 0.5 11z"/>',
    cross: '<path d="M4 0.5h4v3.5h3.5v4H8v3.5H4V8H0.5V4H4z"/>',
    x: '<path d="M2 0.5 6 4.5 10 0.5 11.5 2 7.5 6 11.5 10 10 11.5 6 7.5 2 11.5 0.5 10 4.5 6 0.5 2z"/>',
    star: '<path d="M6 0.5 7.6 4.3 11.7 4.6 8.6 7.3 9.5 11.3 6 9.2 2.5 11.3 3.4 7.3 0.3 4.6 4.4 4.3z"/>',
  };
  svg.innerHTML = shapes[symbol] ?? shapes.circle;
  svg.setAttribute('fill', color);
  svg.style.color = color; // outline shapes stroke with currentColor
  return svg;
}

/**
 * @param {HTMLElement} root
 * @param {{cfg: SurveyConfig, store: Store, points: Point[], groups: string[]}} opts
 */
export function createFilters(root, { cfg, store, points, groups }) {
  /** @type {Record<string, number>} */
  const counts = {};
  for (const p of points) counts[p.group] = (counts[p.group] ?? 0) + 1;

  // --- groups ---------------------------------------------------------------
  /** @type {Map<string, HTMLInputElement>} */
  const boxes = new Map();
  const setGroups = (/** @type {string[]} */ on) => store.set({ groups: on.length === groups.length ? null : on });
  const groupList = h('div', { class: 'check-list' }, groups.map((g) => {
    const input = /** @type {HTMLInputElement} */ (h('input', {
      type: 'checkbox',
      value: g,
      onchange: () => setGroups(groups.filter((x) => boxes.get(x)?.checked)),
    }));
    boxes.set(g, input);
    return h('label', { class: 'check' },
      input,
      symbolIcon(cfg.groups.symbols[g] ?? 'circle', cfg.groups.colors[g] ?? cfg.groups.fallbackColor),
      h('span', { class: 'check-label' }, g),
      h('span', { class: 'count', title: `${counts[g] ?? 0} points in the survey` }, String(counts[g] ?? 0)));
  }));
  const groupSet = h('fieldset', { class: 'filter' },
    h('legend', { class: 'filter-label' }, cfg.groups.label ?? 'Group'),
    groupList,
    h('div', { class: 'link-row' },
      h('button', { type: 'button', class: 'link', onclick: () => setGroups(groups) }, 'All'),
      h('button', { type: 'button', class: 'link', onclick: () => setGroups([]) }, 'None')));

  // --- caveats ("Include") ----------------------------------------------------
  const caveatKeys = Object.keys(cfg.caveats ?? {});
  /** @type {Map<string, HTMLInputElement>} */
  const caveatBoxes = new Map();
  const setExclude = (/** @type {string[]} */ ex) => store.set({ exclude: caveatKeys.filter((k) => ex.includes(k)) });
  const caveatSet = caveatKeys.length ? h('fieldset', { class: 'filter' },
    h('legend', { class: 'filter-label' }, 'Include'),
    h('div', { class: 'check-list' }, caveatKeys.map((k) => {
      const c = /** @type {import('./types.js').Caveat} */ (cfg.caveats?.[k]);
      const flagged = points.filter((p) => p.flags?.includes(k)).length;
      const values = k === 'simulated' ? points.filter((p) => p.simMetrics?.length && !p.flags?.includes(k)).length : 0;
      const input = /** @type {HTMLInputElement} */ (h('input', {
        type: 'checkbox',
        value: k,
        onchange: () => setExclude(caveatKeys.filter((x) => !caveatBoxes.get(x)?.checked)),
      }));
      caveatBoxes.set(k, input);
      const count = values ? `${flagged} + ${values}` : String(flagged);
      return h('label', { class: 'check', title: c.description },
        input,
        symbolIcon('circle-open', '#52514e'),
        h('span', { class: 'check-label' }, c.label),
        h('span', { class: 'count', title: values ? `${flagged} points, plus ${values} points with individual simulated values` : `${flagged} points` }, count));
    })),
    h('div', { class: 'link-row' },
      h('button', { type: 'button', class: 'link', onclick: () => setExclude([]) }, 'All'),
      h('button', { type: 'button', class: 'link', title: 'Hide every simulated, system-level and cryogenic value', onclick: () => setExclude(caveatKeys) }, 'Measured standalone only'))) : null;

  // --- ranges ---------------------------------------------------------------
  const sliders = cfg.filters.map((key) => rangeControl(key, cfg, store, points)).filter((s) => s != null);

  root.replaceChildren(groupSet, ...(caveatSet ? [caveatSet] : []), ...sliders.map((s) => s.el),
    h('button', { type: 'button', class: 'btn btn-quiet', onclick: () => store.set({ groups: null, ranges: {}, exclude: [] }) }, 'Reset filters'));

  return {
    /** @param {AppState} state */
    render(state) {
      const on = new Set(state.groups ?? groups);
      for (const [g, box] of boxes) box.checked = on.has(g);
      for (const [k, box] of caveatBoxes) box.checked = !state.exclude.includes(k);
      for (const s of sliders) s.render(state);
    },
  };
}

/**
 * Dual-thumb slider over the data extent, in log space for log metrics.
 * Both thumbs at the ends = no filter (so nothing is lost to rounding).
 * @param {string} key @param {SurveyConfig} cfg @param {Store} store @param {Point[]} points
 */
function rangeControl(key, cfg, store, points) {
  const m = cfg.metrics[key];
  if (!m) return null;
  const log = m.scale === 'log';
  const ext = extent(points, key, log);
  if (!ext || ext[1] <= ext[0]) return null;
  const [lo, hi] = ext;
  const digits = m.digits ?? 2;
  const steps = digits === 0 ? Math.max(1, Math.round(hi - lo)) : 400;
  const a0 = log ? Math.log10(lo) : lo;
  const a1 = log ? Math.log10(hi) : hi;
  const toValue = (/** @type {number} */ pos) => {
    const t = a0 + (pos / steps) * (a1 - a0);
    return log ? 10 ** t : t;
  };
  const toPos = (/** @type {number} */ v) => {
    const t = log ? Math.log10(Math.max(v, lo)) : v;
    return Math.min(steps, Math.max(0, Math.round(((t - a0) / (a1 - a0)) * steps)));
  };

  const label = metricLabel(m);
  const inputA = /** @type {HTMLInputElement} */ (h('input', { type: 'range', min: 0, max: steps, step: 1, value: 0, 'aria-label': `${label}, minimum` }));
  const inputB = /** @type {HTMLInputElement} */ (h('input', { type: 'range', min: 0, max: steps, step: 1, value: steps, 'aria-label': `${label}, maximum` }));
  const fill = h('div', { class: 'range-fill' });
  const out = h('output', { class: 'range-value' });

  const commit = () => {
    const pa = Number(inputA.value);
    const pb = Number(inputB.value);
    store.set((s) => {
      const ranges = { ...s.ranges };
      if (pa === 0 && pb === steps) delete ranges[key];
      else ranges[key] = [pa === 0 ? lo : toValue(pa), pb === steps ? hi : toValue(pb)];
      return { ranges };
    });
  };
  inputA.addEventListener('input', () => {
    if (Number(inputA.value) > Number(inputB.value)) inputA.value = inputB.value;
    commit();
  });
  inputB.addEventListener('input', () => {
    if (Number(inputB.value) < Number(inputA.value)) inputB.value = inputA.value;
    commit();
  });

  const el = h('div', { class: 'filter range' },
    h('div', { class: 'filter-head' }, h('span', { class: 'filter-label', title: m.description ?? null }, label), out),
    h('div', { class: 'dual-range' }, h('div', { class: 'range-track' }), fill, inputA, inputB),
    log ? h('div', { class: 'range-hint' }, 'log scale') : null);

  return {
    el,
    /** @param {AppState} state */
    render(state) {
      const r = state.ranges[key];
      const pa = r ? toPos(r[0]) : 0;
      const pb = r ? toPos(r[1]) : steps;
      inputA.value = String(pa);
      inputB.value = String(pb);
      // keep the min thumb reachable when both thumbs sit at the right end
      inputA.style.zIndex = pa >= steps - 1 ? '3' : '';
      fill.style.left = `${(pa / steps) * 100}%`;
      fill.style.width = `${((pb - pa) / steps) * 100}%`;
      const [vl, vh] = r ?? [lo, hi];
      out.textContent = `${fmt(vl, digits)} – ${fmt(vh, digits)}${m.unit ? ` ${m.unit}` : ''}`;
      el.classList.toggle('is-active', !!r);
    },
  };
}
