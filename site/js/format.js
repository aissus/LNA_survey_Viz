// @ts-check
/** Formatting and tiny DOM helpers shared by the UI modules. */

/** @typedef {import('./types.js').Metric} Metric */

/**
 * @param {unknown} v
 * @param {number} [digits]
 */
export function fmt(v, digits = 2) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '–';
  if (Math.abs(v) >= 1e5) return v.toExponential(2);
  // a small non-zero value would round to 0.00: show two significant digits instead
  if (v !== 0 && Math.abs(v) < 0.5 * 10 ** -digits) return Number(v.toPrecision(2)).toString();
  return v.toFixed(digits);
}

/** Value with its unit, or an en dash when missing. @param {unknown} v @param {Metric} m */
export function fmtWithUnit(v, m) {
  const s = fmt(v, m.digits ?? 2);
  return s === '–' || !m.unit ? s : `${s} ${m.unit}`;
}

/** "Noise figure (dB)" @param {Metric} m */
export function metricLabel(m) {
  return m.unit ? `${m.label} (${m.unit})` : m.label;
}

/** Escape text for Plotly hover labels and innerHTML. @param {unknown} s */
export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] ?? c);
}

/**
 * Create an element. Attributes starting with "on" and holding a function
 * become listeners; null / false attributes and children are skipped.
 * @param {string} tag
 * @param {Record<string, any> | null} [attrs]
 * @param {...any} children
 * @returns {HTMLElement}
 */
export function h(tag, attrs = null, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs ?? {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, String(v));
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : String(c));
  }
  return el;
}

/** @param {string} selector @returns {HTMLElement} */
export function $(selector) {
  const el = document.querySelector(selector);
  if (!el) throw new Error(`missing element ${selector}`);
  return /** @type {HTMLElement} */ (el);
}
