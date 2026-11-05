// @ts-check
/**
 * User-added design points. Kept in this browser's localStorage (per survey),
 * never sent anywhere. Derived metrics (e.g. FOM) are computed for them too.
 */

import { fmtWithUnit, h, metricLabel } from './format.js';
import { applyDerived, isNum } from './model.js';
import { MANUAL_GROUP } from './charts.js';

/** @typedef {import('./types.js').SurveyConfig} SurveyConfig */
/** @typedef {import('./types.js').Point} Point */

/** @param {string} surveyId */
const storageKey = (surveyId) => `surveyviz:${surveyId}:manual:v1`;

/** @param {SurveyConfig} cfg @returns {Array<Record<string, any>>} */
function load(cfg) {
  try {
    const parsed = JSON.parse(localStorage.getItem(storageKey(cfg.id)) ?? '[]');
    return Array.isArray(parsed) ? parsed.filter((r) => r && typeof r === 'object' && typeof r.label === 'string') : [];
  } catch {
    return [];
  }
}

/** @param {SurveyConfig} cfg @param {Array<Record<string, any>>} rows */
function save(cfg, rows) {
  try {
    localStorage.setItem(storageKey(cfg.id), JSON.stringify(rows));
    return true;
  } catch {
    return false; // private mode / storage disabled: points still work for this visit
  }
}

/**
 * @param {HTMLElement} root
 * @param {{cfg: SurveyConfig, onChange: () => void}} opts
 */
export function createManualPoints(root, { cfg, onChange }) {
  let rows = load(cfg);
  /** @type {Point[]} */
  let points = [];

  const rebuild = () => {
    points = rows.map((r, i) => {
      /** @type {Point} */
      const p = { ...r, id: `manual:${i}`, paperId: null, group: MANUAL_GROUP, manual: true };
      return applyDerived(p, cfg);
    });
  };
  rebuild();

  /** @type {Map<string, HTMLInputElement>} */
  const inputs = new Map();
  const labelInput = /** @type {HTMLInputElement} */ (h('input', { type: 'text', id: 'manual-label', placeholder: 'My design', maxlength: 60, autocomplete: 'off' }));
  const fields = cfg.manual.fields.map((f) => {
    const m = cfg.metrics[f.metric];
    const input = /** @type {HTMLInputElement} */ (h('input', { type: 'number', id: `manual-${f.metric}`, step: 'any', inputmode: 'decimal', placeholder: f.placeholder ?? null, required: !!f.required }));
    inputs.set(f.metric, input);
    return h('label', { class: 'field' }, h('span', null, metricLabel(m), f.required ? h('span', { class: 'req', 'aria-hidden': 'true' }, ' *') : null), input);
  });
  const error = h('p', { class: 'form-error', role: 'alert' });
  const list = h('ul', { class: 'manual-list' });
  const storageNote = h('p', { class: 'muted small' }, 'Saved in this browser only.');

  const form = /** @type {HTMLFormElement} */ (h('form', {
    class: 'manual-form',
    novalidate: true,
    onsubmit: (/** @type {Event} */ e) => {
      e.preventDefault();
      /** @type {Record<string, any>} */
      const row = { label: labelInput.value.trim() || `Point ${rows.length + 1}` };
      for (const f of cfg.manual.fields) {
        const raw = inputs.get(f.metric)?.value.trim() ?? '';
        const v = raw === '' ? null : Number(raw);
        const m = cfg.metrics[f.metric];
        if (v === null && f.required) return showError(`${m.label} is required.`);
        if (v !== null && !isNum(v)) return showError(`${m.label} must be a number.`);
        if (v !== null && m.scale === 'log' && v <= 0) return showError(`${m.label} must be greater than 0.`);
        row[f.metric] = v; // missing stays null, never 0
      }
      rows = [...rows, row];
      persist();
      form.reset();
      showError('');
    },
  },
  h('label', { class: 'field' }, h('span', null, 'Label'), labelInput),
  h('div', { class: 'field-grid' }, fields),
  error,
  h('div', { class: 'btn-row' },
    h('button', { type: 'submit', class: 'btn btn-primary' }, 'Add point'),
    h('button', { type: 'button', class: 'btn btn-quiet', onclick: () => { rows = []; persist(); } }, 'Clear all'))));

  root.replaceChildren(form, list, storageNote);

  /** @param {string} msg */
  function showError(msg) {
    error.textContent = msg;
  }

  function persist() {
    const ok = save(cfg, rows);
    storageNote.textContent = ok ? 'Saved in this browser only.' : 'Could not save in this browser; points last until you leave the page.';
    rebuild();
    renderList();
    onChange();
  }

  function renderList() {
    const derivedKeys = Object.keys(cfg.derived ?? {});
    list.replaceChildren(...points.map((p, i) => h('li', null,
      h('span', null, h('b', null, p.label), ': ',
        [...cfg.manual.fields.map((f) => f.metric), ...derivedKeys]
          .filter((k) => isNum(p[k]))
          .map((k) => `${cfg.metrics[k]?.short ?? cfg.metrics[k]?.label ?? k} ${fmtWithUnit(p[k], cfg.metrics[k] ?? { label: k })}`)
          .join(', ')),
      h('button', {
        type: 'button', class: 'icon-btn', 'aria-label': `Remove ${p.label}`, title: 'Remove',
        onclick: () => { rows = rows.filter((_, j) => j !== i); persist(); },
      }, '×'))));
    if (!points.length) list.replaceChildren(h('li', { class: 'muted' }, 'No points yet.'));
  }
  renderList();

  return {
    /** @returns {Point[]} */
    points: () => points,
  };
}
