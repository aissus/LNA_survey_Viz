// @ts-check
/** Data table: every filtered point, sortable, linked to the selection. */

import { fmt, h, metricLabel } from './format.js';
import { metricFlagsOf, sortBy } from './model.js';

/** @typedef {import('./types.js').SurveyConfig} SurveyConfig */
/** @typedef {import('./types.js').AppState} AppState */
/** @typedef {import('./types.js').Point} Point */
/** @typedef {import('./state.js').Store} Store */

/**
 * @typedef {object} Column
 * @property {string} key
 * @property {string} label
 * @property {boolean} [numeric]
 * @property {(p: Point) => unknown} value
 * @property {(p: Point) => (Node|string)} cell
 */

/**
 * @param {HTMLElement} root
 * @param {{cfg: SurveyConfig, store: Store, papers: Map<string, any>, multiBand?: boolean}} opts
 */
export function createTable(root, { cfg, store, papers, multiBand = true }) {
  const paper = (/** @type {Point} */ p) => papers.get(/** @type {string} */ (p.paperId));

  /** Small caveat badges ("sim", "system", "cryo 4 K", "sim IIP3"). @param {Point} p */
  const badges = (p) => {
    const out = (p.flags ?? []).map((/** @type {string} */ f) => {
      const c = cfg.caveats?.[f];
      const text = f === 'cryogenic' && p.tempK != null ? `${c?.short ?? f} ${p.tempK} K` : c?.short ?? f;
      return h('span', { class: 'badge', title: c?.description ?? f }, text);
    });
    // caveats on single values, named by their measured inputs (not the FOMs derived from them)
    const valueFlags = { ...(p.masked ?? {}), ...metricFlagsOf(p) };
    for (const [f, ms] of Object.entries(valueFlags)) {
      const inputs = ms.filter((k) => !cfg.metrics[k]?.dependsOn);
      if (!inputs.length || p.flags?.includes(f)) continue;
      const c = cfg.caveats?.[f];
      out.push(h('span', { class: 'badge', title: c?.description ?? f },
        `${c?.short ?? f} ${inputs.map((k) => cfg.metrics[k]?.short ?? cfg.metrics[k]?.label ?? k).join(', ')}`));
    }
    return out.length ? h('span', { class: 'badges' }, out) : null;
  };

  /** @type {Column[]} */
  const columns = [
    {
      key: 'title',
      label: 'Paper',
      value: (p) => paper(p).title,
      cell: (p) => h('div', { class: 'paper-cell' },
        h('span', { class: 'paper-title' }, paper(p).title ?? ''),
        h('span', { class: 'paper-cite' }, cfg.paper.citation(paper(p)), badges(p))),
    },
    ...cfg.table.map((key) => {
      const m = cfg.metrics[key];
      return /** @type {Column} */ ({
        key,
        label: metricLabel(m),
        numeric: true,
        value: (p) => p[key],
        cell: (p) => fmt(p[key], m.digits ?? 2),
      });
    }),
    ...(multiBand ? [/** @type {Column} */ ({
      key: 'band',
      label: 'Band',
      numeric: true,
      value: (p) => (p.nBands > 1 ? p.band + 1 : null),
      cell: (p) => (p.nBands > 1 ? `${p.band + 1}/${p.nBands}` : ''),
    })] : []),
    { key: 'year', label: 'Year', numeric: true, value: (p) => p.year, cell: (p) => String(p.year ?? '') },
    { key: 'group', label: cfg.groups.label ?? 'Group', value: (p) => p.group, cell: (p) => p.group },
    ...Object.entries(cfg.categories ?? {}).map(([key, cat]) => /** @type {Column} */ ({
      key, label: cat.label, value: (p) => p[key], cell: (p) => String(p[key] ?? ''),
    })),
    {
      key: 'link',
      label: 'Link',
      value: () => null,
      cell: (p) => {
        const href = cfg.paper.link?.(paper(p));
        return href ? h('a', { href, target: '_blank', rel: 'noopener', onclick: (/** @type {Event} */ e) => e.stopPropagation() }, 'Scholar') : '';
      },
    },
  ];
  const sortable = new Set(columns.filter((c) => c.key !== 'link').map((c) => c.key));
  // text columns sort A-Z first, numbers largest first
  const textKeys = new Set(['title', 'group', ...Object.keys(cfg.categories ?? {})]);

  const table = h('table', { class: 'data-table' });
  const caption = h('p', { class: 'table-caption' });
  const onlySelected = /** @type {HTMLInputElement} */ (h('input', {
    type: 'checkbox',
    onchange: () => store.set({ onlySelected: onlySelected.checked }),
  }));
  root.replaceChildren(
    h('div', { class: 'table-tools' }, caption, h('label', { class: 'toggle' }, onlySelected, ' Only selected papers')),
    h('div', { class: 'table-wrap' }, table),
  );

  /** @param {string} key */
  const sortOn = (key) => store.set((s) => ({
    sort: { key, dir: s.sort.key === key ? /** @type {1|-1} */ (-s.sort.dir) : textKeys.has(key) ? 1 : -1 },
  }));

  return {
    sortKeys: sortable,
    /**
     * @param {Point[]} points  visible (filtered) points
     * @param {AppState} state
     */
    render(points, state) {
      const selected = new Set(state.selected);
      onlySelected.checked = state.onlySelected;
      onlySelected.disabled = selected.size === 0;
      const rows = state.onlySelected && selected.size ? points.filter((p) => selected.has(/** @type {string} */ (p.paperId))) : points;
      const col = columns.find((c) => c.key === state.sort.key) ?? columns[0];
      const sorted = sortBy(rows, col.value, state.sort.dir);
      const nPapers = new Set(sorted.map((p) => p.paperId)).size;
      caption.textContent = `${sorted.length} points from ${nPapers} papers`;

      const head = h('tr', null, columns.map((c) => {
        const active = c.key === state.sort.key;
        const aria = active ? (state.sort.dir > 0 ? 'ascending' : 'descending') : null;
        const label = sortable.has(c.key)
          ? h('button', { type: 'button', class: 'sort', onclick: () => sortOn(c.key) }, c.label, h('span', { class: 'sort-mark', 'aria-hidden': 'true' }, active ? (state.sort.dir > 0 ? ' ▲' : ' ▼') : ''))
          : c.label;
        return h('th', { scope: 'col', class: c.numeric ? 'num' : null, 'aria-sort': aria }, label);
      }));

      const body = document.createDocumentFragment();
      for (const p of sorted) {
        const id = /** @type {string} */ (p.paperId);
        const cls = [selected.has(id) ? 'is-selected' : '', state.focus === id ? 'is-focus' : ''].filter(Boolean).join(' ');
        body.append(h('tr', {
          class: cls || null,
          tabindex: 0,
          onclick: () => store.set({ focus: id }),
          onkeydown: (/** @type {KeyboardEvent} */ e) => { if (e.key === 'Enter') store.set({ focus: id }); },
        }, columns.map((c) => h('td', { class: c.numeric ? 'num' : null }, c.cell(p)))));
      }
      table.replaceChildren(h('thead', null, head), h('tbody', null, body));
    },
  };
}
