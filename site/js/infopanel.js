// @ts-check
/**
 * The detail card: a floating card showing the clicked paper (every band,
 * caveats, links) or the list of selected papers. It stays up while you scroll
 * and until the next click replaces it; × or Esc closes it, and it can be
 * minimised to its title. Docked bottom-right on desktop, a sheet on phones.
 */

import { caveatText } from './charts.js';
import { fmt, h, metricLabel } from './format.js';
import { caveatsOf, isNum, metricFlagsOf } from './model.js';

/** @typedef {import('./types.js').SurveyConfig} SurveyConfig */
/** @typedef {import('./types.js').AppState} AppState */
/** @typedef {import('./types.js').Point} Point */
/** @typedef {import('./state.js').Store} Store */

const MAX_LISTED = 40;

/**
 * @param {HTMLElement} card  the (initially hidden) card element
 * @param {{cfg: SurveyConfig, store: Store, papers: Map<string, any>, byPaper: Map<string, Point[]>, warnings: Map<string, any[]>}} opts
 */
export function createDetailCard(card, { cfg, store, papers, byPaper, warnings }) {
  let minimized = false;
  let shownKey = '';

  const headText = h('div', { class: 'detail-head-text' });
  const minBtn = h('button', { type: 'button', class: 'icon-btn', onclick: () => setMinimized(!minimized) });
  const closeBtn = h('button', { type: 'button', class: 'icon-btn', title: 'Close (Esc)', 'aria-label': 'Close details', onclick: close }, '×');
  const body = h('div', { class: 'detail-body' });
  card.replaceChildren(h('div', { class: 'detail-head' }, headText, minBtn, closeBtn), body);

  /** @param {boolean} v */
  function setMinimized(v) {
    minimized = v;
    card.classList.toggle('is-min', v);
    minBtn.textContent = v ? '▴' : '▾';
    minBtn.title = v ? 'Expand' : 'Minimise';
    minBtn.setAttribute('aria-label', v ? 'Expand details' : 'Minimise details');
    minBtn.setAttribute('aria-expanded', String(!v));
    updateSpacer();
  }
  setMinimized(false);

  /** Close the paper view (back to the selection list, if any), else clear the selection. */
  function close() {
    const s = store.get();
    if (s.focus) store.set({ focus: null });
    else if (s.selected.length) store.set({ selected: [], onlySelected: false });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || card.hidden) return;
    if (/** @type {HTMLElement} */ (e.target).closest?.('input, select, textarea')) return;
    close();
  });

  // On phones the card spans the bottom of the screen: pad the page so its end stays reachable.
  const narrow = window.matchMedia('(max-width: 640px)');
  function updateSpacer() {
    document.body.style.paddingBottom = !card.hidden && narrow.matches ? `${card.offsetHeight + 12}px` : '';
  }
  new ResizeObserver(updateSpacer).observe(card);
  narrow.addEventListener('change', updateSpacer);

  /** @param {Point} p */
  const caveatCell = (p) => caveatText(p, cfg);

  /**
   * @param {string} id @param {Set<string>} visible @param {AppState} state
   * @returns {{head: any[], body: any[]}}
   */
  function paperView(id, visible, state) {
    const paper = papers.get(id);
    if (!paper) return { head: [h('p', { class: 'muted' }, 'Paper not found.')], body: [] };
    const pts = byPaper.get(id) ?? [];
    const link = cfg.paper.link?.(paper);

    // Metrics as rows, bands as columns: fits the narrow card even for 8 bands.
    const metricKeys = cfg.info.filter((k) => pts.some((p) => isNum(p[k])));
    const hasTemp = pts.some((p) => isNum(p.tempK));
    const hasCaveats = pts.some((p) => caveatCell(p));
    const multi = pts.length > 1;
    const bandTable = h('table', { class: 'band-table transposed' },
      multi ? h('thead', null, h('tr', null,
        h('th', { scope: 'col' }, ''),
        pts.map((_, i) => h('th', { scope: 'col', class: 'num' }, `Band ${i + 1}`)))) : null,
      h('tbody', null,
        metricKeys.map((k) => h('tr', null,
          h('th', { scope: 'row', title: cfg.metrics[k].description ?? null }, metricLabel(cfg.metrics[k])),
          pts.map((p) => {
            // italic: this value carries a caveat of its own (e.g. simulated)
            const on = Object.entries(metricFlagsOf(p)).filter(([, ms]) => ms.includes(k)).map(([f]) => cfg.caveats?.[f]?.label ?? f);
            return h('td', { class: on.length ? 'num is-sim' : 'num', title: on.length ? on.join(', ') : null }, fmt(p[k], cfg.metrics[k].digits ?? 2));
          }))),
        Object.entries(cfg.categories ?? {}).map(([key, cat]) => h('tr', null,
          h('th', { scope: 'row' }, cat.label),
          pts.map((p) => h('td', { title: cat.descriptions?.[p[key]] ?? null }, String(p[key] ?? '–'))))),
        hasTemp ? h('tr', null, h('th', { scope: 'row' }, 'Temperature (K)'),
          pts.map((p) => h('td', { class: 'num' }, isNum(p.tempK) ? String(p.tempK) : '–'))) : null,
        hasCaveats ? h('tr', { class: 'caveat-row' }, h('th', { scope: 'row' }, 'Caveats'),
          pts.map((p) => h('td', { class: 'caveat-cell' }, caveatCell(p) || '–'))) : null));

    const flagsHere = new Set(pts.flatMap(caveatsOf));
    const caveatNotes = [...flagsHere].filter((f) => cfg.caveats?.[f]).map((f) => {
      const c = /** @type {import('./types.js').Caveat} */ (cfg.caveats?.[f]);
      return h('li', null, h('strong', null, c.label), ': ', c.description);
    });

    const multiEntry = paper.entries.length > 1;
    const entries = paper.entries.map((/** @type {any} */ e) => {
      const details = cfg.paper.entryDetails?.(e) ?? [];
      const notes = Object.entries(e.sources ?? {})
        .map(([metric, source]) => cfg.sourceNotes?.[metric]?.[/** @type {string} */ (source)])
        .filter(Boolean);
      return h('div', { class: 'entry' },
        multiEntry ? h('h3', { class: 'entry-title' }, `Row ${e.row} of the survey`) : null,
        details.length ? h('dl', { class: 'details' }, details.map(([k, v]) => [h('dt', null, k), h('dd', null, v)])) : null,
        notes.length ? h('p', { class: 'muted small' }, notes.join(' ')) : null);
    });

    const importNotes = (warnings.get(id) ?? []).map((w) =>
      h('li', null, w.column ? `${w.column}: ` : '', w.raw ? h('code', null, w.raw) : '', w.raw ? ' → ' : '', w.issue));

    const hidden = !visible.has(id);
    return {
      head: [
        // Full title, wrapped over as many lines as it needs; it is the Google Scholar link.
        h('h2', { class: 'detail-title' }, link
          ? h('a', { href: link, target: '_blank', rel: 'noopener', title: 'Search this title on Google Scholar (new tab)' },
            paper.title ?? id,
            h('span', { class: 'ext', 'aria-hidden': 'true' }, ' ↗'),
            h('span', { class: 'visually-hidden' }, ' (Google Scholar, opens in a new tab)'))
          : paper.title ?? id),
        h('p', { class: 'paper-cite' }, cfg.paper.citation(paper)),
      ],
      body: [
        state.selected.length > 1
          ? h('button', { type: 'button', class: 'link back-link', onclick: () => store.set({ focus: null }) }, `← All ${state.selected.length} selected papers`)
          : null,
        hidden ? h('p', { class: 'notice' }, 'Hidden by the current filters.') : null,
        h('div', { class: 'table-wrap compact' }, bandTable),
        caveatNotes.length ? h('ul', { class: 'caveat-notes' }, caveatNotes) : null,
        entries,
        importNotes.length ? h('details', { class: 'data-notes' }, h('summary', null, `Import notes (${importNotes.length})`), h('ul', null, importNotes)) : null,
      ],
    };
  }

  /** @param {string[]} ids @param {Set<string>} visible */
  function selectionView(ids, visible) {
    const shown = ids.slice(0, MAX_LISTED);
    return {
      head: [h('h2', { class: 'detail-title' }, `${ids.length} papers selected`),
        h('p', { class: 'paper-cite' }, 'Click one to see its details.')],
      body: [
        h('ul', { class: 'paper-list' }, shown.map((id) => {
          const p = papers.get(id);
          return h('li', null, h('button', { type: 'button', class: 'link', onclick: () => store.set({ focus: id }) }, p?.title ?? id),
            h('span', { class: 'paper-cite' }, ` ${cfg.paper.citation(p)}${visible.has(id) ? '' : ' (filtered out)'}`));
        })),
        ids.length > MAX_LISTED ? h('p', { class: 'muted' }, `… and ${ids.length - MAX_LISTED} more. The table can show only the selected papers.`) : null,
      ],
    };
  }

  return {
    /** @param {AppState} state @param {Set<string>} visiblePapers */
    render(state, visiblePapers) {
      const key = state.focus ? `p:${state.focus}` : state.selected.length ? `s:${state.selected.join(',')}` : '';
      if (!key) {
        card.hidden = true;
        shownKey = '';
        updateSpacer();
        return;
      }
      if (key !== shownKey) {
        // a new click brings the card back up, scrolled to the top
        shownKey = key;
        setMinimized(false);
        body.scrollTop = 0;
      }
      card.hidden = false;
      const view = state.focus ? paperView(state.focus, visiblePapers, state) : selectionView(state.selected, visiblePapers);
      headText.replaceChildren(...view.head.flat().filter(Boolean));
      body.replaceChildren(...view.body.flat(Infinity).filter(Boolean));
      updateSpacer();
    },
  };
}
