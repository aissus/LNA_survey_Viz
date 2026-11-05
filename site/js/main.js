// @ts-check
/** Boot: pick the survey, load its config and data, wire the modules to one state. */

import { DEFAULT_SURVEY, SURVEYS } from '../surveys/index.js';
import { MANUAL_GROUP, columnsFor, createCharts } from './charts.js';
import { exportImage } from './export.js';
import { $, h, metricLabel } from './format.js';
import { createFilters } from './filters.js';
import { createDetailCard } from './infopanel.js';
import { createManualPoints } from './manual.js';
import { excludeCaveats, filterByRanges, groupsInData, joinData } from './model.js';
import { createStore } from './state.js';
import { createTable } from './table.js';
import { decodeState, encodeState, surveyFromSearch } from './url.js';

/** @typedef {import('./types.js').AppState} AppState */
/** @typedef {import('./types.js').SurveyConfig} SurveyConfig */

/** @param {string | Node} message */
function fatal(message) {
  const el = $('#status');
  el.replaceChildren(h('strong', null, 'Could not load the survey. '), message);
  el.classList.add('is-error');
  $('#hint').hidden = true;
}

/** @param {SurveyConfig} cfg @param {any} meta */
function renderChrome(cfg, meta, surveyId) {
  document.title = cfg.title;
  $('#survey-title').textContent = cfg.title;
  $('#survey-subtitle').textContent = cfg.subtitle ?? '';
  $('#survey-source').replaceChildren('Data: ', h('a', { href: cfg.source.url, target: '_blank', rel: 'noopener' }, cfg.source.label));
  $('#citation').textContent = cfg.source.citation ?? '';
  const built = meta?.built_at ? new Date(meta.built_at).toISOString().slice(0, 10) : 'unknown date';
  $('#build-info').textContent = `Built from ${meta?.source_file ?? 'source'} on ${built}: ${meta?.rows_in} spreadsheet rows, ${meta?.papers_out} papers, ${meta?.points_out} points.`;

  const ids = Object.keys(SURVEYS);
  if (ids.length > 1) {
    const select = /** @type {HTMLSelectElement} */ ($('#survey-select'));
    select.replaceChildren(...ids.map((id) => h('option', { value: id, selected: id === surveyId }, SURVEYS[id].label)));
    select.addEventListener('change', () => { location.search = `?s=${encodeURIComponent(select.value)}`; });
    $('#survey-picker').hidden = false;
  }
}

/** @param {any[]} warnings */
function renderImportNotes(warnings) {
  const counts = { dropped: 0, nulled: 0, note: 0 };
  for (const w of warnings) counts[/** @type {'dropped'|'nulled'|'note'} */ (w.severity)] += 1;
  const summary = `Import notes: ${counts.dropped} rows not plotted, ${counts.nulled} values not plotted, ${counts.note} annotations`;
  const order = { dropped: 0, nulled: 1, note: 2 };
  const sorted = [...warnings].sort((a, b) => order[/** @type {'note'} */ (a.severity)] - order[/** @type {'note'} */ (b.severity)] || a.row - b.row);
  $('#import-notes').replaceChildren(
    h('summary', null, summary),
    h('ul', null, sorted.map((w) => h('li', null,
      h('span', { class: `sev sev-${w.severity}` }, w.severity), ` row ${w.row}`, w.column ? `, ${w.column}` : '', ': ',
      w.raw ? [h('code', null, w.raw), ' → '] : '', w.issue))));
}

async function main() {
  if (typeof Plotly === 'undefined') return fatal('The plotting library did not load.');

  const requested = surveyFromSearch(location.search);
  const surveyId = requested && Object.hasOwn(SURVEYS, requested) ? requested : DEFAULT_SURVEY;
  const cfg = (await SURVEYS[surveyId].load()).default;

  let json;
  try {
    const res = await fetch(new URL(`../data/${cfg.id}.json`, import.meta.url));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    json = await res.json();
  } catch (err) {
    return fatal(`data/${cfg.id}.json is missing or unreadable (${err instanceof Error ? err.message : err}). ` +
      'Build it with "uv run scripts/build_data.py" and serve the site over http, not file://.');
  }

  const model = joinData(json, cfg);
  const groups = groupsInData(model.points, cfg.groups.order);
  renderChrome(cfg, model.meta, surveyId);
  renderImportNotes(json.warnings ?? []);

  /** @type {AppState} */
  const defaults = {
    survey: DEFAULT_SURVEY,
    groups: null,
    exclude: [],
    ranges: {},
    selected: [],
    focus: null,
    minorGrid: false,
    explorer: { x: cfg.explorer.x, y: cfg.explorer.y, color: cfg.explorer.color, logX: null, logY: null },
    sort: { key: 'year', dir: -1 },
    onlySelected: false,
    zoomRev: 0,
    manualRev: 0,
  };
  const known = {
    metrics: new Set(Object.keys(cfg.metrics)),
    sortKeys: new Set(['title', 'year', 'band', 'group', ...cfg.table]),
    colorKeys: new Set(['group', ...cfg.explorer.metrics]),
    caveats: new Set(Object.keys(cfg.caveats ?? {})),
  };
  const initial = decodeState(location.search, defaults, known);
  initial.survey = surveyId;
  initial.selected = initial.selected.filter((id) => model.papers.has(id));
  if (initial.focus && !model.papers.has(initial.focus)) initial.focus = null;
  const store = createStore(initial);

  // On phones the sidebar sits above the plots: start it collapsed.
  if (window.matchMedia('(max-width: 900px)').matches) {
    for (const d of document.querySelectorAll('.sidebar details')) d.removeAttribute('open');
  }

  // --- modules ---------------------------------------------------------------
  const linkedEl = $('#linked-figure');
  const explorerEl = $('#explorer-figure');
  const cardEl = $('#detail-card');
  const filters = createFilters($('#filters'), { cfg, store, points: model.points, groups });
  const table = createTable($('#table'), { cfg, store, papers: model.papers });
  const card = createDetailCard(cardEl, { cfg, store, papers: model.papers, byPaper: model.byPaper, warnings: model.warnings });
  const manual = createManualPoints($('#manual'), { cfg, onChange: () => store.set((s) => ({ manualRev: s.manualRev + 1 })) });

  const charts = createCharts({
    onPoint(id) {
      if (id.startsWith('manual:')) return;
      store.set({ focus: id.split('.')[0] }); // the detail card pins it until the next click
    },
    onSelect(ids) {
      const papers = [...new Set(ids.filter((id) => !id.startsWith('manual:')).map((id) => id.split('.')[0]))];
      store.set((s) => ({ selected: papers, focus: papers.length === 1 ? papers[0] : papers.length ? null : s.focus }));
    },
    onLegend(group, isolate) {
      if (group === MANUAL_GROUP || !groups.includes(group)) return;
      store.set((s) => {
        const on = s.groups ?? groups;
        let next;
        if (isolate) next = on.length === 1 && on[0] === group ? groups : [group];
        else next = on.includes(group) ? on.filter((g) => g !== group) : groups.filter((g) => g === group || on.includes(g));
        return { groups: next.length === groups.length ? null : next };
      });
    },
  });

  // --- explorer controls ---------------------------------------------------
  const exX = /** @type {HTMLSelectElement} */ ($('#ex-x'));
  const exY = /** @type {HTMLSelectElement} */ ($('#ex-y'));
  const exColor = /** @type {HTMLSelectElement} */ ($('#ex-color'));
  const exLogX = /** @type {HTMLInputElement} */ ($('#ex-logx'));
  const exLogY = /** @type {HTMLInputElement} */ ($('#ex-logy'));
  const metricOptions = () => cfg.explorer.metrics.map((k) => h('option', { value: k }, metricLabel(cfg.metrics[k])));
  exX.replaceChildren(...metricOptions());
  exY.replaceChildren(...metricOptions());
  exColor.replaceChildren(h('option', { value: 'group' }, cfg.groups.label ?? 'Group'), ...metricOptions());
  const setExplorer = (/** @type {Partial<AppState['explorer']>} */ patch) => store.set((s) => ({ explorer: { ...s.explorer, ...patch } }));
  exX.addEventListener('change', () => setExplorer({ x: exX.value, logX: null }));
  exY.addEventListener('change', () => setExplorer({ y: exY.value, logY: null }));
  exColor.addEventListener('change', () => setExplorer({ color: exColor.value }));
  exLogX.addEventListener('change', () => setExplorer({ logX: exLogX.checked }));
  exLogY.addEventListener('change', () => setExplorer({ logY: exLogY.checked }));

  // --- toolbar -------------------------------------------------------------
  $('#clear-selection').addEventListener('click', () => store.set({ selected: [], focus: null, onlySelected: false }));
  $('#reset-zoom').addEventListener('click', () => store.set((s) => ({ zoomRev: s.zoomRev + 1 })));
  const minor = /** @type {HTMLInputElement} */ ($('#minor-grid'));
  minor.addEventListener('change', () => store.set({ minorGrid: minor.checked }));

  const exportWhich = /** @type {HTMLSelectElement} */ ($('#export-which'));
  const exportFormat = /** @type {HTMLSelectElement} */ ($('#export-format'));
  exportWhich.replaceChildren(
    h('option', { value: 'all' }, 'All panels'),
    ...cfg.panels.map((p, i) => h('option', { value: String(i) }, p.title ?? `${p.y} vs ${p.x}`)),
    h('option', { value: 'explorer' }, 'Explorer plot'));
  /** @type {import('./charts.js').RenderCtx | null} */
  let lastCtx = null;
  $('#export-btn').addEventListener('click', async () => {
    if (!lastCtx) return;
    try {
      await exportImage(exportWhich.value, /** @type {'svg'|'png'} */ (exportFormat.value), lastCtx, linkedEl, explorerEl);
    } catch (err) {
      flash(`Export failed: ${err instanceof Error ? err.message : err}`);
    }
  });

  const flashEl = $('#flash');
  /** @param {string} msg */
  function flash(msg) {
    flashEl.textContent = msg;
    setTimeout(() => { if (flashEl.textContent === msg) flashEl.textContent = ''; }, 2500);
  }
  $('#copy-link').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(location.href);
      flash('Link copied');
    } catch {
      flash('Copy the address bar to share this view');
    }
  });

  // --- render loop ---------------------------------------------------------
  const statusEl = $('#status');
  const hintEl = $('#hint');
  let lastCols = 0;
  function render() {
    const s = store.get();
    const enabled = new Set(s.groups ?? groups);
    const inRanges = filterByRanges(excludeCaveats(model.points, s.exclude), s.ranges);
    const visible = inRanges.filter((p) => enabled.has(p.group));
    const visiblePapers = new Set(visible.map((p) => /** @type {string} */ (p.paperId)));

    lastCtx = { cfg, state: s, points: inRanges, manual: manual.points(), papers: model.papers, groups, enabled, selected: new Set(s.selected) };
    lastCols = charts.render(lastCtx, linkedEl, explorerEl);
    filters.render(s);
    table.render(visible, s);
    card.render(s, visiblePapers);
    hintEl.hidden = !!(s.focus || s.selected.length);

    exX.value = s.explorer.x;
    exY.value = s.explorer.y;
    exColor.value = s.explorer.color;
    exLogX.checked = s.explorer.logX ?? cfg.metrics[s.explorer.x]?.scale === 'log';
    exLogY.checked = s.explorer.logY ?? cfg.metrics[s.explorer.y]?.scale === 'log';
    minor.checked = s.minorGrid;
    statusEl.textContent = `Showing ${visible.length} of ${model.points.length} points from ${visiblePapers.size} of ${model.papers.size} papers` +
      (s.selected.length ? ` · ${s.selected.length} selected` : '');
  }

  // One render per burst of state changes. requestAnimationFrame alone stalls in
  // hidden or embedded views, so a short timer races it.
  let pending = false;
  const schedule = () => {
    if (pending) return;
    pending = true;
    const run = () => {
      if (!pending) return;
      pending = false;
      cancelAnimationFrame(frame);
      clearTimeout(timer);
      render();
    };
    const frame = requestAnimationFrame(run);
    const timer = window.setTimeout(run, 50);
  };
  let urlTimer = 0;
  store.subscribe((s) => {
    schedule();
    clearTimeout(urlTimer);
    urlTimer = window.setTimeout(() => {
      const q = encodeState(s, defaults);
      history.replaceState(null, '', q ? `?${q}` : location.pathname);
    }, 200);
  });
  new ResizeObserver(() => { if (columnsFor(linkedEl.clientWidth) !== lastCols) schedule(); }).observe(linkedEl);

  render();
}

main().catch((err) => {
  console.error(err);
  fatal(err instanceof Error ? err.message : String(err));
});
