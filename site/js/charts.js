// @ts-check
/**
 * Plotly figures.
 *
 * - The main figure draws every panel of `cfg.panels` in ONE Plotly figure.
 *   Panels that share an x metric get `matches`, so Plotly keeps their zoom in
 *   sync itself (no relayout listeners, no timers).
 * - The explorer is one free "any metric vs any metric" panel.
 *
 * Highlighting is done by re-rendering with per-point size/opacity arrays;
 * Plotly.react keeps it cheap and `uirevision` keeps the user's zoom.
 */

import { escapeHtml, fmtWithUnit, metricLabel } from './format.js';
import { hasCaveat, isNum } from './model.js';

/** @typedef {import('./types.js').SurveyConfig} SurveyConfig */
/** @typedef {import('./types.js').AppState} AppState */
/** @typedef {import('./types.js').Point} Point */
/** @typedef {import('./types.js').PanelSpec} PanelSpec */

/**
 * @typedef {object} RenderCtx
 * @property {SurveyConfig} cfg
 * @property {AppState} state
 * @property {Point[]} points        range-filtered points of every group
 * @property {Point[]} manual        user-added points
 * @property {Map<string, any>} papers
 * @property {string[]} groups       every group in the data, display order
 * @property {Set<string>} enabled   groups switched on
 * @property {Set<string>} selected  selected paper ids
 */

/** @typedef {PanelSpec & {logX?: boolean, logY?: boolean, color?: string}} PanelDraw */

// Keep in sync with the tokens in css/app.css.
export const THEME = {
  surface: '#fcfcfb',
  text: '#0b0b0b',
  text2: '#52514e',
  border: '#d9d8d3',
  grid: 'rgba(11, 11, 11, 0.09)',
  gridMinor: 'rgba(11, 11, 11, 0.045)',
  axis: 'rgba(11, 11, 11, 0.35)',
  missing: '#b8b7b1',
  font: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
};
// Sequential blue ramp (light step kept >= 2:1 against the surface for markers).
const SEQUENTIAL = [[0, '#86b6ef'], [0.5, '#2a78d6'], [1, '#0d366b']];
const SIZE = { base: 9, focus: 15, manual: 16 };
const OPACITY = { normal: 0.85, dimmed: 0.15 };
export const MANUAL_GROUP = '__manual';
const CAVEAT_KEY = '__caveats';

/** @param {SurveyConfig} cfg @param {string} key */
export const defaultLog = (cfg, key) => cfg.metrics[key]?.scale === 'log';

/** @param {unknown} v @param {boolean|undefined} log */
const usable = (v, log) => isNum(v) && (!log || v > 0);

/** @type {WeakMap<Point, string>} */
const hoverCache = new WeakMap();

/** @param {Point} p @param {RenderCtx} ctx */
function hoverText(p, ctx) {
  const cached = hoverCache.get(p);
  if (cached) return cached;
  const { cfg } = ctx;
  let head;
  if (p.manual) {
    head = `<b>${escapeHtml(p.label)}</b><br>Your point`;
  } else {
    const paper = ctx.papers.get(p.paperId);
    const t = paper.title ?? '';
    const title = t.length > 72 ? `${t.slice(0, 70)}…` : t;
    head = `<b>${escapeHtml(title)}</b><br>${escapeHtml(cfg.paper.citation(paper))}`;
    if (p.nBands > 1) head += ` · band ${p.band + 1} of ${p.nBands}`;
  }
  const parts = cfg.hover.map((k) => {
    const m = cfg.metrics[k];
    return `${escapeHtml(m.short ?? m.label)} ${escapeHtml(fmtWithUnit(p[k], m))}`;
  });
  const lines = [];
  for (let i = 0; i < parts.length; i += 3) lines.push(parts.slice(i, i + 3).join(' · '));
  const notes = caveatText(p, cfg);
  const text = `${head}<br>${lines.join('<br>')}${p.manual ? '' : `<br>${escapeHtml(p.group)}`}` +
    (notes ? `<br><i>${escapeHtml(notes)}</i>` : '');
  hoverCache.set(p, text);
  return text;
}

/**
 * "Simulated IIP3, FOM · Cryogenic (4 K)" for a point, or "" when it has no caveats.
 * @param {Point} p @param {SurveyConfig} cfg
 */
export function caveatText(p, cfg) {
  const parts = (p.flags ?? []).map((/** @type {string} */ f) => {
    const label = cfg.caveats?.[f]?.label ?? f;
    return f === 'cryogenic' && isNum(p.tempK) ? `${label} (${p.tempK} K)` : label;
  });
  const sim = p.simMetrics?.length ? p.simMetrics : p.maskedSim;
  if (sim?.length && !p.flags?.includes('simulated')) {
    const names = sim.filter((/** @type {string} */ k) => k !== 'nfmin' || !sim.includes('nf'))
      .map((/** @type {string} */ k) => cfg.metrics[k]?.short ?? cfg.metrics[k]?.label ?? k);
    parts.push(`Simulated ${names.join(', ')}${p.maskedSim ? ' (hidden)' : ''}`);
  }
  return parts.join(' · ');
}

/**
 * Per-point marker arrays. Points with a caveat for the metrics on this plot
 * are hollow ("-open" symbol, drawn as an outline in the group colour); the
 * focused paper is larger with a dark outline; a selection dims the rest.
 * @param {Point[]} pts @param {RenderCtx} ctx
 * @param {string} symbol group symbol @param {string[]} keys metrics on this plot
 * @param {string} stroke outline colour for hollow markers
 */
function markerStyle(pts, ctx, symbol, keys, stroke) {
  const focus = ctx.state.focus;
  const hasSelection = ctx.selected.size > 0;
  /** @type {string[]} */ const sym = [];
  /** @type {number[]} */ const size = [];
  /** @type {number[]} */ const opacity = [];
  /** @type {number[]} */ const width = [];
  /** @type {string[]} */ const color = [];
  for (const p of pts) {
    const isFocus = focus != null && p.paperId === focus;
    const hollow = hasCaveat(p, keys);
    sym.push(hollow ? `${symbol}-open` : symbol);
    size.push(isFocus ? SIZE.focus : SIZE.base);
    opacity.push(isFocus || !hasSelection ? (isFocus ? 1 : OPACITY.normal) : ctx.selected.has(/** @type {string} */ (p.paperId)) ? 1 : OPACITY.dimmed);
    width.push(isFocus ? 2.5 : hollow ? 1.6 : 1);
    color.push(isFocus ? THEME.text : hollow ? stroke : THEME.surface);
  }
  return { symbol: sym, size, opacity, line: { width, color } };
}

/** @param {Point[]} pts @param {string} x @param {string} y @param {RenderCtx} ctx */
function xy(pts, x, y, ctx) {
  return {
    x: pts.map((p) => p[x]),
    y: pts.map((p) => p[y]),
    customdata: pts.map((p) => p.id),
    text: pts.map((p) => hoverText(p, ctx)),
    hovertemplate: '%{text}<extra></extra>',
  };
}

/**
 * Colour-by-metric helpers: values (log10 for log metrics) and the colour bar.
 * @param {string} key @param {RenderCtx} ctx
 */
function colorScale(key, ctx) {
  const m = ctx.cfg.metrics[key];
  const log = defaultLog(ctx.cfg, key);
  const value = (/** @type {Point} */ p) => (usable(p[key], log) ? (log ? Math.log10(p[key]) : p[key]) : null);
  const vals = ctx.points.map(value).filter(isNum);
  if (!vals.length) return null;
  const cmin = Math.min(...vals);
  const cmax = Math.max(...vals);
  /** @type {Record<string, any>} */
  const colorbar = { title: { text: metricLabel(m), side: 'right' }, thickness: 12, len: 0.8, outlinewidth: 0, tickfont: { size: 11, color: THEME.text2 } };
  if (log && m.ticks) {
    const ticks = m.ticks.filter((t) => Math.log10(t) >= cmin - 1e-9 && Math.log10(t) <= cmax + 1e-9);
    colorbar.tickvals = ticks.map(Math.log10);
    colorbar.ticktext = ticks.map(String);
  }
  return { value, cmin, cmax, colorbar, label: m.label };
}

/**
 * Traces of one panel: one per group (+ user points). Disabled groups stay in
 * the legend as "legendonly" so they can be clicked back on.
 * @param {PanelDraw} spec
 * @param {RenderCtx} ctx
 * @param {{xaxis?: string, yaxis?: string}} axes
 * @param {boolean} showlegend
 */
export function panelTraces(spec, ctx, axes, showlegend) {
  const { cfg } = ctx;
  const { x, y, logX, logY } = spec;
  const ok = (/** @type {Point} */ p) => usable(p[x], logX) && usable(p[y], logY);
  const scale = spec.color && spec.color !== 'group' ? colorScale(spec.color, ctx) : null;
  /** @type {any[]} */
  const traces = [];

  const keys = [x, y, ...(scale ? [/** @type {string} */ (spec.color)] : [])];
  let anyHollow = false;
  for (const g of ctx.groups) {
    const pts = ctx.points.filter((p) => p.group === g && ok(p));
    const base = { type: 'scatter', mode: 'markers', legendgroup: g, visible: ctx.enabled.has(g) ? true : 'legendonly', ...axes };
    const symbol = cfg.groups.symbols[g] ?? 'circle';
    if (ctx.enabled.has(g)) anyHollow ||= pts.some((p) => hasCaveat(p, keys));
    if (!scale) {
      const color = cfg.groups.colors[g] ?? cfg.groups.fallbackColor;
      traces.push({ ...base, name: g, showlegend, ...xy(pts, x, y, ctx), marker: { color, ...markerStyle(pts, ctx, symbol, keys, color) } });
      continue;
    }
    const coloured = pts.filter((p) => scale.value(p) != null);
    const missing = pts.filter((p) => scale.value(p) == null);
    traces.push({
      ...base, name: g, showlegend, ...xy(coloured, x, y, ctx),
      marker: { color: coloured.map(scale.value), colorscale: SEQUENTIAL, cmin: scale.cmin, cmax: scale.cmax, showscale: false, ...markerStyle(coloured, ctx, symbol, keys, THEME.text2) },
    });
    if (missing.length) {
      traces.push({
        ...base, name: `${g}, no ${scale.label}`, showlegend: false, ...xy(missing, x, y, ctx),
        marker: { color: THEME.missing, ...markerStyle(missing, ctx, symbol, keys, THEME.missing) },
      });
    }
  }

  if (anyHollow && showlegend && cfg.caveats) {
    // Legend key only: explains the hollow markers.
    const names = Object.values(cfg.caveats).map((c) => c.label.toLowerCase());
    const list = names.length > 1 ? `${names.slice(0, -1).join(', ')} or ${names.at(-1)}` : names[0];
    traces.push({
      type: 'scatter', mode: 'markers', x: [null], y: [null], ...axes, hoverinfo: 'skip',
      name: `Hollow: ${list}`, legendgroup: CAVEAT_KEY, showlegend: true, legendrank: 2000,
      marker: { symbol: 'circle-open', size: SIZE.base, color: THEME.text2, line: { width: 1.6, color: THEME.text2 } },
    });
  }

  if (scale) {
    // A data-less trace carries the colour bar, so hiding a group never hides it.
    traces.push({
      type: 'scatter', mode: 'markers', x: [null], y: [null], ...axes, showlegend: false, hoverinfo: 'skip',
      marker: { color: [scale.cmin], colorscale: SEQUENTIAL, cmin: scale.cmin, cmax: scale.cmax, showscale: true, colorbar: scale.colorbar },
    });
  }

  const mine = ctx.manual.filter(ok);
  if (mine.length) {
    traces.push({
      type: 'scatter', mode: 'markers', name: 'Your points', legendgroup: MANUAL_GROUP, showlegend, ...axes,
      ...xy(mine, x, y, ctx),
      marker: { symbol: 'star', size: SIZE.manual, color: THEME.text, line: { width: 1.5, color: THEME.surface } },
    });
  }
  return traces;
}

/**
 * @param {RenderCtx} ctx @param {string} key @param {boolean} log
 * @param {Record<string, any>} [extra]
 */
function axis(ctx, key, log, extra = {}) {
  const m = ctx.cfg.metrics[key];
  /** @type {Record<string, any>} */
  const a = {
    title: { text: metricLabel(m), font: { size: 12, color: THEME.text2 }, standoff: 8 },
    type: log ? 'log' : 'linear',
    showgrid: true,
    gridcolor: THEME.grid,
    zeroline: false,
    showline: true,
    linecolor: THEME.axis,
    ticks: 'outside',
    tickcolor: THEME.axis,
    ticklen: 4,
    tickfont: { size: 11, color: THEME.text2 },
    minor: ctx.state.minorGrid ? { showgrid: true, gridcolor: THEME.gridMinor, ticks: '' } : { showgrid: false },
    automargin: true,
    ...extra,
  };
  if (log && m.ticks) {
    a.tickmode = 'array';
    a.tickvals = m.ticks;
    a.ticktext = m.ticks.map(String);
  }
  return a;
}

/** @param {RenderCtx} ctx @param {number} selectionRev */
function baseLayout(ctx, selectionRev) {
  return {
    paper_bgcolor: THEME.surface,
    plot_bgcolor: THEME.surface,
    font: { family: THEME.font, size: 12, color: THEME.text },
    hovermode: 'closest',
    dragmode: 'zoom',
    hoverlabel: { bgcolor: '#ffffff', bordercolor: THEME.border, font: { family: THEME.font, size: 12, color: THEME.text }, align: 'left' },
    legend: { orientation: 'h', x: 0, xanchor: 'left', y: 1.02, yanchor: 'bottom', font: { size: 12, color: THEME.text2 } },
    // A new selection revision drops Plotly's own selection state; ours is in the marker arrays.
    selectionrevision: selectionRev,
  };
}

/** Number of columns for the main figure at a given width. @param {number} width */
export const columnsFor = (width) => (width >= 820 ? 2 : 1);

/**
 * All `cfg.panels` in one figure.
 * @param {RenderCtx} ctx @param {number} width @param {number} selectionRev
 */
export function linkedFigure(ctx, width, selectionRev) {
  const { cfg } = ctx;
  const panels = cfg.panels;
  const cols = columnsFor(width);
  const rows = Math.ceil(panels.length / cols);
  const panelH = cols === 2 ? 320 : 280;
  const gapY = 100;
  const mt = 76;
  const mb = 56;
  const height = rows * panelH + (rows - 1) * gapY + mt + mb;
  const plotH = height - mt - mb;
  const ph = panelH / plotH;
  const gy = gapY / plotH;
  const gx = cols === 2 ? 0.1 : 0;
  const pw = (1 - gx * (cols - 1)) / cols;

  /** @type {Record<string, any>} */
  const layout = baseLayout(ctx, selectionRev);
  layout.height = height;
  layout.margin = { t: mt, b: mb, l: 64, r: 16 };
  layout.legend.y = 1 + 46 / plotH;
  layout.uirevision = `${ctx.state.zoomRev}:${cols}`;

  /** @type {any[]} */ const data = [];
  /** @type {any[]} */ const annotations = [];
  /** @type {Record<string, string>} */ const firstAxis = {};
  panels.forEach((spec, i) => {
    const n = i === 0 ? '' : String(i + 1);
    const row = Math.floor(i / cols);
    const col = i % cols;
    const top = 1 - row * (ph + gy);
    const logX = defaultLog(cfg, spec.x);
    const logY = defaultLog(cfg, spec.y);
    const x0 = col * (pw + gx);
    const xa = axis(ctx, spec.x, logX, { anchor: `y${n}`, domain: [x0, Math.min(1, x0 + pw)] });
    const matchKey = `${spec.x}:${logX}`;
    if (firstAxis[matchKey]) xa.matches = firstAxis[matchKey];
    else firstAxis[matchKey] = `x${n}`;
    layout[`xaxis${n}`] = xa;
    layout[`yaxis${n}`] = axis(ctx, spec.y, logY, { anchor: `x${n}`, domain: [Math.max(0, top - ph), top] });
    data.push(...panelTraces({ ...spec, logX, logY }, ctx, { xaxis: `x${n}`, yaxis: `y${n}` }, i === 0));
    annotations.push({
      text: `<b>${escapeHtml(spec.title ?? `${cfg.metrics[spec.y].label} vs ${cfg.metrics[spec.x].label}`)}</b>`,
      xref: `x${n} domain`, yref: `y${n} domain`, x: 0, y: 1, xanchor: 'left', yanchor: 'bottom', yshift: 8,
      showarrow: false, font: { size: 13, color: THEME.text },
    });
  });
  layout.annotations = annotations;
  return { data, layout, cols };
}

/**
 * A single panel (the explorer, or one main panel exported on its own).
 * @param {PanelDraw} spec @param {RenderCtx} ctx @param {number} selectionRev
 * @param {{title?: string, height?: number}} [opts]
 */
export function panelFigure(spec, ctx, selectionRev, opts = {}) {
  const { cfg } = ctx;
  const logX = spec.logX ?? defaultLog(cfg, spec.x);
  const logY = spec.logY ?? defaultLog(cfg, spec.y);
  const hasScale = !!spec.color && spec.color !== 'group';
  /** @type {Record<string, any>} */
  const layout = baseLayout(ctx, selectionRev);
  layout.height = opts.height ?? 460;
  layout.margin = { t: opts.title ? 92 : 48, b: 56, l: 64, r: hasScale ? 24 : 16 };
  layout.xaxis = axis(ctx, spec.x, logX);
  layout.yaxis = axis(ctx, spec.y, logY);
  layout.uirevision = `${ctx.state.zoomRev}:${spec.x}:${spec.y}:${logX}:${logY}`;
  if (opts.title) {
    layout.title = { text: `<b>${escapeHtml(opts.title)}</b>`, x: 0, xanchor: 'left', xref: 'paper', y: 0.98, yanchor: 'top', yref: 'container', font: { size: 15 } };
  }
  return { data: panelTraces({ ...spec, logX, logY }, ctx, {}, true), layout };
}

/**
 * @param {{onPoint: (id: string) => void, onSelect: (ids: string[]) => void, onLegend: (group: string, isolate: boolean) => void}} handlers
 */
export function createCharts(handlers) {
  /** @type {WeakSet<HTMLElement>} */
  const bound = new WeakSet();
  let selectionKey = '';
  let selectionRev = 0;

  /** @param {string} filename */
  const plotConfig = (filename) => ({
    displaylogo: false,
    responsive: true,
    toImageButtonOptions: { format: 'svg', filename, scale: 2 },
    modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian'],
  });

  /** @param {any} gd */
  function bind(gd) {
    if (bound.has(gd)) return;
    bound.add(gd);
    gd.on('plotly_click', (/** @type {any} */ ev) => {
      const id = ev?.points?.[0]?.customdata;
      if (typeof id === 'string') handlers.onPoint(id);
    });
    gd.on('plotly_selected', (/** @type {any} */ ev) => {
      if (!ev) return; // fired with nothing when a drag ends outside the data
      handlers.onSelect(ev.points.map((/** @type {any} */ p) => p.customdata).filter((/** @type {any} */ c) => typeof c === 'string'));
    });
    gd.on('plotly_deselect', () => handlers.onSelect([]));
    gd.on('plotly_legendclick', (/** @type {any} */ ev) => {
      handlers.onLegend(ev.data[ev.curveNumber].legendgroup, false);
      return false;
    });
    gd.on('plotly_legenddoubleclick', (/** @type {any} */ ev) => {
      handlers.onLegend(ev.data[ev.curveNumber].legendgroup, true);
      return false;
    });
  }

  return {
    /**
     * @param {RenderCtx} ctx
     * @param {HTMLElement} linkedEl
     * @param {HTMLElement} explorerEl
     */
    render(ctx, linkedEl, explorerEl) {
      const key = [...ctx.selected].sort().join(',');
      if (key !== selectionKey) {
        selectionKey = key;
        selectionRev += 1;
      }
      const linked = linkedFigure(ctx, linkedEl.clientWidth, selectionRev);
      // responsive mode fills the container, so the container must carry the height
      linkedEl.style.height = `${linked.layout.height}px`;
      Plotly.react(linkedEl, linked.data, linked.layout, plotConfig(`${ctx.cfg.id}-panels`));
      const e = ctx.state.explorer;
      const ex = panelFigure({ x: e.x, y: e.y, color: e.color, logX: e.logX ?? undefined, logY: e.logY ?? undefined }, ctx, selectionRev);
      explorerEl.style.height = `${ex.layout.height}px`;
      Plotly.react(explorerEl, ex.data, ex.layout, plotConfig(`${ctx.cfg.id}-explorer`));
      bind(linkedEl);
      bind(explorerEl);
      return linked.cols;
    },
  };
}
