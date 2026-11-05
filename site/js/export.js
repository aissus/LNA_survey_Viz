// @ts-check
/**
 * SVG / PNG export of the whole figure, the explorer, or one panel on its own.
 * Exports always draw SVG markers: a WebGL (scattergl) plot is redrawn off
 * screen first, so a vector export never embeds a bitmap of the points.
 */

import { linkedFigure, panelFigure } from './charts.js';

/** @typedef {import('./charts.js').RenderCtx} RenderCtx */

const PANEL_SIZE = { width: 900, height: 560 };

/** @param {string} s */
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

/**
 * The current zoom of panel i of the main figure, or null when autoscaled.
 * @param {any} gd @param {string} axisName e.g. "xaxis3"
 */
function currentRange(gd, axisName) {
  const ax = gd?._fullLayout?.[axisName];
  return ax && ax.autorange === false && Array.isArray(ax.range) ? [...ax.range] : null;
}

/**
 * @param {string} which   "all" | "explorer" | panel index
 * @param {'svg'|'png'} format
 * @param {RenderCtx} ctx
 * @param {HTMLElement} linkedEl
 * @param {HTMLElement} explorerEl
 */
export async function exportImage(which, format, ctx, linkedEl, explorerEl) {
  const scale = format === 'png' ? 2 : 1;
  const id = ctx.cfg.id;
  const svgCtx = { ...ctx, gl: false };
  if (which === 'all' || which === 'explorer') {
    const gd = /** @type {any} */ (which === 'all' ? linkedEl : explorerEl);
    const filename = `${id}-${which === 'all' ? 'all-panels' : 'explorer'}`;
    const size = { width: gd.clientWidth, height: gd._fullLayout?.height ?? gd.clientHeight };
    if (!ctx.gl) return Plotly.downloadImage(gd, { format, scale, filename, ...size });
    const e = ctx.state.explorer;
    const fig = which === 'all'
      ? linkedFigure(svgCtx, size.width, 0)
      : panelFigure({ x: e.x, y: e.y, color: e.color, logX: e.logX ?? undefined, logY: e.logY ?? undefined }, svgCtx, 0);
    for (const axisName of Object.keys(fig.layout).filter((k) => /^[xy]axis\d*$/.test(k))) {
      const range = currentRange(gd, axisName);
      if (range) Object.assign(fig.layout[axisName], { range, autorange: false });
    }
    return offscreen(fig, size, { format, scale, filename });
  }

  const i = Number(which);
  const spec = ctx.cfg.panels[i];
  if (!spec) return undefined;
  const fig = panelFigure(spec, svgCtx, 0, { title: spec.title, height: PANEL_SIZE.height });
  const n = i === 0 ? '' : String(i + 1);
  for (const [src, dst] of [[`xaxis${n}`, 'xaxis'], [`yaxis${n}`, 'yaxis']]) {
    const range = currentRange(linkedEl, src);
    if (range) Object.assign(fig.layout[dst], { range, autorange: false });
  }
  return offscreen(fig, PANEL_SIZE, { format, scale, filename: `${id}-${slug(spec.title ?? `${spec.y}-vs-${spec.x}`)}` });
}

/**
 * Draw a figure in a hidden div, download it, and clean up.
 * @param {{data: any[], layout: Record<string, any>}} fig
 * @param {{width: number, height: number}} size
 * @param {{format: 'svg'|'png', scale: number, filename: string}} opts
 */
async function offscreen(fig, size, opts) {
  const tmp = document.createElement('div');
  Object.assign(tmp.style, { position: 'fixed', left: '-10000px', top: '0', width: `${size.width}px`, height: `${size.height}px` });
  document.body.append(tmp);
  try {
    await Plotly.newPlot(tmp, fig.data, { ...fig.layout, ...size }, { staticPlot: true });
    await Plotly.downloadImage(tmp, { ...opts, ...size });
  } finally {
    Plotly.purge(tmp);
    tmp.remove();
  }
  return undefined;
}
