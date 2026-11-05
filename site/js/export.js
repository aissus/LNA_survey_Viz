// @ts-check
/** SVG / PNG export of the whole figure, the explorer, or one panel on its own. */

import { panelFigure } from './charts.js';

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
  if (which === 'all' || which === 'explorer') {
    const gd = /** @type {any} */ (which === 'all' ? linkedEl : explorerEl);
    return Plotly.downloadImage(gd, {
      format, scale, filename: `${id}-${which === 'all' ? 'all-panels' : 'explorer'}`,
      width: gd.clientWidth, height: gd._fullLayout?.height ?? gd.clientHeight,
    });
  }

  const i = Number(which);
  const spec = ctx.cfg.panels[i];
  if (!spec) return undefined;
  const fig = panelFigure(spec, ctx, 0, { title: spec.title, height: PANEL_SIZE.height });
  const n = i === 0 ? '' : String(i + 1);
  for (const [src, dst] of [[`xaxis${n}`, 'xaxis'], [`yaxis${n}`, 'yaxis']]) {
    const range = currentRange(linkedEl, src);
    if (range) Object.assign(fig.layout[dst], { range, autorange: false });
  }

  const tmp = document.createElement('div');
  Object.assign(tmp.style, { position: 'fixed', left: '-10000px', top: '0', width: `${PANEL_SIZE.width}px`, height: `${PANEL_SIZE.height}px` });
  document.body.append(tmp);
  try {
    await Plotly.newPlot(tmp, fig.data, { ...fig.layout, ...PANEL_SIZE }, { staticPlot: true });
    await Plotly.downloadImage(tmp, { format, scale, filename: `${id}-${slug(spec.title ?? `${spec.y}-vs-${spec.x}`)}`, ...PANEL_SIZE });
  } finally {
    Plotly.purge(tmp);
    tmp.remove();
  }
  return undefined;
}
