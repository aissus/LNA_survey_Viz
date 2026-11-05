// @ts-check
/**
 * Shareable state in the query string. Pure functions, no DOM access.
 *
 *   s=lna            survey (omitted for the default survey)
 *   g=SiGe,CMOS SOI  enabled groups (omitted = all, empty = none)
 *   x=simulated      excluded caveat flags
 *   r.freq=1~100     narrowed range per metric
 *   sel=ab12cd34,..  selected paper ids      f=ab12cd34  focused paper
 *   minor=1          minor grid              only=1      table: selected only
 *   ex=x,y,color     explorer axes           exlog=10    explorer log X/Y (1/0/- = default)
 *   sort=-year       table sort (leading "-" = descending)
 */

/** @typedef {import('./types.js').AppState} AppState */

const SEP = '~';

/** @param {number} v */
const num = (v) => String(Number(v.toPrecision(6)));
/** @param {boolean|null} v */
const flag = (v) => (v == null ? '-' : v ? '1' : '0');
/** @param {string|undefined} c @returns {boolean|null} */
const unflag = (c) => (c === '1' ? true : c === '0' ? false : null);

/** Survey id requested by the URL, if any. @param {string} search */
export function surveyFromSearch(search) {
  return new URLSearchParams(search).get('s');
}

/**
 * @param {AppState} s
 * @param {AppState} d defaults
 * @returns {string} query string without the leading "?"
 */
export function encodeState(s, d) {
  const q = new URLSearchParams();
  if (s.survey !== d.survey) q.set('s', s.survey);
  if (s.groups) q.set('g', s.groups.join(','));
  if (s.exclude.length) q.set('x', s.exclude.join(','));
  for (const [k, [lo, hi]] of Object.entries(s.ranges)) q.set(`r.${k}`, `${num(lo)}${SEP}${num(hi)}`);
  if (s.selected.length) q.set('sel', s.selected.join(','));
  if (s.focus) q.set('f', s.focus);
  if (s.minorGrid) q.set('minor', '1');
  if (s.onlySelected) q.set('only', '1');
  const e = s.explorer;
  const de = d.explorer;
  if (e.x !== de.x || e.y !== de.y || e.color !== de.color) q.set('ex', [e.x, e.y, e.color].join(','));
  if (e.logX != null || e.logY != null) q.set('exlog', flag(e.logX) + flag(e.logY));
  if (s.sort.key !== d.sort.key || s.sort.dir !== d.sort.dir) q.set('sort', (s.sort.dir < 0 ? '-' : '') + s.sort.key);
  return q.toString();
}

/**
 * Read state from a query string on top of the defaults. Anything unknown or
 * malformed is ignored, so an old or hand-edited link never breaks the page.
 * @param {string} search
 * @param {AppState} d defaults
 * @param {{metrics: Set<string>, sortKeys: Set<string>, colorKeys: Set<string>, caveats?: Set<string>}} known
 * @returns {AppState}
 */
export function decodeState(search, d, known) {
  const q = new URLSearchParams(search);
  /** @type {AppState} */
  const s = structuredClone(d);
  const list = (/** @type {string} */ k) => (q.get(k) ?? '').split(',').map((x) => x.trim()).filter(Boolean);

  if (q.has('s')) s.survey = /** @type {string} */ (q.get('s'));
  if (q.has('g')) s.groups = list('g');
  s.exclude = list('x').filter((k) => known.caveats?.has(k));
  for (const [k, v] of q) {
    if (!k.startsWith('r.') || !known.metrics.has(k.slice(2))) continue;
    const [lo, hi] = v.split(SEP).map(Number);
    if (Number.isFinite(lo) && Number.isFinite(hi) && lo <= hi) s.ranges[k.slice(2)] = [lo, hi];
  }
  s.selected = list('sel').filter((id) => /^[0-9a-f]{8}$/.test(id));
  const f = q.get('f');
  if (f && /^[0-9a-f]{8}$/.test(f)) s.focus = f;
  s.minorGrid = q.get('minor') === '1';
  s.onlySelected = q.get('only') === '1';

  const [x, y, color] = list('ex');
  if (x && known.metrics.has(x)) s.explorer.x = x;
  if (y && known.metrics.has(y)) s.explorer.y = y;
  if (color && known.colorKeys.has(color)) s.explorer.color = color;
  const exlog = q.get('exlog') ?? '';
  s.explorer.logX = unflag(exlog[0]);
  s.explorer.logY = unflag(exlog[1]);

  const sort = q.get('sort');
  if (sort) {
    const key = sort.replace(/^-/, '');
    if (known.sortKeys.has(key)) s.sort = { key, dir: sort.startsWith('-') ? -1 : 1 };
  }
  return s;
}
