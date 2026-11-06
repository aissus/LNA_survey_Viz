// @ts-check
/**
 * Pure data functions: joining the build output into plot points, derived
 * metrics, filtering and sorting. No DOM access, so Node can unit-test it.
 */

/** @typedef {import('./types.js').SurveyConfig} SurveyConfig */
/** @typedef {import('./types.js').Point} Point */

/** @param {unknown} v @returns {v is number} */
export function isNum(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

/**
 * Evaluate the survey's derived metrics (e.g. FOM) on a point, in place.
 * A derived value that is not a finite number becomes null.
 * @template {Record<string, any>} T
 * @param {T} p
 * @param {SurveyConfig} cfg
 * @returns {T}
 */
export function applyDerived(p, cfg) {
  for (const [key, fn] of Object.entries(cfg.derived ?? {})) {
    const v = fn(p);
    /** @type {Record<string, any>} */ (p)[key] = isNum(v) ? v : null;
  }
  return p;
}

/**
 * Join data/<survey>.json (papers -> entries -> points) into flat points.
 * @param {any} json
 * @param {SurveyConfig} cfg
 */
export function joinData(json, cfg) {
  /** @type {Map<string, any>} */
  const papers = new Map(json.papers.map((/** @type {any} */ p) => [p.id, p]));

  /** @type {Map<string, any[]>} */
  const warnings = new Map();
  for (const w of json.warnings ?? []) {
    const list = warnings.get(w.paper) ?? [];
    list.push(w);
    warnings.set(w.paper, list);
  }

  /** @type {Point[]} */
  const points = json.points.map((/** @type {any} */ raw) => {
    const paper = papers.get(raw.paper);
    const entry = paper.entries[raw.entry];
    // caveat -> metrics it applies to; a derived metric inherits its inputs' caveats
    /** @type {Record<string, string[]>} */
    const metricFlags = {};
    for (const [flag, keys] of Object.entries(entry.metric_flags ?? {})) {
      metricFlags[flag] = withDependents(/** @type {string[]} */ (keys), cfg);
    }
    /** @type {Point} */
    const p = {
      ...raw,
      id: `${raw.paper}.${raw.entry}.${raw.band}`,
      paperId: raw.paper,
      group: entry.group,
      nBands: entry.n_bands,
      sources: entry.sources ?? {},
      flags: raw.flags ?? [],
      tempK: raw.temp_k ?? null,
      metricFlags,
      simMetrics: metricFlags.simulated ?? [],
      manual: false,
    };
    for (const f of cfg.paperFields ?? []) p[f] = paper[f];
    return applyDerived(p, cfg);
  });

  /** @type {Map<string, Point[]>} */
  const byPaper = new Map();
  for (const p of points) {
    const list = byPaper.get(p.paperId) ?? [];
    list.push(p);
    byPaper.set(/** @type {string} */ (p.paperId), list);
  }
  return { papers, points, byPaper, warnings, meta: json.meta };
}

/**
 * Simulated metrics plus every derived metric computed from one of them
 * (a FOM that uses a simulated IIP3 is itself simulated).
 * @param {string[]} metrics
 * @param {SurveyConfig} cfg
 * @returns {string[]}
 */
export function withDependents(metrics, cfg) {
  if (!metrics.length) return [];
  const out = new Set(metrics);
  for (const [key, m] of Object.entries(cfg.metrics ?? {})) {
    if (m.dependsOn?.some((d) => out.has(d))) out.add(key);
  }
  return [...out];
}

/**
 * Is this point a caveat on a plot of x vs y? Flagged points always are;
 * individually simulated values only on the plots that show them.
 * @param {Point} p @param {string[]} keys metric keys shown
 */
export function hasCaveat(p, keys) {
  return (p.flags?.length ?? 0) > 0 || keys.some((k) => p.simMetrics?.includes(k));
}

/**
 * Drop points carrying an excluded flag. Excluding "simulated" also blanks
 * individually simulated values (and what is derived from them) on points
 * that are otherwise measured, so "measured only" really means that.
 * @param {Point[]} points
 * @param {string[]} exclude
 */
export function excludeCaveats(points, exclude) {
  if (!exclude?.length) return points;
  const ex = new Set(exclude);
  /** @type {Point[]} */
  const out = [];
  for (const p of points) {
    if (p.flags?.some((/** @type {string} */ f) => ex.has(f))) continue;
    if (ex.has('simulated') && p.simMetrics?.length) {
      /** @type {Point} */
      const q = { ...p, simMetrics: [], maskedSim: p.simMetrics };
      for (const k of p.simMetrics) q[k] = null;
      out.push(q);
    } else {
      out.push(p);
    }
  }
  return out;
}

/**
 * [min, max] of the finite values of `key`, or null if there are none.
 * @param {Point[]} points
 * @param {string} key
 * @param {boolean} [positiveOnly] ignore values <= 0 (for log scales)
 * @returns {[number, number] | null}
 */
export function extent(points, key, positiveOnly = false) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of points) {
    const v = p[key];
    if (isNum(v) && (!positiveOnly || v > 0)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo <= hi ? [lo, hi] : null;
}

/**
 * Points inside every narrowed range. A point with no value for a metric is
 * kept: a paper that did not report IIP3 is not hidden by an IIP3 range.
 * @param {Point[]} points
 * @param {Record<string, [number, number]>} ranges
 */
export function filterByRanges(points, ranges) {
  const rs = Object.entries(ranges ?? {});
  if (rs.length === 0) return points;
  return points.filter((p) => rs.every(([k, [lo, hi]]) => !isNum(p[k]) || (p[k] >= lo && p[k] <= hi)));
}

/**
 * @param {Point[]} points
 * @param {string[] | null} groups  null = all groups
 */
export function filterByGroups(points, groups) {
  if (!groups) return points;
  const enabled = new Set(groups);
  return points.filter((p) => enabled.has(p.group));
}

/**
 * Groups present in the data: config order first, unknown groups after, A-Z.
 * @param {Point[]} points
 * @param {string[]} order
 */
export function groupsInData(points, order) {
  const present = new Set(points.map((p) => p.group));
  const extra = [...present].filter((g) => !order.includes(g)).sort();
  return [...order.filter((g) => present.has(g)), ...extra];
}

/**
 * Stable sort; missing values always last, whatever the direction.
 * @template T
 * @param {T[]} rows
 * @param {(row: T) => unknown} valueOf
 * @param {1|-1} dir
 */
export function sortBy(rows, valueOf, dir) {
  return rows
    .map((row, i) => ({ row, i, v: valueOf(row) }))
    .sort((a, b) => {
      const ma = a.v == null || a.v === '';
      const mb = b.v == null || b.v === '';
      if (ma || mb) return ma === mb ? a.i - b.i : ma ? 1 : -1;
      const c = typeof a.v === 'number' && typeof b.v === 'number'
        ? a.v - b.v
        : String(a.v).localeCompare(String(b.v));
      return c * dir || a.i - b.i;
    })
    .map((x) => x.row);
}
