# Survey Visualizer — project brief

## Goal
A static website, hosted entirely on **GitHub Pages**, that visualizes IC block surveys
(published-paper performance data). Today: **LNA**. Next: **PA**, then **phased array**,
possibly more blocks later. Adding a new survey should mean adding its data + a small config,
not copying code.

Hard constraint: **no server at runtime.** Everything the browser needs must be static files.
Python is fine at *build time* (GitHub Actions), not at serve time.

## Current state of this repo (aissus/LNA_survey_Viz)
Milestone 1 is implemented on branch `refactor/static-modular-viewer` (see README for layout).
- `scripts/` (Python, run with `uv`): `parsing.py` (pure helpers), `survey_sources.py` (how to read
  each xlsx), `build_data.py` (xlsx -> `site/data/<id>.json`), `fetch_vendor.py` (pinned Plotly).
- Data model in the JSON: paper (year+first author+title; may span several xlsx rows, because older
  survey versions list one row per band) -> entries (one per row) -> points (one per band/mode).
- `site/`: generic viewer as native ES modules (`js/`), survey display config in `surveys/lna.js`,
  survey registry in `surveys/index.js`, Plotly cartesian 3.7.0 vendored in `vendor/`.
- Tests: `uv run pytest` (pipeline, real xlsx); `npm test` (node:test, viewer + config consistency);
  `npm run typecheck` (JSDoc via tsc, CI only needs Node). Locally Node can come from
  `uvx --from "nodejs-wheel>=22,<25" node` (no system Node on this machine).
- `.github/workflows/deploy.yml`: test + build, publish only `site/` on `main`.
- Removed: `app_dash.py`, root `index.html`, `data/lna_data.csv`, `requirements.txt`.
- Decisions: FOM axis is log (values span ~6 decades); 3 process groups get hues, smaller groups
  share a grey and differ by marker shape (colour-blind-safe scatter palette).
- Caveats (`scripts/caveats.py`): points carry flags simulated / system / cryogenic (cryogenic per band
  from "T:300K/4K" notes); entries carry `sim_metrics` for individually simulated values. The viewer
  draws them hollow and can exclude them (`x=` in the URL). Derived metrics declare `dependsOn` so a
  FOM built from a simulated value counts as simulated.
- Paper details live in a floating card (`site/js/infopanel.js`) that stays until the next click.
- PA (ETH PA-Survey-v11, ~6.1k points, 10 sheets) and phased array (ETH v3, 151 points) are added on
  branch `surveys/phased-array-and-pa` (kept off `main` until ETH answers the republishing question).
  Their xlsx are downloaded at build time (`SurveySource.url` + pinned `sha256`), not committed.
  Per-survey rules live in `scripts/source_<id>.py` (`adjust(ctx)` hook); registry in `scripts/registry.py`.
- Viewer config also has `categories` (filters; one may set marker shape), `refs` (reference curves on
  panels), value-level caveats (`entry.metric_flags`) and `webgl` (PA: scattergl + gl2d bundle; exports
  redraw as SVG). Only 3 hues validate for an all-pairs scatter: more groups = grey + shape.
- Shell gotcha on this machine: the Bash tool collapses `\\` in heredocs, so write files containing
  backslashes (regexes, CSS escapes) with the file tools, not `cat <<EOF`.
- Roadmap agreed with the owner (2026-11-05): pinned card + caveats (done) -> phased array + PA (done
  on the branch above) -> publication export dialog (IEEE sizes, paper style, PDF, "This work" labels,
  numbered refs) -> overlays (SOTA envelope) and polish.

## Known bugs (verified) — all fixed in milestone 1
1. **Multi-band values are mis-parsed on the live site.** 44 of 472 rows contain values like
   `7,35/10,65/14,75/11,45` (European decimal commas, `/`-separated bands) in fc, f@NFmin, NF,
   NFmin, Gain, IIP3, Power. JS `parseFloat("7,35/10,65")` returns `7`, so these papers are
   plotted with wrong values and no warning.
2. **`parseFloat(x) || null` drops genuine zeros** (e.g. IIP3 = 0 dBm, 2 rows) and turns
   `"0,85"` into missing.
3. **The two implementations disagree:** Dash uses only `fc` (~142 plotted papers);
   `index.html` falls back to `f@NFmin` (~471). Decide the rule explicitly
   (proposal: use `fc`, fall back to `f@NFmin`, and record which was used).
4. Excel encoding artifacts in titles, e.g. `Variable_x0002_Gain` (2 rows).
5. README install step says `cd examples/Survey_Viz` — leftover path.
6. Hard-coded fallbacks like `yearMax: 2025` in `getFilters()`.

## Target architecture
1. **Build-time data pipeline (Python, run in GitHub Actions):**
   `scripts/build_data.py` reads each survey xlsx → normalizes decimal commas → splits
   multi-band entries into one point per band (sharing a `paper_id`, with a `band` index) →
   cleans text → writes `site/data/<survey>.json`. Log/flag rows it can't parse instead of
   dropping them silently. Unit-test the parsing on the real problem values above.
2. **One generic viewer + a config per survey** (`site/surveys/lna.js`, later `pa.js`, ...):
   columns, units, which axes are log, derived metrics (LNA FOM =
   G_lin·f / (P·(F_lin−1)); PA will need PAE/Psat-based FOMs), filters, process grouping,
   default plots. The viewer code must not contain LNA-specific logic.
3. **Split the single HTML file into ES modules** (data loading, filters, charts, selection,
   info panel, manual points). Native ES modules work on Pages with no bundler.
   Only introduce Vite/TypeScript if there's a clear need.
4. **Plotly:** use the `plotly-cartesian` dist bundle (only scatter is needed) instead of the full one.
5. **Nice to have:** filter/selection state in the URL (shareable links); manual points in
   `localStorage` (wrapped in try/catch).
6. **Deploy workflow:** install Python deps → run `build_data.py` → upload only `site/`
   (not the repo root, not the xlsx unless intended).

## First milestone
- [x] `scripts/build_data.py` + tests for the parsing bugs above; produce `site/data/lna.json`.
- [x] Generic viewer as ES modules reproducing every current feature, driven by `surveys/lna.js`.
- [x] Check the multi-band papers now plot at correct values (spot-check the
      `7,35/10,65/14,75/11,45` row: 4 points at 7.35/10.65/14.75/11.45 GHz).
- [x] Update `deploy.yml` to build then publish `site/`.
- [x] Remove `app_dash.py`, fix README.
- [x] Verify locally with a static server — serve the repo root and open `/site/` to mimic the
      Pages subpath; `fetch` does not work from `file://`.

Then: add the PA survey: its xlsx under `sources/pa/`, an entry in `scripts/survey_sources.py`,
`site/surveys/pa.js`, and a line in `site/surveys/index.js`.

## Open questions for the owner
- Do the ETH IDEAS survey terms allow republishing the raw xlsx/CSV? If not, publish only
  the processed JSON or link to the source.
- Where do the PA and phased-array survey files come from, and in what format?
- ~~Multi-band papers: every band or one point?~~ Decided: every band, highlighted together.
