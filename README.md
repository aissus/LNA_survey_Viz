# IC Survey Visualizer

Interactive plots of published IC block surveys, served as a static site on
GitHub Pages: **LNA**, **PA** and **phased arrays**, picked from the header. The
viewer is generic, so a new survey is its spreadsheet plus two small config files.

**[Open the visualizer →](https://aissus.github.io/LNA_survey_Viz/)**

## Data sources and citation

All three surveys are compiled by **ETH Zürich Integrated Devices, Electronics,
And Systems (IDEAS)**. If you use them, please cite them as the surveys ask:

- **LNA** ([ideas.ethz.ch/Surveys/lna-survey.html](https://ideas.ethz.ch/Surveys/lna-survey.html)):
  Hua Wang, Argyro Kamperi, Boce Lin, Basem Abdelaziz Abdelmagid Ali, "Low Noise
  Amplifier Noise Figure Survey 2018-present," [Online]. Available:
  https://ideas.ethz.ch/Surveys/lna-survey.html
- **PA**, the "ETH PA Survey" (v11, 14.08.2026;
  [ideas.ethz.ch/Surveys/pa-survey.html](https://ideas.ethz.ch/Surveys/pa-survey.html)):
  Hua Wang et al., "Power Amplifiers Performance Survey 2000-Present," [Online].
  Available: https://ideas.ethz.ch/Surveys/pa-survey.html
- **Phased arrays** (v3.0, 2024-01-17;
  [ideas.ethz.ch/Surveys/phased-array-survey.html](https://ideas.ethz.ch/Surveys/phased-array-survey.html)):
  Hua Wang, Yuqi Liu, Naga Sasikanth Mannem, "Phased Array Survey 2010-Present,"
  [Online]. Available: https://ideas.ethz.ch/Surveys/phased-array-survey.html

The PA and phased-array spreadsheets are not in this repository: the build
downloads them from ETH and checks them against a pinned sha256.

## What the site does

- Five frequency plots (NF, gain, power, IIP3, FOM) plus gain vs NF, in one
  figure. The frequency axes are linked, so zooming one zooms all five.
- An explorer plot: any metric against any other, coloured by process or by a
  third metric.
- Filters: process group, and ranges for frequency (log scale), year, NF, gain
  and power.
- Click a point or a table row to pin its paper in a floating card: every band
  it reports, process details, caveats, a Google Scholar link, and any notes from
  the data import. The card stays up while you scroll until the next click
  replaces it. Close it with × or Esc, or minimise it to its title. On phones it
  is a panel along the bottom of the screen.
- Use the box or lasso tool to select several papers. The selection is
  highlighted in every plot and in the table.
- Caveats are marked: simulated results, system-level measurements (a receiver
  front-end, array or system, so gain, NF and power include more than the LNA),
  and cryogenic measurements. They are drawn as hollow markers, labelled in hover
  text, the table and the card, and can be hidden. "Measured standalone only"
  hides all three. Hiding "Simulated" also blanks individually simulated values,
  such as a simulated IIP3, and the FOM computed from them.
- Your own design points. They are stored in your browser only (localStorage),
  and the FOM is computed for them too.
- Shareable links: filters, selection, explorer axes and table sort live in the
  URL.
- Export: the whole figure, one panel on its own (keeping its zoom), or the
  explorer, as SVG or PNG.

## How the data is read

`scripts/build_data.py` turns the survey spreadsheet into `site/data/lna.json`
at build time:

- European decimal commas: `0,85` → 0.85. A genuine `0` stays 0 and is not
  treated as missing.
- Multi-band cells: `7,35/10,65/14,75/11,45` becomes four points that share one
  paper. A single value in another column applies to every band. A column whose
  count fits neither case is left out for that paper and flagged.
- Several values at one frequency (operating modes) become several points at
  that frequency.
- Older survey versions list a multi-band paper as one row per band. Rows with
  the same year, first author and title are one paper, so selecting it
  highlights all of its points.
- Frequency is `fc`, else `f@NFmin`. NF is `NF`, else `NFmin`. The info panel
  says which one was used.
- Placeholders (`N/A`, `*`, `-`) are missing values. Annotations (`9,9*`,
  `0,41(0,38)`, `>25`) keep the number and are noted.
- Process strings are grouped by ordered rules. "SiGe BiCMOS" is SiGe, not
  CMOS Bulk.
- Caveats come from the Comment column and titles (`scripts/caveats.py`):
  - `Simulated` marks the whole entry.
  - `IIP3 simulated` or `Simulated NF` marks only those values.
  - `Receiver Frontend`, `System`, `Rx consumption`, `Array` and `Beamformer`
    mark system-level entries.
  - `T:300K/4K` gives one temperature per band, so only the cold bands are
    marked cryogenic.
  - A cryogenic, quantum or qubit title marks the entry cryogenic when there is
    no temperature note.
- Every spreadsheet row is accounted for: it is plotted, or listed with a
  reason. The page footer lists all import notes, and the build fails if a row
  goes missing.

### PA survey

- The workbook has one sheet per technology (CMOS, SiGe, GaN, GaAs, InP, LDMOS,
  Others) plus product datasheets, frequency multipliers and THz oscillators.
  They are read as one table of about 6,100 points; each point records its
  sheet. Sheet headers that differ in spelling are mapped to one name, and
  vertically merged cells are filled down.
- Colour is the technology family (CMOS, SiGe, III-V; LDMOS and others in grey).
  Marker shape is the exact technology: bulk, SOI or FinFET CMOS; GaN, GaAs or
  InP.
- Efficiencies that the survey reports as drain, collector or system
  efficiency rather than PAE are marked "Not PAE", and so is any FOM computed
  from them. Other caveats come from the notes: system-level, off-chip output
  network, pulsed, estimated and simulated.
- Impossible values, such as a positive EVM in dB or a 2065 dB gain, are left
  out and listed in the import notes. So are datasheets whose band exists only
  in a text note.
- ITRS FOM = Psat[W]·G·PAE·f²[GHz²], with gain linear and PAE as a fraction.
  Power density = Psat / PA core area (W/mm²).
- With thousands of points the plots are drawn with WebGL, and the table shows
  200 rows at a time. Exports are redrawn as SVG.

### Phased-array survey

- Each spreadsheet row (an array, or one of its TX/RX modes, bands or sizes) is
  one point.
- Types such as "TX FE" are split into the type (TX / RX / TRX / oscillator /
  relay; the marker shape) and "front-end only" (a filter).
- An RX Pdc of 0 next to a TX Pdc means the two were reported together (the
  survey's Readme 6.4): the zero is left out and TX Pdc is marked.
- For passive reflectarrays and RIS, the survey's "TX conversion gain" and "TX
  efficiency" are antenna gain (dBi) and aperture efficiency, shown as their own
  metrics.
- Elements are the survey's TX + RX count; passive relays, which have no TX/RX
  chains, count their unit cells.
- The Note column marks passive arrays, commercial beamformer ICs, non-silicon
  parts and cryogenic operation, plus, on the affected values only, DC power
  that covers more than the array, TX power that includes RX, and array results
  taken from another paper.
- Aperture area, area per antenna and thermal densities are recomputed in the
  viewer from the sheet's inputs (they match the sheet exactly). EIRP/Pdc and
  Pout/Pdc follow the survey's Readme 6.9. Reference lines show the (λ/2)² grid
  cell and 20·log₁₀N EIRP scaling.
- Cells with spreadsheet errors (#VALUE!, #REF!, #DIV/0!) are read as missing
  and listed in the import notes.

## Repository layout

```
sources/lna/LNA_Survey_v3.0.xlsx   LNA spreadsheet (input; not published to Pages)
sources/<id>/                      spreadsheets downloaded at build time (gitignored)
scripts/parsing.py                 pure parsing helpers (decimal commas, bands, text, Excel errors)
scripts/caveats.py                 simulated / system-level / cryogenic notes -> flags
scripts/survey_sources.py          SurveySource + RowContext, and the LNA source
scripts/source_<id>.py             how to read each further survey (+ its adjust hook)
scripts/registry.py                every survey the build knows
scripts/build_data.py              xlsx -> site/data/<survey>.json
scripts/fetch_vendor.py            reproduce / upgrade the vendored Plotly bundles
tests/                             pytest (pipeline) and tests/js (viewer, node:test)
site/                              everything that is published
  index.html, css/app.css
  js/                              generic viewer (ES modules, no bundler)
  surveys/index.js                 survey registry (the picker in the header)
  surveys/<id>.js                  one display config per survey
  surveys/palette.js               process colours shared by every survey
  vendor/plotly-*.min.js           pinned Plotly bundles: cartesian (SVG) and gl2d (WebGL)
  data/                            generated, not committed
```

## Run it locally

Prerequisites: [uv](https://docs.astral.sh/uv/). Node is only needed for the
viewer's type check and unit tests.

```bash
uv run scripts/build_data.py
uv run python -m http.server 8000
```

Then open <http://localhost:8000/site/>. Serving from the repo root on purpose
mimics the GitHub Pages subpath (`/LNA_survey_Viz/`). `fetch` does not work from
`file://`, so the page must be served.

## Tests

```bash
uv run pytest          # data pipeline, against the real spreadsheet
npm ci
npm run typecheck      # JSDoc types of site/js and site/surveys
npm test               # viewer unit tests + survey config consistency
```

## Adding a survey

1. Describe how to read the spreadsheet in `scripts/source_<id>.py` (a
   `SurveySource`): sheet, columns, fallbacks, which columns are multi-band,
   and process grouping rules. Either commit the file under `sources/<id>/`, or
   give its `url` and `sha256`: the build then downloads it (and CI caches it)
   instead of republishing it in the repository. Add one line to
   `scripts/registry.py`.
2. Rules the generic parser cannot know go in the source's `adjust(ctx)` hook,
   which sees every parsed row: a unit that differs on some rows, a zero that
   means "included elsewhere", caveats written in a notes column. A caveat can
   cover a whole point (`ctx.flags`) or only some values (`ctx.metric_flags`).
3. Describe how to show it in `site/surveys/<id>.js`: metrics with units and
   scales, derived metrics such as FOMs (with `dependsOn`), filters, panels
   (optionally with reference curves, `refs`), categories (each gets a filter;
   one can pick the marker shape), table columns, caveats, and group colours
   from `site/surveys/palette.js`. Set `webgl: true` for thousands of points.
   Register it in `site/surveys/index.js`.
4. Run `uv run pytest`, `npm test` and the local server. The config test fails
   if the config references a metric or category value it does not define.

Colours: only three hues stay distinct for colour-blind readers when every
pair can sit next to each other in a scatter (checked with the palette
validator). Further groups share a neutral grey and are told apart by marker
shape, the hover text, the table and the filters.

## Deployment

`.github/workflows/deploy.yml` runs on every push and pull request. It tests
and builds the data, type-checks and tests the viewer, and on `main` publishes
only `site/` to GitHub Pages. In the repository settings, Pages must use
**GitHub Actions** as its source.

## License

Viewer code: provided for educational and research purposes. Survey data:
belongs to ETH Zürich IDEAS; see their page for terms. Plotly.js is MIT-licensed.
