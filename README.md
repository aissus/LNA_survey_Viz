# IC Survey Visualizer

Interactive plots of published IC block surveys, served as a static site on
GitHub Pages. Today: **LNA**. The viewer is generic, so a new survey (PA, phased
array, ...) is its spreadsheet plus two small config files.

**[Open the visualizer →](https://aissus.github.io/LNA_survey_Viz/)**

## Data source and citation

The LNA data is the survey compiled by **ETH Zürich Integrated Devices,
Electronics, And Systems (IDEAS)**:
[ideas.ethz.ch/Surveys/lna-survey.html](https://ideas.ethz.ch/Surveys/lna-survey.html).
If you use it, please cite it as the survey asks:

> Hua Wang, Argyro Kamperi, Boce Lin, Basem Abdelaziz Abdelmagid Ali, "Low Noise
> Amplifier Noise Figure Survey 2018-present," [Online]. Available:
> https://ideas.ethz.ch/Surveys/lna-survey.html

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

## Repository layout

```
sources/lna/LNA_Survey_v3.0.xlsx   survey spreadsheet (input; not published)
scripts/parsing.py                 pure parsing helpers (decimal commas, bands, text)
scripts/caveats.py                 simulated / system-level / cryogenic notes -> flags
scripts/survey_sources.py          how to read each survey's spreadsheet
scripts/build_data.py              xlsx -> site/data/<survey>.json
scripts/fetch_vendor.py            reproduce / upgrade the vendored Plotly bundle
tests/                             pytest (pipeline) and tests/js (viewer, node:test)
site/                              everything that is published
  index.html, css/app.css
  js/                              generic viewer (ES modules, no bundler)
  surveys/index.js, surveys/lna.js survey registry + the LNA display config
  vendor/plotly-cartesian-*.min.js pinned Plotly bundle (scatter only)
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

1. Put the spreadsheet in `sources/<id>/`.
2. Describe how to read it in `scripts/survey_sources.py`: sheet, columns,
   fallbacks, which columns are multi-band, and process grouping rules.
3. Describe how to show it in `site/surveys/<id>.js`: metrics with units and
   scales, derived metrics such as FOMs, filters, panels, table columns, and
   group colours. Register it in `site/surveys/index.js`.
4. Run `uv run pytest`, `npm test` and the local server. The config test fails
   if the config references a metric it does not define.

## Deployment

`.github/workflows/deploy.yml` runs on every push and pull request. It tests
and builds the data, type-checks and tests the viewer, and on `main` publishes
only `site/` to GitHub Pages. In the repository settings, Pages must use
**GitHub Actions** as its source.

## License

Viewer code: provided for educational and research purposes. Survey data:
belongs to ETH Zürich IDEAS; see their page for terms. Plotly.js is MIT-licensed.
