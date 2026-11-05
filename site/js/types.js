// @ts-check
/**
 * Shared JSDoc types for the viewer. No runtime code.
 *
 * A survey is described by one SurveyConfig (site/surveys/<id>.js). The viewer
 * modules only ever read survey-specific facts from that object.
 *
 * @typedef {object} Metric
 * @property {string} label            e.g. "Noise figure"
 * @property {string} [short]          compact label for hover text, e.g. "NF"
 * @property {string} [unit]           e.g. "dB"
 * @property {'linear'|'log'} [scale]  default axis / slider scale
 * @property {number} [digits]         decimals when displayed
 * @property {number[]} [ticks]        explicit tick values for log axes
 * @property {string} [description]    shown as a tooltip next to the label
 * @property {string[]} [dependsOn]    for derived metrics: inputs (a simulated input makes it simulated)
 *
 * @typedef {object} Caveat
 * @property {string} label            e.g. "Simulated"
 * @property {string} short            badge text, e.g. "sim"
 * @property {string} description      why it is not directly comparable
 *
 * @typedef {object} PanelSpec
 * @property {string} x
 * @property {string} y
 * @property {string} [title]
 *
 * @typedef {object} ManualField
 * @property {string} metric
 * @property {boolean} [required]
 * @property {string} [placeholder]
 *
 * @typedef {object} SurveyConfig
 * @property {string} id                       matches data/<id>.json
 * @property {string} title
 * @property {string} [subtitle]
 * @property {{label: string, url: string, citation?: string}} source
 * @property {Record<string, Metric>} metrics
 * @property {string[]} paperFields            paper fields copied onto every point (e.g. "year")
 * @property {Record<string, (p: Record<string, any>) => (number|null)>} [derived]
 * @property {{order: string[], colors: Record<string, string>, symbols: Record<string, string>, fallbackColor: string, label?: string}} groups
 * @property {string[]} filters                metrics that get a range slider
 * @property {PanelSpec[]} panels              main figure; panels sharing an x metric zoom together
 * @property {{metrics: string[], x: string, y: string, color: string}} explorer
 * @property {string[]} hover                  metrics in the hover text
 * @property {string[]} table                  metric columns of the data table
 * @property {string[]} info                   metrics in the info panel band table
 * @property {{citation: (paper: any) => string, link?: (paper: any) => (string|null), entryDetails?: (entry: any) => Array<[string, string]>}} paper
 * @property {Record<string, Record<string, string>>} [sourceNotes]  metric -> source column -> explanation
 * @property {Record<string, Caveat>} [caveats]  point flags set by the build (simulated, system, ...);
 *   flagged points are drawn hollow and can be excluded
 * @property {{fields: ManualField[]}} manual
 *
 * A plotted point: one band of one entry of one paper (or a user-added point).
 * @typedef {Record<string, any> & {id: string, paperId: (string|null), group: string, manual: boolean}} Point
 *
 * @typedef {object} ExplorerState
 * @property {string} x
 * @property {string} y
 * @property {string} color          "group" or a metric key
 * @property {boolean|null} logX     null = metric default
 * @property {boolean|null} logY
 *
 * @typedef {object} AppState
 * @property {string} survey
 * @property {string[]|null} groups               enabled groups; null = all
 * @property {string[]} exclude                   caveat flags hidden from plots and table
 * @property {Record<string, [number, number]>} ranges   only narrowed ranges are present
 * @property {string[]} selected                  paper ids (box / lasso selection)
 * @property {string|null} focus                  paper id shown in the info panel
 * @property {boolean} minorGrid
 * @property {ExplorerState} explorer
 * @property {{key: string, dir: 1|-1}} sort
 * @property {boolean} onlySelected               table shows only selected papers
 * @property {number} zoomRev                     bump to reset zoom
 * @property {number} manualRev                   bump when user points change
 */
export {};
