// @ts-check
/**
 * The surveys this site can show. Only ids listed here are ever loaded, so a
 * hand-edited ?s= value cannot make the page import an arbitrary module.
 * Adding a survey: add its entry here + site/surveys/<id>.js, and its source
 * in scripts/survey_sources.py.
 */

/** @type {Record<string, {label: string, load: () => Promise<{default: import('../js/types.js').SurveyConfig}>}>} */
export const SURVEYS = {
  lna: { label: 'LNA', load: () => import('./lna.js') },
  pa: { label: 'PA', load: () => import('./pa.js') },
};

export const DEFAULT_SURVEY = 'lna';
