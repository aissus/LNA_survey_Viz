// @ts-check
/** The single app state object, with change subscriptions. */

/** @typedef {import('./types.js').AppState} AppState */

/**
 * @param {AppState} initial
 */
export function createStore(initial) {
  let state = initial;
  /** @type {Set<(s: AppState) => void>} */
  const subscribers = new Set();
  return {
    /** @returns {AppState} */
    get: () => state,
    /** @param {Partial<AppState> | ((s: AppState) => Partial<AppState>)} patch */
    set(patch) {
      const p = typeof patch === 'function' ? patch(state) : patch;
      state = { ...state, ...p };
      for (const fn of subscribers) fn(state);
    },
    /** @param {(s: AppState) => void} fn */
    subscribe(fn) {
      subscribers.add(fn);
      return () => subscribers.delete(fn);
    },
  };
}

/** @typedef {ReturnType<typeof createStore>} Store */
