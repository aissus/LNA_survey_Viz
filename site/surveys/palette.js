// @ts-check
/**
 * Colours shared by every survey, so a process looks the same in each of them.
 * These three hues are validated together for a scatter (every pair stays
 * distinct for colour-blind readers). Smaller groups share NEUTRAL and are told
 * apart by marker shape.
 */

/** @type {Record<string, string>} */
export const PROCESS_COLORS = {
  'CMOS Bulk': '#2a78d6',
  SiGe: '#eb6834',
  'CMOS SOI': '#1baf7a',
};

export const NEUTRAL = '#6f6e69';
