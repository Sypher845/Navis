/**
 * time.js — Temporal utility functions
 */

/**
 * Format ISO datetime string to human-readable IST format.
 */
export function formatTime(isoStr) {
  if (!isoStr) return '—';
  try {
    const d = new Date(isoStr);
    return d.toLocaleString('en-IN', {
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
    });
  } catch {
    return isoStr;
  }
}

/**
 * Format duration in minutes to human-readable string.
 */
export function formatDuration(minutes) {
  if (minutes == null || minutes <= 0) return '—';
  if (minutes < 60) return `${Math.round(minutes)}m`;
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

/**
 * Get day of week name from 0-6 index.
 */
const DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
export function dayName(index) {
  return DAY_NAMES[index] || '—';
}

/**
 * Get hour label in 12-hour format.
 */
export function hourLabel(hour) {
  if (hour === 0) return '12 AM';
  if (hour < 12) return `${hour} AM`;
  if (hour === 12) return '12 PM';
  return `${hour - 12} PM`;
}

/**
 * Check if two ISO date strings are on the same calendar day.
 */
export function sameDay(iso1, iso2) {
  if (!iso1 || !iso2) return false;
  return iso1.substring(0, 10) === iso2.substring(0, 10);
}
