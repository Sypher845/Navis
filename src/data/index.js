/**
 * data/index.js — In-memory query engine for ASTraM incident data
 *
 * Loads the preprocessed JSON and provides fast query methods
 * for spatial, temporal, and categorical filtering.
 */

import { haversine, boundingBox, inBoundingBox } from '../utils/geo.js';

let _data = null;     // raw data object
let _incidents = [];   // array of all incidents
let _indices = {};     // pre-built indices for fast lookups

/**
 * Load and index the preprocessed data.
 */
export async function loadData() {
  const response = await fetch('/data.json');
  _data = await response.json();
  _incidents = _data.incidents;

  // Build indices
  _indices = {
    byCause: groupBy(_incidents, 'cause'),
    byCorridor: groupBy(_incidents, 'corridor'),
    byZone: groupBy(_incidents, 'zone'),
    byType: groupBy(_incidents, 'type'),
    byPriority: groupBy(_incidents, 'priority'),
    byStatus: groupBy(_incidents, 'status'),
    byHour: groupBy(_incidents, 'hourIST'),
    byDayOfWeek: groupBy(_incidents, 'dayOfWeek'),
    byPoliceStation: groupBy(_incidents, 'policeStation'),
  };

  return _data.meta;
}

/**
 * Get all incidents.
 */
export function getAllIncidents() {
  return _incidents;
}

/**
 * Get metadata.
 */
export function getMeta() {
  return _data ? _data.meta : null;
}

/**
 * Get incidents near a point within a radius (km).
 * Uses bounding box pre-filter + Haversine for accuracy.
 */
export function getIncidentsNear(lat, lng, radiusKm) {
  const bbox = boundingBox(lat, lng, radiusKm);
  const results = [];
  for (const inc of _incidents) {
    if (inBoundingBox(inc.lat, inc.lng, bbox)) {
      const dist = haversine(lat, lng, inc.lat, inc.lng);
      if (dist <= radiusKm) {
        results.push({ ...inc, _distance: dist });
      }
    }
  }
  return results;
}

/**
 * Get incidents matching a set of filters.
 * Filters: { cause, corridor, zone, type, priority, hourRange, dayOfWeek }
 */
export function queryIncidents(filters = {}) {
  let results = _incidents;

  if (filters.cause) {
    const causes = Array.isArray(filters.cause) ? filters.cause : [filters.cause];
    results = results.filter((inc) => causes.includes(inc.cause));
  }

  if (filters.corridor) {
    results = results.filter((inc) => inc.corridor === filters.corridor);
  }

  if (filters.zone) {
    results = results.filter((inc) => inc.zone === filters.zone);
  }

  if (filters.type) {
    results = results.filter((inc) => inc.type === filters.type);
  }

  if (filters.priority) {
    results = results.filter((inc) => inc.priority === filters.priority);
  }

  if (filters.hourRange) {
    const [start, end] = filters.hourRange;
    results = results.filter(
      (inc) => inc.hourIST != null && inc.hourIST >= start && inc.hourIST <= end
    );
  }

  if (filters.dayOfWeek != null) {
    results = results.filter((inc) => inc.dayOfWeek === filters.dayOfWeek);
  }

  return results;
}

/**
 * Get unique corridor names sorted by frequency.
 */
export function getCorridors() {
  const entries = Object.entries(_indices.byCorridor || {});
  return entries
    .map(([name, items]) => ({ name, count: items.length }))
    .sort((a, b) => b.count - a.count);
}

/**
 * Get unique zone names sorted by frequency.
 */
export function getZones() {
  const entries = Object.entries(_indices.byZone || {});
  return entries
    .map(([name, items]) => ({ name, count: items.length }))
    .sort((a, b) => b.count - a.count);
}

/**
 * Get cause distribution.
 */
export function getCauseDistribution() {
  return Object.entries(_indices.byCause || {})
    .map(([cause, items]) => ({ cause, count: items.length }))
    .sort((a, b) => b.count - a.count);
}

/**
 * Get hourly distribution.
 */
export function getHourlyDistribution() {
  const dist = new Array(24).fill(0);
  for (const inc of _incidents) {
    if (inc.hourIST != null) dist[inc.hourIST]++;
  }
  return dist;
}

/**
 * Get median resolution time for a given filter set (in minutes).
 */
export function getMedianResolution(filters = {}) {
  const incidents = queryIncidents(filters);
  const times = incidents
    .map((inc) => inc.resolutionMin)
    .filter((t) => t != null && t > 0 && t < 1440) // exclude > 24h outliers
    .sort((a, b) => a - b);

  if (times.length === 0) return null;
  const mid = Math.floor(times.length / 2);
  return times.length % 2 === 0
    ? (times[mid - 1] + times[mid]) / 2
    : times[mid];
}

/**
 * Get incidents that are event-related (public_event, procession, etc).
 */
export function getEventIncidents() {
  const eventCauses = ['public_event', 'procession', 'vip_movement', 'protest'];
  return _incidents.filter((inc) => eventCauses.includes(inc.cause));
}

/**
 * Get incidents index by a field.
 */
export function getIndex(field) {
  return _indices[field] || {};
}

// --- Internal helpers ---

function groupBy(items, key) {
  const groups = {};
  for (const item of items) {
    const val = item[key];
    if (val == null) continue;
    if (!groups[val]) groups[val] = [];
    groups[val].push(item);
  }
  return groups;
}
