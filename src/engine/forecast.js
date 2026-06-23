/**
 * engine/forecast.js — Event impact prediction engine
 *
 * Predicts the traffic impact of a given event based on historical
 * patterns from the ASTraM dataset. No ML framework — pure statistical
 * inference on actual incident distributions.
 *
 * Algorithm:
 * 1. Find all historical incidents near the event location (within radius)
 * 2. Weight by: event cause similarity, time-of-day match, day-of-week match
 * 3. Compute severity score (0-100) from density + priority + road-closure rate
 * 4. Identify affected corridors ranked by historical incident density
 * 5. Estimate manpower needs from resolution time patterns
 */

import { getIncidentsNear, getMeta, getAllIncidents } from '../data/index.js';

/**
 * Compute the number of calendar days the dataset spans.
 * Used to normalize per-day averages instead of hardcoding.
 */
function getDatasetDays() {
  const meta = getMeta();
  if (!meta || !meta.dateRange || !meta.dateRange.start || !meta.dateRange.end) return 152;
  const start = new Date(meta.dateRange.start);
  const end = new Date(meta.dateRange.end);
  const diffMs = end.getTime() - start.getTime();
  return Math.max(1, Math.round(diffMs / (1000 * 60 * 60 * 24)));
}

// Radius tiers for impact analysis (km)
const INNER_RADIUS = 1.5;
const MIDDLE_RADIUS = 3;
const OUTER_RADIUS = 5;

// Weight multipliers for different factors
const WEIGHTS = {
  causeSimilarity: 0.3,
  timeOfDay: 0.25,
  dayOfWeek: 0.15,
  density: 0.2,
  priorityMix: 0.1,
};

/**
 * Main prediction function.
 *
 * @param {string} eventCause - Type of event (public_event, procession, etc.)
 * @param {number} lat - Event latitude
 * @param {number} lng - Event longitude
 * @param {number} hour - Expected hour (IST, 0-23)
 * @param {number} dayOfWeek - Day of week (0=Mon, 6=Sun)
 * @param {number} durationHours - Expected duration in hours
 * @returns {Object} Prediction result
 */
export function predictImpact(eventCause, lat, lng, hour, dayOfWeek, durationHours = 4) {
  // 1. Gather historical incidents in concentric rings
  const innerIncidents = getIncidentsNear(lat, lng, INNER_RADIUS);
  const middleIncidents = getIncidentsNear(lat, lng, MIDDLE_RADIUS);
  const outerIncidents = getIncidentsNear(lat, lng, OUTER_RADIUS);

  // 2. Compute sub-scores

  // Density score: how many incidents historically occur here?
  const densityScore = computeDensityScore(innerIncidents.length, middleIncidents.length, outerIncidents.length);

  // Cause similarity: what fraction of nearby incidents are similar causes?
  const relatedCauses = getRelatedCauses(eventCause);
  const causeSimilarityScore = computeCauseSimilarity(middleIncidents, relatedCauses);

  // Time match: do incidents here spike at this hour?
  const timeScore = computeTimeScore(middleIncidents, hour);

  // Day match: are certain days worse?
  const dayScore = computeDayScore(middleIncidents, dayOfWeek);

  // Priority mix: what fraction are High priority?
  const priorityScore = computePriorityScore(middleIncidents);

  // 3. Weighted severity score (0–100)
  const severity = Math.round(
    densityScore * WEIGHTS.density * 100 +
    causeSimilarityScore * WEIGHTS.causeSimilarity * 100 +
    timeScore * WEIGHTS.timeOfDay * 100 +
    dayScore * WEIGHTS.dayOfWeek * 100 +
    priorityScore * WEIGHTS.priorityMix * 100
  );

  const clampedSeverity = Math.min(100, Math.max(0, severity));

  // 4. Identify affected corridors
  const affectedCorridors = identifyAffectedCorridors(outerIncidents);

  // 5. Road closure probability
  const closureProbability = computeClosureProbability(middleIncidents, eventCause);

  // 6. Estimated incident spike (days derived from actual dataset range)
  const datasetDays = getDatasetDays();
  const avgIncidentsPerDay = middleIncidents.length / datasetDays;
  const predictedExtraIncidents = Math.round(
    avgIncidentsPerDay * (durationHours / 24) * (1 + clampedSeverity / 50)
  );

  // 7. Impact zones for map rendering
  const impactZones = [
    { radiusKm: INNER_RADIUS, severity: 'critical', opacity: 0.4 },
    { radiusKm: MIDDLE_RADIUS, severity: 'high', opacity: 0.25 },
    { radiusKm: OUTER_RADIUS, severity: 'moderate', opacity: 0.1 },
  ];

  return {
    severity: clampedSeverity,
    severityLevel: getSeverityLevel(clampedSeverity),
    affectedCorridors,
    closureProbability: Math.round(closureProbability * 100),
    predictedExtraIncidents,
    estimatedDuration: durationHours,
    impactZones,
    historicalIncidents: middleIncidents.length,
    scores: {
      density: Math.round(densityScore * 100),
      causeSimilarity: Math.round(causeSimilarityScore * 100),
      timeMatch: Math.round(timeScore * 100),
      dayMatch: Math.round(dayScore * 100),
      priorityMix: Math.round(priorityScore * 100),
    },
  };
}

// --- Sub-score computation functions ---

function computeDensityScore(inner, middle, outer) {
  // Normalize relative to the total dataset size so the score
  // stays meaningful regardless of how many incidents the data contains.
  const totalIncidents = getAllIncidents().length || 1;
  const baseline = totalIncidents * 0.08; // top ~8% of locations should score ~1.0
  const weighted = inner * 3 + middle * 1.5 + outer * 0.5;
  return Math.min(1, weighted / baseline);
}

function computeCauseSimilarity(incidents, relatedCauses) {
  if (incidents.length === 0) return 0;
  const related = incidents.filter((inc) => relatedCauses.includes(inc.cause));
  return related.length / incidents.length;
}

function computeTimeScore(incidents, targetHour) {
  if (incidents.length === 0) return 0.5;
  // What fraction of incidents happen within ±2 hours?
  const nearby = incidents.filter(
    (inc) => inc.hourIST != null && Math.abs(inc.hourIST - targetHour) <= 2
  );
  return nearby.length / incidents.length;
}

function computeDayScore(incidents, targetDay) {
  if (incidents.length === 0) return 0.5;
  const sameDay = incidents.filter((inc) => inc.dayOfWeek === targetDay);
  // Normalize: if uniform, each day gets ~14.3% of incidents
  const ratio = sameDay.length / incidents.length;
  return Math.min(1, ratio * 7); // scale so uniform = 1.0
}

function computePriorityScore(incidents) {
  if (incidents.length === 0) return 0;
  const highPriority = incidents.filter((inc) => inc.priority === 'High');
  return highPriority.length / incidents.length;
}

function identifyAffectedCorridors(incidents) {
  const corridorCounts = {};
  for (const inc of incidents) {
    if (inc.corridor && inc.corridor !== 'Non-corridor') {
      corridorCounts[inc.corridor] = (corridorCounts[inc.corridor] || 0) + 1;
    }
  }
  return Object.entries(corridorCounts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);
}

function computeClosureProbability(incidents, eventCause) {
  // Data-derived: compute closure rate for this cause type from nearby incidents
  const relatedCauses = getRelatedCauses(eventCause);
  const sameCause = incidents.filter((inc) => relatedCauses.includes(inc.cause));
  const sameCauseClosed = sameCause.filter((inc) => inc.roadClosure).length;
  const causeRate = sameCause.length >= 5
    ? sameCauseClosed / sameCause.length
    : null;

  // Location-specific: overall closure rate at this location
  const localClosed = incidents.filter((inc) => inc.roadClosure).length;
  const localRate = incidents.length > 0 ? localClosed / incidents.length : 0;

  // Blend: cause-specific rate (if enough data) + location rate
  if (causeRate !== null) {
    return causeRate * 0.6 + localRate * 0.4;
  }
  // Fallback: just location rate with a small cause-type boost for known event causes
  const eventBoost = ['public_event', 'procession', 'vip_movement', 'protest'].includes(eventCause)
    ? 0.15
    : 0;
  return Math.min(1, localRate + eventBoost);
}

function getRelatedCauses(eventCause) {
  const groups = {
    public_event: ['public_event', 'congestion', 'vip_movement'],
    procession: ['procession', 'congestion', 'public_event'],
    vip_movement: ['vip_movement', 'congestion', 'public_event'],
    protest: ['protest', 'congestion'],
    construction: ['construction', 'road_conditions', 'congestion'],
  };
  return groups[eventCause] || [eventCause, 'congestion'];
}

function getSeverityLevel(score) {
  if (score >= 75) return 'critical';
  if (score >= 50) return 'high';
  if (score >= 25) return 'medium';
  return 'low';
}
