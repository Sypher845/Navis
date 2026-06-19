/**
 * engine/learning.js — Post-event analysis engine
 */

import { getIncidentsNear, getMedianResolution, getEventIncidents, getMeta } from '../data/index.js';

export function analyzeEvent(event) {
  const incidents = getIncidentsNear(event.lat, event.lng, 3);
  const eventStart = event.start ? new Date(event.start) : null;
  const eventEnd = event.end ? new Date(event.end) : null;

  // Incidents during event window
  const duringEvent = incidents.filter(inc => {
    if (!inc.start || !eventStart) return false;
    const t = new Date(inc.start);
    const windowStart = new Date(eventStart.getTime() - 2 * 3600000);
    const windowEnd = eventEnd || new Date(eventStart.getTime() + 6 * 3600000);
    return t >= windowStart && t <= windowEnd;
  });

  // Baseline: same area, same hour range, on non-event days
  const baselineHour = eventStart ? eventStart.getHours() : 12;
  const baselineIncidents = incidents.filter(inc => {
    if (inc.hourIST == null) return false;
    return Math.abs(inc.hourIST - baselineHour) <= 2 && !duringEvent.includes(inc);
  });

  // Compute actual dataset days from date range
  const meta = getMeta();
  let datasetDays = 152;
  if (meta && meta.dateRange && meta.dateRange.start && meta.dateRange.end) {
    const diffMs = new Date(meta.dateRange.end).getTime() - new Date(meta.dateRange.start).getTime();
    datasetDays = Math.max(1, Math.round(diffMs / (1000 * 60 * 60 * 24)));
  }
  const avgBaseline = baselineIncidents.length / datasetDays;
  const eventDayCount = duringEvent.length;
  const spike = avgBaseline > 0 ? eventDayCount / avgBaseline : 0;

  // Resolution comparison
  const eventResolutions = duringEvent.filter(i => i.resolutionMin > 0).map(i => i.resolutionMin);
  const baseResolutions = baselineIncidents.filter(i => i.resolutionMin > 0).map(i => i.resolutionMin);
  const eventMedian = median(eventResolutions);
  const baseMedian = median(baseResolutions);

  const effectivenessScore = baseMedian && eventMedian
    ? Math.round((1 - (eventMedian - baseMedian) / baseMedian) * 100)
    : null;

  // Cause breakdown during event
  const causeBreakdown = {};
  for (const inc of duringEvent) {
    causeBreakdown[inc.cause] = (causeBreakdown[inc.cause] || 0) + 1;
  }

  return {
    event,
    incidentsDuringEvent: eventDayCount,
    baselineAvg: Math.round(avgBaseline * 10) / 10,
    spikeMultiplier: Math.round(spike * 10) / 10,
    eventMedianResolution: eventMedian ? Math.round(eventMedian) : null,
    baselineMedianResolution: baseMedian ? Math.round(baseMedian) : null,
    effectivenessScore,
    causeBreakdown,
    incidents: duringEvent,
  };
}

export function getPastEvents() {
  return getEventIncidents().filter(e => e.type === 'planned' && e.desc);
}

function median(arr) {
  if (!arr.length) return null;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}
