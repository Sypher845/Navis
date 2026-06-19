/**
 * engine/resource.js — Resource deployment recommender
 *
 * v2: Junction-based barricading, data-driven diversions via corridor
 * co-occurrence, police station routing, and decision-tree-scaled manpower.
 */

import { getMedianResolution, getIncidentsNear, getAllIncidents } from '../data/index.js';

/** Reference to loaded model intelligence (set externally). */
let _modelData = null;

export function setModelData(modelData) {
  _modelData = modelData;
}

export function recommendResources(prediction, lat, lng, eventCause, decision) {
  const { severity, affectedCorridors, closureProbability } = prediction;
  const closureFrac = closureProbability / 100;

  // Scale manpower by the ML decision action, not just severity heuristics
  const actionClass = decision ? decision.action : null;
  const manpower = computeManpower(severity, affectedCorridors.length, closureFrac, actionClass);

  // Junction-based barricade positions (data-driven)
  const barricadePoints = computeJunctionBarricades(lat, lng, eventCause);

  // Data-driven diversions from corridor co-occurrence
  const diversions = computeSmartDiversions(affectedCorridors);

  // Equipment scaled by action class and vehicle type distribution
  const equipment = computeEquipment(severity, closureFrac, actionClass);

  // Police stations that should be alerted
  const respondingStations = computeRespondingStations(lat, lng);

  const medianResolution = getMedianResolution({ cause: eventCause });

  return {
    manpower,
    barricadePoints,
    diversions,
    equipment,
    respondingStations,
    readinessTimeMin: medianResolution ? Math.round(medianResolution * 0.3) : 30,
    medianResolutionMin: medianResolution ? Math.round(medianResolution) : null,
  };
}


// --- Manpower (decision-tree-scaled) ---

const MANPOWER_PROFILES = {
  'monitor':      { baseTP: 2,  cdScale: 0,   volScale: 0,   closureBonus: 0   },
  'deploy_light': { baseTP: 6,  cdScale: 0.5, volScale: 0.3, closureBonus: 4   },
  'deploy_heavy': { baseTP: 14, cdScale: 1.0, volScale: 0.7, closureBonus: 8   },
  'full_closure': { baseTP: 20, cdScale: 1.5, volScale: 1.0, closureBonus: 12  },
};

function computeManpower(severity, corridorCount, closureProb, actionClass) {
  const profile = MANPOWER_PROFILES[actionClass] || MANPOWER_PROFILES['deploy_light'];

  const tp = Math.max(profile.baseTP, profile.baseTP + corridorCount * 2);
  const cd = Math.round(severity / 20 * profile.cdScale);
  const vol = Math.round(severity / 15 * profile.volScale);
  const extra = closureProb > 0.4 ? profile.closureBonus : 0;

  return {
    trafficPolice: tp + extra,
    civilDefense: cd,
    volunteers: vol,
    total: tp + extra + cd + vol,
  };
}


// --- Junction-based barricading ---

function computeJunctionBarricades(lat, lng, eventCause) {
  // Strategy: Use junction intelligence from trained model if available
  const junctionIntel = _modelData ? _modelData.junctionIntelligence : null;

  if (junctionIntel) {
    return junctionBarricadesFromModel(lat, lng, eventCause, junctionIntel);
  }

  // Fallback: Use incident data to find nearby junction names
  return junctionBarricadesFromIncidents(lat, lng);
}


function junctionBarricadesFromModel(lat, lng, eventCause, junctionIntel) {
  // Find junctions within ~3km of event location, ranked by incident severity
  const radiusKm = 3;
  const results = [];

  for (const [jn, data] of Object.entries(junctionIntel)) {
    const dist = haversineQuick(lat, lng, data.lat, data.lng);
    if (dist <= radiusKm) {
      // Compute a barricade priority score:
      // Higher = more critical to barricade
      const score = data.count * 0.3 +
                    data.highPriorityRate * 30 +
                    data.closureRate * 50 +
                    (1 / (dist + 0.1)) * 10;

      results.push({
        junction: jn,
        lat: data.lat,
        lng: data.lng,
        count: data.count,
        highPriorityRate: data.highPriorityRate,
        closureRate: data.closureRate,
        corridors: data.corridors,
        policeStations: data.policeStations,
        medianResolutionMin: data.medianResolutionMin,
        type: data.closureRate > 0.1 ? 'primary' : 'secondary',
        score: Math.round(score),
        distanceKm: Math.round(dist * 100) / 100,
      });
    }
  }

  return results
    .sort((a, b) => b.score - a.score)
    .slice(0, 8);
}


function junctionBarricadesFromIncidents(lat, lng) {
  // Fallback: cluster nearby incidents by junction name
  const incidents = getIncidentsNear(lat, lng, 3);
  const junctionMap = {};

  for (const inc of incidents) {
    if (!inc.junction) continue;
    if (!junctionMap[inc.junction]) {
      junctionMap[inc.junction] = {
        junction: inc.junction,
        lat: 0, lng: 0, count: 0,
        closures: 0, corridors: new Set(),
      };
    }
    const j = junctionMap[inc.junction];
    j.lat += inc.lat;
    j.lng += inc.lng;
    j.count++;
    if (inc.roadClosure) j.closures++;
    if (inc.corridor && inc.corridor !== 'Non-corridor') j.corridors.add(inc.corridor);
  }

  return Object.values(junctionMap)
    .map(j => ({
      junction: j.junction,
      lat: j.lat / j.count,
      lng: j.lng / j.count,
      count: j.count,
      closureRate: Math.round(j.closures / j.count * 100) / 100,
      corridors: [...j.corridors].slice(0, 2),
      type: j.closures > 0 ? 'primary' : 'secondary',
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);
}


// --- Smart diversions (data-driven co-occurrence) ---

function computeSmartDiversions(affectedCorridors) {
  const diversions = _modelData ? _modelData.corridorDiversions : null;

  if (diversions) {
    return smartDiversionsFromModel(affectedCorridors, diversions);
  }

  // Fallback to static table
  return staticDiversions(affectedCorridors);
}


function smartDiversionsFromModel(affectedCorridors, diversions) {
  const affected = new Set(affectedCorridors.map(c => c.name));
  const result = [];
  const seen = new Set();

  for (const c of affectedCorridors) {
    const data = diversions[c.name];
    if (!data) continue;

    for (const alt of data.bestDiversions) {
      if (!affected.has(alt) && !seen.has(alt)) {
        seen.add(alt);
        // Find co-occurrence rate for explanation
        const detail = data.details
          ? data.details.find(d => d.corridor === alt)
          : null;
        const coRate = detail ? detail.coRate : null;

        result.push({
          from: c.name,
          to: alt,
          confidence: coRate !== null
            ? Math.round((1 - coRate) * 100)
            : null,
          reason: coRate !== null
            ? `${Math.round((1 - coRate) * 100)}% independent (co-occur only ${Math.round(coRate * 100)}% of days)`
            : 'Data-driven suggestion',
        });
      }
    }
  }

  return result.slice(0, 6);
}


const CORRIDOR_ALTS_STATIC = {
  'CBD 2': ['CBD 1', 'Old Madras Road'],
  'Bellary Road 1': ['Bellary Road 2', 'Tumkur Road'],
  'Mysore Road': ['West of Chord Road', 'Magadi Road'],
  'Tumkur Road': ['West of Chord Road', 'Bellary Road 1'],
  'Hosur Road': ['Bannerghata Road', 'ORR East 1'],
  'ORR East 1': ['ORR East 2', 'Hosur Road'],
  'ORR North 1': ['ORR North 2', 'Hennur Main Road'],
  'Bannerghata Road': ['Hosur Road', 'ORR West 1'],
  'Old Madras Road': ['CBD 1', 'ORR East 2'],
  'Magadi Road': ['Mysore Road', 'West of Chord Road'],
};

function staticDiversions(affectedCorridors) {
  const affected = new Set(affectedCorridors.map(c => c.name));
  const result = [];
  const seen = new Set();
  for (const c of affectedCorridors) {
    for (const alt of (CORRIDOR_ALTS_STATIC[c.name] || [])) {
      if (!affected.has(alt) && !seen.has(alt)) {
        seen.add(alt);
        result.push({ from: c.name, to: alt });
      }
    }
  }
  return result.slice(0, 5);
}


// --- Police station routing ---

function computeRespondingStations(lat, lng) {
  const psIntel = _modelData ? _modelData.policeStationIntelligence : null;
  if (!psIntel) return [];

  const results = [];
  for (const [name, data] of Object.entries(psIntel)) {
    const dist = haversineQuick(lat, lng, data.lat, data.lng);
    if (dist <= 5) {
      results.push({
        name,
        distanceKm: Math.round(dist * 100) / 100,
        incidentCount: data.count,
        medianResolutionMin: data.medianResolutionMin,
        topJunctions: data.topJunctions ? data.topJunctions.slice(0, 3) : [],
      });
    }
  }

  return results
    .sort((a, b) => a.distanceKm - b.distanceKm)
    .slice(0, 4);
}


// --- Equipment (decision-tree-scaled) ---

function computeEquipment(severity, closureProb, actionClass) {
  const items = [];

  // Scale by action class
  const isHeavy = actionClass === 'deploy_heavy' || actionClass === 'full_closure';
  const barricadeMultiplier = isHeavy ? 2.0 : 1.0;

  if (closureProb > 0.2 || isHeavy) {
    items.push({
      name: 'Barricades',
      count: Math.max(4, Math.round(closureProb * 20 * barricadeMultiplier)),
      icon: '🚧',
    });
  }

  items.push({
    name: 'Traffic Cones',
    count: Math.max(10, Math.round(severity / 3)),
    icon: '🔶',
  });

  items.push({
    name: 'Diversion Signs',
    count: Math.max(4, Math.round(severity / 10)),
    icon: '↪️',
  });

  if (severity >= 30 || isHeavy) {
    items.push({
      name: 'Tow Trucks',
      count: isHeavy ? Math.max(2, Math.ceil(severity / 25)) : Math.ceil(severity / 35),
      icon: '🚛',
    });
  }

  items.push({
    name: 'Radio Sets',
    count: Math.max(4, Math.round(severity / 15)),
    icon: '📻',
  });

  if (actionClass === 'full_closure') {
    items.push({
      name: 'Ambulances',
      count: 2,
      icon: '🚑',
    });
  }

  return items;
}


// --- Utility ---

function haversineQuick(lat1, lng1, lat2, lng2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}
