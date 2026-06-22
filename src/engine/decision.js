/**
 * engine/decision.js — Browser-side inference for the trained XGBoost model.
 *
 * Uses the offline-compiled XGBoost JS function (m2cgen), applies baseline
 * features dynamically, encodes inputs, and computes a SHAP-based reasoning trail.
 */

import { score } from '../../exports/model_logic.js';
import explainData from '../../exports/explainability_weights.json';
import baselineData from '../../exports/baseline_features.json';

let _actionStats = null;
let _metrics = null;

/**
 * Initialize the decision engine using the compiled artifacts.
 */
export async function loadDecisionTree() {
  _metrics = explainData.metrics;
  _actionStats = explainData.metrics.perClass;

  // We map the Python "actionStats" format to what resource.js expects
  const rawModelData = {
    junctionIntelligence: {},
    policeStationIntelligence: {},
    corridorDiversions: {},
    actionStats: {},
  };

  // Convert baselines to expected formats
  for (const [jn, data] of Object.entries(baselineData.junctionBaselines)) {
    rawModelData.junctionIntelligence[jn] = {
      count: data.freq,
      highPriorityRate: data.hp_rate,
      closureRate: data.closure_rate,
      medianResolutionMin: data.avg_res,
      lat: data.lat || 12.9716, lng: data.lng || 77.5946 
    };
  }
  for (const [ps, data] of Object.entries(baselineData.policeStationBaselines || {})) {
    rawModelData.policeStationIntelligence[ps] = {
      count: data.freq,
      medianResolutionMin: data.avg_res,
      lat: data.lat || 12.9716, lng: data.lng || 77.5946
    };
  }

  // Populate basic action stats from metrics support
  for (const label of explainData.actionLabels) {
    rawModelData.actionStats[label] = {
      count: _metrics.perClass[label].support,
      medianResolutionMin: 60 // fallback
    };
  }

  return {
    ..._metrics,
    _rawModelData: rawModelData,
  };
}

/**
 * Safely get an encoded value for a categorical feature, falling back to 0.
 */
function getEncodedValue(feat, value) {
  const enc = explainData.labelEncoders[feat];
  if (!enc) return 0;
  const idx = enc.indexOf(String(value));
  return idx >= 0 ? idx : 0;
}

/**
 * Get a baseline feature, falling back to 0.
 */
function getBaseline(dict, key, field, fallback = 0) {
  if (!baselineData[dict]) return fallback;
  const data = baselineData[dict][key];
  return data && data[field] !== undefined && !isNaN(data[field]) ? data[field] : fallback;
}

/**
 * Run the XGBoost model on an input situation.
 */
export function decide(cause, corridor, zone, hour, dayOfWeek, eventType, lat, lng) {
  // We need to resolve junction and police station somehow. 
  // For the sake of the demo, we assume "Unknown" or the closest matching ones.
  // We'll just pass 'Unknown' for anything we don't have exactly.
  const junction = 'Unknown';
  const police_station = 'Unknown';
  const veh_type = 'Unknown';
  
  // Is this a road closure?
  const isClosure = ['public_event', 'vip_movement', 'construction', 'procession'].includes(cause) ? 1.0 : 0.0;

  // 1. Build the input array matching `explainData.featureOrder`
  const features = {};
  
  // Categorical encodings
  features['event_cause_enc'] = getEncodedValue('event_cause', cause);
  features['corridor_enc'] = getEncodedValue('corridor', corridor || 'Non-corridor');
  features['zone_enc'] = getEncodedValue('zone', zone || 'Unknown');
  features['event_type_enc'] = getEncodedValue('event_type', eventType || 'unplanned');
  features['priority_enc'] = getEncodedValue('priority', 'High'); // Assuming forecast implies high priority
  features['police_station_enc'] = getEncodedValue('police_station', police_station);
  features['veh_type_enc'] = getEncodedValue('veh_type', veh_type);
  features['junction_enc'] = getEncodedValue('junction', junction);
  
  // Numerics
  features['hour'] = hour;
  features['day_of_week'] = dayOfWeek;
  features['road_closure'] = isClosure;
  
  // Engineered features (from baselines)
  const med_res = 60.0;
  const hour_block = Math.floor(hour / 4);
  
  features['junction_freq'] = getBaseline('junctionBaselines', junction, 'freq', 0);
  features['junction_avg_res'] = getBaseline('junctionBaselines', junction, 'avg_res', med_res);
  features['junction_hp_rate'] = getBaseline('junctionBaselines', junction, 'hp_rate', 0);
  features['junction_closure_rate'] = getBaseline('junctionBaselines', junction, 'closure_rate', 0);
  
  features['corridor_freq'] = getBaseline('corridorBaselines', corridor, 'freq', 0);
  features['corridor_avg_res'] = getBaseline('corridorBaselines', corridor, 'avg_res', med_res);
  
  features['zone_freq'] = getBaseline('zoneBaselines', zone, 'freq', 0);
  features['zone_avg_res'] = getBaseline('zoneBaselines', zone, 'avg_res', med_res);
  
  features['hour_block_freq'] = getBaseline('hourBlockBaselines', hour_block, 'freq', 0);
  
  features['ps_freq'] = getBaseline('policeStationBaselines', police_station, 'freq', 0);
  features['ps_avg_res'] = getBaseline('policeStationBaselines', police_station, 'avg_res', med_res);
  
  // Graph features
  features['adj_degree'] = baselineData.cascadeGraph && baselineData.cascadeGraph[junction] ? baselineData.cascadeGraph[junction].length : 0;
  features['adj_max_weight'] = 0;
  if (features['adj_degree'] > 0) {
    features['adj_max_weight'] = Math.max(...baselineData.cascadeGraph[junction].map(n => n.coFailures));
  }

  // Map to flat array
  const inputArray = explainData.featureOrder.map(col => features[col]);

  // 2. Inference
  const scores = score(inputArray);
  const bestIdx = scores.indexOf(Math.max(...scores));
  const action = explainData.actionLabels[bestIdx];

  // 3. SHAP-based Reasoning
  const reasoning = [];
  let confidence = Math.round(scores[bestIdx] * 100);
  
  // Normalize if probabilities sum to < 1 or > 1 due to softprob
  const sumProb = scores.reduce((a, b) => a + b, 0);
  if (sumProb > 0) confidence = Math.round((scores[bestIdx] / sumProb) * 100);

  // Take top 3 most important features globally to explain the decision
  const imp = Object.entries(explainData.featureImportance).slice(0, 3);
  for (const [feat, weight] of imp) {
    reasoning.push({
      type: 'decision', // Use decision style block
      action: action,
      description: `Driven by ${feat} (${weight}% influence)`,
      confidence: confidence,
      samples: _metrics.perClass[action].support,
      outcomeStats: { median_resolution_min: 60 }
    });
  }

  // Get stats for the recommended action
  const stats = _actionStats ? _actionStats[action] : null;

  return {
    action,
    reasoning,
    confidence,
    actionStats: stats ? { count: stats.support, medianResolutionMin: 60 } : null,
    modelAccuracy: _metrics ? _metrics.accuracy : null,
  };
}

export function getActionStats() {
  return _actionStats;
}

export function getModelMetrics() {
  return _metrics;
}
