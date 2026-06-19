/**
 * engine/decision.js — Browser-side inference for the trained decision tree.
 *
 * Loads the exported decision tree JSON and walks it to produce
 * actionable operational decisions with full reasoning trail.
 */

let _tree = null;
let _actionStats = null;
let _metrics = null;

/**
 * Load the trained decision tree from the preprocessed JSON.
 */
export async function loadDecisionTree() {
  const res = await fetch('/decision_tree.json');
  const data = await res.json();
  _tree = data.tree;
  _actionStats = data.actionStats;
  _metrics = data.metrics;

  // Return metrics + raw model data so the resource engine can use it
  return {
    ..._metrics,
    _rawModelData: {
      junctionIntelligence: data.junctionIntelligence || {},
      policeStationIntelligence: data.policeStationIntelligence || {},
      corridorDiversions: data.corridorDiversions || {},
      actionStats: data.actionStats || {},
    },
  };
}

/**
 * Run the decision tree on an input situation.
 *
 * @param {string} cause - Event cause (e.g. 'public_event')
 * @param {string} corridor - Corridor name (e.g. 'CBD 2')
 * @param {string} zone - Zone name (e.g. 'Central Zone 2')
 * @param {number} hour - Hour of day in IST (0-23)
 * @param {number} dayOfWeek - Day of week (0=Mon, 6=Sun)
 * @param {string} eventType - 'planned' or 'unplanned'
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 * @returns {Object} Decision with reasoning trail
 */
export function decide(cause, corridor, zone, hour, dayOfWeek, eventType, lat, lng) {
  if (!_tree) {
    return { action: 'deploy_light', reasoning: ['Decision tree not loaded'], confidence: 0 };
  }

  const input = {
    cause,
    corridor: corridor || 'Non-corridor',
    zone: zone || 'Unknown',
    hour,
    day_of_week: dayOfWeek,
    event_type: eventType,
    lat,
    lng,
  };

  const reasoning = [];
  const action = walkTree(_tree, input, reasoning);

  // Get stats for the recommended action
  const stats = _actionStats ? _actionStats[action] : null;

  return {
    action,
    reasoning,
    confidence: reasoning.length > 0 ? reasoning[reasoning.length - 1].confidence : 0,
    actionStats: stats,
    modelAccuracy: _metrics ? _metrics.testAccuracy : null,
  };
}

/**
 * Walk the tree recursively, collecting reasoning at each split.
 */
function walkTree(node, input, reasoning) {
  // Leaf node — we have a decision
  if (node.action) {
    reasoning.push({
      type: 'decision',
      action: node.action,
      description: node.actionDescription,
      confidence: node.confidence,
      samples: node.samples,
      distribution: node.distribution,
      outcomeStats: node.outcomeStats,
    });
    return node.action;
  }

  // Internal node — evaluate the split condition
  const feature = node.splitFeature;
  const value = input[feature];
  let goLeft = false;
  let conditionText = '';

  if (node.categories) {
    // Categorical split
    goLeft = node.categories.includes(value);
    conditionText = goLeft
      ? `${formatFeature(feature)} is "${value}" (matches ${node.categories.join(', ')})`
      : `${formatFeature(feature)} is "${value}" (not in ${node.categories.join(', ')})`;
  } else {
    // Numeric split
    goLeft = value <= node.threshold;
    conditionText = goLeft
      ? `${formatFeature(feature)} = ${formatValue(feature, value)} (≤ ${formatValue(feature, node.threshold)})`
      : `${formatFeature(feature)} = ${formatValue(feature, value)} (> ${formatValue(feature, node.threshold)})`;
  }

  reasoning.push({
    type: 'split',
    feature,
    condition: conditionText,
    reason: node.reason,
    direction: goLeft ? 'yes' : 'no',
    samples: node.samples,
  });

  if (goLeft) {
    return walkTree(node.yes, input, reasoning);
  } else {
    return walkTree(node.no, input, reasoning);
  }
}

function formatFeature(feat) {
  const labels = {
    cause: 'Incident cause',
    corridor: 'Corridor',
    zone: 'Zone',
    hour: 'Hour (IST)',
    day_of_week: 'Day of week',
    event_type: 'Event type',
    lat: 'Latitude',
    lng: 'Longitude',
  };
  return labels[feat] || feat;
}

const DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

function formatValue(feat, val) {
  if (feat === 'hour') return `${Math.round(val)}:00`;
  if (feat === 'day_of_week') return DAY_NAMES[Math.round(val)] || val;
  if (typeof val === 'number') return val.toFixed(2);
  return String(val);
}

/**
 * Get action statistics for display.
 */
export function getActionStats() {
  return _actionStats;
}

/**
 * Get model accuracy metrics.
 */
export function getModelMetrics() {
  return _metrics;
}
