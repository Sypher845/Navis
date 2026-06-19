/**
 * config.js — Application configuration
 *
 * When you receive your Mappls API key, set it here and change
 * MAP_PROVIDER to 'mappls'. Everything else adapts automatically.
 */

export const MAP_PROVIDER = 'leaflet'; // 'leaflet' | 'mappls'
export const MAPPLS_KEY = ''; // paste your Mappls API key here

// Bengaluru center coordinates
export const BENGALURU_CENTER = { lat: 12.9716, lng: 77.5946 };
export const DEFAULT_ZOOM = 12;

// Incident cause → color mapping (matches CSS variables)
export const CAUSE_COLORS = {
  vehicle_breakdown: '#fb923c',
  accident: '#ef4444',
  construction: '#fbbf24',
  water_logging: '#3b82f6',
  tree_fall: '#22c55e',
  pot_holes: '#a78bfa',
  public_event: '#f472b6',
  procession: '#e879f9',
  congestion: '#ef4444',
  vip_movement: '#fbbf24',
  protest: '#f87171',
  road_conditions: '#94a3b8',
  others: '#64748b',
  Debris: '#94a3b8',
  test_demo: '#64748b',
  'Fog / Low Visibility': '#94a3b8',
};

// Human-readable cause labels
export const CAUSE_LABELS = {
  vehicle_breakdown: 'Vehicle Breakdown',
  accident: 'Accident',
  construction: 'Construction',
  water_logging: 'Water Logging',
  tree_fall: 'Tree Fall',
  pot_holes: 'Potholes',
  public_event: 'Public Event',
  procession: 'Procession',
  congestion: 'Congestion',
  vip_movement: 'VIP Movement',
  protest: 'Protest',
  road_conditions: 'Road Conditions',
  others: 'Others',
  Debris: 'Debris',
  test_demo: 'Test',
  'Fog / Low Visibility': 'Fog / Low Visibility',
};

// Event causes that represent "events" (for the forecasting engine)
export const EVENT_CAUSES = [
  'public_event',
  'procession',
  'vip_movement',
  'protest',
  'construction', // long-running planned events
];
