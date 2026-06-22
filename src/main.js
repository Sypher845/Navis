/**
 * main.js — Application entry point
 */

import './index.css';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.heat';
import 'leaflet.markercluster';
import Chart from 'chart.js/auto';

import { loadData, getAllIncidents, getCauseDistribution, getHourlyDistribution, getCorridors, getEventIncidents, getIncidentsNear } from './data/index.js';
import { predictImpact } from './engine/forecast.js';
import { recommendResources, setModelData } from './engine/resource.js';
import { analyzeEvent, getPastEvents } from './engine/learning.js';
import { loadDecisionTree, decide } from './engine/decision.js';
import * as Simulator from './engine/simulator.js';
import { CAUSE_COLORS, CAUSE_LABELS, BENGALURU_CENTER } from './config.js';
import { formatTime, formatDuration, hourLabel } from './utils/time.js';

// State
let map = null;
let heatLayer = null;
let markerCluster = null;
let impactLayers = [];
let currentMode = 'forecast';
let selectedLocation = null;
let charts = {};

// --- Bootstrap ---
async function init() {
  const [meta, treeData] = await Promise.all([loadData(), loadDecisionTree()]);
  console.log('Decision tree loaded:', treeData);

  // Wire model intelligence data into the resource engine
  if (treeData && treeData._rawModelData) {
    setModelData(treeData._rawModelData);
  }

  initMap();
  renderIncidents();
  updateTopbarStats(meta);
  renderSidebar();
  renderRightPanel();
  setupModeSwitch();
  document.getElementById('loading-overlay').classList.add('hidden');
}

// --- Map ---
function initMap() {
  map = L.map('map', {
    center: [BENGALURU_CENTER.lat, BENGALURU_CENTER.lng],
    zoom: 12,
    zoomControl: true,
    attributionControl: false,
  });

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19,
  }).addTo(map);

  L.control.attribution({ position: 'bottomright', prefix: 'Navis | Leaflet' }).addTo(map);

  // Click to pick location in forecast mode
  map.on('click', (e) => {
    if (currentMode === 'forecast') {
      selectedLocation = { lat: e.latlng.lat, lng: e.latlng.lng };
      updateLocationMarker();
      document.getElementById('loc-display').textContent =
        `${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)}`;
    }
  });
}

let locationMarker = null;
function updateLocationMarker() {
  if (locationMarker) map.removeLayer(locationMarker);
  if (!selectedLocation) return;
  locationMarker = L.marker([selectedLocation.lat, selectedLocation.lng], {
    icon: L.divIcon({
      className: '',
      html: '<div style="width:20px;height:20px;border-radius:50%;background:#3b82f6;border:3px solid white;box-shadow:0 0 12px rgba(59,130,246,0.6);"></div>',
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    }),
  }).addTo(map);
}

// Global incident markers for simulator
let simulatorMarkers = [];

function renderIncidents() {
  const incidents = currentMode === 'simulator' ? Simulator.getCurrentIncidents() : getAllIncidents();

  // Heatmap layer
  if (heatLayer) map.removeLayer(heatLayer);
  if (markerCluster) map.removeLayer(markerCluster);
  simulatorMarkers.forEach(m => map.removeLayer(m));
  simulatorMarkers = [];

  if (currentMode === 'simulator') {
    // Render individual glowing markers for the simulator sequence
    for (const inc of incidents) {
      const color = CAUSE_COLORS[inc.cause] || '#64748b';
      const m = L.circleMarker([inc.lat, inc.lng], {
        radius: 6, fillColor: color, fillOpacity: 0.9, color: 'white', weight: 1,
      }).addTo(map);
      m.bindPopup(buildPopup(inc));
      simulatorMarkers.push(m);
    }
    document.getElementById('visible-count').textContent = `Simulator: ${incidents.length} incidents`;
    return; // Skip standard heatmaps in simulator
  }

  // Standard rendering
  const heatPoints = incidents.map(i => [i.lat, i.lng, 0.5]);
  heatLayer = L.heatLayer(heatPoints, {
    radius: 18, blur: 25, maxZoom: 15,
    gradient: { 0.2: '#22d3ee', 0.4: '#3b82f6', 0.6: '#fbbf24', 0.8: '#fb923c', 1: '#ef4444' },
  }).addTo(map);

  // Marker cluster
  markerCluster = L.markerClusterGroup({
    maxClusterRadius: 50,
    disableClusteringAtZoom: 16,
    spiderfyOnMaxZoom: true,
  });

  for (const inc of incidents) {
    const color = CAUSE_COLORS[inc.cause] || '#64748b';
    const marker = L.circleMarker([inc.lat, inc.lng], {
      radius: 4, fillColor: color, fillOpacity: 0.7, stroke: false,
    });
    marker.bindPopup(buildPopup(inc), { maxWidth: 280 });
    markerCluster.addLayer(marker);
  }
  map.addLayer(markerCluster);

  // Legend
  renderLegend();
  document.getElementById('visible-count').textContent = `${incidents.length.toLocaleString()} incidents`;
}

function buildPopup(inc) {
  const label = CAUSE_LABELS[inc.cause] || inc.cause;
  const desc = inc.desc ? `<p>${inc.desc.substring(0, 120)}</p>` : '';
  return `<div class="incident-popup">
    <h4>${label}</h4>
    <p>📍 ${inc.corridor || 'Non-corridor'}</p>
    <p>🕐 ${formatTime(inc.start)}</p>
    ${inc.resolutionMin ? `<p>⏱ Resolved in ${formatDuration(inc.resolutionMin)}</p>` : ''}
    ${desc}
    <div class="popup-badges">
      <span class="badge badge-${inc.priority === 'High' ? 'high' : 'low'}">${inc.priority || '—'}</span>
      <span class="badge badge-${inc.type}">${inc.type}</span>
    </div>
  </div>`;
}

function renderLegend() {
  const container = document.getElementById('legend-items');
  const topCauses = getCauseDistribution().slice(0, 8);
  container.innerHTML = topCauses.map(({ cause }) => {
    const color = CAUSE_COLORS[cause] || '#64748b';
    const label = CAUSE_LABELS[cause] || cause;
    return `<div class="legend-item"><span class="legend-dot" style="background:${color}"></span>${label}</div>`;
  }).join('');
}

// --- Top Bar ---
function updateTopbarStats(meta) {
  document.querySelector('#stat-total .stat-value').textContent = meta.totalRecords.toLocaleString();
  const events = getEventIncidents();
  document.querySelector('#stat-events .stat-value').textContent = events.length;
  const corridors = getCorridors().filter(c => c.name !== 'Non-corridor');
  document.querySelector('#stat-corridors .stat-value').textContent = corridors.length;
}

// --- Mode Switching ---
function setupModeSwitch() {
  document.getElementById('mode-switcher').addEventListener('click', (e) => {
    const btn = e.target.closest('.mode-btn');
    if (!btn) return;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
    renderSidebar();
    
    if (currentMode === 'simulator') {
      Simulator.setFrame(0);
      document.getElementById('panel-content').innerHTML = `
        <div style="padding: 20px; text-align: center; color: var(--text-muted);">
          Drag the slider on the left to start the Gridlock Simulation.
        </div>`;
    } else {
      renderRightPanel();
    }
    renderIncidents();
  });
}

// --- Sidebar ---
function renderSidebar() {
  const el = document.getElementById('sidebar-content');
  if (currentMode === 'forecast') el.innerHTML = forecastSidebar();
  else if (currentMode === 'explore') el.innerHTML = exploreSidebar();
  else if (currentMode === 'simulator') el.innerHTML = simulatorSidebar();
  else el.innerHTML = learningSidebar();
  attachSidebarListeners();
}

function simulatorSidebar() {
  const data = Simulator.getSimulationData();
  if (!data) return `<div class="card"><div class="card-body">Simulator data not loaded.</div></div>`;
  
  return `
    <div class="card">
      <div class="card-header">
        <h3>Time-Machine Simulator</h3>
        <span class="badge badge-planned">${data.date}</span>
      </div>
      <div class="card-body">
        <p style="font-size:0.75rem;color:var(--text-secondary);margin-bottom:12px;">
          Scrub through the most chaotic day in the dataset. Watch the AI engine react to cascading failures in real-time.
        </p>
        
        <div class="metric-row" style="margin-bottom: 16px;">
          <div class="metric-card">
            <div class="metric-value" style="color:var(--accent-red)" id="sim-active-count">0</div>
            <div class="metric-label">Active Incidents</div>
          </div>
          <div class="metric-card">
            <div class="metric-value" style="color:var(--accent-cyan)" id="sim-time">--:--</div>
            <div class="metric-label">Time (IST)</div>
          </div>
        </div>

        <div class="form-group">
          <label class="form-label" style="display:flex; justify-content:space-between;">
            <span>Timeline</span>
            <span id="sim-progress" style="color:var(--accent-blue)">0 / ${data.totalIncidents}</span>
          </label>
          <input type="range" class="form-input" id="sim-slider" min="0" max="${data.totalIncidents}" value="0" style="padding:0; cursor:pointer;" />
        </div>
        
        <button class="btn btn-primary btn-block" id="btn-sim-play" style="margin-top: 10px;">Play Sequence</button>
      </div>
    </div>
    <div id="sim-result"></div>`;
}

function forecastSidebar() {
  return `
    <div class="card">
      <div class="card-header"><h3>Event Configuration</h3></div>
      <div class="card-body" style="display:flex;flex-direction:column;gap:12px;">
        <div class="form-group">
          <label class="form-label">Event Type</label>
          <select class="form-select" id="event-type">
            <option value="public_event">Public Event (Rally/Match)</option>
            <option value="procession">Procession / Festival</option>
            <option value="vip_movement">VIP Movement</option>
            <option value="protest">Protest / Demonstration</option>
            <option value="construction">Construction Activity</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Location (click map)</label>
          <div id="loc-display" style="padding:8px 12px;background:var(--bg-elevated);border-radius:6px;font-size:0.75rem;color:var(--text-secondary);">
            Click on the map to select
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">Expected Hour (IST)</label>
          <select class="form-select" id="event-hour">
            ${Array.from({ length: 24 }, (_, i) => `<option value="${i}" ${i === 18 ? 'selected' : ''}>${hourLabel(i)}</option>`).join('')}
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Day of Week</label>
          <select class="form-select" id="event-day">
            <option value="0">Monday</option><option value="1">Tuesday</option>
            <option value="2">Wednesday</option><option value="3">Thursday</option>
            <option value="4" selected>Friday</option><option value="5">Saturday</option>
            <option value="6">Sunday</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Duration (hours)</label>
          <input type="number" class="form-input" id="event-duration" value="4" min="1" max="48" />
        </div>
        <button class="btn btn-primary btn-block" id="btn-predict">Predict Impact</button>
      </div>
    </div>
    <div id="forecast-result"></div>`;
}

function exploreSidebar() {
  const corridors = getCorridors().filter(c => c.name !== 'Non-corridor').slice(0, 15);
  return `
    <div class="card">
      <div class="card-header"><h3>Filters</h3></div>
      <div class="card-body" style="display:flex;flex-direction:column;gap:12px;">
        <div class="form-group">
          <label class="form-label">Cause</label>
          <select class="form-select" id="filter-cause">
            <option value="">All</option>
            ${getCauseDistribution().map(c => `<option value="${c.cause}">${CAUSE_LABELS[c.cause] || c.cause} (${c.count})</option>`).join('')}
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Corridor</label>
          <select class="form-select" id="filter-corridor">
            <option value="">All</option>
            ${corridors.map(c => `<option value="${c.name}">${c.name} (${c.count})</option>`).join('')}
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Priority</label>
          <select class="form-select" id="filter-priority">
            <option value="">All</option>
            <option value="High">High</option>
            <option value="Low">Low</option>
          </select>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><h3>Recent Incidents</h3></div>
      <div class="card-body">
        <div class="timeline-feed" id="timeline-feed"></div>
      </div>
    </div>`;
}

function learningSidebar() {
  const events = getPastEvents().slice(0, 20);
  return `
    <div class="card">
      <div class="card-header"><h3>Past Events</h3></div>
      <div class="card-body">
        <p style="font-size:0.75rem;color:var(--text-muted);margin-bottom:12px;">Select an event to analyze its impact</p>
        <div class="timeline-feed" id="event-list">
          ${events.map((e, i) => `
            <div class="timeline-item" data-event-idx="${i}">
              <div class="timeline-dot" style="background:${CAUSE_COLORS[e.cause] || '#64748b'}"></div>
              <div class="timeline-body">
                <div class="timeline-title">${CAUSE_LABELS[e.cause] || e.cause}</div>
                <div class="timeline-meta">${formatTime(e.start)} · ${e.corridor || 'Non-corridor'}</div>
                ${e.desc ? `<div class="timeline-desc">${e.desc.substring(0, 80)}</div>` : ''}
              </div>
            </div>`).join('')}
        </div>
      </div>
    </div>
    <div id="learning-result"></div>`;
}

function attachSidebarListeners() {
  const predictBtn = document.getElementById('btn-predict');
  if (predictBtn) {
    predictBtn.addEventListener('click', runForecast);
  }

  // Explore mode filters
  ['filter-cause', 'filter-corridor', 'filter-priority'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', updateExploreTimeline);
  });
  if (currentMode === 'explore') updateExploreTimeline();

  // Learning mode event selection
  const eventList = document.getElementById('event-list');
  if (eventList) {
    eventList.addEventListener('click', (e) => {
      const item = e.target.closest('.timeline-item');
      if (!item) return;
      const idx = parseInt(item.dataset.eventIdx);
      const events = getPastEvents().slice(0, 20);
      if (events[idx]) runLearningAnalysis(events[idx]);
    });
  }

  // Simulator mode
  const simSlider = document.getElementById('sim-slider');
  if (simSlider) {
    simSlider.addEventListener('input', (e) => {
      const frame = parseInt(e.target.value);
      Simulator.setFrame(frame);
      document.getElementById('sim-progress').textContent = `${frame} / ${simSlider.max}`;
      renderIncidents(); // this will redraw the glowing dots up to currentFrame
      
      const res = Simulator.runSimulationFrame();
      if (res) {
        // center map on incident
        map.setView([res.incident.lat, res.incident.lng], 14);
        
        // update sidebar metric
        document.getElementById('sim-time').textContent = formatTime(res.incident.start);
        document.getElementById('sim-active-count').textContent = frame;

        // Render decision
        const el = document.getElementById('sim-result');
        if (el) {
          el.innerHTML = '';
          const tempDiv = document.createElement('div');
          // Temporarily override the main element to reuse renderForecastResult HTML
          tempDiv.id = 'forecast-result';
          el.appendChild(tempDiv);
          // Redefine getElementById temporarily? No, renderForecastResult uses document.getElementById('forecast-result')
          // Let's just create the DOM node with that ID inside sim-result
          renderForecastResult(res.prediction, res.resources, res.decision);
          renderImpactZones(res.prediction);
          renderBarricadeMarkers(res.resources.barricadePoints);
          renderRightPanel(res.prediction, res.resources, res.decision);
        }
      } else {
        document.getElementById('sim-result').innerHTML = '';
        clearImpactLayers();
      }
    });
  }

  const simPlayBtn = document.getElementById('btn-sim-play');
  if (simPlayBtn && simSlider) {
    let playInterval = null;
    simPlayBtn.addEventListener('click', () => {
      if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
        simPlayBtn.textContent = 'Play Sequence';
      } else {
        simPlayBtn.textContent = 'Pause Sequence';
        playInterval = setInterval(() => {
          let val = parseInt(simSlider.value);
          if (val >= parseInt(simSlider.max)) {
            clearInterval(playInterval);
            playInterval = null;
            simPlayBtn.textContent = 'Play Sequence';
            return;
          }
          simSlider.value = val + 1;
          simSlider.dispatchEvent(new Event('input'));
        }, 1000); // 1 frame per second
      }
    });
  }
}

// --- Forecast Execution ---
function findNearestCorridor(lat, lng) {
  // Find the corridor most represented among nearby incidents
  const nearby = getIncidentsNear(lat, lng, 2);
  const counts = {};
  for (const inc of nearby) {
    if (inc.corridor && inc.corridor !== 'Non-corridor') {
      counts[inc.corridor] = (counts[inc.corridor] || 0) + 1;
    }
  }
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return sorted.length > 0 ? sorted[0][0] : 'Non-corridor';
}

function findNearestZone(lat, lng) {
  const nearby = getIncidentsNear(lat, lng, 2);
  const counts = {};
  for (const inc of nearby) {
    if (inc.zone) counts[inc.zone] = (counts[inc.zone] || 0) + 1;
  }
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return sorted.length > 0 ? sorted[0][0] : 'Unknown';
}

function runForecast() {
  if (!selectedLocation) {
    alert('Please click on the map to select an event location.');
    return;
  }
  const cause = document.getElementById('event-type').value;
  const hour = parseInt(document.getElementById('event-hour').value);
  const day = parseInt(document.getElementById('event-day').value);
  const duration = parseFloat(document.getElementById('event-duration').value) || 4;

  // Statistical impact prediction
  const prediction = predictImpact(cause, selectedLocation.lat, selectedLocation.lng, hour, day, duration);
  // ML decision tree -- what action should we take?
  const corridor = findNearestCorridor(selectedLocation.lat, selectedLocation.lng);
  const zone = findNearestZone(selectedLocation.lat, selectedLocation.lng);
  const eventType = cause === 'construction' ? 'planned' : 'unplanned';
  const decision = decide(cause, corridor, zone, hour, day, eventType, selectedLocation.lat, selectedLocation.lng);

  const resources = recommendResources(prediction, selectedLocation.lat, selectedLocation.lng, cause, decision);

  renderForecastResult(prediction, resources, decision);
  renderImpactZones(prediction);
  renderBarricadeMarkers(resources.barricadePoints);
  renderRightPanel(prediction, resources, decision);
}

function renderForecastResult(prediction, resources, decision) {
  const el = document.getElementById('forecast-result');
  const levelClass = { critical: 'score-critical', high: 'score-high', medium: 'score-medium', low: 'score-low' };
  const actionColors = { monitor: 'var(--accent-green)', deploy_light: 'var(--accent-amber)', deploy_heavy: 'var(--accent-orange)', full_closure: 'var(--accent-red)' };
  const actionIcons = { monitor: '👁️', deploy_light: '🚔', deploy_heavy: '🚧', full_closure: '🚨' };

  // Build reasoning trail HTML
  const reasoningHtml = decision.reasoning.map(step => {
    if (step.type === 'split') {
      return `<div style="display:flex;align-items:flex-start;gap:8px;padding:6px 0;border-bottom:1px solid var(--border-color);">
        <span style="color:${step.direction === 'yes' ? 'var(--accent-green)' : 'var(--accent-red)'};font-weight:600;font-size:0.7rem;margin-top:2px;">${step.direction === 'yes' ? '✓' : '✗'}</span>
        <span style="font-size:0.72rem;color:var(--text-secondary);">${step.condition}</span>
      </div>`;
    } else {
      return `<div style="padding:8px 10px;background:var(--bg-primary);border-radius:6px;margin-top:6px;">
        <div style="font-size:0.75rem;font-weight:600;color:${actionColors[step.action] || 'var(--text-primary)'};">
          ${actionIcons[step.action] || ''} ${step.description}
        </div>
        <div style="font-size:0.68rem;color:var(--text-muted);margin-top:4px;">
          ${step.confidence}% confidence (${step.samples} historical cases)
          ${step.outcomeStats?.median_resolution_min ? ` · Median resolution: ${Math.round(step.outcomeStats.median_resolution_min)}min` : ''}
        </div>
      </div>`;
    }
  }).join('');

  el.innerHTML = `
    <div class="card forecast-result">
      <div class="card-header"><h3>🧠 ML Decision</h3><span class="badge badge-planned" style="font-size:0.65rem;">${Math.round((decision.modelAccuracy || 0) * 100)}% model accuracy</span></div>
      <div class="card-body">
        <div style="padding:10px;background:var(--bg-elevated);border-radius:8px;border-left:3px solid ${actionColors[decision.action] || 'var(--accent-blue)'};margin-bottom:12px;">
          <div style="font-size:0.85rem;font-weight:700;color:${actionColors[decision.action]};">
            ${actionIcons[decision.action] || ''} ${decision.action.replace('_', ' ').toUpperCase()}
          </div>
          ${decision.actionStats ? `<div style="font-size:0.7rem;color:var(--text-muted);margin-top:4px;">Historical: ${decision.actionStats.count} similar cases, median resolution ${decision.actionStats.medianResolutionMin}min</div>` : ''}
        </div>
        <div class="form-label" style="margin-bottom:4px;">Model Reasoning</div>
        <div style="max-height:200px;overflow-y:auto;">${reasoningHtml}</div>
      </div>
    </div>
    <div class="card forecast-result" style="margin-top:12px;">
      <div class="card-header"><h3>Impact Prediction</h3></div>
      <div class="card-body">
        <div class="impact-score">
          <div class="score-circle ${levelClass[prediction.severityLevel] || 'score-low'}">${prediction.severity}</div>
          <div class="score-details">
            <h4>${prediction.severityLevel.charAt(0).toUpperCase() + prediction.severityLevel.slice(1)} Impact</h4>
            <p>${prediction.historicalIncidents} historical incidents in area</p>
            <p>${prediction.closureProbability}% road closure probability</p>
          </div>
        </div>
      </div>
    </div>
    <div class="card forecast-result" style="margin-top:12px;">
      <div class="card-header"><h3>Resource Deployment</h3></div>
      <div class="card-body">
        <div class="metric-row">
          <div class="metric-card"><div class="metric-value" style="color:var(--accent-cyan)">${resources.manpower.trafficPolice}</div><div class="metric-label">Traffic Police</div></div>
          <div class="metric-card"><div class="metric-value" style="color:var(--accent-amber)">${resources.manpower.total}</div><div class="metric-label">Total Personnel</div></div>
        </div>
        <div style="margin-top:12px;">
          <div class="form-label" style="margin-bottom:8px;">Equipment</div>
          <div class="resource-list">
            ${resources.equipment.map(eq => `
              <div class="resource-item">
                <div class="resource-icon">${eq.icon}</div>
                <div class="resource-info"><h5>${eq.name}</h5></div>
                <div class="resource-count">${eq.count}</div>
              </div>`).join('')}
          </div>
        </div>
        ${resources.diversions.length > 0 ? `
          <div style="margin-top:12px;">
            <div class="form-label" style="margin-bottom:8px;">Diversions (Data-Driven)</div>
            ${resources.diversions.map(d => `
              <div style="padding:6px 10px;background:var(--bg-elevated);border-radius:6px;margin-bottom:4px;font-size:0.75rem;">
                <span style="color:var(--accent-red);">${d.from}</span> → <span style="color:var(--accent-green);">${d.to}</span>
                ${d.confidence ? `<span style="color:var(--text-muted);font-size:0.65rem;margin-left:4px;">${d.confidence}% independent</span>` : ''}
              </div>`).join('')}
          </div>` : ''}
        ${resources.respondingStations && resources.respondingStations.length > 0 ? `
          <div style="margin-top:12px;">
            <div class="form-label" style="margin-bottom:8px;">Responding Police Stations</div>
            ${resources.respondingStations.map(ps => `
              <div style="padding:6px 10px;background:var(--bg-elevated);border-radius:6px;margin-bottom:4px;font-size:0.75rem;display:flex;justify-content:space-between;align-items:center;">
                <span>🏛️ ${ps.name}</span>
                <span style="color:var(--text-muted);font-size:0.65rem;">${ps.distanceKm}km · ${ps.incidentCount} cases</span>
              </div>`).join('')}
          </div>` : ''}
        ${resources.barricadePoints && resources.barricadePoints.length > 0 && resources.barricadePoints[0].junction ? `
          <div style="margin-top:12px;">
            <div class="form-label" style="margin-bottom:8px;">Junction Barricades</div>
            ${resources.barricadePoints.slice(0, 5).map(bp => `
              <div style="padding:6px 10px;background:var(--bg-elevated);border-radius:6px;margin-bottom:4px;font-size:0.75rem;display:flex;justify-content:space-between;align-items:center;">
                <span>🚧 ${bp.junction}</span>
                <span style="color:var(--text-muted);font-size:0.65rem;">${bp.count} incidents${bp.closureRate > 0 ? ` · ${Math.round(bp.closureRate * 100)}% closure` : ''}</span>
              </div>`).join('')}
          </div>` : ''}
      </div>
    </div>`;
}

function renderImpactZones(prediction) {
  clearImpactLayers();
  if (!selectedLocation) return;

  const colors = { critical: '#ef4444', high: '#fb923c', moderate: '#fbbf24' };
  for (const zone of prediction.impactZones) {
    const circle = L.circle([selectedLocation.lat, selectedLocation.lng], {
      radius: zone.radiusKm * 1000,
      fillColor: colors[zone.severity] || '#3b82f6',
      fillOpacity: zone.opacity,
      stroke: true,
      color: colors[zone.severity] || '#3b82f6',
      weight: 1,
      opacity: 0.5,
    }).addTo(map);
    impactLayers.push(circle);
  }
}

function renderBarricadeMarkers(points) {
  for (const pt of points) {
    const hasJunction = pt.junction;
    const m = L.marker([pt.lat, pt.lng], {
      icon: L.divIcon({
        className: '',
        html: `<div style="font-size:16px;text-shadow:0 0 4px black;">🚧</div>`,
        iconSize: [20, 20], iconAnchor: [10, 10],
      }),
    }).addTo(map);

    const popupContent = hasJunction
      ? `<b>${pt.junction}</b><br>${pt.type} priority<br>${pt.count} historical incidents${pt.corridors && pt.corridors.length ? '<br>Corridors: ' + pt.corridors.join(', ') : ''}${pt.policeStations && pt.policeStations.length ? '<br>PS: ' + pt.policeStations.join(', ') : ''}${pt.medianResolutionMin ? '<br>Median resolution: ' + Math.round(pt.medianResolutionMin) + 'min' : ''}`
      : `<b>Barricade Point</b><br>${pt.type} priority<br>${pt.count} historical incidents`;

    m.bindPopup(popupContent);
    impactLayers.push(m);
  }
}

function clearImpactLayers() {
  for (const layer of impactLayers) map.removeLayer(layer);
  impactLayers = [];
}

// --- Explore Mode ---
function updateExploreTimeline() {
  const incidents = getAllIncidents();
  const cause = document.getElementById('filter-cause')?.value;
  const corridor = document.getElementById('filter-corridor')?.value;
  const priority = document.getElementById('filter-priority')?.value;

  let filtered = incidents;
  if (cause) filtered = filtered.filter(i => i.cause === cause);
  if (corridor) filtered = filtered.filter(i => i.corridor === corridor);
  if (priority) filtered = filtered.filter(i => i.priority === priority);

  const feed = document.getElementById('timeline-feed');
  if (!feed) return;
  const items = filtered.slice(-50).reverse();
  feed.innerHTML = items.map(inc => {
    const color = CAUSE_COLORS[inc.cause] || '#64748b';
    return `<div class="timeline-item">
      <div class="timeline-dot" style="background:${color}"></div>
      <div class="timeline-body">
        <div class="timeline-title">${CAUSE_LABELS[inc.cause] || inc.cause}</div>
        <div class="timeline-meta">${formatTime(inc.start)} · ${inc.corridor || '—'}</div>
        ${inc.desc ? `<div class="timeline-desc">${inc.desc.substring(0, 80)}</div>` : ''}
      </div>
    </div>`;
  }).join('');

  document.getElementById('visible-count').textContent = `${filtered.length.toLocaleString()} filtered`;
}

// --- Learning Mode ---
function runLearningAnalysis(event) {
  const result = analyzeEvent(event);
  const el = document.getElementById('learning-result');
  if (!el) return;

  el.innerHTML = `
    <div class="card forecast-result" style="margin-top:12px;">
      <div class="card-header"><h3>Event Analysis</h3></div>
      <div class="card-body">
        <div class="metric-row">
          <div class="metric-card"><div class="metric-value" style="color:var(--accent-red)">${result.incidentsDuringEvent}</div><div class="metric-label">Incidents During</div></div>
          <div class="metric-card"><div class="metric-value" style="color:var(--accent-cyan)">${result.baselineAvg}</div><div class="metric-label">Baseline Avg</div></div>
        </div>
        <div class="metric-row" style="margin-top:8px;">
          <div class="metric-card"><div class="metric-value" style="color:var(--accent-amber)">${result.spikeMultiplier}x</div><div class="metric-label">Spike</div></div>
          <div class="metric-card"><div class="metric-value" style="color:${(result.effectivenessScore || 0) >= 70 ? 'var(--accent-green)' : 'var(--accent-red)'}">${result.effectivenessScore != null ? result.effectivenessScore + '%' : '—'}</div><div class="metric-label">Effectiveness</div></div>
        </div>
        ${result.eventMedianResolution ? `<p style="margin-top:12px;font-size:0.75rem;color:var(--text-secondary);">Event resolution: ${formatDuration(result.eventMedianResolution)} vs baseline ${formatDuration(result.baselineMedianResolution)}</p>` : ''}
      </div>
    </div>`;

  // Center map on event
  map.setView([event.lat, event.lng], 14);
  clearImpactLayers();
  const circle = L.circle([event.lat, event.lng], {
    radius: 3000, fillColor: '#f472b6', fillOpacity: 0.15, color: '#f472b6', weight: 1,
  }).addTo(map);
  impactLayers.push(circle);
}

// --- Right Panel ---
function renderRightPanel(prediction, resources, decision) {
  const el = document.getElementById('panel-content');

  // Hourly distribution chart
  const hourlyData = getHourlyDistribution();
  const causeData = getCauseDistribution().slice(0, 8);
  const corridorData = getCorridors().filter(c => c.name !== 'Non-corridor').slice(0, 10);

  let html = '';

  // Affected corridors (if forecast ran)
  if (prediction && prediction.affectedCorridors.length > 0) {
    const maxCount = prediction.affectedCorridors[0]?.count || 1;
    html += `<div class="card">
      <div class="card-header"><h3>Affected Corridors</h3></div>
      <div class="card-body" style="display:flex;flex-direction:column;gap:6px;">
        ${prediction.affectedCorridors.map(c => {
          const pct = Math.round((c.count / maxCount) * 100);
          return `<div class="corridor-bar">
            <div class="corridor-bar-name">${c.name}</div>
            <div class="corridor-bar-track"><div class="corridor-bar-fill" style="width:${pct}%;background:var(--accent-red);"></div></div>
            <div class="corridor-bar-value">${c.count}</div>
          </div>`;
        }).join('')}
      </div>
    </div>`;
  }

  // Hourly chart
  html += `<div class="card">
    <div class="card-header"><h3>Hourly Pattern (IST)</h3></div>
    <div class="card-body"><div class="chart-container"><canvas id="chart-hourly"></canvas></div></div>
  </div>`;

  // Cause chart
  html += `<div class="card">
    <div class="card-header"><h3>Incident Causes</h3></div>
    <div class="card-body"><div class="chart-container"><canvas id="chart-cause"></canvas></div></div>
  </div>`;

  // Top corridors
  html += `<div class="card">
    <div class="card-header"><h3>Top Corridors</h3></div>
    <div class="card-body" style="display:flex;flex-direction:column;gap:6px;">
      ${corridorData.map(c => {
        const pct = Math.round((c.count / corridorData[0].count) * 100);
        return `<div class="corridor-bar">
          <div class="corridor-bar-name">${c.name}</div>
          <div class="corridor-bar-track"><div class="corridor-bar-fill" style="width:${pct}%;background:var(--accent-blue);"></div></div>
          <div class="corridor-bar-value">${c.count}</div>
        </div>`;
      }).join('')}
    </div>
  </div>`;

  el.innerHTML = html;

  // Render charts after DOM update
  requestAnimationFrame(() => {
    renderHourlyChart(hourlyData);
    renderCauseChart(causeData);
  });
}

function renderHourlyChart(data) {
  const ctx = document.getElementById('chart-hourly');
  if (!ctx) return;
  if (charts.hourly) charts.hourly.destroy();

  charts.hourly = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Array.from({ length: 24 }, (_, i) => hourLabel(i)),
      datasets: [{
        data,
        backgroundColor: data.map((_, i) => i >= 4 && i <= 11 ? 'rgba(59,130,246,0.7)' : 'rgba(59,130,246,0.3)'),
        borderRadius: 3,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 9 }, maxRotation: 45 }, grid: { display: false } },
        y: { ticks: { color: '#64748b', font: { size: 9 } }, grid: { color: 'rgba(148,163,184,0.08)' } },
      },
    },
  });
}

function renderCauseChart(data) {
  const ctx = document.getElementById('chart-cause');
  if (!ctx) return;
  if (charts.cause) charts.cause.destroy();

  charts.cause = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: data.map(d => CAUSE_LABELS[d.cause] || d.cause),
      datasets: [{
        data: data.map(d => d.count),
        backgroundColor: data.map(d => CAUSE_COLORS[d.cause] || '#64748b'),
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '65%',
      plugins: {
        legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 10, padding: 8 } },
      },
    },
  });
}

// --- Start ---
init().catch(console.error);
