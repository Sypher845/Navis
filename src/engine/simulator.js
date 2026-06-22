import playbackData from '../../exports/simulation_playback.json';
import { decide } from './decision.js';
import { recommendResources } from './resource.js';
import { predictImpact } from './forecast.js';

let currentFrame = 0;
let totalFrames = playbackData.incidents.length;

export function getSimulationData() {
  return playbackData;
}

export function setFrame(frame) {
  currentFrame = Math.max(0, Math.min(frame, totalFrames));
  return currentFrame;
}

export function getCurrentIncidents() {
  return playbackData.incidents.slice(0, currentFrame);
}

export function runSimulationFrame() {
  if (currentFrame === 0) return null;
  const currentIncident = playbackData.incidents[currentFrame - 1];
  
  const hour = new Date(currentIncident.start).getHours();
  const day = new Date(currentIncident.start).getDay();
  // JS getDay() is 0 (Sun) to 6 (Sat). Our Python code used 0 (Mon) to 6 (Sun).
  const pythonDay = day === 0 ? 6 : day - 1;

  // ML decision tree -- what action should we take?
  const decision = decide(
    currentIncident.cause, 
    currentIncident.corridor, 
    currentIncident.zone, 
    hour, 
    pythonDay, 
    currentIncident.type, 
    currentIncident.lat, 
    currentIncident.lng
  );

  // Statistical impact prediction
  const prediction = predictImpact(
    currentIncident.cause, 
    currentIncident.lat, 
    currentIncident.lng, 
    hour, 
    pythonDay, 
    4 // assume 4 hr duration
  );

  const resources = recommendResources(
    prediction, 
    currentIncident.lat, 
    currentIncident.lng, 
    currentIncident.cause, 
    decision
  );

  return {
    incident: currentIncident,
    decision,
    prediction,
    resources
  };
}
