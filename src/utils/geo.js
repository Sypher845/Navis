/**
 * geo.js — Geospatial utility functions
 *
 * All distance calculations use the Haversine formula.
 * No external dependencies.
 */

const EARTH_RADIUS_KM = 6371;

/**
 * Calculate distance between two points in kilometers.
 */
export function haversine(lat1, lng1, lat2, lng2) {
  const toRad = (deg) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return EARTH_RADIUS_KM * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/**
 * Get bounding box around a point for fast spatial filtering.
 * Returns { minLat, maxLat, minLng, maxLng }.
 */
export function boundingBox(lat, lng, radiusKm) {
  const latDelta = radiusKm / 111.32;
  const lngDelta = radiusKm / (111.32 * Math.cos((lat * Math.PI) / 180));
  return {
    minLat: lat - latDelta,
    maxLat: lat + latDelta,
    minLng: lng - lngDelta,
    maxLng: lng + lngDelta,
  };
}

/**
 * Simple point-in-bounding-box check.
 */
export function inBoundingBox(lat, lng, bbox) {
  return (
    lat >= bbox.minLat &&
    lat <= bbox.maxLat &&
    lng >= bbox.minLng &&
    lng <= bbox.maxLng
  );
}

/**
 * Generate circle polygon points for map rendering.
 * Returns array of {lat, lng} points.
 */
export function circlePoints(centerLat, centerLng, radiusKm, segments = 64) {
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const angle = (2 * Math.PI * i) / segments;
    const dLat = (radiusKm / 111.32) * Math.cos(angle);
    const dLng =
      (radiusKm / (111.32 * Math.cos((centerLat * Math.PI) / 180))) *
      Math.sin(angle);
    points.push({ lat: centerLat + dLat, lng: centerLng + dLng });
  }
  return points;
}
