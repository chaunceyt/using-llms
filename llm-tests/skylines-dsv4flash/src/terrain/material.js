// terrain/material.js — photographic splat-PBR ground material.
//
// A single custom GLSL shader blends four procedural albedo layers
// (grass / sand / dirt / rock) by slope + height, sampled with triplanar
// mapping so steep cliffs don't stretch. Lighting is driven by a sun
// direction/color computed from the time of day, plus a hemisphere ambient.
// The terrain is self-shaded (environment module is a stub), and the renderer's
// global ACESFilmic tone mapping + sRGB output conversion are applied to the
// linear color we output.
import * as THREE from 'three';

// Physical-ish sun state for a given time of day (seconds since midnight).
export function computeSun(timeOfDaySec) {
  const hours = timeOfDaySec / 3600;
  const lat = 0.6;          // ~34 deg latitude
  const decl = 0.35 * Math.sin(((timeOfDaySec - 82 * 86400) / (365 * 86400)) * Math.PI * 2);
  const hAngle = (hours - 12) * (Math.PI / 12);

  const sinEl = Math.sin(decl) * Math.sin(lat)
    + Math.cos(decl) * Math.cos(lat) * Math.cos(hAngle);
  let elev = Math.asin(Math.max(-0.25, Math.min(1, sinEl)));

  // Azimuth (approx): south at noon; negative in the afternoon (west).
  const cosAz = (Math.sin(decl) - Math.sin(elev) * Math.sin(lat))
    / (Math.cos(elev) * Math.cos(lat));
  let azim = Math.acos(Math.max(-1, Math.min(1, cosAz)));
  if (hAngle > 0) azim = -azim;

  const ce = Math.cos(elev);
  const dir = new THREE.Vector3(Math.sin(azim) * ce, Math.sin(elev), Math.cos(azim) * ce);

  // Color: warm at low elevation, near-white overhead.
  const t = THREE.MathUtils.smoothstep(-0.08, 0.32, elev);
  const color = new THREE.Color(1, 1, 1)
    .lerp(new THREE.Color(1.0, 0.58, 0.34), 1 - t);   // warm tint at sunrise/set
  color.multiplyScalar(Math.max(elev * 2.2, 0.05));    // dim when sun low/below horizon

  return { dir, color };
}

const vertexShader = /* glsl */`
// 'position' and 'normal' attributes are injected by three.js — do not redeclare.
varying vec3 vWorldPos;
varying vec3 vNormal;

void main() {
  vWorldPos = position;                 // geometry is authored in world coords
  vNormal = normalize(normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = /* glsl */`
precision highp float;

varying vec3 vWorldPos;
varying vec3 vNormal;

uniform sampler2D mapGrass;
uniform sampler2D mapSand;
uniform sampler2D mapDirt;
uniform sampler2D mapRock;

uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;
uniform float uTile;            // metres per texture tile
uniform vec3 uCamPos;

// ---- tiny procedural value noise (for detail + micro-bump) ----
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash(i), b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0)), d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p){
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + 17.7; a *= 0.5; }
  return s;
}

// Triplanar sample of an albedo map (avoids stretching on cliffs).
vec3 tri(sampler2D m, vec3 P, vec3 N){
  float t = uTile;
  vec3 w = abs(N);
  w = pow(w, vec3(1.5));
  w /= (w.x + w.y + w.z);
  vec3 col =
      w.x * texture2D(m, P.zy / t).rgb
    + w.y * texture2D(m, P.xz / t).rgb
    + w.z * texture2D(m, P.xy / t).rgb;
  return pow(col, vec3(2.2));   // sRGB -> linear for correct lighting
}

void main(){
  vec3 P = vWorldPos;
  vec3 N = normalize(vNormal);
  float h = P.y;

  // micro-bump from procedural detail so the ground reads rough up close.
  float det = fbm(P.xz * 0.11) + fbm(P.xz * 0.045) * 1.6;
  vec3 bump = normalize(vec3(-dFdx(det), 1.0, -dFdy(det)));
  N = normalize(N * 0.55 + bump * 0.45);

  float slope = clamp(1.0 - N.y, 0.0, 1.0);

  vec3 grass = tri(mapGrass, P, N);
  vec3 sand  = tri(mapSand,  P, N);
  vec3 dirt  = tri(mapDirt,  P, N);
  vec3 rock  = tri(mapRock,  P, N);

  // ---- layer weights from height + slope ----
  float wetness   = smoothstep(42.0, 6.0, h);                 // low ground -> sand/mud (floodplain)
  float sandW     = wetness * (1.0 - slope * 0.4);
  float rockBySlope = smoothstep(0.16, 0.42, slope);
  float rockByH    = smoothstep(150.0, 225.0, h);
  float rockW      = clamp(rockBySlope + rockByH * 0.6, 0.0, 1.0);
  float dirtW      = (1.0 - sandW) * smoothstep(0.10, 0.32, slope) * (1.0 - rockW);
  float grassW     = clamp(1.0 - sandW - rockW - dirtW, 0.0, 1.0);

  float wsum = grassW + sandW + dirtW + rockW;
  vec3 albedo = (grass * grassW + sand * sandW + dirt * dirtW + rock * rockW) / max(wsum, 0.001);

  // broad color variation so it isn't one flat green.
  albedo *= mix(0.9, 1.08, fbm(P.xz * 0.006));

  // ---- lighting: hemisphere ambient + directional sun ----
  float amb = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 ambCol = mix(uGroundColor, uSkyColor, amb);
  float ndl = dot(N, uSunDir);
  float diff = max(ndl, 0.0) * 0.7
             + pow(max(ndl + 0.18, 0.0), 3.0) * 0.5;   // soft wrap for a gentle light-dark side
  vec3 col = albedo * (ambCol * 1.15 + uSunColor * diff);

  // simple height/slope AO for crevice depth.
  float ao = clamp(1.0 - slope * 0.32 - smoothstep(6.0, 26.0, h) * 0.05, 0.55, 1.0);
  col *= ao;

  // cheap distance fog toward the horizon for aerial depth (fades to sky).
  float dist = length(P.xz - uCamPos.xz);
  float fog = smoothstep(1800.0, 3200.0, dist);
  vec3 fogCol = uSkyColor * 1.6;
  col = mix(col, fogCol, clamp(fog, 0.0, 0.8));

  gl_FragColor = vec4(col, 1.0);
}
`;

export function createTerrainMaterial(maps) {
  const mat = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      mapGrass: { value: maps.grass },
      mapSand:  { value: maps.sand },
      mapDirt:  { value: maps.dirt },
      mapRock:  { value: maps.rock },
      uSunDir:  { value: new THREE.Vector3(0.5, 1.0, 0.4).normalize() },
      uSunColor:{ value: new THREE.Color(1, 0.96, 0.9) },
      uSkyColor:{ value: new THREE.Color(0.35, 0.5, 0.72) },
      uGroundColor:{ value: new THREE.Color(0.16, 0.19, 0.15) },
      uTile:    { value: 70.0 },
      uCamPos:  { value: new THREE.Vector3(0, 100, 0) },
    },
  });
  return mat;
}
