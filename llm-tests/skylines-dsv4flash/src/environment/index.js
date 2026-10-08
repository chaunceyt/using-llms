// ===========================================================================
// environment — sun / sky dome / lighting / weather / fog
//
// Owns:
//   * a physically-based sun arc driven by time-of-day (seconds since midnight)
//   * a procedural analytic sky-dome shader (gradient + twilight band + mie glow
//     around the sun + deterministic star field at night)
//   * a key shadow-casting DirectionalLight that tracks the sun, plus a
//     HemisphereLight modulated by time/weather
//   * an optional procedural cloud dome (single instanced shell, cheap)
//   * time/weather-matched fog and scene background
//
// Weather ids per ARCHITECTURE: clear | overcast | rain | snow | fog.
// Listens to bus `timeofday:changed` / `weather:changed` so external drives
// (e.g. the screenshot tool) update lighting live.
// ===========================================================================
import * as THREE from 'three';
import { bus } from '../core/index.js';

export const id = 'environment';

// ---------------------------------------------------------------------------
// Weather table. Each entry tunes fog, sun dimming, sky grey-out, cloud cover
// and star visibility.
// ---------------------------------------------------------------------------
const WEATHER = {
  clear:    { fogDensity: 0.00016, sunDim: 1.0,  skyGray: 0.0,  cloud: 0.30, stars: 1.0 },
  overcast: { fogDensity: 0.00050, sunDim: 0.55, skyGray: 0.80, cloud: 0.90, stars: 0.15 },
  rain:     { fogDensity: 0.00075, sunDim: 0.40, skyGray: 0.95, cloud: 1.0,  stars: 0.03 },
  snow:     { fogDensity: 0.00060, sunDim: 0.50, skyGray: 0.98, cloud: 1.0,  stars: 0.03 },
  fog:      { fogDensity: 0.00140, sunDim: 0.70, skyGray: 0.95, cloud: 1.0,  stars: 0.0 },
};

const SKY_RADIUS = 5200;
const CLOUD_RADIUS = 4400;

const state = {
  timeOfDaySec: 9 * 3600,
  weather: 'clear',
  elapsed: 0,
};

// last-computed fog params, re-asserted each frame (environment owns fog).
const fogState = { color: new THREE.Color(0x070a16), density: 0.0002 };

let tempDomeFound = false;

let worldRef = null;
let skyDome = null, cloudDome = null, sunSprite = null;
let keyLight = null, hemi = null, backgroundCol = null;
let skyU = null, cloudU = null;

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function smoothstep(a, b, x) { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }

// ---------------------------------------------------------------------------
// Sun arc: solar elevation is a cosine that peaks at noon and crosses the
// horizon around ~06:24 / ~18:36; azimuth sweeps east(ish) -> southwest so the
// setting sun sits in view of the sunset camera preset.
// ---------------------------------------------------------------------------
function computeSun(sec) {
  const noon = 12 * 3600;
  const h = (sec - noon) / 3600;          // hours since solar noon
  const half = 6.5;
  const x = Math.min(Math.abs(h) / half, 1.4);
  const elev = (Math.PI / 2) * Math.cos(x * (Math.PI / 2));   // rad, -~53deg..+90deg
  const hc = clamp(h, -half, half);
  const az = Math.PI * 0.75 + (hc / half) * (Math.PI * 0.5);  // 45deg..225deg
  const sd = new THREE.Vector3(
    Math.cos(elev) * Math.sin(az),
    Math.sin(elev),
    Math.cos(elev) * Math.cos(az),
  ).normalize();
  return { sunDir: sd, elev };
}

// ---------------------------------------------------------------------------
// Sky shaders
// ---------------------------------------------------------------------------
const DOME_VERT = `
varying vec3 vWorldPos;
void main(){
  vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}`;

const SKY_FRAG = `
uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform float uSunElevation;
uniform vec3 uZenith;
uniform vec3 uHorizon;
uniform vec3 uSunsetGlow;
uniform float uTwilight;
uniform float uNight;
uniform float uStars;
varying vec3 vWorldPos;
const float PI = 3.14159265359;

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453123); }

void main(){
  vec3 dir = normalize(vWorldPos - uCamPos);
  float y = dir.y;

  // base vertical gradient: horizon -> zenith
  float up = clamp(y, 0.0, 1.0);
  vec3 sky = mix(uHorizon, uZenith, pow(up, 0.55));

  // warm twilight band hugging the horizon at sunrise/sunset
  float tb = exp(-abs(y) * 8.0) * uTwilight;
  sky += uSunsetGlow * tb;

  // below-horizon falls to a dark ground tone
  float grd = smoothstep(0.0, -0.35, y);
  sky = mix(sky, vec3(0.030, 0.038, 0.058), grd);

  // sun mie halo + soft wide twilight haze on the sun-side hemisphere
  float cosSun = clamp(dot(dir, uSunDir), -1.0, 1.0);
  {
    float g = 0.85; float gg = g * g;
    float hg = (1.0 - gg) / (4.0 * PI * pow(max(1.0 + gg - 2.0 * g * cosSun, 1e-4), 1.5));
    vec3 glowCol = mix(uSunsetGlow, vec3(1.0, 0.98, 0.90), smoothstep(-0.05, 0.35, uSunElevation));
    float amt = hg * 0.028 + pow(max(cosSun, 0.0), 7.0) * uTwilight * 0.6;
    sky += glowCol * amt;
  }

  // deterministic star field, only when the sun is below the horizon
  if (uStars > 0.01) {
    float th = acos(clamp(dir.y, -1.0, 1.0));
    float ph = atan(dir.x, dir.z);
    vec2 g2 = vec2(th, ph) * 90.0;
    vec2 cell = floor(g2);
    float r = hash(cell + 12.9898);
    float tw = (r > 0.9935) ? pow(r, 18.0) : 0.0;
    sky += uStars * 4.0 * tw * vec3(1.0, 1.0, 1.03);
  }

  gl_FragColor = vec4(sky, 1.0);
}`;

// ---------------------------------------------------------------------------
// Cloud dome — a transparent upper hemisphere with fbm coverage; lit by the sun
// side so clouds catch warm light at sunset.
// ---------------------------------------------------------------------------
const CLOUD_FRAG = `
uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform float uTime;
uniform float uAmount;
varying vec3 vWorldPos;

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453123); }
float noise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  f = f*f*(3.0-2.0*f);
  return mix(mix(hash(i), hash(i+vec2(1.0,0.0)), f.x),
             mix(hash(i+vec2(0.0,1.0)), hash(i+vec2(1.0,1.0)), f.x), f.y);
}
float fbm(vec2 p){
  float v = 0.0, a = 0.5;
  mat2 r = mat2(1.6, 1.2, -1.2, 1.6);
  for (int i = 0; i < 4; i++){ v += a*noise(p); p = r*p + vec2(0.3,0.7); a *= 0.5; }
  return v;
}
void main(){
  vec3 dir = normalize(vWorldPos - uCamPos);
  if (dir.y < -0.05) discard;

  float th = acos(clamp(dir.y, -1.0, 1.0));
  float ph = atan(dir.x, dir.z);
  vec2 uv = vec2(ph * 2.4, th * 3.0);
  float n = fbm(uv + uTime * 0.004);

  // thin near horizon and near the dome apex
  float vf = smoothstep(0.03, 0.20, dir.y) * smoothstep(0.99, 0.72, dir.y);
  float cov = smoothstep(0.40, 0.68, n) * uAmount * vf;
  if (cov < 0.01) discard;

  float sunSide = clamp(dot(dir, uSunDir), 0.0, 1.0);
  vec3 col = mix(vec3(0.28, 0.32, 0.42), vec3(1.00, 0.97, 0.90), 0.30 + 0.70 * sunSide) * 1.4;
  col *= mix(0.55, 1.0, smoothstep(0.02, 0.22, dir.y));
  gl_FragColor = vec4(col, cov);
}`;

function makeSunTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0.00, 'rgba(255,252,240,1)');
  gr.addColorStop(0.22, 'rgba(255,236,200,0.95)');
  gr.addColorStop(0.55, 'rgba(255,205,150,0.45)');
  gr.addColorStop(1.00, 'rgba(255,180,110,0)');
  g.fillStyle = gr;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

function ensureFog() {
  const fog = worldRef && worldRef.scene ? worldRef.scene.fog : null;
  if (!(fog instanceof THREE.FogExp2)) {
    worldRef.scene.fog = new THREE.FogExp2(0x070a16, 0.0002);
  }
}

// ---------------------------------------------------------------------------
// Recompute all sky + light parameters from current time/weather. Cheap; called
// on init, on setTimeOfDay/setWeather and via bus events.
// ---------------------------------------------------------------------------
function applyEnvironment() {
  if (!worldRef || !skyDome) return;
  const { sunDir, elev } = computeSun(state.timeOfDaySec);
  const w = WEATHER[state.weather] || WEATHER.clear;

  const dayFrac = smoothstep(-0.05, 0.20, elev);
  const nightFrac = smoothstep(0.03, -0.06, elev);

  // ---- sky colors ---------------------------------------------------------
  const zN = new THREE.Color(0x040b18);        // deep navy (night zenith)
  const zD = new THREE.Color(0x2f57c0);        // day blue
  const zen = zN.clone().lerp(zD, dayFrac);

  const hN = new THREE.Color(0x070a16);
  const hD = new THREE.Color(0xbfd3e2);
  const warm = new THREE.Color(1.0, 0.42, 0.16);
  let hor = hN.clone().lerp(hD, dayFrac);

  const tw = Math.exp(-Math.abs(elev) * 6.5);        // peaks at sunrise/sunset
  hor.lerp(warm, clamp(tw * (0.45 + 0.55 * dayFrac), 0, 1));

  // weather grey-out
  const gcol = new THREE.Color(0x9aa2ac);
  zen.lerp(gcol, w.skyGray);
  hor.lerp(gcol, w.skyGray * 0.7);

  skyU.uZenith.value.copy(zen);
  skyU.uHorizon.value.copy(hor);
  skyU.uSunsetGlow.value.copy(warm).multiplyScalar(1.2);
  skyU.uSunDir.value.copy(sunDir);
  skyU.uSunElevation.value = elev;
  skyU.uTwilight.value = tw;
  skyU.uNight.value = nightFrac;
  skyU.uStars.value = nightFrac * w.stars;

  cloudU.uSunDir.value.copy(sunDir);
  cloudU.uAmount.value = w.cloud * clamp(dayFrac * 1.4, 0.15, 1.0);

  // scene background + fog match the horizon so edges blend seamlessly.
  // Re-assert a FogExp2 each time (environment owns fog; effects may try to
  // clear/replace it — see update()).
  backgroundCol.copy(hor);
  let density = w.fogDensity;
  if (elev < -0.10) density *= 0.5;                  // thinner haze at night
  ensureFog();
  worldRef.scene.fog.color.copy(hor);
  worldRef.scene.fog.density = density;
  fogState.color.copy(hor);
  fogState.density = density;

  // ---- lights -------------------------------------------------------------
  const warmLight = new THREE.Color(1.0, 0.55, 0.32);
  const white = new THREE.Color(1.0, 0.97, 0.92);
  keyLight.color.copy(warmLight).lerp(white, smoothstep(-0.02, 0.3, elev));
  let sunI = 2.9 * w.sunDim * clamp((elev + 0.12) / 0.22, 0.05, 1.0);
  if (elev < 0) keyLight.color.setRGB(0.42, 0.48, 0.62);   // cool moonlight
  keyLight.intensity = sunI;
  keyLight.position.copy(sunDir).multiplyScalar(2500);

  const hs = hD.clone().lerp(gcol, w.skyGray * 0.5);
  hemi.color.copy(hs);
  hemi.groundColor.copy(new THREE.Color(0x2a3433));
  hemi.intensity = (0.85 * dayFrac + 0.16 * nightFrac) * clamp(w.sunDim, 0.3, 1.0);

  // sun disc sprite
  const cam = worldRef.camera;
  if (cam) {
    sunSprite.position.copy(cam.position).addScaledVector(sunDir, 3800);
  }
  sunSprite.material.opacity = smoothstep(-0.04, 0.10, elev) * w.sunDim;
  const rad = THREE.MathUtils.lerp(1100, 340, smoothstep(-0.02, 0.40, elev));
  sunSprite.scale.set(rad, rad, 1);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
export function init(world) {
  worldRef = world;
  const cam = world.camera;

  if (world.meta && typeof world.meta.timeOfDaySec === 'number') state.timeOfDaySec = world.meta.timeOfDaySec;
  if (world.meta && world.meta.weather && WEATHER[world.meta.weather]) state.weather = world.meta.weather;

  backgroundCol = new THREE.Color(0x0a1117);
  world.scene.background = backgroundCol;

  // ---- sky dome ------------------------------------------------------------
  skyU = {
    uCamPos:       { value: new THREE.Vector3() },
    uSunDir:       { value: new THREE.Vector3(0, 1, 0) },
    uSunElevation: { value: 0.5 },
    uZenith:       { value: new THREE.Color(0x2f57c0) },
    uHorizon:      { value: new THREE.Color(0xbfd3e2) },
    uSunsetGlow:   { value: new THREE.Color(1.0, 0.42, 0.16) },
    uTwilight:     { value: 0 },
    uNight:        { value: 0 },
    uStars:        { value: 0 },
  };
  const skyMat = new THREE.ShaderMaterial({
    vertexShader: DOME_VERT,
    fragmentShader: SKY_FRAG,
    uniforms: skyU,
    side: THREE.BackSide,
    depthWrite: false,
  });
  skyDome = new THREE.Mesh(new THREE.SphereGeometry(SKY_RADIUS, 40, 24), skyMat);
  skyDome.frustumCulled = false;
  // render after effects' temporary atmosphere dome (renderOrder -2) so the
  // real sky supersedes it; before clouds/sprite.
  skyDome.renderOrder = 0;
  world.scene.add(skyDome);

  // ---- cloud dome ------------------------------------------------------------
  cloudU = {
    uCamPos: { value: new THREE.Vector3() },
    uSunDir: { value: new THREE.Vector3(0, 1, 0) },
    uTime:   { value: 0 },
    uAmount: { value: 0.3 },
  };
  const cloudMat = new THREE.ShaderMaterial({
    vertexShader: DOME_VERT,
    fragmentShader: CLOUD_FRAG,
    uniforms: cloudU,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false,
  });
  cloudDome = new THREE.Mesh(
    new THREE.SphereGeometry(CLOUD_RADIUS, 32, 12, 0, Math.PI * 2, 0, Math.PI * 0.5),
    cloudMat,
  );
  cloudDome.frustumCulled = false;
  cloudDome.renderOrder = 10;
  world.scene.add(cloudDome);

  // ---- sun disc sprite -----------------------------------------------------
  const smat = new THREE.SpriteMaterial({
    map: makeSunTexture(),
    color: 0xffffff,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  sunSprite = new THREE.Sprite(smat);
  sunSprite.renderOrder = 11;
  world.scene.add(sunSprite);

  // ---- lights ---------------------------------------------------------------
  keyLight = new THREE.DirectionalLight(0xffffff, 2.9);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  const sc = keyLight.shadow.camera;
  sc.left = -1600; sc.right = 1600; sc.top = 1600; sc.bottom = -1600;
  sc.near = 200; sc.far = 4200;
  keyLight.shadow.bias = -0.0004;
  keyLight.shadow.normalBias = 1.2;
  keyLight.shadow.radius = 6;
  const target = new THREE.Object3D();
  world.scene.add(target);
  keyLight.target = target;
  world.scene.add(keyLight);

  if (world.renderer) {
    world.renderer.shadowMap.enabled = true;
    world.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  }

  hemi = new THREE.HemisphereLight(0x9db8e6, 0x2a3433, 0.7);
  world.scene.add(hemi);

  // ---- fog ------------------------------------------------------------------
  world.scene.fog = new THREE.FogExp2(backgroundCol.getHex(), 0.0002);

  // ---- bus listeners (external drives update lighting live) -------------------
  if (bus) {
    bus.on('timeofday:changed', ({ sec }) => { state.timeOfDaySec = sec; applyEnvironment(); });
    bus.on('weather:changed', ({ id }) => { if (WEATHER[id]) { state.weather = id; applyEnvironment(); } });
  }

  applyEnvironment();
}

export function update(dt, world) {
  const cam = world && world.camera;
  if (!cam || !skyDome) return;
  state.elapsed += dt || 0;

  skyU.uCamPos.value.copy(cam.position);
  cloudU.uCamPos.value.copy(cam.position);
  cloudU.uTime.value = state.elapsed;

  skyDome.position.copy(cam.position);
  cloudDome.position.copy(cam.position);
  sunSprite.position.copy(cam.position).addScaledVector(skyU.uSunDir.value, 3800);

  // Re-assert fog ownership each frame: environment owns the fog, so if another
  // module clears/replaces scene.fog we restore our FogExp2 with current params.
  ensureFog();
  world.scene.fog.color.copy(fogState.color);
  world.scene.fog.density = fogState.density;

  // The effects module ships a temporary atmosphere dome marked as superseded
  // once the environment lands. Hide it so the real sky (ours) is what renders.
  if (!tempDomeFound) {
    const child = world.scene.children.find(c => c.name === 'effectsAtmosphereDomeTEMP');
    if (child) { child.visible = false; tempDomeFound = true; }
  }
}

export function setTimeOfDay(sec) {
  state.timeOfDaySec = sec;
  applyEnvironment();
}

export function setWeather(id) {
  if (!WEATHER[id]) return;
  state.weather = id;
  applyEnvironment();
}

export function showcase(container) {
  try {
    if (container) {
      container.innerHTML = `<div style="padding:12px;font-family:sans-serif;color:#cfe0ff;
        background:#0a1117">environment — ${state.weather}, TOD ${(state.timeOfDaySec / 3600).toFixed(1)}h</div>`;
    }
  } catch (e) { /* isolation */ }
}
