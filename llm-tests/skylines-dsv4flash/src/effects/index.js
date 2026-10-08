// =============================================================================
// effects — particle + atmosphere module (Skylines)
//
// Believable rain / snow / mist driven by weather, plus a cheap in-scene
// atmosphere stand-in (gradient sky dome with an analytic sun glow + vignette).
// All particles are instanced-ish single draw calls (Lines/Points), frustum +
// distance cheap, one shared approach: buffer-geometry updates on CPU.
//
// Weather ids: clear | overcast | rain | snow | fog  (default 'clear').
// Listens for `weather:changed` so external setWeather() (screenshot tool)
// turns the systems on. URL param `?fx=rain|snow` triggers a self-test override
// at boot for verification (shipped default stays clear).
//
// TEMPORARY atmosphere dome: environment module owns the real sky when it ships.
// Disable with  world.modules.effects.atmosphereDome.visible = false  once env lands.
// =============================================================================
import { THREE, bus, setWeather, mulberry32 } from '../core/index.js';

export const id = 'effects';

// Effects randomness MUST NOT consume the shared deterministic stream: rain/snow
// respawn happens per-frame, so drawing from world.rng() would make how many draws
// occur depend on framerate — breaking "same seed → same city" for downstream
// consumers (zoning/buildings). Use a module-local seeded PRNG instead.
const localRng = mulberry32(0x5eed0007);

const WEATHERS = ['clear', 'overcast', 'rain', 'snow', 'fog'];
const RAIN_N   = 4200;          // number of rain streaks (one LineSegments draw)
const SNOW_N   = 2400;          // drifting flakes (one Points draw)
const MIST_N   = 140;           // low drifting haze blobs
const BOX_HALF = 62;            // half-width of the region anchored to camera (m)
const Y_TOP    = 48;
const Y_BOT    = -58;

// ---------------------------------------------------------------------------
let worldRef = null;
let weather = 'clear';

// Systems
let rainSys = null;   // LineSegments
let snowSys = null;   // Points
let mistSys = null;   // Points
let dome = null;      // gradient sky + sun glow (BackSide shader sphere)
let vigMesh = null;   // in-canvas vignette (parented to camera)

// Rain state arrays
let rPx, rPy, rPz, rLen, rSpd;
let rainPosArr, rainAttr, windX;

// Snow state arrays
let sPx, sPy, sPz, sPh, sWd, sSpd;
let snowPosArr, snowAttr;

// Mist state
let mPx, mPy, mPz, mDrift;

const V3 = new THREE.Vector3();
const SUN = new THREE.Vector3();

// ---------------------------------------------------------------------------
function makeRadialTexture(inner, outer) {
  // Soft radial dot for flakes / mist. Static (non-random) procedural asset.
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, inner);
  grad.addColorStop(1, outer);
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// ---------------------------------------------------------------------------
function buildRain(world) {
  rPx = new Float32Array(RAIN_N);
  rPy = new Float32Array(RAIN_N);
  rPz = new Float32Array(RAIN_N);
  rLen = new Float32Array(RAIN_N);
  rSpd = new Float32Array(RAIN_N);
  windX = (localRng() - 0.5) * 1.6;   // deterministic prevailing tilt

  for (let i = 0; i < RAIN_N; i++) {
    rPx[i] = (localRng() * 2 - 1) * BOX_HALF;
    rPy[i] = Y_BOT + localRng() * (Y_TOP - Y_BOT);
    rPz[i] = (localRng() * 2 - 1) * BOX_HALF;
    rLen[i] = 0.7 + localRng() * 2.6;          // streak length (m)
    rSpd[i] = 18 + localRng() * 12;            // fall speed (m/s)
  }

  rainPosArr = new Float32Array(RAIN_N * 2 * 3);
  rainAttr = new THREE.BufferAttribute(rainPosArr, 3);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', rainAttr);

  const mat = new THREE.LineBasicMaterial({
    color: 0xd7e4f4,
    transparent: true,
    opacity: 0.5,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    fog: false,          // keep streaks bright against fogged sky
  });

  rainSys = new THREE.LineSegments(geo, mat);
  rainSys.frustumCulled = false;
  rainSys.renderOrder = 3;
  rainSys.visible = false;
  world.scene.add(rainSys);

  writeRain(world); // initial buffer fill
}

function writeRain() {
  const pos = rainPosArr;
  for (let i = 0; i < RAIN_N; i++) {
    const a = i * 6;
    const x = rPx[i], y = rPy[i], z = rPz[i], L = rLen[i];
    pos[a]     = x;
    pos[a + 1] = y;
    pos[a + 2] = z;
    pos[a + 3] = x + windX * (L / rSpd[i]) * 22;
    pos[a + 4] = y - L;
    pos[a + 5] = z;
  }
  rainAttr.needsUpdate = true;
}

function updateRain(dt) {
  for (let i = 0; i < RAIN_N; i++) {
    rPy[i] -= rSpd[i] * dt;
    if (rPy[i] < Y_BOT) {
      rPy[i] = Y_TOP;
      // deterministic respawn within the camera-anchored box
      rPx[i] = (localRng() * 2 - 1) * BOX_HALF;
      rPz[i] = (localRng() * 2 - 1) * BOX_HALF;
    }
  }
  writeRain();
}

// ---------------------------------------------------------------------------
function buildSnow(world) {
  sPx = new Float32Array(SNOW_N);
  sPy = new Float32Array(SNOW_N);
  sPz = new Float32Array(SNOW_N);
  sPh = new Float32Array(SNOW_N);   // wobble phase
  sWd = new Float32Array(SNOW_N);   // wobble rate
  sSpd = new Float32Array(SNOW_N);

  for (let i = 0; i < SNOW_N; i++) {
    const r = localRng;
    sPx[i] = (r() * 2 - 1) * BOX_HALF;
    sPy[i] = Y_BOT + r() * (Y_TOP - Y_BOT);
    sPz[i] = (r() * 2 - 1) * BOX_HALF;
    sPh[i] = r() * Math.PI * 2;
    sWd[i] = 0.6 + r() * 1.4;
    sSpd[i] = 0.9 + r() * 1.3;      // slow drifting fall
  }

  snowPosArr = new Float32Array(SNOW_N * 3);
  snowAttr = new THREE.BufferAttribute(snowPosArr, 3);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', snowAttr);

  const tex = makeRadialTexture('rgba(255,255,255,1)', 'rgba(255,255,255,0)');
  const mat = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.42,
    map: tex,
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    sizeAttenuation: true,
    fog: false,
  });

  snowSys = new THREE.Points(geo, mat);
  snowSys.frustumCulled = false;
  snowSys.renderOrder = 4;
  snowSys.visible = false;
  world.scene.add(snowSys);

  writeSnow();
}

function writeSnow() {
  for (let i = 0; i < SNOW_N; i++) {
    const a = i * 3;
    snowPosArr[a]     = sPx[i];
    snowPosArr[a + 1] = sPy[i];
    snowPosArr[a + 2] = sPz[i];
  }
  snowAttr.needsUpdate = true;
}

function updateSnow(dt, tSec) {
  for (let i = 0; i < SNOW_N; i++) {
    const wobble = Math.sin(tSec * sWd[i] + sPh[i]) * 0.9;
    sPy[i] -= sSpd[i] * dt;
    sPx[i] += wobble * dt;
    if (sPy[i] < Y_BOT) {
      sPy[i] = Y_TOP;
      sPx[i] = (localRng() * 2 - 1) * BOX_HALF;
      sPz[i] = (localRng() * 2 - 1) * BOX_HALF;
    }
  }
  writeSnow();
}

// ---------------------------------------------------------------------------
function buildMist(world) {
  mPx = new Float32Array(MIST_N);
  mPy = new Float32Array(MIST_N);
  mPz = new Float32Array(MIST_N);
  for (let i = 0; i < MIST_N; i++) {
    const r = localRng;
    mPx[i] = (r() * 2 - 1) * BOX_HALF * 1.6;
    mPy[i] = Y_BOT + r() * 20;
    mPz[i] = (r() * 2 - 1) * BOX_HALF * 1.6;
  }
  const pos = new Float32Array(MIST_N * 3);
  for (let i = 0; i < MIST_N; i++) {
    pos[i * 3]     = mPx[i];
    pos[i * 3 + 1] = mPy[i];
    pos[i * 3 + 2] = mPz[i];
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const tex = makeRadialTexture('rgba(200,210,222,0.35)', 'rgba(200,210,222,0)');
  const mat = new THREE.PointsMaterial({
    color: 0xbdc8d4,
    size: 9,
    map: tex,
    transparent: true,
    opacity: 0.5,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    sizeAttenuation: true,
    fog: false,
  });
  mistSys = new THREE.Points(geo, mat);
  mistSys.frustumCulled = false;
  mistSys.renderOrder = 2;
  mistSys.visible = false;
  world.scene.add(mistSys);
}

function updateMist(dt) {
  for (let i = 0; i < MIST_N; i++) {
    mPx[i] += dt * 1.4;
    if (mPx[i] > BOX_HALF * 1.6) mPx[i] = -BOX_HALF * 1.6;
    const a = i * 3;
    mistSys.geometry.attributes.position.array[a]     = mPx[i];
    mistSys.geometry.attributes.position.array[a + 2] = mPz[i];
  }
  mistSys.geometry.attributes.position.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// Temporary atmosphere dome: gradient sky + analytic sun glow (0 extra draw).
function buildDome(world) {
  const vert = `
    varying vec3 vDir;
    void main(){
      vDir = normalize(position);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
    }`;
  const frag = `
    uniform vec3 uZenith;
    uniform vec3 uHorizon;
    uniform vec3 uSunDir;
    uniform float uGlow;      // sun glow intensity
    uniform vec3 uGlowColor;
    varying vec3 vDir;
    void main(){
      vec3 dir = normalize(vDir);
      float h = clamp(dir.y, -1.0, 1.0);
      float f = smoothstep(-0.06, 0.22, h);          // horizon band blend
      vec3 col = mix(uHorizon, uZenith, f);
      // cheap light-scattering glow near the sun
      float s = max(dot(dir, normalize(uSunDir)), 0.0);
      float glow = pow(s, 6.0) * uGlow;
      col += uGlowColor * glow;
      gl_FragColor = vec4(col, 1.0);
    }`;

  const geo = new THREE.SphereGeometry(6900, 32, 20);
  const mat = new THREE.ShaderMaterial({
    vertexShader: vert,
    fragmentShader: frag,
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    uniforms: {
      uZenith: { value: new THREE.Color(0xffffff) },
      uHorizon: { value: new THREE.Color(0xffffff) },
      uSunDir: { value: new THREE.Vector3(0, 1, 0) },
      uGlow: { value: 0.0 },
      uGlowColor: { value: new THREE.Color(1.0, 0.45, 0.18) },
    },
  });
  dome = new THREE.Mesh(geo, mat);
  dome.frustumCulled = false;
  dome.renderOrder = -2;            // draw first (behind everything)
  dome.name = 'effectsAtmosphereDomeTEMP';
  world.scene.add(dome);

  const SKY = {
    clear:    { top: [0x2367b8, 0x3f83cf], hor: [0xcfe2ee, 0xf2e9d8] },
    overcast: { top: [0x59626f, 0x6a7481], hor: [0x9fa6ae, 0xb4bac0] },
    rain:     { top: [0x40485a, 0x565f70], hor: [0x8b93a3, 0x99a2b0] },
    snow:     { top: [0x9fb4c9, 0xb7c7d8], hor: [0xe1e8ee, 0xf2f5f8] },
    fog:      { top: [0x777d85, 0x868c93], hor: [0xaeb3b9, 0xbfc4c9] },
  };
  dome.userData.sky = SKY;
}

function skyFor(w) {
  const d = w || 'clear';
  return (dome && dome.userData.sky[d]) ? dome.userData.sky[d] : dome.userData.sky.clear;
}

// ---------------------------------------------------------------------------
// In-canvas vignette, parented to the camera (1 draw, no post composer).
function buildVignette(world) {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(128, 128, 60, 128, 128, 181);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, 'rgba(4,8,14,0.55)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(c);

  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    blending: THREE.NormalBlending,
  });
  vigMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
  vigMesh.frustumCulled = false;
  vigMesh.renderOrder = 999;

  const cam = world.camera;
  if (cam) {
    cam.add(vigMesh);
    sizeVignette();
  }
}

function sizeVignette() {
  const cam = worldRef ? worldRef.camera : null;
  if (!cam || !vigMesh) return;
  const dist = 0.6;                                  // in front of camera
  const fovR = (THREE.MathUtils.degToRad(cam.fov || 55));
  const h = 2 * dist * Math.tan(fovR / 2);
  const w = h * ((cam.aspect) || 1.6);
  vigMesh.scale.set(w * 1.15, h * 1.15, 1);
  vigMesh.position.set(0, 0, -dist);
}

// ---------------------------------------------------------------------------
function applyFog() {
  const scene = worldRef ? worldRef.scene : null;
  if (!scene) return;
  const cfg = {
    clear:    null,
    overcast: { c: 0x8b9096, n: 1500, f: 4200 },
    rain:     { c: 0x5a6270, n: 320,  f: 1400 },
    snow:     { c: 0xd7dfe6, n: 220,  f: 900 },
    fog:      { c: 0x979da4, n: 120,  f: 700 },
  }[weather];
  if (cfg) scene.fog = new THREE.Fog(cfg.c, cfg.n, cfg.f);
  else scene.fog = null;
}

function updateAtmosphere(tSec) {
  if (!dome) return;
  const sky = skyFor(weather);
  const dayFrac = ((tSec % 86400) + 86400) % 86400;

  // Sun azimuth/elevation (analytic). 06:00 horizon east, noon overhead.
  const a = Math.PI * (dayFrac - 6 * 3600) / (12 * 3600);   // [-π/2..? ]
  let elev = Math.sin(a);
  elev = THREE.MathUtils.clamp(elev, -0.05, 1.0);
  const sunRiseGlow = THREE.MathUtils.smoothstep(Math.abs(elev), 0.02, 0.30);

  // warm tint near the horizon at sunrise/sunset
  const warm = (1.0 - Math.abs(elev)) * 0.9;
  SUN.set(Math.cos(a) > 0 ? 1 : -1, elev, 0).normalize();

  const mat = dome.material;
  mat.uniforms.uSunDir.value.copy(SUN);
  mat.uniforms.uGlow.value = Math.max(elev, 0) * 0.85;

  // interpolate zenith / horizon colors by elevation (day vs low sun)
  const topDay = new THREE.Color(sky.top[0]);
  const topLow = new THREE.Color(sky.top[1]).lerp(new THREE.Color(0xd08a52), 0.35);
  mat.uniforms.uZenith.value.copy(topDay).lerp(topLow, warm * 0.4);

  const horDay = new THREE.Color(sky.hor[0]);
  const horLow = new THREE.Color(sky.hor[1]).lerp(new THREE.Color(0xe89a5a), 0.75);
  mat.uniforms.uHorizon.value.copy(horDay).lerp(horLow, warm);

  mat.uniforms.uGlowColor.value.setRGB(1.0, 0.45 + 0.15 * sunRiseGlow, 0.22);
}

// ---------------------------------------------------------------------------
function applyWeather() {
  if (rainSys) rainSys.visible = weather === 'rain';
  if (snowSys) snowSys.visible = weather === 'snow';
  const misty = weather === 'rain' || weather === 'fog' || weather === 'overcast' || weather === 'snow';
  if (mistSys) mistSys.visible = misty && weather !== 'snow';
  applyFog();
}

// ---------------------------------------------------------------------------
export function init(world) {
  try {
    worldRef = world;
    weather = WEATHERS.includes(world.meta?.weather) ? world.meta.weather : 'clear';

    buildDome(world);
    buildRain(world);
    buildSnow(world);
    buildMist(world);
    buildVignette(world);

    // respond to external setWeather() (screenshot tool / environment)
    const off = bus.on('weather:changed', ({ id: wid }) => {
      if (WEATHERS.includes(wid)) { weather = wid; applyWeather(); }
    });
    worldRef._effectsOff = off;

    // Self-test hook for verification: ?fx=rain | ?fx=snow  (default stays clear)
    try {
      const fx = new URLSearchParams(location.search).get('fx');
      if (fx === 'rain' || fx === 'snow') setWeather(fx);
    } catch (_) {}

    applyWeather();
  } catch (e) {
    console.error('[effects] init error', e);
  }
}

export function update(dtSec, world) {
  try {
    const cam = world.camera;
    if (!cam) return;
    const paused = !!(world.meta && world.meta.paused);
    const tSec = world.meta ? (world.meta.timeOfDaySec || 0) : 0;
    const dt = Math.min(dtSec || 0.016, 0.1);

    // Anchor the camera-relative region so precipitation always surrounds view.
    if (rainSys && rainSys.visible) {
      rainSys.position.copy(cam.position);
      if (!paused) updateRain(dt);
    }
    if (snowSys && snowSys.visible) {
      snowSys.position.copy(cam.position);
      if (!paused) updateSnow(dt, tSec + performance.now() * 0.001);
    }
    if (mistSys && mistSys.visible) {
      mistSys.position.copy(cam.position);
      if (!paused) updateMist(dt);
    }

    updateAtmosphere(tSec);
    sizeVignette();
  } catch (e) {
    console.error('[effects] update error', e);
  }
}

// Optional showcase — renders a simple labelled panel (demo only, not screenshots).
export function showcase(container) {
  try {
    if (!container) return;
    const div = document.createElement('div');
    div.style.cssText =
      'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;' +
      'font-family:system-ui,sans-serif;color:#dfe7f0;background:radial-gradient(circle at 50% 30%,#22324a,#0b1018);';
    div.innerHTML =
      '<div style="text-align:center">' +
      '<h2 style="margin:0 0 6px;font-weight:600">Weather FX</h2>' +
      '<p style="margin:0;opacity:.85">rain · snow · mist — camera-anchored, single-draw-call particles</p>' +
      '<p style="margin:8px 0 0;font-size:12px;opacity:.6">demo panel (screenshots use the live scene)</p>' +
      '</div>';
    container.appendChild(div);
  } catch (_) {}
}
