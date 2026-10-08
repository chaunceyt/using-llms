// ---------------------------------------------------------------------------
// tools — camera controls (orbit / pan / zoom) + build·zone·demolish brushes.
//
// Camera:
//   - left-drag            rotate/orbit around a target point (no brush active)
//   - right/middle-drag    pan
//   - wheel                zoom (clamped altitude so you can go overview->street)
//   - WASD / arrows        pan; Q/E orbit yaw; +/− or PageUp/PageDown zoom
// When a build tool is ACTIVE, left-click places the brush instead of orbiting
// (standard city-builder behaviour); right/middle-drag still pans and Q/E rotate.
//
// Brushes write through safe guards to world.zones / world.buildings so the
// (still-stubbed) zoning/buildings modules can later consume them. We never
// throw: everything is guarded against missing pieces of `world`.
//
// Programmatic API for demo/screenshot tool exposed on window.__skylines.tools:
//   setMode/getMode/clearMode, applyBrushAt(x,z), flyTo(pos,target,dur),
//   orbitBy(yaw,pitch), panBy(dx,dz), zoomBy(factor), syncFromCamera(),
//   getState(), heightAt(x,z)
// ---------------------------------------------------------------------------

import * as THREE from 'three';
import { bus } from '../core/index.js';

export const id = 'tools';

// ---- tuning ---------------------------------------------------------------
const CELL = 32;                       // zone grid cell size (m)
const MIN_ALT = 4;                     // camera altitude floor (m)
const MAX_ALT = 1900;                  // camera altitude ceiling (m)
const MIN_RADIUS = 16;
const MAX_RADIUS = 3400;
const BRUSH_PLACE_RADIUS = 30;         // demolition hit radius (m)
const CLICK_MOVE_PX = 6;               // max drag px to still count as a click
const FLY_EPS = 1.5;                   // m — flyTo considered complete below this
const KEY_PAN_SPEED = 1.15;            // pan speed factor (multiplied by radius)
const KEY_ORBIT_SPEED = 0.9;           // deg per second for Q/E yaw
const KEY_ZOOM_STEP = 1.18;            // wheel / key zoom multiplier

// ---- module state ---------------------------------------------------------
let world = null;
let cam = null;                        // THREE.PerspectiveCamera (world.camera)
let scene = null;

// authoritative orbit state (spherical around `target`)
const target = new THREE.Vector3(0, 40, 0);
let radius = 900;
let yaw = 0;                           // radians, camera azimuth about target
let pitch = 0.4;                       // radians, elevation of camera

// input
let mode = null;                       // active brush: residential|commercial|industrial|demolish
const keys = new Set();
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const _v = new THREE.Vector3();

let rayTargets = [];                   // cached meshes to raycast against (terrain)
let hoverPos = null;                   // {x,z} last hovered ground
let brushMesh = null;                  // preview circle

// pointer gesture state
let pointers = {
  active: false, btn: -1,
  sx: 0, sy: 0, lx: 0, ly: 0,
  moved: 0, modeAtDown: null,
};

// flyTo animation
let flying = null;                     // {fromPos,toPos,fromTar,toTar,t,dur}
const tmpFrom = new THREE.Vector3();
const tmpTo = new THREE.Vector3();

let started = false;
let busOffs = [];
const disposers = [];

function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }

// ---------------------------------------------------------------------------
// Ground picking — raycast onto the scene's terrain meshes.
// ---------------------------------------------------------------------------
function collectRayTargets() {
  // Re-scan lazily; mesh *objects* persist across LOD geometry swaps, so the
  // cached set stays valid. Refreshed if it ever comes up empty (terrain not up yet).
  if (!scene || rayTargets.length) return;
  const out = [];
  scene.traverse((o) => {
    if (o.isMesh && o.visible) out.push(o);
  });
  // keep only meshes we can actually hit — filter nothing, let raycaster decide
  rayTargets = out;
}

// Raycast from camera through a client pixel; returns {x,y,z} on the ground or null.
function pickGround(clientX, clientY) {
  if (!scene || !cam) return null;
  const w = window.innerWidth || 1;
  const h = window.innerHeight || 1;
  ndc.x = (clientX / w) * 2 - 1;
  ndc.y = -(clientY / h) * 2 + 1;
  raycaster.setFromCamera(ndc, cam);
  collectRayTargets();
  if (!rayTargets.length) {
    // fallback: intersect an analytic horizontal plane at the world's heightAt
    const groundY = typeof world.heightAt === 'function' ? world.heightAt(0, 0) : 40;
    const dir = raycaster.ray.direction;
    if (Math.abs(dir.y) < 1e-4) return null;
    const t = (groundY - raycaster.ray.origin.y) / dir.y;
    if (t <= 0) return null;
    const p = raycaster.ray.origin.clone().addScaledVector(dir, t);
    const py = typeof world.heightAt === 'function' ? world.heightAt(p.x, p.z) : groundY;
    return { x: p.x, y: py, z: p.z };
  }
  const hits = raycaster.intersectObjects(rayTargets, false);
  if (!hits.length) return null;
  const pt = hits[0].point;
  // snap the surface to the analytic height field for consistency
  let gy = pt.y;
  if (typeof world.heightAt === 'function') {
    try { gy = world.heightAt(pt.x, pt.z); } catch (_e) {}
  }
  return { x: pt.x, y: gy, z: pt.z };
}

// ---------------------------------------------------------------------------
// Camera math
// ---------------------------------------------------------------------------
function applyCamera() {
  if (!cam) return;
  const cp = Math.cos(pitch);
  _v.set(
    target.x + radius * cp * Math.sin(yaw),
    target.y + radius * Math.sin(pitch),
    target.z + radius * cp * Math.cos(yaw),
  );
  // altitude clamp (keep overview->street exploration sane)
  const minY = typeof world.heightAt === 'function'
    ? Math.max(world.heightAt(_v.x, _v.z) + MIN_ALT, MIN_ALT)
    : MIN_ALT;
  _v.y = clamp(_v.y, minY, MAX_ALT);
  cam.position.copy(_v);
  cam.lookAt(target);
}

// Recompute yaw/pitch/radius from the current camera position relative to target.
function syncFromCamera() {
  if (!cam) return;
  const off = cam.position.clone().sub(target);
  radius = clamp(off.length(), MIN_RADIUS, MAX_RADIUS);
  const rx = Math.hypot(off.x, off.z) || 1e-6;
  yaw = Math.atan2(-off.x, -off.z);
  pitch = clamp(Math.asin(clamp(off.y / radius, -1, 1)), -Math.PI / 2 + 0.02, Math.PI / 2 - 0.02);
}

// Recompute the orbit target from where the camera is currently pointing.
function setTargetFromCamera() {
  if (!cam) return;
  const d = cam.getWorldDirection(_v).negate();   // direction camera looks
  const dist = radius || clamp(cam.position.length(), MIN_RADIUS, MAX_RADIUS);
  target.copy(cam.position).addScaledVector(d, dist);
  syncFromCamera();
}

function panBy(worldDX, worldDZ) {
  if (!cam) return;
  // horizontal right vector from camera
  _v.set(1, 0, 0).applyQuaternion(cam.quaternion); // not exactly right; compute properly below
  const right = new THREE.Vector3().setFromMatrixColumn(cam.matrix, 0);
  right.y = 0;
  if (right.lengthSq() < 1e-6) right.set(1, 0, 0);
  right.normalize();
  // horizontal forward vector
  const fwd = new THREE.Vector3().setFromMatrixColumn(cam.matrix, 2).negate();
  fwd.y = 0;
  if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
  fwd.normalize();
  const d = new THREE.Vector3()
    .addScaledVector(right, worldDX)
    .addScaledVector(fwd, worldDZ);
  target.add(d);
  cam.position.add(d);
}

function orbitBy(dYaw, dPitch) {
  yaw += dYaw;
  pitch = clamp(pitch + dPitch, -Math.PI / 2 + 0.02, Math.PI / 2 - 0.02);
  applyCamera();
}

function zoomBy(factor) {
  radius = clamp(radius * factor, MIN_RADIUS, MAX_RADIUS);
  applyCamera();
}

// Fly the camera (position & target) with a simple ease animation.
function flyTo(toPos, toTarget, duration) {
  if (!cam) return;
  const toP = Array.isArray(toPos)
    ? new THREE.Vector3(toPos[0], toPos[1], toPos[2])
    : (toPos && toPos.isVector3 ? toPos.clone() : new THREE.Vector3(...(toPos || [0, 60, 0])));
  const toT = Array.isArray(toTarget)
    ? new THREE.Vector3(toTarget[0], toTarget[1], toTarget[2])
    : (toTarget && toTarget.isVector3 ? toTarget.clone() : target.clone());
  flying = {
    fromPos: cam.position.clone(),
    toPos: toP,
    fromTar: target.clone(),
    toTar: toT,
    t: 0,
    dur: Math.max(0.2, duration || 1.6),
  };
}

function tickFly(dt) {
  if (!flying || !cam) return;
  flying.t += (dt || 0.016) / flying.dur;
  const f = clamp(flying.t, 0, 1);
  const e = f < 0.5 ? 2 * f * f : -1 + (4 - 2 * f) * f; // easeInOutQuad
  tmpFrom.lerpVectors(flying.fromPos, flying.toPos, e);
  cam.position.copy(tmpFrom);
  tmpTo.lerpVectors(flying.fromTar, flying.toTar, e);
  target.copy(tmpTo);
  if (f >= 1) { cam.lookAt(target); flying = null; syncFromCamera(); }
  else cam.lookAt(target);
}

// ---------------------------------------------------------------------------
// Brush application (safe guards — never mutate a missing/non-array world).
// ---------------------------------------------------------------------------
function ensureArr(obj, key) {
  if (!obj || !Array.isArray(obj[key])) obj[key] = [];
  return obj[key];
}

function applyBrush(x, z, brushMode) {
  if (!world || x == null || z == null) return;
  const m = brushMode || mode;
  if (!m) return;

  if (m === 'demolish') {
    let removed = 0;
    const blds = world.buildings && Array.isArray(world.buildings) ? world.buildings : [];
    for (let i = blds.length - 1; i >= 0; i--) {
      const b = blds[i];
      if (!b || typeof b.x !== 'number') continue;
      const dx = b.x - x, dz = b.z - z;
      if ((dx * dx + dz * dz) <= BRUSH_PLACE_RADIUS * BRUSH_PLACE_RADIUS) {
        const bCopy = { ...b };
        blds.splice(i, 1);
        removed++;
        try { bus.emit('building:remove', { b: bCopy }); } catch (_e) {}
      }
    }
    if (removed > 0) emitToast(`Demolished ${removed} structure${removed === 1 ? '' : 's'}.`, 'info');
    else emitToast('No buildings here to demolish.', 'warn');
    return;
  }

  // zone brush
  const cells = ensureArr(world, 'zones');
  const ix = Math.floor(x / CELL);
  const iz = Math.floor(z / CELL);
  const cx = ix * CELL + CELL / 2;
  const cz = iz * CELL + CELL / 2;
  let cell = cells.find((c) => c && c.ix === ix && c.iz === iz);
  if (cell) {
    if (cell.type !== m) {
      cell.type = m;
      cell.growth = 0;
    }
    cell.density = Math.min(3, (cell.density || 1));
  } else {
    cell = { ix, iz, x: cx, z: cz, type: m, density: 1, growth: 0 };
    cells.push(cell);
  }
  try { bus.emit('zone:changed', { cell }); } catch (_e) {}
  emitToast(`Zoned ${m} at (${cx}, ${cz}).`, 'info');
}

function emitToast(text, level) {
  try { bus.emit('ui:toast', { text, level: level || 'info' }); } catch (_e) {}
}

// ---------------------------------------------------------------------------
// Brush hover preview
// ---------------------------------------------------------------------------
function ensureBrushMesh() {
  if (brushMesh || !scene) return;
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(CELL / 2 - 1, CELL / 2, 48),
    new THREE.MeshBasicMaterial({ color: 0x9fe8ff, transparent: true, opacity: 0.65, depthWrite: false, side: THREE.DoubleSide }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.renderOrder = 999;
  brushMesh = ring;
  scene.add(ring);
}

function updateHover(clientX, clientY) {
  if (!mode || mode === 'demolish') { hideBrush(); return; }
  const p = pickGround(clientX, clientY);
  ensureBrushMesh();
  if (!p || !brushMesh) return;
  hoverPos = { x: p.x, z: p.z };
  const ix = Math.floor(p.x / CELL) * CELL + CELL / 2;
  const iz = Math.floor(p.z / CELL) * CELL + CELL / 2;
  brushMesh.position.set(ix, (typeof world.heightAt === 'function' ? world.heightAt(ix, iz) : p.y) + 1.6, iz);
  brushMesh.visible = true;
}

function hideBrush() {
  if (brushMesh) brushMesh.visible = false;
  hoverPos = null;
}

// ---------------------------------------------------------------------------
// Pointer / wheel / key handlers
// ---------------------------------------------------------------------------
function onPointerDown(e) {
  try {
    // ignore interactions with HUD controls
    const t = e.target;
    if (t && (t.closest('.tool-btn, .ctl-btn, .speed-pill, button, input, textarea') ||
      (typeof t.matches === 'function' && t.matches('button,input,textarea,a')))) return;

    // right/middle => pan; left => orbit-or-place
    const isPan = e.button === 2 || e.button === 1;
    if (isPan) { pointers.active = true; pointers.btn = e.button; }
    else if (e.button === 0) {
      pointers.active = true; pointers.btn = 0; pointers.modeAtDown = mode;
    } else return;

    pointers.sx = pointers.lx = e.clientX;
    pointers.sy = pointers.ly = e.clientY;
    pointers.moved = 0;
  } catch (_e) {}
}

function onPointerMove(e) {
  try {
    const dx = e.clientX - pointers.lx;
    const dy = e.clientY - pointers.ly;
    if (pointers.active && pointers.btn === 2 || (pointers.active && pointers.btn === 1)) {
      // pan
      const scale = Math.max(radius * 0.0026, 0.5);
      panBy(dx * scale, -dy * scale);
      pointers.lx = e.clientX; pointers.ly = e.clientY;
    } else if (pointers.active && pointers.btn === 0) {
      // orbit — but only when no build brush is active
      if (!pointers.modeAtDown) {
        const ayaw = dx * 0.0055;
        const apitch = dy * 0.0042;
        orbitBy(ayaw, apitch);
      }
      pointers.moved += Math.abs(dx) + Math.abs(dy);
      pointers.lx = e.clientX; pointers.ly = e.clientY;
    } else {
      // passive hover preview
      updateHover(e.clientX, e.clientY);
    }
  } catch (_e) {}
}

function onPointerUp(e) {
  try {
    const wasClick = pointers.active && pointers.moved < CLICK_MOVE_PX &&
      (e.clientX === undefined || (Math.abs(e.clientX - pointers.sx) + Math.abs(e.clientY - pointers.sy)) < CLICK_MOVE_PX);
    // place brush on a clean click with an active build tool
    if (wasClick && pointers.modeAtDown && e.button === 0) {
      const p = pickGround(e.clientX, e.clientY);
      if (p) applyBrush(p.x, p.z, pointers.modeAtDown);
    }
    pointers.active = false; pointers.btn = -1;
  } catch (_e) {}
}

function onWheel(e) {
  try {
    e.preventDefault();
    const factor = e.deltaY > 0 ? KEY_ZOOM_STEP : (1 / KEY_ZOOM_STEP);
    zoomBy(factor);
  } catch (_e) {}
}

function onContextMenu(e) { try { e.preventDefault(); } catch (_e) {} }

function onKeyDown(e) {
  const tag = (e.target && e.target.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) return;
  keys.add((e.key || '').toLowerCase());
  try {
    if (e.key === 'Escape') clearMode();
    // wheel-like zoom keys
    if (e.key === '+' || e.key === '=') { zoomBy(1 / KEY_ZOOM_STEP); }
    else if (e.key === '-' || e.key === '_') { zoomBy(KEY_ZOOM_STEP); }
  } catch (_e) {}
}

function onKeyUp(e) {
  keys.delete((e.key || '').toLowerCase());
}

// keyboard-held camera motion (WASD pan, Q/E orbit)
function tickKeys(dt) {
  if (!cam || !dt) return;
  let dx = 0, dz = 0, ayaw = 0;
  if (keys.has('w')) dz -= 1;
  if (keys.has('s')) dz += 1;
  if (keys.has('a')) dx -= 1;
  if (keys.has('d')) dx += 1;
  if (keys.has('q')) ayaw -= KEY_ORBIT_SPEED * Math.PI / 180;
  if (keys.has('e')) ayaw += KEY_ORBIT_SPEED * Math.PI / 180;

  // arrow keys rotate camera too
  if (keys.has('arrowleft')) ayaw -= KEY_ORBIT_SPEED * Math.PI / 180;
  if (keys.has('arrowright')) ayaw += KEY_ORBIT_SPEED * Math.PI / 180;

  const scale = Math.max(radius * KEY_PAN_SPEED * dt, 0.5 * dt);
  if (dx || dz) panBy(dx * scale, dz * scale);
  if (ayaw) orbitBy(ayaw, 0);
}

// ---------------------------------------------------------------------------
// Public tool API (also exposed on window.__skylines.tools for demo/screenshot)
// ---------------------------------------------------------------------------
const api = {
  setMode(m) { mode = m; try { bus.emit('tool:mode', { mode: m }); } catch (_e) {} },
  getMode() { return mode; },
  clearMode() {
    if (mode) { try { bus.emit('tool:mode', { mode: null }); } catch (_e) {} }
    mode = null;
    hideBrush();
  },
  applyBrushAt(x, z, brushMode) { applyBrush(x, z, brushMode || mode); },
  flyTo(pos, target, duration) { flyTo(pos, target, duration); },
  orbitBy(yawDeg, pitchDeg) {
    if (cam) { orbitBy(yawDeg * Math.PI / 180, pitchDeg * Math.PI / 180); }
  },
  panBy(dx, dz) { panBy(dx, dz); },
  zoomBy(factor) { zoomBy(factor || 1); },
  syncFromCamera() { syncFromCamera(); applyCamera(); },
  setTarget(p) {
    if (Array.isArray(p)) target.set(p[0], p[1], p[2]);
    else if (p && p.isVector3) target.copy(p);
    else target.set(0, 40, 0);
    syncFromCamera(); applyCamera();
  },
  getState() {
    return {
      mode,
      camera: cam ? [cam.position.x, cam.position.y, cam.position.z] : null,
      target: [target.x, target.y, target.z],
      radius, yaw, pitch,
      zoneCells: world && Array.isArray(world.zones) ? world.zones.length : 0,
      buildings: world && Array.isArray(world.buildings) ? world.buildings.length : 0,
    };
  },
};

// ---------------------------------------------------------------------------
// Module lifecycle
// ---------------------------------------------------------------------------
export function init(w, renderer, sceneArg, cameraArg) {
  try {
    world = w || {};
    scene = sceneArg || (world && world.scene);
    cam = cameraArg || (world && world.camera);

    // initial orbit state from the bootstrap camera (overview look)
    if (cam) { syncFromCamera(); applyCamera(); }

    // listen to tool selection emitted by the UI palette
    const b = bus && typeof bus.on === 'function' ? bus : null;
    if (b) {
      busOffs.push(b.on('tool:mode', ({ mode: m }) => {
        mode = (m === null || m === undefined) ? null : String(m);
        if (!mode) hideBrush();
      }));
    }

    // pointer / wheel on window so input works even over the HUD overlay
    const tgt = typeof window !== 'undefined' ? window : null;
    if (tgt) {
      tgt.addEventListener('pointerdown', onPointerDown, { passive: true });
      tgt.addEventListener('pointermove', onPointerMove, { passive: true });
      tgt.addEventListener('pointerup', onPointerUp, { passive: true });
      tgt.addEventListener('wheel', onWheel, { passive: false });
      tgt.addEventListener('contextmenu', onContextMenu);
      tgt.addEventListener('keydown', onKeyDown);
      tgt.addEventListener('keyup', onKeyUp);
    }

    // expose for demo / screenshot tool to drive camera + brushes.
    // main.js assigns window.__skylines AFTER modules load, so it overwrites any
    // property we set here — use an independent global and re-attach each frame.
    if (typeof window !== 'undefined') {
      try { window.skylinesTools = api; } catch (_e) {}
    }
    started = true;
  } catch (_e) {
    /* never throw on boot */
  }
}

export function update(dtSec, worldArg) {
  try {
    if (!started) return;
    // keep refs fresh in case they were replaced
    if (worldArg && worldArg !== world) world = worldArg;
    if (world && world.camera && world.camera !== cam) { cam = world.camera; }
    if (world && world.scene && world.scene !== scene) { scene = world.scene; rayTargets = []; }

    // attach to window.__skylines once main.js has set it up (it clobbers on boot)
    if (typeof window !== 'undefined') {
      try {
        if (window.__skylines && !window.__skylines.tools) window.__skylines.tools = api;
        if (!window.skylinesTools) window.skylinesTools = api;
      } catch (_e) {}
    }

    const dt = Number.isFinite(dtSec) ? dtSec : 0.016;
    tickFly(dt);
    tickKeys(dt);

    // refresh hover preview if a zone brush is active and pointer idle-ish
    if (mode && mode !== 'demolish' && hoverPos == null && typeof window !== 'undefined') {
      // lightweight: keep last hover; full update happens on pointermove
    }
  } catch (_e) {
    /* never throw in update */
  }
}

export function showcase(container) {
  try {
    if (!container || typeof container.appendChild !== 'function') return;
    const d = document.createElement('div');
    d.style.cssText = 'font-family:sans-serif;color:#cfd8dc;padding:16px;line-height:1.5';
    d.innerHTML =
      `<b>Tools</b> — camera + build brushes.<br>
       <span style="color:#7fa3b8;font-size:13px">
         Left-drag orbit · right/middle-drag pan · wheel zoom · WASD/QE keys.<br>
         Brush modes (residential/commercial/industrial/demolish) via the UI palette.
       </span>`;
    container.appendChild(d);
  } catch (_e) { /* ignore */ }
}
