import * as THREE from 'three';
import { CONFIG } from './config.js';

const W = CONFIG.TRACK.WIDTH;
const CURB = CONFIG.TRACK.CURB_WIDTH;
const WALL = CONFIG.TRACK.WALL_MARGIN;
const SAMPLES = CONFIG.TRACK.SAMPLES;

// Hand-authored closed circuit (XZ plane). Bottom straight = start, with an
// S-bend / hairpin up the left side and a sweep around the right.
const CONTROL_POINTS = [
  [0, -120], [70, -112], [132, -62], [148, 18], [104, 92], [44, 132],
  [-18, 122], [-40, 62], [-100, 92], [-142, 30], [-130, -52], [-72, -112],
];

function roadTexture(trackLength) {
  const c = document.createElement('canvas');
  c.width = 128; c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#3a3f46';
  g.fillRect(0, 0, 128, 256);
  for (let i = 0; i < 900; i++) {
    g.fillStyle = 'rgba(255,255,255,0.035)';
    g.fillRect(Math.random() * 128, Math.random() * 256, 2, 2);
  }
  g.fillStyle = '#e8e8e8';
  g.fillRect(2, 0, 4, 256);
  g.fillRect(122, 0, 4, 256);
  g.fillStyle = '#f2c94c';
  for (let y = 0; y < 256; y += 64) g.fillRect(60, y + 10, 8, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(1, trackLength / 10);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function curbTexture() {
  const c = document.createElement('canvas');
  c.width = 32; c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = '#e23b3b'; g.fillRect(0, 0, 32, 32);
  g.fillStyle = '#f5f5f5'; g.fillRect(0, 32, 32, 32);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function ribbon(curve, segs, inner, outer, y, texture, vScale) {
  const positions = [], uvs = [], indices = [];
  let v = 0;
  let prev = curve.getPointAt(0);
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const p = curve.getPointAt(t);
    const tan = curve.getTangentAt(t);
    const nx = -tan.z, nz = tan.x;
    positions.push(p.x + nx * inner, y, p.z + nz * inner);
    positions.push(p.x + nx * outer, y, p.z + nz * outer);
    uvs.push(0, v, 1, v);
    if (i > 0) v += p.distanceTo(prev);
    prev = p;
    if (i < segs) {
      const a = i * 2;
      indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  texture.repeat.set(1, vScale);
  const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ map: texture }));
  mesh.receiveShadow = true;
  return mesh;
}

export function buildTrack(scene) {
  const curve = new THREE.CatmullRomCurve3(
    CONTROL_POINTS.map((p) => new THREE.Vector3(p[0], 0, p[1])), true, 'catmullrom', 0.5);

  const sampleCount = SAMPLES;
  const centerlineXZ = new Float32Array(sampleCount * 2);
  let trackLength = 0;
  for (let i = 0; i < sampleCount; i++) {
    const t = i / sampleCount;
    const p = curve.getPointAt(t);
    centerlineXZ[i * 2] = p.x;
    centerlineXZ[i * 2 + 1] = p.z;
    if (i > 0) {
      const pp = curve.getPointAt((i - 1) / sampleCount);
      trackLength += p.distanceTo(pp);
    }
  }
  trackLength += curve.getPointAt((sampleCount - 1) / sampleCount).distanceTo(curve.getPointAt(0));

  const cl = centerlineXZ;
  const n = sampleCount;

  function nearestIndex(x, z, hintT) {
    const win = Math.max(6, Math.floor(n * 0.04));
    const start = Math.floor(((hintT % 1) + 1) % 1 * n);
    let best = Infinity, bestI = -1;
    for (let k = -win; k <= win; k++) {
      const i = ((start + k) % n + n) % n;
      const dx = x - cl[i * 2], dz = z - cl[i * 2 + 1];
      const d = dx * dx + dz * dz;
      if (d < best) { best = d; bestI = i; }
    }
    if (bestI < 0 || best > 625) {
      best = Infinity;
      for (let i = 0; i < n; i += 4) {
        const dx = x - cl[i * 2], dz = z - cl[i * 2 + 1];
        const d = dx * dx + dz * dz;
        if (d < best) { best = d; bestI = i; }
      }
    }
    return bestI;
  }

  function nearestT(x, z, hintT) {
    const i = nearestIndex(x, z, hintT);
    return i < 0 ? 0 : i / n;
  }

  function offRoad(x, z) {
    let min = Infinity;
    for (let i = 0; i < n; i += 8) {
      const dx = x - cl[i * 2], dz = z - cl[i * 2 + 1];
      const d = dx * dx + dz * dz;
      if (d < min) min = d;
    }
    return Math.sqrt(min) > W / 2 + CURB;
  }

  function wallPush(x, z, state) {
    const limit = W / 2 + CURB + WALL;
    let min = Infinity, bx = 0, bz = 0;
    for (let i = 0; i < n; i += 8) {
      const dx = x - cl[i * 2], dz = z - cl[i * 2 + 1];
      const d = dx * dx + dz * dz;
      if (d < min) { min = d; bx = cl[i * 2]; bz = cl[i * 2 + 1]; }
    }
    const dist = Math.sqrt(min);
    if (dist > limit && dist > 0.001) {
      const dx = x - bx, dz = z - bz;
      const inv = 1 / dist;
      state.x = bx + dx * inv * limit;
      state.z = bz + dz * inv * limit;
      state.speed *= 0.85;
    }
  }

  // ---- Road + curbs ----
  scene.add(ribbon(curve, 400, W / 2, -W / 2, 0.01, roadTexture(trackLength), trackLength / 10));
  scene.add(ribbon(curve, 400, W / 2, W / 2 + CURB, 0.03, curbTexture(), trackLength / 2));
  scene.add(ribbon(curve, 400, -W / 2, -W / 2 - CURB, 0.03, curbTexture(), trackLength / 2));

  // ---- Start / finish ----
  const sp = curve.getPointAt(0);
  const st = curve.getTangentAt(0);
  const sx = -st.z, sz = st.x; // lateral normal
  const heading = Math.atan2(st.x, st.z);

  // checkerboard strip
  const cc = document.createElement('canvas');
  cc.width = 128; cc.height = 32;
  const cg = cc.getContext('2d');
  const cs = 16;
  for (let x = 0; x < 8; x++) for (let y = 0; y < 2; y++) {
    cg.fillStyle = (x + y) % 2 ? '#111' : '#fff';
    cg.fillRect(x * cs, y * cs, cs, cs);
  }
  const startTex = new THREE.CanvasTexture(cc);
  startTex.colorSpace = THREE.SRGBColorSpace;
  const startGroup = new THREE.Group();
  const startPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(W, 4),
    new THREE.MeshLambertMaterial({ map: startTex }));
  startPlane.rotation.x = -Math.PI / 2;
  startGroup.add(startPlane);
  startGroup.position.set(sp.x, 0.04, sp.z);
  startGroup.rotation.y = heading;
  scene.add(startGroup);

  // gantry
  const poleMat = new THREE.MeshLambertMaterial({ color: 0xd8d8d8 });
  const poleGeo = new THREE.CylinderGeometry(0.35, 0.35, 9, 8);
  const poleL = new THREE.Mesh(poleGeo, poleMat);
  poleL.position.set(sp.x + sx * (W / 2 + 1.2), 4.5, sp.z + sz * (W / 2 + 1.2));
  poleL.castShadow = true;
  const poleR = new THREE.Mesh(poleGeo, poleMat);
  poleR.position.set(sp.x - sx * (W / 2 + 1.2), 4.5, sp.z - sz * (W / 2 + 1.2));
  poleR.castShadow = true;

  const bc = document.createElement('canvas');
  bc.width = 512; bc.height = 96;
  const bg = bc.getContext('2d');
  bg.fillStyle = '#e63946';
  bg.fillRect(0, 0, 512, 96);
  bg.fillStyle = '#fff';
  bg.font = 'bold 64px sans-serif';
  bg.textAlign = 'center';
  bg.textBaseline = 'middle';
  bg.fillText('START', 256, 52);
  const bannerTex = new THREE.CanvasTexture(bc);
  bannerTex.colorSpace = THREE.SRGBColorSpace;
  const banner = new THREE.Mesh(
    new THREE.BoxGeometry(W + 2.4, 2.4, 0.5),
    new THREE.MeshLambertMaterial({ map: bannerTex }));
  banner.position.set(sp.x, 9.2, sp.z);
  banner.rotation.y = heading;
  banner.castShadow = true;
  scene.add(poleL, poleR, banner);

  return {
    curve,
    startT: 0,
    sampleCount: n,
    centerlineXZ: cl,
    TRACK_WIDTH: W,
    trackLength,
    offRoad,
    wallPush,
    nearestT,
    pointAt(t) { return curve.getPointAt(((t % 1) + 1) % 1); },
    tangentAt(t) { return curve.getTangentAt(((t % 1) + 1) % 1); },
  };
}
