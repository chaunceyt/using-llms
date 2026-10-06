import * as THREE from 'three';
import { CONFIG } from './config.js';

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildEnvironment(scene, track) {
  const rng = mulberry32(20260930);
  const W = track.TRACK_WIDTH;
  const corridor = W / 2 + CONFIG.TRACK.CURB_WIDTH + 6;
  const cl = track.centerlineXZ;
  const n = track.sampleCount;

  function distCoarse(x, z) {
    let min = Infinity;
    for (let i = 0; i < n; i += 8) {
      const dx = x - cl[i * 2], dz = z - cl[i * 2 + 1];
      const d = dx * dx + dz * dz;
      if (d < min) min = d;
    }
    return Math.sqrt(min);
  }

  function placeInGrass(rMin, rMax) {
    for (let tries = 0; tries < 50; tries++) {
      const a = rng() * Math.PI * 2;
      const r = rMin + rng() * (rMax - rMin);
      const x = Math.cos(a) * r, z = Math.sin(a) * r;
      if (distCoarse(x, z) > corridor) return { x, z };
    }
    return null;
  }

  // ---- Ground ----
  function grassTexture() {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const g = c.getContext('2d');
    g.fillStyle = '#3f9e4d';
    g.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 5000; i++) {
      g.fillStyle = rng() < 0.5 ? 'rgba(53,135,63,0.55)' : 'rgba(74,168,86,0.45)';
      g.fillRect(rng() * 256, rng() * 256, 2, 2);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(70, 70);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(1300, 1300),
    new THREE.MeshLambertMaterial({ map: grassTexture() })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const m4 = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const pos = new THREE.Vector3();
  const scl = new THREE.Vector3();
  const Y = new THREE.Vector3(0, 1, 0);

  // ---- Trees (instanced trunk + foliage) ----
  const TREE_COUNT = 140;
  const trunks = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.3, 0.45, 2.4, 6),
    new THREE.MeshLambertMaterial({ color: 0x6b4423 }), TREE_COUNT);
  const leaves = new THREE.InstancedMesh(
    new THREE.ConeGeometry(2.3, 5.5, 8),
    new THREE.MeshLambertMaterial({ color: 0x2d7a3a }), TREE_COUNT);
  trunks.castShadow = leaves.castShadow = true;
  leaves.receiveShadow = true;
  let treePlaced = 0, treeGuard = 0;
  while (treePlaced < TREE_COUNT && treeGuard < TREE_COUNT * 60) {
    treeGuard++;
    const p = placeInGrass(22, 290);
    if (!p) continue;
    const s = 0.7 + rng() * 1.0;
    q.setFromAxisAngle(Y, rng() * Math.PI * 2);
    pos.set(p.x, 1.2 * s, p.z); scl.set(s, s, s);
    m4.compose(pos, q, scl); trunks.setMatrixAt(treePlaced, m4);
    pos.set(p.x, (2.4 + 2.4) * s, p.z);
    m4.compose(pos, q, scl); leaves.setMatrixAt(treePlaced, m4);
    treePlaced++;
  }
  trunks.count = treePlaced; leaves.count = treePlaced;
  trunks.instanceMatrix.needsUpdate = leaves.instanceMatrix.needsUpdate = true;
  scene.add(trunks, leaves);

  // ---- Rocks (instanced) ----
  const ROCK_COUNT = 40;
  const rocks = new THREE.InstancedMesh(
    new THREE.DodecahedronGeometry(0.9, 0),
    new THREE.MeshLambertMaterial({ color: 0x8a8f98, flatShading: true }), ROCK_COUNT);
  rocks.castShadow = rocks.receiveShadow = true;
  let rockPlaced = 0, rockGuard = 0;
  while (rockPlaced < ROCK_COUNT && rockGuard < ROCK_COUNT * 60) {
    rockGuard++;
    const p = placeInGrass(18, 260);
    if (!p) continue;
    const s = 0.5 + rng() * 1.4;
    q.setFromAxisAngle(Y, rng() * Math.PI * 2);
    pos.set(p.x, 0.4 * s, p.z); scl.set(s, s * (0.7 + rng() * 0.5), s);
    m4.compose(pos, q, scl); rocks.setMatrixAt(rockPlaced, m4);
    rockPlaced++;
  }
  rocks.count = rockPlaced;
  rocks.instanceMatrix.needsUpdate = true;
  scene.add(rocks);

  // ---- Flowers (instanced, per-instance color) ----
  const FLOWER_COUNT = 90;
  const flowers = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.28, 6, 5),
    new THREE.MeshLambertMaterial({ color: 0xffffff }), FLOWER_COUNT);
  const flowerPalette = [0xf7b32b, 0xf15bb5, 0xffffff, 0x9b5de5, 0xff6b6b];
  const cTmp = new THREE.Color();
  let flPlaced = 0, flGuard = 0;
  while (flPlaced < FLOWER_COUNT && flGuard < FLOWER_COUNT * 60) {
    flGuard++;
    const p = placeInGrass(16, 240);
    if (!p) continue;
    const s = 0.7 + rng() * 0.8;
    pos.set(p.x, 0.25, p.z); scl.set(s, s, s); q.identity();
    m4.compose(pos, q, scl); flowers.setMatrixAt(flPlaced, m4);
    flowers.setColorAt(flPlaced, cTmp.setHex(flowerPalette[flPlaced % flowerPalette.length]));
    flPlaced++;
  }
  flowers.count = flPlaced;
  flowers.instanceMatrix.needsUpdate = true;
  if (flowers.instanceColor) flowers.instanceColor.needsUpdate = true;
  scene.add(flowers);

  // ---- Clouds (static clusters) ----
  const cloudMat = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.92 });
  for (let i = 0; i < 12; i++) {
    const cloud = new THREE.Group();
    const blobs = 3 + Math.floor(rng() * 3);
    for (let b = 0; b < blobs; b++) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(6 + rng() * 5, 8, 6), cloudMat);
      s.position.set((b - blobs / 2) * 7, rng() * 3, rng() * 6);
      s.scale.y = 0.5;
      cloud.add(s);
    }
    const a = rng() * Math.PI * 2, r = 120 + rng() * 260;
    cloud.position.set(Math.cos(a) * r, 60 + rng() * 60, Math.sin(a) * r);
    scene.add(cloud);
  }

  // ---- Distant mountains ----
  for (let i = 0; i < 10; i++) {
    const hgt = 60 + rng() * 90;
    const mt = new THREE.Mesh(
      new THREE.ConeGeometry(45 + rng() * 45, hgt, 5),
      new THREE.MeshLambertMaterial({ color: 0x6d7f9c, flatShading: true }));
    const a = (i / 10) * Math.PI * 2 + rng() * 0.4;
    const r = 380 + rng() * 140;
    mt.position.set(Math.cos(a) * r, hgt / 2 - 6, Math.sin(a) * r);
    scene.add(mt);
  }

  // ---- A couple of far-field lakes ----
  const lakeMat = new THREE.MeshLambertMaterial({ color: 0x2a8fd8, transparent: true, opacity: 0.9 });
  for (let i = 0; i < 2; i++) {
    const lake = new THREE.Mesh(new THREE.CircleGeometry(24 + rng() * 14, 24), lakeMat);
    lake.rotation.x = -Math.PI / 2;
    const a = rng() * Math.PI * 2, r = 240 + rng() * 120;
    lake.position.set(Math.cos(a) * r, 0.15, Math.sin(a) * r);
    scene.add(lake);
  }
}
