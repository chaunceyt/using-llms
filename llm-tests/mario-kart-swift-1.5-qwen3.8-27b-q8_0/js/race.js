import * as THREE from 'three';
import { CONFIG } from './config.js';

const PHYS = CONFIG.PHYS;
const ITEMS = CONFIG.ITEMS;
const LAPS = CONFIG.RACE.LAPS;

function makeItemBox() {
  const g = new THREE.Group();
  const box = new THREE.Mesh(
    new THREE.BoxGeometry(1.6, 1.6, 1.6),
    new THREE.MeshLambertMaterial({ color: 0xe91e8c, transparent: true, opacity: 0.82, emissive: 0x7a0f3e }));
  box.castShadow = true;
  g.add(box);
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 46px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('?', 32, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const qmat = new THREE.MeshBasicMaterial({ map: tex, transparent: true });
  for (let f = 0; f < 4; f++) {
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(1.05, 1.05), qmat);
    const ang = (f * Math.PI) / 2;
    plane.position.set(Math.sin(ang) * 0.82, 0, Math.cos(ang) * 0.82);
    plane.rotation.y = ang;
    g.add(plane);
  }
  return g;
}

function makeShell() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.SphereGeometry(0.72, 10, 8),
    new THREE.MeshLambertMaterial({ color: 0x2fa84f }));
  body.position.y = 0.35;
  body.castShadow = true;
  g.add(body);
  const finMat = new THREE.MeshLambertMaterial({ color: 0x1e6b34 });
  for (let i = 0; i < 3; i++) {
    const fin = new THREE.Mesh(new THREE.ConeGeometry(0.26, 0.5, 4), finMat);
    const a = (i / 3) * Math.PI * 2;
    fin.position.set(Math.sin(a) * 0.5, 0.35, Math.cos(a) * 0.5);
    g.add(fin);
  }
  return g;
}

function makeBanana() {
  const g = new THREE.Group();
  const b = new THREE.Mesh(
    new THREE.SphereGeometry(0.6, 8, 6),
    new THREE.MeshLambertMaterial({ color: 0xf7d02b }));
  b.scale.set(1.5, 0.5, 0.7);
  b.position.y = 0.22;
  b.castShadow = true;
  g.add(b);
  return g;
}

export function createRace(karts, track, scene, events) {
  const E = events || {};
  const fire = (name, ...args) => { if (E[name]) E[name](...args); };

  let racing = false;
  let time = 0;
  let countdown = 3.0;
  let lastCount = 4;
  let results = null;
  let raceOverFired = false;
  let playerGrace = 0;
  const positions = [];

  // ---- item boxes ----
  const boxes = [];
  for (const tz of [0.12, 0.4, 0.68, 0.9]) {
    for (let k = 0; k < 4; k++) {
      const col = k % 2, row = k >> 1;
      const tt = (tz + (col ? 0.004 : -0.004)) % 1;
      const p = track.pointAt(tt);
      const tan = track.tangentAt(tt);
      const nx = -tan.z, nz = tan.x;
      const lat = (col ? 1 : -1) * 2.2 + (row ? 0.6 : -0.6);
      const group = makeItemBox();
      group.position.set(p.x + nx * lat, 1.2, p.z + nz * lat);
      scene.add(group);
      boxes.push({ group, active: true, timer: 0, baseY: 1.2, seed: Math.random() * 6 });
    }
  }
  const shells = [];
  const bananas = [];

  function reset() {
    racing = false; time = 0; countdown = 3.0; lastCount = 4;
    results = null; raceOverFired = false; playerGrace = 0;
    for (const k of karts) {
      k.lap = 0; k.nextCheckpoint = 0; k.finished = false;
      k.finishTime = null; k.place = 0; k._wasStar = false;
    }
    for (const b of boxes) { b.active = true; b.timer = 0; b.group.visible = true; }
    for (const s of shells) scene.remove(s.group);
    shells.length = 0;
    for (const b of bananas) scene.remove(b.group);
    bananas.length = 0;
  }

  function crossed(prevT, nt, cp) {
    if (prevT <= nt) return prevT < cp && nt >= cp;
    return prevT < cp || nt >= cp;
  }

  function updateProgress(k) {
    const prevT = k.t;
    const nt = track.nearestT(k.state.x, k.state.z, k.t);
    k.t = nt;
    const cps = [0.25, 0.5, 0.75];
    if (k.nextCheckpoint < 3 && crossed(prevT, nt, cps[k.nextCheckpoint])) k.nextCheckpoint++;
    if (prevT > 0.9 && nt < 0.1) {
      if (k.nextCheckpoint >= 3) {
        k.lap++;
        k.nextCheckpoint = 0;
        fire('onLap', k, k.lap);
        if (k.lap > LAPS && !k.finished) {
          k.finished = true;
          k.finishTime = time;
          k.place = karts.filter((x) => x.finished).length;
          fire('onPlace', k, k.place);
          if (k.isPlayer) playerGrace = time + 8;
        }
      }
    }
  }

  function pickItem(k) {
    const rank = positions.indexOf(k.index) + 1;
    let pool;
    if (rank <= 2) pool = ['shell', 'shell', 'mushroom', 'banana'];
    else if (rank <= 5) pool = ['shell', 'mushroom', 'banana', 'banana'];
    else pool = ['mushroom', 'mushroom', 'banana', 'mushroom'];
    return pool[Math.floor(Math.random() * pool.length)];
  }

  function updateBoxes(dt) {
    for (const b of boxes) {
      if (!b.active) {
        b.timer += dt;
        if (b.timer >= ITEMS.BOX_RESPAWN) { b.active = true; b.group.visible = true; }
        continue;
      }
      b.group.rotation.y += dt * 1.6;
      b.group.position.y = b.baseY + Math.sin(time * 3 + b.seed) * 0.2;
      for (const k of karts) {
        if (k.item) continue;
        const dx = k.state.x - b.group.position.x;
        const dz = k.state.z - b.group.position.z;
        if (dx * dx + dz * dz < 2.2 * 2.2) {
          k.item = pickItem(k);
          b.active = false; b.group.visible = false; b.timer = 0;
          fire('onItem', k, k.item);
          break;
        }
      }
    }
  }

  function useItems() {
    for (const k of karts) {
      if (!(k.wantUseItem && k.item)) continue;
      const it = k.item;
      const h = k.state.heading;
      if (it === 'mushroom') {
        k.boost(PHYS.BOOST_DURATION, PHYS.BOOST_SPEED);
        fire('onBoost', k);
      } else if (it === 'star') {
        k.setStar(true);
        fire('onStar', k, true);
      } else if (it === 'banana') {
        const g = makeBanana();
        g.position.set(k.state.x - Math.sin(h) * 2.5, 0, k.state.z - Math.cos(h) * 2.5);
        scene.add(g);
        bananas.push({ group: g, life: ITEMS.BANANA_LIFETIME });
      } else if (it === 'shell') {
        const g = makeShell();
        g.position.set(k.state.x + Math.sin(h) * 2, 0, k.state.z + Math.cos(h) * 2);
        scene.add(g);
        shells.push({ group: g, x: g.position.x, z: g.position.z, heading: h, life: ITEMS.SHELL_LIFETIME, owner: k });
      }
      k.item = null;
      k.wantUseItem = false;
    }
  }

  function updateShells(dt) {
    for (let i = shells.length - 1; i >= 0; i--) {
      const s = shells[i];
      s.life -= dt;
      let target = null, best = ITEMS.SHELL_RANGE;
      const shx = Math.sin(s.heading), shz = Math.cos(s.heading);
      for (const k of karts) {
        if (k === s.owner) continue;
        const dx = k.state.x - s.x, dz = k.state.z - s.z;
        const along = dx * shx + dz * shz;
        if (along > 0 && along < best) {
          const lat = dx * -shz + dz * shx;
          const score = along + Math.abs(lat) * 3;
          if (score < best) { best = score; target = k; }
        }
      }
      if (target) {
        const desired = Math.atan2(target.state.x - s.x, target.state.z - s.z);
        let da = Math.atan2(Math.sin(desired - s.heading), Math.cos(desired - s.heading));
        s.heading += Math.max(-1.6 * dt, Math.min(1.6 * dt, da));
      }
      s.x += Math.sin(s.heading) * ITEMS.SHELL_SPEED * dt;
      s.z += Math.cos(s.heading) * ITEMS.SHELL_SPEED * dt;
      s.group.position.x = s.x;
      s.group.position.z = s.z;
      s.group.rotation.y = s.heading;
      let hit = false;
      for (const k of karts) {
        if (k === s.owner || k.state.starTime > 0) continue;
        const dx = k.state.x - s.x, dz = k.state.z - s.z;
        if (dx * dx + dz * dz < 4) {
          k.spinOut();
          fire('onSpin', k); fire('onHit', k);
          hit = true;
          break;
        }
      }
      if (hit || s.life <= 0) { scene.remove(s.group); shells.splice(i, 1); }
    }
  }

  function updateBananas(dt) {
    for (let i = bananas.length - 1; i >= 0; i--) {
      const b = bananas[i];
      b.life -= dt;
      let hit = false;
      for (const k of karts) {
        if (k.state.starTime > 0) continue;
        const dx = k.state.x - b.group.position.x;
        const dz = k.state.z - b.group.position.z;
        if (dx * dx + dz * dz < 2.25) {
          k.spinOut();
          fire('onSpin', k); fire('onHit', k);
          hit = true;
          break;
        }
      }
      if (hit || b.life <= 0) { scene.remove(b.group); bananas.splice(i, 1); }
    }
  }

  function updateCollisions(dt) {
    for (let i = 0; i < karts.length; i++) {
      for (let j = i + 1; j < karts.length; j++) {
        const a = karts[i], b = karts[j];
        const dx = b.state.x - a.state.x, dz = b.state.z - a.state.z;
        const d = Math.hypot(dx, dz);
        const minD = 2 * PHYS.KART_RADIUS;
        if (d < minD && d > 0.0001) {
          if (a.state.starTime > 0) { b.spinOut(); fire('onSpin', b); }
          else if (b.state.starTime > 0) { a.spinOut(); fire('onSpin', a); }
          else {
            const overlap = minD - d;
            const push = Math.min(overlap * 0.5, PHYS.KART_PUSH * dt);
            const nx = dx / d, nz = dz / d;
            a.state.x -= nx * push; a.state.z -= nz * push;
            b.state.x += nx * push; b.state.z += nz * push;
            a.state.speed *= 0.985; b.state.speed *= 0.985;
          }
        }
      }
    }
  }

  function updatePositions() {
    const idx = karts.map((k) => k.index);
    idx.sort((a, b) => {
      const ka = karts[a], kb = karts[b];
      if (ka.finished && kb.finished) return ka.finishTime - kb.finishTime;
      if (ka.finished) return -1;
      if (kb.finished) return 1;
      return (kb.lap + kb.t) - (ka.lap + ka.t);
    });
    positions.length = 0;
    for (const i of idx) positions.push(i);
  }

  function checkRaceOver() {
    if (raceOverFired) return;
    const allDone = karts.every((k) => k.finished);
    const playerDone = karts[0].finished;
    if (!(allDone || (playerDone && time >= playerGrace))) return;
    const finished = karts.filter((k) => k.finished).sort((a, b) => a.finishTime - b.finishTime);
    const notFin = karts.filter((k) => !k.finished)
      .sort((a, b) => (b.lap + b.t) - (a.lap + a.t));
    results = finished.concat(notFin)
      .map((k, i) => ({
        index: k.index,
        name: k.isPlayer ? 'You' : 'Kart #' + k.index,
        time: k.finished ? k.finishTime : null,
        place: i + 1,
      }));
    raceOverFired = true;
    fire('onRaceOver', results);
  }

  function update(dt) {
    if (!racing) {
      countdown -= dt;
      const c = Math.ceil(countdown);
      if (c < lastCount && c >= 0) { fire('onCountdown', c); lastCount = c; }
      if (countdown <= 0) { racing = true; fire('onGo'); }
      for (const b of boxes) if (b.active) b.group.rotation.y += dt * 1.6;
      return;
    }
    time += dt;
    for (const k of karts) {
      if (!k.finished) updateProgress(k);
      if (k.state.starTime > 0) k._wasStar = true;
      else if (k._wasStar) { k._wasStar = false; fire('onStar', k, false); }
    }
    useItems();
    updateBoxes(dt);
    updateShells(dt);
    updateBananas(dt);
    updateCollisions(dt);
    updatePositions();
    checkRaceOver();
  }

  return {
    get racing() { return racing; },
    get time() { return time; },
    positions,
    get results() { return results; },
    reset,
    update,
  };
}
