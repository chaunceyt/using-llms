import * as THREE from 'three';
import { CONFIG } from './config.js';

const PHYS = CONFIG.PHYS;
const TURBO_COLORS = [0x4cc9f0, 0xff9e00, 0xff3b3b];

export function createKart(index, isPlayer) {
  const group = new THREE.Group();

  const chassisMat = new THREE.MeshLambertMaterial({ color: CONFIG.COLORS.KARTS[index] });
  const chassis = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.5, 2.0), chassisMat);
  chassis.position.y = 0.55;
  chassis.castShadow = true;
  group.add(chassis);

  const nose = new THREE.Mesh(new THREE.BoxGeometry(0.85, 0.32, 0.7), chassisMat);
  nose.position.set(0, 0.52, 1.15);
  nose.castShadow = true;
  group.add(nose);

  const spoilerMat = new THREE.MeshLambertMaterial({ color: 0x22262b });
  const spoiler = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.1, 0.42), spoilerMat);
  spoiler.position.set(0, 0.98, -0.95);
  group.add(spoiler);
  for (const side of [-0.42, 0.42]) {
    const strut = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.34, 0.3), spoilerMat);
    strut.position.set(side, 0.82, -0.95);
    group.add(strut);
  }

  const torso = new THREE.Mesh(
    new THREE.SphereGeometry(0.34, 10, 8),
    new THREE.MeshLambertMaterial({ color: 0x2b2f36 }));
  torso.position.set(0, 0.98, -0.12);
  torso.scale.set(1, 1.15, 1);
  group.add(torso);
  const helmet = new THREE.Mesh(
    new THREE.SphereGeometry(0.29, 12, 10),
    new THREE.MeshLambertMaterial({ color: 0xffffff }));
  helmet.position.set(0, 1.36, -0.12);
  helmet.castShadow = true;
  group.add(helmet);
  const visor = new THREE.Mesh(
    new THREE.SphereGeometry(0.2, 10, 8, 0, Math.PI),
    new THREE.MeshLambertMaterial({ color: 0x18202a }));
  visor.position.set(0, 1.38, 0.06);
  visor.scale.set(1, 0.55, 1);
  group.add(visor);

  const wheelGeo = new THREE.CylinderGeometry(0.33, 0.33, 0.32, 12);
  wheelGeo.rotateZ(Math.PI / 2);
  const hubGeo = new THREE.CylinderGeometry(0.14, 0.14, 0.34, 8);
  hubGeo.rotateZ(Math.PI / 2);
  const wheelMat = new THREE.MeshLambertMaterial({ color: 0x15181c });
  const hubMat = new THREE.MeshLambertMaterial({ color: 0xd9dde3 });
  const wheelPos = [
    [0.66, 0.33, 0.78], [-0.66, 0.33, 0.78],
    [0.66, 0.33, -0.78], [-0.66, 0.33, -0.78],
  ];
  const wheels = [];
  for (const wp of wheelPos) {
    const w = new THREE.Group();
    const tire = new THREE.Mesh(wheelGeo, wheelMat);
    tire.castShadow = true;
    w.add(tire);
    w.add(new THREE.Mesh(hubGeo, hubMat));
    w.position.set(wp[0], wp[1], wp[2]);
    group.add(w);
    wheels.push(w);
  }

  const state = {
    x: 0, y: 0, z: 0, heading: 0, speed: 0,
    drifting: false, driftDir: 1, driftCharge: 0,
    boostTime: 0, starTime: 0, spinTime: 0,
  };
  let boostSpeed = PHYS.BOOST_SPEED;
  let driftSparkTimer = 0;
  let dustTimer = 0;
  let starPulse = 0;
  let _particles = null;

  function boost(duration, extra) {
    state.boostTime = Math.max(state.boostTime, duration);
    boostSpeed = extra || PHYS.BOOST_SPEED;
    if (_particles) _particles.boost(state.x, state.z, state.heading);
  }
  function spinOut() {
    state.spinTime = PHYS.SPIN_DURATION;
    state.speed *= 0.3;
  }
  function setStar(on) {
    state.starTime = on ? PHYS.STAR_DURATION : 0;
  }
  function reset(pos, heading, t) {
    state.x = pos.x; state.z = pos.z; state.heading = heading; state.speed = 0;
    state.drifting = false; state.driftCharge = 0;
    state.boostTime = 0; state.starTime = 0; state.spinTime = 0;
    boostSpeed = PHYS.BOOST_SPEED;
    item = null; wantUseItem = false;
    kart.t = t; kart.lap = 0;
    group.position.set(pos.x, 0, pos.z);
    group.rotation.y = heading;
  }
  function setColor(hex) {
    chassisMat.color.setHex(hex);
  }
  function updateIdle() {
    state.speed *= 0.9;
  }

  function update(dt, track, particles, canDrive) {
    _particles = particles;
    const s = state;
    if (s.boostTime > 0) s.boostTime -= dt;
    if (s.starTime > 0) s.starTime -= dt;
    if (s.spinTime > 0) s.spinTime -= dt;
    if (driftSparkTimer > 0) driftSparkTimer -= dt;
    if (dustTimer > 0) dustTimer -= dt;

    if (s.starTime > 0) {
      starPulse += dt * 9;
      chassisMat.emissive.setHex(CONFIG.ITEMS.STAR_COLOR);
      chassisMat.emissiveIntensity = 0.45 + Math.sin(starPulse) * 0.3;
    } else {
      chassisMat.emissive.setHex(0x000000);
      chassisMat.emissiveIntensity = 1;
    }

    const inp = input;

    if (s.spinTime > 0) {
      s.speed *= Math.max(0, 1 - 3 * dt);
    } else if (!canDrive) {
      s.speed *= Math.max(0, 1 - 8 * dt);
    } else {
      let maxSpeed = PHYS.MAX_SPEED;
      const off = track.offRoad(s.x, s.z);
      if (off) maxSpeed = PHYS.OFFROAD_MAX_SPEED;
      if (s.boostTime > 0) maxSpeed += boostSpeed;
      if (s.starTime > 0) maxSpeed += PHYS.STAR_SPEED;

      if (inp.throttle) {
        s.speed += PHYS.ACCEL * (1 - Math.max(0, s.speed) / maxSpeed) * dt;
      }
      if (inp.brake) {
        if (s.speed > 0.5) s.speed -= PHYS.BRAKE * dt;
        else s.speed -= PHYS.ACCEL * 0.6 * dt;
        if (s.speed < PHYS.REVERSE_MAX) s.speed = PHYS.REVERSE_MAX;
      }
      if (!inp.throttle && !inp.brake) {
        s.speed -= Math.sign(s.speed) * Math.min(Math.abs(s.speed), PHYS.FRICTION * dt);
      }
      if (off) {
        s.speed -= Math.sign(s.speed) * Math.min(Math.abs(s.speed), PHYS.OFFROAD_DRAG * dt);
        if (dustTimer <= 0 && Math.abs(s.speed) > 2) {
          particles.dust(s.x, s.z);
          dustTimer = 0.05;
        }
      }
      if (s.speed > maxSpeed) s.speed = Math.max(maxSpeed, s.speed - 30 * dt);

      const sf = Math.min(1, Math.abs(s.speed) / 12);
      if (Math.abs(s.speed) > PHYS.STEER_MIN_SPEED) {
        s.heading += inp.steer * PHYS.STEER_RATE * dt * sf * Math.sign(s.speed);
      }

      const wantDrift = inp.drift && Math.abs(s.speed) > PHYS.DRIFT_MIN_SPEED && Math.abs(inp.steer) > 0.1;
      if (wantDrift && !s.drifting) {
        s.drifting = true;
        s.driftDir = inp.steer > 0 ? 1 : -1;
        s.driftCharge = 0;
      }
      if (s.drifting) {
        if (!wantDrift) {
          const c = s.driftCharge;
          let tier = -1;
          if (c >= PHYS.DRIFT_TURBO_TIME[2]) tier = 2;
          else if (c >= PHYS.DRIFT_TURBO_TIME[1]) tier = 1;
          else if (c >= PHYS.DRIFT_TURBO_TIME[0]) tier = 0;
          if (tier >= 0) {
            particles.drift(s.x, s.z, TURBO_COLORS[tier]);
            boost(PHYS.DRIFT_TURBO_DURATION, PHYS.DRIFT_TURBO_BOOST[tier]);
          }
          s.drifting = false;
          s.driftCharge = 0;
        } else {
          s.driftCharge += dt;
          if (driftSparkTimer <= 0) {
            const tier = s.driftCharge >= PHYS.DRIFT_TURBO_TIME[2] ? 2
              : s.driftCharge >= PHYS.DRIFT_TURBO_TIME[1] ? 1
              : s.driftCharge >= PHYS.DRIFT_TURBO_TIME[0] ? 0 : -1;
            const col = tier >= 0 ? TURBO_COLORS[tier] : 0x8ecae6;
            particles.drift(s.x - Math.sin(s.heading), s.z - Math.cos(s.heading), col);
            driftSparkTimer = 0.06;
          }
        }
      }
    }

    const fx = Math.sin(s.heading), fz = Math.cos(s.heading);
    s.x += fx * s.speed * dt;
    s.z += fz * s.speed * dt;
    track.wallPush(s.x, s.z, s);

    group.position.set(s.x, 0, s.z);
    let visHead = s.heading;
    if (s.drifting) visHead += s.driftDir * 0.35 * Math.min(1, Math.abs(s.speed) / 30);
    if (s.spinTime > 0) visHead += s.spinTime * 7;
    group.rotation.y = visHead;

    const roll = (s.speed * dt) / 0.33;
    for (const w of wheels) {
      w.children[0].rotation.x += roll;
      w.children[1].rotation.x += roll;
    }
    const steerVis = inp.steer * 0.42;
    wheels[0].rotation.y = steerVis;
    wheels[1].rotation.y = steerVis;
  }

  const input = { throttle: 0, brake: 0, steer: 0, drift: 0, useItem: 0 };
  let item = null;
  let wantUseItem = false;

  const kart = {
    group, isPlayer, index, state,
    get item() { return item; },
    set item(v) { item = v; },
    get wantUseItem() { return wantUseItem; },
    set wantUseItem(v) { wantUseItem = v; },
    t: 0, lap: 0,
    input,
    reset, setColor, boost, spinOut, setStar, update, updateIdle,
  };

  return kart;
}

export function setupPlayerInput(kart) {
  let left = false, right = false;
  const setSteer = () => { kart.input.steer = (right ? 1 : 0) - (left ? 1 : 0); };
  const gameKeys = new Set([
    'KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
    'Space', 'ShiftLeft', 'ShiftRight', 'KeyE',
  ]);
  window.addEventListener('keydown', (e) => {
    if (gameKeys.has(e.code)) e.preventDefault();
    switch (e.code) {
      case 'KeyW': case 'ArrowUp': kart.input.throttle = 1; break;
      case 'KeyS': case 'ArrowDown': kart.input.brake = 1; break;
      case 'KeyA': case 'ArrowLeft': left = true; setSteer(); break;
      case 'KeyD': case 'ArrowRight': right = true; setSteer(); break;
      case 'Space': case 'ShiftLeft': case 'ShiftRight': kart.input.drift = 1; break;
      case 'KeyE': kart.input.useItem = 1; break;
    }
  });
  window.addEventListener('keyup', (e) => {
    switch (e.code) {
      case 'KeyW': case 'ArrowUp': kart.input.throttle = 0; break;
      case 'KeyS': case 'ArrowDown': kart.input.brake = 0; break;
      case 'KeyA': case 'ArrowLeft': left = false; setSteer(); break;
      case 'KeyD': case 'ArrowRight': right = false; setSteer(); break;
      case 'Space': case 'ShiftLeft': case 'ShiftRight': kart.input.drift = 0; break;
    }
  });
}
