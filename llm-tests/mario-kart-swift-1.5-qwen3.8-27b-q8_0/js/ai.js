import { CONFIG } from './config.js';

const AI = CONFIG.AI;
const PHYS = CONFIG.PHYS;

// Deterministic per-index hash in [0,1) — gives each AI kart a stable
// lane offset / skill / personality without a shared RNG.
function hash01(n) {
  let h = (Math.imul(n, 374761393) + 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function wrapAngle(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

const AVOID_DURATION = 2.0;   // seconds of lateral bias after spotting a kart ahead
const AVOID_OFFSET = 2.6;     // meters of extra lateral aim
const SHARP_ANGLE = 0.45;     // rad of heading change over 30m that counts as "sharp"
const STEER_SCALE = 0.5;      // rad of heading error for full steer authority
const STEER_DEADZONE = 0.15;

export function createAIController(kart, track, index) {
  const trackLength = track.trackLength;
  const laneOffset = (hash01(index + 1) * 2 - 1) * AI.LANE_SPREAD;
  const skill = 1 + (hash01(index + 7) * 2 - 1) * AI.SKILL_JITTER;
  const drifts = hash01(index + 13) < 0.85; // a few karts stay clean through corners

  const maxOffset = track.TRACK_WIDTH / 2 - 1.6;

  let avoidTimer = 0;
  let avoidDir = 0;
  let itemTimer = 0;
  let lastItem = null;

  function update(dt, karts, race) {
    const st = kart.state;
    const inp = kart.input;
    inp.steer = 0;
    inp.throttle = 0;
    inp.brake = 0;
    inp.drift = 0;

    if (st.spinTime > 0) return; // physics owns the spin; keep inputs flat

    if (kart.item !== lastItem) {
      lastItem = kart.item;
      itemTimer = 0.5 + Math.random() * 1.5;
    }

    const heading = st.heading;
    const fx = Math.sin(heading);
    const fz = Math.cos(heading);
    const nx = -fz; // left normal
    const nz = fx;

    const t0 = ((kart.t % 1) + 1) % 1;

    // --- corner sharpness: heading change over the next 30m ---
    const tangNow = track.tangentAt(t0);
    const tangAhead = track.tangentAt((t0 + 30 / trackLength) % 1);
    const dot = clamp(tangNow.x * tangAhead.x + tangNow.z * tangAhead.z, -1, 1);
    const sharp = Math.acos(dot) > SHARP_ANGLE;

    // --- avoidance: kart within 6m ahead in a similar lane ---
    if (avoidTimer > 0) avoidTimer -= dt;
    let effOffset = laneOffset;
    for (let i = 0; i < karts.length; i++) {
      const o = karts[i];
      if (o === kart) continue;
      const rx = o.state.x - st.x;
      const rz = o.state.z - st.z;
      const along = rx * fx + rz * fz;
      if (along > 0 && along < 6) {
        const latSigned = rx * nx + rz * nz;
        if (Math.abs(latSigned) < 2.6) {
          // steer away from the threat; if dead ahead, lean into our own lane side
          avoidDir = latSigned > 0.4 ? -1 : latSigned < -0.4 ? 1 : (laneOffset >= 0 ? 1 : -1);
          avoidTimer = AVOID_DURATION;
          break;
        }
      }
    }
    if (avoidTimer > 0) effOffset = clamp(effOffset + avoidDir * AVOID_OFFSET, -maxOffset, maxOffset);

    // --- steering: proportional toward the look-ahead aim point ---
    const aimT = (t0 + AI.LOOKAHEAD / trackLength) % 1;
    const aim = track.pointAt(aimT);
    const aimTan = track.tangentAt(aimT);
    const ax = aim.x + -aimTan.z * effOffset;
    const az = aim.z + aimTan.x * effOffset;
    const desired = Math.atan2(ax - st.x, az - st.z);
    let raw = wrapAngle(desired - heading) / STEER_SCALE;
    raw = clamp(raw, -1, 1);
    if (raw > -STEER_DEADZONE && raw < STEER_DEADZONE) raw = 0;
    inp.steer = raw;

    // --- throttle: full, eased in sharp corners, rubber-banded vs leader ---
    let throttle = skill;
    if (sharp) throttle = Math.min(throttle, 0.6);
    if (race && race.positions && race.positions.length) {
      const leader = karts[race.positions[0]];
      if (leader && leader !== kart) {
        const gap = (leader.lap + leader.t - (kart.lap + kart.t)) * trackLength;
        if (gap > 40) throttle = 1;              // far behind: ride full throttle
        else if (gap < -25) throttle *= AI.RUBBERBAND_AHEAD; // far ahead: ease off
      }
    }
    inp.throttle = clamp(throttle, 0, 1);

    // --- drift for mini-turbos in sharp corners at speed ---
    if (drifts && sharp && Math.abs(raw) > 0.25 && st.speed > PHYS.DRIFT_MIN_SPEED) {
      inp.drift = 1;
    }

    // --- item usage ---
    if (kart.item) {
      const it = kart.item;
      if (it === 'mushroom' || it === 'star') {
        itemTimer -= dt;
        if (itemTimer <= 0) kart.wantUseItem = true;
      } else if (it === 'shell') {
        for (let i = 0; i < karts.length; i++) {
          const o = karts[i];
          if (o === kart) continue;
          const along = (o.state.x - st.x) * fx + (o.state.z - st.z) * fz;
          if (along > 0 && along < 30) { kart.wantUseItem = true; break; }
        }
      } else if (it === 'banana') {
        let clear = true;
        for (let i = 0; i < karts.length; i++) {
          const o = karts[i];
          if (o === kart) continue;
          const along = (o.state.x - st.x) * fx + (o.state.z - st.z) * fz;
          if (along < 0 && along > -15) { clear = false; break; }
        }
        if (clear) kart.wantUseItem = true;
      }
    }
  }

  return { update };
}
