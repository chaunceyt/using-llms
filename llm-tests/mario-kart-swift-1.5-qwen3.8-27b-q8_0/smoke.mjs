// Headless smoke test: stubs browser APIs, drives the logic modules.
function makeCtx2d() {
  return new Proxy({}, {
    get: (_t, p) => (p === 'canvas' ? {} : () => {}),
    set: () => true,
  });
}
function makeEl() {
  const el = {
    style: {}, children: [], dataset: {},
    className: '', textContent: '', innerHTML: '',
    appendChild(c) { this.children.push(c); return c; },
    querySelector() { return makeEl(); },
    addEventListener() {}, removeEventListener() {},
    getContext() { return makeCtx2d(); },
    get offsetWidth() { return 100; },
  };
  return el;
}
globalThis.document = {
  createElement: (tag) => (tag === 'canvas' ? { width: 0, height: 0, getContext: () => makeCtx2d() } : makeEl()),
};
globalThis.window = {
  addEventListener() {}, removeEventListener() {},
  innerWidth: 1280, innerHeight: 720, devicePixelRatio: 1,
};
globalThis.performance = globalThis.performance || { now: () => Date.now() };

const results = [];
function check(name, cond, extra) {
  results.push({ name, ok: !!cond, extra });
  console.log((cond ? 'PASS' : 'FAIL') + '  ' + name + (extra ? '  [' + extra + ']' : ''));
}

try {
  const THREE = await import('three');
  const { buildTrack } = await import('./js/track.js');
  const { createKart } = await import('./js/kart.js');
  const { createRace } = await import('./js/race.js');
  const { createAIController } = await import('./js/ai.js');
  const { createParticles } = await import('./js/particles.js');
  const { createCameraRig } = await import('./js/camera.js');

  const fakeScene = { add() {}, remove() {} };

  // ---- track ----
  const track = buildTrack(fakeScene);
  check('track built', !!track && !!track.curve);
  check('track length sane', track.trackLength > 300 && track.trackLength < 1200, track.trackLength.toFixed(0) + 'm');
  check('centerline size', track.centerlineXZ.length === track.sampleCount * 2);

  // nearestT self-consistency
  const pMid = track.pointAt(0.3);
  const tBack = track.nearestT(pMid.x, pMid.z, 0.3);
  check('nearestT ~ identity', Math.abs(tBack - 0.3) < 0.02, tBack.toFixed(3));

  // offRoad
  check('on-road not offroad', track.offRoad(pMid.x, pMid.z) === false);
  const far = { x: pMid.x + 60, z: pMid.z + 60 };
  check('far point offroad', track.offRoad(far.x, far.z) === true);

  // wallPush clamps
  const st = { x: 0, z: 0, speed: 40 };
  // push a point far out then wallPush should pull it back near the road
  const edge = track.pointAt(0.5);
  st.x = edge.x + 40; st.z = edge.z + 40; st.speed = 40;
  track.wallPush(st.x, st.z, st);
  const afterT = track.nearestT(st.x, st.z, 0.5);
  const afterP = track.pointAt(afterT);
  const distAfter = Math.hypot(st.x - afterP.x, st.z - afterP.z);
  // wall limit is 17.6m; coarse sampling allows a couple meters of slack
  check('wallPush pulls back to road', distAfter < 22, distAfter.toFixed(1) + 'm (was 56m)');

  // ---- karts ----
  const karts = [];
  for (let i = 0; i < 8; i++) karts.push(createKart(i, i === 0));
  const start = track.pointAt(0);
  const startTan = track.tangentAt(0);
  const heading = Math.atan2(startTan.x, startTan.z);
  karts.forEach((k, i) => {
    const pos = new THREE.Vector3(start.x - (i % 2) * 3, 0, start.z - Math.floor(i / 2) * 6 - 4);
    k.reset(pos, heading, 0.99);
  });

  const stubParticles = { dust() {}, drift() {}, boost() {}, hit() {}, confetti() {} };

  // ---- race ----
  let events = {};
  const cbs = {
    onCountdown: (n) => { events.countdown = (events.countdown || 0) + 1; events.lastCount = n; },
    onGo: () => { events.go = (events.go || 0) + 1; },
    onLap: () => { events.lap = (events.lap || 0) + 1; },
    onRaceOver: () => { events.over = (events.over || 0) + 1; },
  };
  const race = createRace(karts, track, fakeScene, cbs);
  race.reset();
  check('race starts in countdown', race.racing === false);
  karts[0].input.throttle = 1; // player drives

  // run 6 seconds (past the 3s countdown)
  const ai = [];
  for (let i = 1; i < 8; i++) ai.push(createAIController(karts[i], track, i));
  for (let f = 0; f < 360; f++) {
    const dt = 1 / 60;
    for (const c of ai) c.update(dt, karts, race);
    for (const k of karts) k.update(dt, track, stubParticles, race.racing);
    race.update(dt);
  }
  check('race went to racing', race.racing === true, 'go=' + events.go);
  check('countdown fired 4 beats', events.countdown === 4, events.countdown);
  check('race time advanced', race.time > 2.5, race.time.toFixed(2) + 's');
  check('positions has 8', race.positions.length === 8, race.positions.length);
  check('player speed > 0 after GO', karts[0].state.speed > 1, karts[0].state.speed.toFixed(1));

  // give player input and run more
  karts[0].input.throttle = 1;
  for (let f = 0; f < 600; f++) {
    const dt = 1 / 60;
    for (const c of ai) c.update(dt, karts, race);
    for (const k of karts) k.update(dt, track, stubParticles, race.racing);
    race.update(dt);
  }
  check('player progressed in track t', karts[0].t > 0 && karts[0].t < 1, karts[0].t.toFixed(3));
  check('karts stay near road', (() => {
    let worst = 0;
    for (const k of karts) {
      const t = track.nearestT(k.state.x, k.state.z, k.t);
      const p = track.pointAt(t);
      worst = Math.max(worst, Math.hypot(k.state.x - p.x, k.state.z - p.z));
    }
    return worst < 18;
  })(), 'worst ' + 'm');

  // ---- particles (real THREE scene) ----
  const pScene = new THREE.Scene();
  const particles = createParticles(pScene);
  particles.dust(0, 0); particles.drift(1, 1, 0xff0000); particles.boost(2, 2, 0);
  particles.hit(3, 3); particles.confetti(4, 4);
  for (let f = 0; f < 120; f++) particles.update(1 / 60);
  check('particles run', true);

  // ---- camera ----
  const cam = new THREE.PerspectiveCamera(60, 16 / 9, 0.5, 800);
  const rig = createCameraRig(cam);
  rig.setMode('chase');
  for (let f = 0; f < 60; f++) rig.update(1 / 60, karts[0], track, 'racing', race);
  check('camera moved (chase)', cam.position.length() > 1, cam.position.toArray().map((v) => v.toFixed(0)).join(','));
  rig.setMode('menu');
  for (let f = 0; f < 60; f++) rig.update(1 / 60, karts[0], track, 'menu', race);
  check('camera menu orbit', cam.position.length() > 1);

  // ---- item grant + use ----
  karts[0].item = 'mushroom';
  karts[0].wantUseItem = true;
  const speedBefore = karts[0].state.speed;
  race.update(1 / 60);
  check('mushroom consumed + boost', karts[0].item === null && karts[0].state.boostTime > 0,
    'boostTime=' + karts[0].state.boostTime.toFixed(2));

  // ---- spinout ----
  karts[1].spinOut();
  check('spinout sets timer', karts[1].state.spinTime > 0);

} catch (e) {
  console.log('THREW  ' + e.stack);
  process.exitCode = 1;
}

const failed = results.filter((r) => !r.ok).length;
console.log('\n' + (results.length - failed) + '/' + results.length + ' checks passed');
if (failed) process.exitCode = 1;
