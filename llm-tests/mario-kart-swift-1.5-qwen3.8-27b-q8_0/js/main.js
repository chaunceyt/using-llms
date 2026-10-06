import * as THREE from 'three';
import { CONFIG } from './config.js';
import { createScene } from './scene.js';
import { buildEnvironment } from './environment.js';
import { buildTrack } from './track.js';
import { createKart, setupPlayerInput } from './kart.js';
import { createAIController } from './ai.js';
import { createCameraRig } from './camera.js';
import { createRace } from './race.js';
import { createHUD } from './hud.js';
import { createAudio } from './audio.js';
import { createParticles } from './particles.js';

const gameEl = document.getElementById('game');
const hudEl = document.getElementById('hud');

const PHYS = CONFIG.PHYS;

// Wrap a curve parameter into [0,1) — `t % 1` alone stays negative for t < 0.
function wrapT(t) {
  return ((t % 1) + 1) % 1;
}

let sceneBundle, scene, camera;
let track, karts, aiControllers, race, hud, audio, particles, camRig;
let gameState = 'menu'; // 'menu' | 'racing' | 'results'
let selectedColor = 0;
let clock = new THREE.Clock();

// ---------- setup ----------

function init() {
  sceneBundle = createScene(gameEl);
  scene = sceneBundle.scene;
  camera = sceneBundle.camera;

  track = buildTrack(scene);
  buildEnvironment(scene, track);

  particles = createParticles(scene);
  audio = createAudio();
  camRig = createCameraRig(camera);
  hud = createHUD(hudEl, track);

  karts = [];
  for (let i = 0; i < CONFIG.RACE.KART_COUNT; i++) {
    const kart = createKart(i, i === 0);
    scene.add(kart.group);
    karts.push(kart);
  }

  aiControllers = [];
  for (let i = 1; i < karts.length; i++) {
    aiControllers.push(createAIController(karts[i], track, i));
  }

  setupPlayerInput(karts[0]);

  race = createRace(karts, track, scene, {
    onCountdown: (n) => { hud.setCountdown(n); audio.beep(n === 0); },
    onGo: () => { hud.setCountdown('GO'); },
    onLap: (kart, lap) => {
      if (kart.isPlayer) { hud.flashLap(lap); audio.lap(); }
    },
    onItem: (kart, item) => { if (kart.isPlayer) audio.item(); },
    onBoost: (kart) => { if (kart.isPlayer) audio.boost(); },
    onSpin: (kart) => { if (kart.isPlayer) audio.spin(); },
    onHit: (kart) => {
      particles.hit(kart.state.x, kart.state.z);
      if (kart.isPlayer) audio.hit();
    },
    onStar: (kart, on) => { if (kart.isPlayer) audio.star(on); },
    onPlace: (kart) => {
      if (kart.isPlayer) {
        audio.finish();
        particles.confetti(kart.state.x, kart.state.z);
      }
    },
    onRaceOver: (results) => {
      gameState = 'results';
      hud.showResults(results);
      camRig.setMode('results');
    },
  });

  resetToMenu();
  clock.start();
  requestAnimationFrame(loop);
}

function gridPosition(i) {
  // 2 columns x 4 rows behind the start line
  const col = i % 2;
  const row = Math.floor(i / 2);
  const t = wrapT(track.startT - (0.004 + row * 0.011));
  const p = track.curve.getPointAt(t);
  const tangent = track.curve.getTangentAt(t);
  const normal = new THREE.Vector3(-tangent.z, 0, tangent.x);
  const lateral = (col === 0 ? -1 : 1) * (CONFIG.TRACK.WIDTH * 0.22);
  return new THREE.Vector3(p.x + normal.x * lateral, 0, p.z + normal.z * lateral);
}

function placeKarts() {
  for (let i = 0; i < karts.length; i++) {
    const pos = gridPosition(i);
    const t = wrapT(track.startT - (0.004 + Math.floor(i / 2) * 0.011));
    const tangent = track.curve.getTangentAt(t);
    karts[i].reset(pos, Math.atan2(tangent.x, tangent.z), t);
  }
}

function resetToMenu() {
  gameState = 'menu';
  placeKarts();
  camRig.setMode('menu');
  hud.showMenu();
  hud.setSelectedColor(selectedColor);
  karts[0].setColor(CONFIG.COLORS.KARTS[selectedColor]);
}

function startRace() {
  if (gameState !== 'menu') return;
  audio.init();
  karts[0].setColor(CONFIG.COLORS.KARTS[selectedColor]);
  placeKarts();
  race.reset();
  camRig.setMode('chase');
  hud.hideMenu();
  gameState = 'racing';
}

// ---------- input (global, menu/results) ----------

window.addEventListener('keydown', (e) => {
  if (gameState === 'menu') {
    if (e.code === 'Enter' || e.code === 'Space') {
      startRace();
    } else if (e.code === 'ArrowLeft' || e.code === 'KeyA') {
      selectedColor = (selectedColor + CONFIG.COLORS.KARTS.length - 1) % CONFIG.COLORS.KARTS.length;
      hud.setSelectedColor(selectedColor);
      karts[0].setColor(CONFIG.COLORS.KARTS[selectedColor]);
    } else if (e.code === 'ArrowRight' || e.code === 'KeyD') {
      selectedColor = (selectedColor + 1) % CONFIG.COLORS.KARTS.length;
      hud.setSelectedColor(selectedColor);
      karts[0].setColor(CONFIG.COLORS.KARTS[selectedColor]);
    }
  } else if (gameState === 'results') {
    if (e.code === 'KeyR' || e.code === 'Enter') {
      resetToMenu();
    }
  }
});

// ---------- main loop ----------

function loop() {
  requestAnimationFrame(loop);
  const dt = Math.min(clock.getDelta(), 0.05);

  if (gameState === 'racing') {
    // AI drives itself
    for (const ai of aiControllers) ai.update(dt, karts, race);

    // physics for every kart
    for (const kart of karts) kart.update(dt, track, particles, gameState === 'racing' && race.racing);

    // race logic: laps, items, projectiles, rankings
    race.update(dt);

    // engine audio follows player speed
    const p = karts[0];
    audio.setEngine(Math.min(1, Math.abs(p.state.speed) / (PHYS.MAX_SPEED + PHYS.BOOST_SPEED)));
  } else if (gameState === 'menu') {
    for (const kart of karts) kart.updateIdle(dt);
    audio.setEngine(0);
  }

  // camera
  camRig.update(dt, karts[0], track, gameState, race);

  // particles
  particles.update(dt);

  // HUD
  if (gameState === 'racing' || gameState === 'results') {
    hud.update({
      mode: gameState,
      time: race.time,
      lap: karts[0].lap,
      laps: CONFIG.RACE.LAPS,
      position: race.positions.findIndex((k) => k === karts[0]) + 1,
      totalKarts: karts.length,
      speed: karts[0].state.speed,
      item: karts[0].item,
      karts,
    });
  }

  sceneBundle.render();
}

init();
