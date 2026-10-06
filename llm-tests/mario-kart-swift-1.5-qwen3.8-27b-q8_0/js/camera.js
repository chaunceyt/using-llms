import * as THREE from 'three';
import { CONFIG } from './config.js';

const CAM = CONFIG.CAMERA;
const PHYS = CONFIG.PHYS;

export function createCameraRig(camera) {
  let mode = 'menu';
  let orbit = 0;
  let shake = 0;
  let prevBoost = 0, prevSpin = 0;

  const camPos = new THREE.Vector3(0, 32, -70);
  const lookAt = new THREE.Vector3(0, 0, 0);
  const tPos = new THREE.Vector3();
  const tLook = new THREE.Vector3();

  function setMode(m) { mode = m; }

  function damp(cur, target, lambda, dt) {
    cur.lerp(target, 1 - Math.exp(-lambda * dt));
  }

  function update(dt, kart, track, gameState, race) {
    if (gameState) mode = gameState === 'menu' ? 'menu' : gameState === 'results' ? 'results' : 'chase';
    const s = kart.state;

    if (mode === 'chase') {
      const fx = Math.sin(s.heading), fz = Math.cos(s.heading);
      tPos.set(s.x - fx * CAM.CHASE_DISTANCE, CAM.CHASE_HEIGHT, s.z - fz * CAM.CHASE_DISTANCE);
      tLook.set(s.x + fx * 6, 1.5, s.z + fz * 6);

      if (s.boostTime > prevBoost) shake = Math.max(shake, 0.28);
      if (s.spinTime > prevSpin) shake = Math.max(shake, 0.55);
      prevBoost = s.boostTime; prevSpin = s.spinTime;
      shake = Math.max(0, shake - dt * 1.4);
      if (shake > 0) {
        tPos.x += (Math.random() - 0.5) * shake;
        tPos.y += (Math.random() - 0.5) * shake;
        tPos.z += (Math.random() - 0.5) * shake;
      }

      damp(camPos, tPos, CAM.CHASE_LERP, dt);
      damp(lookAt, tLook, CAM.LOOK_LERP, dt);
      camera.position.copy(camPos);
      camera.lookAt(lookAt);

      const ratio = Math.min(1, Math.abs(s.speed) / (PHYS.MAX_SPEED + PHYS.BOOST_SPEED));
      const fov = CAM.FOV_MIN + (CAM.FOV_MAX - CAM.FOV_MIN) * ratio;
      if (Math.abs(camera.fov - fov) > 0.05) {
        camera.fov = fov;
        camera.updateProjectionMatrix();
      }
    } else if (mode === 'menu') {
      orbit += dt * 0.15;
      tPos.set(Math.cos(orbit) * 90, 35, Math.sin(orbit) * 90);
      tLook.set(0, 0, 0);
      damp(camPos, tPos, 2, dt);
      damp(lookAt, tLook, 2, dt);
      camera.position.copy(camPos);
      camera.lookAt(lookAt);
      if (camera.fov !== 55) { camera.fov = 55; camera.updateProjectionMatrix(); }
    } else {
      orbit += dt * 0.4;
      tPos.set(s.x + Math.cos(orbit) * 14, 6, s.z + Math.sin(orbit) * 14);
      tLook.set(s.x, 1.5, s.z);
      damp(camPos, tPos, 2, dt);
      damp(lookAt, tLook, 2, dt);
      camera.position.copy(camPos);
      camera.lookAt(lookAt);
      if (camera.fov !== 60) { camera.fov = 60; camera.updateProjectionMatrix(); }
    }
  }

  return { update, setMode };
}
