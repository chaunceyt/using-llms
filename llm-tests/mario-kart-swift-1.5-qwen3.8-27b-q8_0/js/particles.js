import * as THREE from 'three';

const COUNT = 250;

export function createParticles(scene) {
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const mat = new THREE.MeshBasicMaterial({
    depthWrite: false,
    transparent: true,
    blending: THREE.AdditiveBlending,
  });
  const mesh = new THREE.InstancedMesh(geo, mat, COUNT);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.frustumCulled = false;
  scene.add(mesh);

  const data = [];
  for (let i = 0; i < COUNT; i++) {
    data.push({
      active: false,
      pos: new THREE.Vector3(0, -1000, 0),
      vel: new THREE.Vector3(),
      life: 0, maxLife: 1, scale: 1,
      color: new THREE.Color(0xffffff),
      gravity: 0,
    });
  }

  const m4 = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const p = new THREE.Vector3();
  const white = new THREE.Color(0xffffff);
  for (let i = 0; i < COUNT; i++) {
    m4.makeScale(0, 0, 0);
    mesh.setMatrixAt(i, m4);
    mesh.setColorAt(i, white);
  }
  mesh.instanceMatrix.needsUpdate = true;

  let cursor = 0;
  function spawn(x, y, z, vx, vy, vz, life, scale, color, gravity) {
    const d = data[cursor];
    cursor = (cursor + 1) % COUNT;
    d.active = true;
    d.pos.set(x, y, z);
    d.vel.set(vx, vy, vz);
    d.life = life; d.maxLife = life;
    d.scale = scale;
    d.color.set(color);
    d.gravity = gravity;
  }

  function rand(a, b) { return a + Math.random() * (b - a); }

  function update(dt) {
    for (let i = 0; i < COUNT; i++) {
      const d = data[i];
      if (!d.active) continue;
      d.life -= dt;
      if (d.life <= 0) {
        d.active = false;
        m4.makeScale(0, 0, 0);
        mesh.setMatrixAt(i, m4);
        continue;
      }
      d.vel.y -= d.gravity * dt;
      d.pos.addScaledVector(d.vel, dt);
      const frac = d.life / d.maxLife;
      const sc = d.scale * (0.4 + 0.6 * frac);
      p.copy(d.pos);
      s.set(sc, sc, sc);
      q.identity();
      m4.compose(p, q, s);
      mesh.setMatrixAt(i, m4);
      mesh.setColorAt(i, d.color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  function dust(x, z) {
    const n = 3 + Math.floor(Math.random() * 3);
    for (let i = 0; i < n; i++) {
      spawn(x + rand(-0.5, 0.5), 0.3, z + rand(-0.5, 0.5),
        rand(-1, 1), rand(0.5, 1.6), rand(-1, 1),
        rand(0.35, 0.55), rand(0.5, 0.9), 0x9c8a6a, 2);
    }
  }

  function drift(x, z, color) {
    const n = 4 + Math.floor(Math.random() * 3);
    for (let i = 0; i < n; i++) {
      spawn(x + rand(-0.4, 0.4), 0.25, z + rand(-0.4, 0.4),
        rand(-2, 2), rand(1, 3), rand(-2, 2),
        rand(0.2, 0.35), rand(0.35, 0.55), color, 6);
    }
  }

  function boost(x, z, heading) {
    const fx = Math.sin(heading), fz = Math.cos(heading);
    const n = 9 + Math.floor(Math.random() * 3);
    for (let i = 0; i < n; i++) {
      spawn(x - fx * 1.3 + rand(-0.4, 0.4), 0.6, z - fz * 1.3 + rand(-0.4, 0.4),
        -fx * rand(4, 9) + rand(-1, 1), rand(0, 2), -fz * rand(4, 9) + rand(-1, 1),
        rand(0.3, 0.5), rand(0.4, 0.75), i % 2 ? 0xffffff : 0xff9e00, 3);
    }
  }

  function hit(x, z) {
    for (let i = 0; i < 10; i++) {
      const a = Math.random() * Math.PI * 2;
      spawn(x, 0.5, z,
        Math.cos(a) * rand(3, 7), rand(1, 5), Math.sin(a) * rand(3, 7),
        rand(0.35, 0.55), rand(0.35, 0.6), i % 2 ? 0xffff66 : 0xffffff, 8);
    }
  }

  function confetti(x, z) {
    const cols = [0xe63946, 0x2a6df5, 0xf7b32b, 0x2fa84f, 0x9b5de5, 0xf15bb5];
    for (let i = 0; i < 30; i++) {
      spawn(x + rand(-2, 2), 8, z + rand(-2, 2),
        rand(-2, 2), rand(-1, 0), rand(-2, 2),
        rand(1.5, 2.5), rand(0.3, 0.6), cols[i % cols.length], 3);
    }
  }

  return { update, dust, drift, boost, hit, confetti };
}
