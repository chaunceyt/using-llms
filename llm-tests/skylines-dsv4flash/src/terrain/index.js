// terrain — large procedural heightfield with chunked distance-LOD, and a
// photographic splat-PBR ground material. Pure function of the seed.
import * as THREE from 'three';
import { createNoise } from './noise.js';
import { createGroundTextures } from './textures.js';
import { createTerrainMaterial, computeSun } from './material.js';

export const id = 'terrain';

// Chunk size (m) and LOD table: [max view distance, segments-per-chunk-side].
const CHUNK = 200;
const LODS = [
  { maxDist: 420, segs: 64 },    // near — ~3.1 m spacing
  { maxDist: 1000, segs: 26 },   // mid  — ~7.7 m spacing
  { maxDist: Infinity, segs: 12 },// far  — ~16.6 m spacing
];
const SKIRT = 90;               // downward skirt drop (m) hides LOD seams

const state = {
  world: null,
  size: 2048,
  half: 1024,
  noise: null,
  heightAt: () => 0,
  normalAt: () => new THREE.Vector3(0, 1, 0),
  group: null,
  material: null,
  chunks: [],                  // { mesh, ix, iz, lod }
  geoCache: new Map(),         // `${ix}_${iz}_l${lod}` -> BufferGeometry
};

// ---------------------------------------------------------------------------
// Height field (the analytic source of truth for the whole terrain + roads).
// Gentle hills / valleys, ridged bluffs, and a meandering river floodplain.
// ---------------------------------------------------------------------------
function makeHeightAt(noise, size) {
  const S = size;
  const { fbm, ridge } = noise;
  return function heightAt(x, z) {
    const nx = x / S, nz = z / S;

    // large rolling countryside
    let e = fbm(nx * 2.1 + 3.7, nz * 2.1, 5) * 1.0;
    e += fbm(nx * 5.3 + 11.9, nz * 5.3, 4) * 0.25;   // mid-scale rolling detail
    const r = ridge(nx * 2.6 + 7.3, nz * 2.6);       // ridged bluffs ~[0,1]
    e += (r - 0.5) * 1.5;

    let h = 80 + e * 70;

    // meandering river / floodplain valley running mostly along X, near origin.
    const rz = 30.0 + 180.0 * Math.sin(x / 260.0);
    const rdist = Math.abs(z - rz);
    const flood = Math.exp(-(rdist * rdist) / (2.0 * 150.0 * 150.0));
    h -= flood * (55.0 + 12.0 * fbm(nx * 14 + 5.0, nz * 14, 3));

    return Math.max(h, 6.0);
  };
}

// ---------------------------------------------------------------------------
// Chunk geometry (one BufferGeometry per chunk/LOD; skirts hide LOD seams).
// ---------------------------------------------------------------------------
function buildChunkGeometry(ix, iz, segs) {
  const size = state.size;
  const half = state.half;
  const w = CHUNK;
  const x0 = ix * w - half;
  const z0 = iz * w - half;
  const n = segs + 2;                 // grid points incl. skirt border
  const heightAt = state.heightAt;

  const positions = new Float32Array(n * n * 3);
  const normals = new Float32Array(n * n * 3);
  let vi = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let fx = (i - 1) / segs;
      let fz = (j - 1) / segs;
      const border = i === 0 || i === n - 1 || j === 0 || j === n - 1;

      if (border) {
        // push the skirt slightly outward so neighbours' walls don't Z-fight
        if (i === 0) fx = -0.6 / segs; else if (i === n - 1) fx = 1 + 0.6 / segs;
        if (j === 0) fz = -0.6 / segs; else if (j === n - 1) fz = 1 + 0.6 / segs;
      }
      const x = x0 + fx * w;
      const z = z0 + fz * w;

      let y, nxv = 0, nyv = 0, nzv = 0;
      if (border) {
        // skirt sits just under the surface along the chunk perimeter
        const ex = Math.min(Math.max(fx, 0), 1);
        const ez = Math.min(Math.max(fz, 0), 1);
        y = heightAt(x0 + ex * w, z0 + ez * w) - SKIRT;
        nxv = (i === 0 ? -1 : i === n - 1 ? 1 : 0) * 0.6;
        nzv = (j === 0 ? -1 : j === n - 1 ? 1 : 0) * 0.6;
        nyv = -1;
      } else {
        y = heightAt(x, z);
        const d = 3.0;
        const hxp = heightAt(x + d, z), hxm = heightAt(x - d, z);
        const hzp = heightAt(x, z + d), hzm = heightAt(x, z - d);
        const gx = (hxp - hxm) / (2 * d);
        const gz = (hzp - hzm) / (2 * d);
        const len = Math.sqrt(gx * gx + 1 + gz * gz);
        nxv = -gx / len; nyv = 1 / len; nzv = -gz / len;
      }

      const k = vi * 3;
      positions[k] = x; positions[k + 1] = y; positions[k + 2] = z;
      normals[k] = nxv; normals[k + 1] = nyv; normals[k + 2] = nzv;
      vi++;
    }
  }

  const cells = n - 1;
  const indices = [];
  for (let j = 0; j < cells; j++) {
    for (let i = 0; i < cells; i++) {
      const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
      indices.push(a, c, b, b, c, d);   // upward-facing
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geo.setIndex(indices);
  return geo;
}

// ---------------------------------------------------------------------------
// Module API
// ---------------------------------------------------------------------------
export function heightAt(x, z) {
  return state.heightAt(x, z);
}
export function normalAt(x, z) {
  return state.normalAt(x, z);
}

export function generate(world, opts = {}) {
  const size = (world && world.terrain && world.terrain.size) || 2048;
  const seed = (world && world.meta && world.meta.seed) || 1337;

  state.world = world;
  state.size = size;
  state.half = size / 2;

  // deterministic noise from the seed
  state.noise = createNoise(seed);
  state.heightAt = makeHeightAt(state.noise, size);

  const d = 3.0;
  state.normalAt = (x, z) => {
    const hxp = state.heightAt(x + d, z), hxm = state.heightAt(x - d, z);
    const hzp = state.heightAt(x, z + d), hzm = state.heightAt(x, z - d);
    const gx = (hxp - hxm) / (2 * d);
    const gz = (hzp - hzm) / (2 * d);
    return new THREE.Vector3(-gx, 1, -gz).normalize();
  };

  // wire convenience accessors so roads/buildings can query ground height
  if (world) {
    if (world.terrain) { world.terrain.heightAt = state.heightAt; world.terrain.normalAt = state.normalAt; }
    world.heightAt = state.heightAt;
    world.normalAt = state.normalAt;
  }

  // tear down previous mesh, if any
  if (state.group) {
    if (state.group.parent) state.group.parent.remove(state.group);
    state.group = null;
  }
  state.geoCache.clear();
  state.chunks.length = 0;

  const maps = createGroundTextures(seed);
  state.material = createTerrainMaterial(maps);

  state.group = new THREE.Group();
  const cps = Math.ceil(size / CHUNK);
  for (let cz = 0; cz < cps; cz++) {
    for (let cx = 0; cx < cps; cx++) {
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), state.material);
      mesh.frustumCulled = true;
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      state.group.add(mesh);
      state.chunks.push({ mesh, ix: cx, iz: cz, lod: -1 });
    }
  }

  if (world && world.scene) world.scene.add(state.group);

  // build initial LODs against the current camera
  update(0, world);
}

export function init(world) {
  generate(world, {});
}

// Keep the terrain inside budget: distance-LOD + three's automatic frustum
// culling (each chunk has correct world-space bounds). Cheap when static.
export function update(dt, world) {
  const cam = world && world.camera;
  if (!cam || !state.group || !state.material) return;

  // refresh sun/lighting + camera uniforms from the time of day
  const sun = computeSun(world.meta ? world.meta.timeOfDaySec : 9 * 3600);
  state.material.uniforms.uSunDir.value.copy(sun.dir);
  state.material.uniforms.uSunColor.value.copy(sun.color);
  state.material.uniforms.uCamPos.value.copy(cam.position);

  const w = CHUNK, half = state.half;
  for (const c of state.chunks) {
    const cx = c.ix * w - half + w / 2;
    const cz = c.iz * w - half + w / 2;
    const dist = cam.position.distanceTo(_v0.set(cx, 0, cz));

    let lod = 0;
    for (let k = 0; k < LODS.length; k++) if (dist >= LODS[k].maxDist) lod = k + 1;
    const lodDef = LODS[Math.min(lod, LODS.length - 1)];

    if (c.lod !== lodDef.segs) {
      c.lod = lodDef.segs;
      const key = `${c.ix}_${c.iz}_l${lodDef.segs}`;
      let geo = state.geoCache.get(key);
      if (!geo) {
        geo = buildChunkGeometry(c.ix, c.iz, lodDef.segs);
        state.geoCache.set(key, geo);
      }
      c.mesh.geometry = geo;
    }
  }
}

// Minimal showcase for the demo harness: describe the terrain in the container.
export function showcase(container) {
  if (!container) return;
  const h0 = state.heightAt(0, 30);
  try {
    container.innerHTML =
      `<div style="font-family:sans-serif;color:#cfd8dc;padding:16px">
        <b>Terrain</b> — seeded heightfield (2048 m), chunked LOD,
        splat-PBR (grass/sand/dirt/rock).<br>
        Center height ≈ ${h0.toFixed(1)} m. Seed-driven, deterministic.
      </div>`;
  } catch { /* ignore */ }
}

const _v0 = new THREE.Vector3();
