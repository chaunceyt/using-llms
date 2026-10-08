// terrain/textures.js — procedural PBR albedo maps (grass / sand / dirt / rock),
// generated at runtime from the seed into a canvas. Deterministic (seeded noise
// only). The fragment shader blends these four layers by slope + height.
import * as THREE from 'three';
import { createNoise } from './noise.js';

const SIZE = 512;

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// Fill one channel of albedo: base color plus layered seeded noise for mottling.
// `base`/`vary` are [r,g,b] in 0..255; `vary2` is the fine-speckle amplitude.
function makeCanvas(seed, base, vary, vary2) {
  const noise = createNoise(seed ^ 0x5a5a);
  const c = document.createElement('canvas');
  c.width = SIZE; c.height = SIZE;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(SIZE, SIZE);
  const d = img.data;
  const sc = 3.0 / SIZE;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const u = x * sc, v = y * sc;
      // coarse patches
      let n = noise.fbm(u + 2.1, v + 5.7, 4);              // ~[-1,1]
      // fine speckle (independent lattice via offset)
      let n2 = noise.valueNoise2(u * 6.0 + 9.3, v * 6.0 + 13.7);
      const f = [
        base[0] + vary[0] * n + vary2 * n2,
        base[1] + vary[1] * n + vary2 * n2,
        base[2] + vary[2] * n + vary2 * n2,
      ];
      const idx = (y * SIZE + x) * 4;
      d[idx] = clamp255(f[0]);
      d[idx + 1] = clamp255(f[1]);
      d[idx + 2] = clamp255(f[2]);
      d[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// Build the four layer textures + a shared detail normal-ish texture.
export function createGroundTextures(seed) {
  const texSeed = (seed | 0) ^ 0x3c6ef37;
  const grass = new THREE.CanvasTexture(
    makeCanvas(texSeed ^ 1, [92, 128, 76], [22, 26, 18], 9)
  );
  const sand = new THREE.CanvasTexture(
    makeCanvas(texSeed ^ 2, [198, 182, 138], [20, 16, 14], 7)
  );
  const dirt = new THREE.CanvasTexture(
    makeCanvas(texSeed ^ 3, [128, 99, 68], [26, 22, 16], 8)
  );
  const rock = new THREE.CanvasTexture(
    makeCanvas(texSeed ^ 4, [124, 120, 114], [26, 26, 26], 10)
  );

  const maps = { grass, sand, dirt, rock };
  for (const k in maps) {
    const t = maps[k];
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.colorSpace = THREE.SRGBColorSpace;
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearMipmapLinearFilter;
  }
  return maps;
}
