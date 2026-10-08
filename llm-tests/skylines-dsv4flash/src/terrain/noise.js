// terrain/noise.js — seeded, deterministic value-noise primitives.
//
// All randomness is drawn from a mulberry32 PRNG derived ONLY from the world
// seed (plus a fixed salt). Nothing uses Math.random, so the heightfield and
// every texture are a pure function of the seed. A dedicated PRNG stream (not
// world.rng itself) keeps terrain generation independent of module call order,
// which preserves cross-module determinism for roads/zoning later.
import { mulberry32 } from '../core/index.js';

const SALT = 0x9e3779b9;

// Build a noise context (permutation table + value-noise / fbm / ridge) from a seed.
export function createNoise(seed) {
  const rng = mulberry32(((seed | 0) ^ SALT) >>> 0);
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  // Fisher-Yates shuffle with the seeded PRNG.
  for (let i = 255; i > 0; i--) {
    const j = (rng() * (i + 1)) | 0;
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  const perm = new Uint8Array(512);
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];

  // Lattice hash -> scalar in [-1, 1].
  function hash(ix, iz) {
    return perm[(perm[ix & 255] + (iz & 255)) & 255] / 127.5 - 1.0;
  }

  // Smooth interpolated value noise over continuous x,z.
  function valueNoise2(x, z) {
    const xi = Math.floor(x), zi = Math.floor(z);
    const xf = x - xi, zf = z - zi;
    const u = xf * xf * (3 - 2 * xf);      // smoothstep
    const v = zf * zf * (3 - 2 * zf);
    const a = hash(xi, zi), b = hash(xi + 1, zi);
    const c = hash(xi, zi + 1), d = hash(xi + 1, zi + 1);
    const ab = a + (b - a) * u;
    const cd = c + (d - c) * u;
    return ab + (cd - ab) * v;             // in [-1, 1]
  }

  // Fractal Brownian motion: sum of octaves -> normalized to approx [-1, 1].
  function fbm(x, z, oct = 5) {
    let s = 0, n = 0, a = 1, f = 1;
    for (let o = 0; o < oct; o++) {
      s += valueNoise2(x * f, z * f) * a;
      n += a;
      a *= 0.5;
      f *= 2.0;
    }
    return s / n;
  }

  // Ridged multifractal -> sharp crests/lines (bluffs, ridges). Range ~[0,1].
  function ridge(x, z) {
    let s = 0, n = 0, a = 1, f = 1;
    for (let o = 0; o < 4; o++) {
      let v = valueNoise2(x * f, z * f);
      v = 1 - Math.abs(v);                 // ridged
      s += v * v * a;
      n += a;
      a *= 0.5;
      f *= 2.2;
    }
    return s / n;
  }

  return { valueNoise2, fbm, ridge };
}
