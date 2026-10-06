// Deterministic seeded PRNG. The only source of randomness in the game.
// mulberry32 core; forks derive independent streams from (seed, label) via FNV-1a.
function fnv1a(str, seed) {
  let h = (seed ^ 0x811c9dc5) >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35) >>> 0; h ^= h >>> 16;
  return h >>> 0;
}

export class RNG {
  constructor(seed) {
    this.seed = (seed >>> 0) || 0x9e3779b9;
    this._s = this.seed;
  }
  _next() {
    let t = (this._s += 0x6d2b79f5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  next() { return this._next(); }
  range(a, b) { return a + (b - a) * this._next(); }
  int(a, b) { return Math.floor(this.range(a, b + 1)); }
  pick(arr) { return arr[this.int(0, arr.length - 1)]; }
  chance(p) { return this._next() < p; }
  // 2D value in [0,1) from a pair of ints (for grid hashing)
  hash2(ix, iz) {
    let h = this.seed;
    h = Math.imul(h ^ ix, 0x27d4eb2d) >>> 0;
    h = Math.imul(h ^ iz, 0x165667b1) >>> 0;
    h ^= h >>> 15; h = Math.imul(h, 0x85ebca6b) >>> 0; h ^= h >>> 13;
    return (h >>> 0) / 4294967296;
  }
  fork(label) { return new RNG(fnv1a(String(label), this.seed)); }
}

export function makeRNG(seed) { return new RNG((seed >>> 0) || 0x9e3779b9); }
