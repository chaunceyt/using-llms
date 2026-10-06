// terrain — deterministic heightfield land + animated water.
//
// The heightmap is generated once (seeded fBm) into the SHARED world.terrain
// buffer, so roads/buildings/props can snap to the ground via getHeight/getNormal.
//
// Landform (Round 4): a broad, gently-rolling PLAIN in the centre (the
// buildable city core, r<150m stays flat and dry), rising through hills to a
// RING of mountains, then a CLIFFED COAST that drops below the waterline.
// The coastline is NOT a circle: the effective land radius is modulated by
// low-frequency angular fBm (sampled on the unit circle, so it is periodic in
// theta), jaggling the shore into natural bays and headlands. A separate map-
// edge mask guarantees the land fades to sea before the heightfield border so
// the square map edge never reads as a straight wall.
//
// Water: a translucent physical sheet (sky IBL gives fresnel reflection at
// grazing angles) over a deep-sea floor plane whose vertex colours darken with
// radius (depth gradient). A per-vertex-alpha shallow-water ring tints the
// near-shore water turquoise, and a thin foam mesh (patchy white band) breaks
// the waterline at the cliff foot and wet-sand edge.
//
// Cliffs: vertex-coloured strata (sin-band sandstone/shale tones warped by
// noise), large-scale pink/grey rock drift, and noise-patch vegetation breaks
// on the lower walls, with a sandy beach band at the cliff foot.
//
// Determinism: only the seeded noise (world.seed) drives the layout. No Math.random.
// Performance: 5 draw calls in the live scene (seafloor + terrain + shallow + water + foam).
//
// Public API: getHeight(x,z), getNormal(x,z), waterLevel. Emits 'terrain:ready'.
import * as THREE from 'three';

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;

// ---- deterministic value noise + fBm, seeded from an integer ----------------
function makeNoise(seed) {
  const hash = (x, y) => {
    let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seed, 974634799)) >>> 0;
    h = (h ^ (h >>> 13)) >>> 0; h = Math.imul(h, 0x5bd1e995) >>> 0; h ^= h >>> 15;
    return (h >>> 0) / 4294967296;
  };
  const noise2 = (x, y) => {
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = x - ix, fy = y - iy;
    const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
  const fbm = (x, y, oct = 5) => {
    let sum = 0, amp = 0.5, f = 1, norm = 0;
    for (let i = 0; i < oct; i++) { sum += amp * noise2(x * f, y * f); norm += amp; amp *= 0.5; f *= 2.03; }
    return sum / norm; // 0..1
  };
  const ridged = (x, y, oct = 4) => {
    let sum = 0, amp = 0.5, f = 1, norm = 0;
    for (let i = 0; i < oct; i++) { const nn = 1 - Math.abs(2 * noise2(x * f, y * f) - 1); sum += amp * nn * nn; norm += amp; amp *= 0.5; f *= 2.01; }
    return sum / norm; // 0..1
  };
  return { fbm, ridged };
}

// Coastline radius modulation, periodic in theta: sample 2D fBm on the unit
// circle (scaled) so the value wraps seamlessly around the shore. 1.0 = the
// old circular coast; the result jags ~17% in and out (bays + headlands) while
// the minimum (0.825) keeps the land fade starting at >=0.85*0.825*half
// (~180m), safely outside the r<150m city basin.
function coastR(th, N) {
  const c = Math.cos(th), s = Math.sin(th);
  const p1 = (N.fbm(c * 1.35 + 4.2, s * 1.35 - 9.1, 3) - 0.5) * 2; // broad bays
  const p2 = (N.fbm(c * 2.70 - 6.6, s * 2.70 + 1.8, 4) - 0.5) * 2; // finer inlets
  return 1 + 0.12 * p1 + 0.055 * p2; // ~0.825 .. 1.175
}

// height at normalised landmass coords (nx,nz in -1..1). `land` is a 0..1 mask.
// Radial landform in EFFECTIVE radius dEff = d / coastR(theta): a flat city
// plain in the core, a mountain ring tracking the (irregular) coast, then the
// seabed. The plain/hill blend keying on dEff means the basin stays centered
// and dry while the shore jags.
function heightAt(nx, nz, land, N) {
  const d = Math.hypot(nx, nz);
  const dEff = d / coastR(Math.atan2(nz, nx), N);
  const base = N.fbm(nx * 1.6 + 11.3, nz * 1.6 - 7.1, 5);
  const detail = N.fbm(nx * 4.2 - 3.7, nz * 4.2 + 5.2, 4);
  const ridge = N.ridged(nx * 2.4 + 2.0, nz * 2.4 - 1.5, 4);

  const rim = smooth(0.50, 0.85, dEff);                 // 0 in the plain -> 1 at the mountain ring
  const plain = 4 + (base - 0.5) * 8;                   // ~0..8m gently rolling city plain
  const mtn = (base * 0.70 + detail * 0.30) * 95 + ridge * 42; // 0..~137m mountain field
  const relief = plain * (1 - rim) + mtn * rim;         // flat core blended into the rim

  const landH = relief * land;                          // surface on land (plain sits above water)
  const seabed = -33;                                   // open ocean floor
  return landH * land + seabed * (1 - land);
}

export default class Terrain {
  name = 'terrain';
  constructor() {
    this._root = null; this._mesh = null; this._water = null; this._shallow = null;
    this._foam = null; this._floor = null;
    this._waterMat = null; this._shallowMat = null; this._foamMat = null; this._floorMat = null;
    this._liveScene = null; this._showExtras = [];
    this._res = 0; this._half = 0; this._cell = 0; this._noise = null;
  }

  async init(world, ctx) {
    const T = ctx.three;
    this.ctx = ctx; this.world = world;
    const tn = world.terrain;
    this._res = tn.res; this._half = tn.size / 2; this._cell = tn.size / tn.res;
    this._noise = makeNoise((world.seed ^ 0x9e3779b9) >>> 0);

    this._generate();
    this._build();
    this._liveScene = ctx.scene;
    ctx.scene.add(this._root);
    ctx.events.emit('terrain:ready', { waterLevel: tn.waterLevel, size: tn.size });
  }

  _generate() {
    const { res, heights, normals } = this.world.terrain;
    const n = res + 1;
    const N = this._noise;
    for (let iz = 0; iz < n; iz++) {
      for (let ix = 0; ix < n; ix++) {
        const x = (ix - res / 2) * this._cell;
        const z = (iz - res / 2) * this._cell;
        const nx = x / this._half, nz = z / this._half;
        const d = Math.hypot(nx, nz);
        // Irregular coast: fade to sea at dEff 0.85..0.98, where dEff scales the
        // radius by the angular modulation. The second term forces land -> 0 at
        // the actual map border (d 0.90..0.995) so bays that reach the edge
        // drop off underwater instead of cliffing at the square border.
        const dEff = d / coastR(Math.atan2(nz, nx), N);
        const land = (1 - smooth(0.85, 0.98, dEff)) * (1 - smooth(0.90, 0.995, d));
        heights[iz * n + ix] = heightAt(nx, nz, land, N);
      }
    }
    const s = this._cell;
    for (let iz = 0; iz < n; iz++) {
      for (let ix = 0; ix < n; ix++) {
        const i = iz * n + ix;
        const xl = heights[iz * n + Math.max(0, ix - 1)];
        const xr = heights[iz * n + Math.min(res, ix + 1)];
        const zd = heights[Math.max(0, iz - 1) * n + ix];
        const zu = heights[Math.min(res, iz + 1) * n + ix];
        const dx = (xr - xl) / (2 * s), dz = (zu - zd) / (2 * s);
        const inv = 1 / Math.hypot(dx, 1, dz);
        normals[i * 3] = -dx * inv;
        normals[i * 3 + 1] = inv;
        normals[i * 3 + 2] = -dz * inv;
      }
    }
  }

  // Vertex albedo: varied grass (meadow / dry / soil patches) over the plain,
  // a muted urban-park tone toward the city core, a sandy shoreline band
  // (including a beach at the cliff foot), strata-banded rock on the cliffs
  // with pink/grey drift, noise-patch vegetation breaks on the lower walls,
  // and a sand -> deep-blue fade underwater. Weights are normalised.
  _tint(h, slope, x, z, out) {
    const N = this._noise;
    const wl = this.world.terrain.waterLevel;
    const r = Math.hypot(x, z);

    // ---- varied grass (plain + low hills) ---------------------------------
    const meadow = N.fbm(x * 0.045 + 3.3, z * 0.045 - 8.8, 4);  // mid-freq patches
    const dry    = N.fbm(x * 0.030 - 2.1, z * 0.030 + 5.9, 4);  // dry / tan grass
    const soil   = N.fbm(x * 0.060 + 11.7, z * 0.060 - 4.4, 3); // dark soil, low spots
    let gr = 0.14 + 0.13 * meadow;
    let gg = 0.38 + 0.20 * meadow;
    let gb = 0.08 + 0.06 * meadow;
    const wDry = smooth(0.55, 0.78, dry) * 0.75;
    gr = gr * (1 - wDry) + 0.53 * wDry;
    gg = gg * (1 - wDry) + 0.45 * wDry;
    gb = gb * (1 - wDry) + 0.23 * wDry;
    const wSoil = (1 - smooth(1.5, 6, h)) * smooth(0.58, 0.82, soil) * 0.55;
    gr = gr * (1 - wSoil) + 0.30 * wSoil;
    gg = gg * (1 - wSoil) + 0.25 * wSoil;
    gb = gb * (1 - wSoil) + 0.16 * wSoil;
    // Urban park tone near the city core: mute the green (less saturated).
    const wPark = (1 - smooth(40, 130, r)) * 0.6;
    gr = gr * (1 - wPark) + 0.32 * wPark;
    gg = gg * (1 - wPark) + 0.42 * wPark;
    gb = gb * (1 - wPark) + 0.30 * wPark;

    // ---- layer weights -----------------------------------------------------
    const wRock = clamp(smooth(0.28, 0.55, slope) * 0.9 + smooth(46, 66, h), 0, 1);
    const wDirt = clamp(smooth(0, 4, h) * (1 - smooth(10, 22, h)) + smooth(0.25, 0.5, slope) * 0.4, 0, 1);
    const wGrass = 1 - wRock;
    // Warm sand at the shoreline: gentle slopes within ~1.7m of the waterline,
    // plus a beach band wrapping the cliff foot (slope-independent) so the
    // wall meets the sea on sand, not a hard colour break.
    const wSandSlope = (1 - smooth(wl + 0.15, wl + 1.7, h)) * (1 - wRock);
    const wSandFoot = (1 - smooth(wl + 0.15, wl + 2.6, h)) * 0.85;
    const wSand = Math.max(wSandSlope, wSandFoot);

    // ---- rock with horizontal strata + large-scale colour drift -------------
    // Two sin bands (fine strata + broad tone drift), warped by low-freq noise
    // so the bedding dips and folds; a pink<->grey drift keeps the wall from
    // reading as one uniform tan slab.
    const warp  = N.fbm(x * 0.02 + 5.1, z * 0.02 - 2.9, 3);
    const band  = 0.5 + 0.5 * Math.sin(h * 0.50 + warp * 3.2);
    const band2 = 0.5 + 0.5 * Math.sin(h * 0.16 + warp * 1.8 + 2.1);
    const shade = 0.78 + 0.46 * (band * 0.65 + band2 * 0.35);   // 0.78..1.24
    const drift = N.fbm(x * 0.013 - 4.4, z * 0.013 + 6.6, 2);    // 0..1 pink<->grey
    const rock = [
      (0.355 + 0.075 * drift) * shade,
      (0.340 - 0.005 * drift) * shade,
      (0.315 - 0.055 * drift) * shade,
    ];
    const dirt = [0.46, 0.37, 0.24];
    const sand = [0.76, 0.69, 0.51];

    const sum = wGrass + wDirt + wRock + wSand;
    out[0] = (gr * wGrass + dirt[0] * wDirt + rock[0] * wRock + sand[0] * wSand) / sum;
    out[1] = (gg * wGrass + dirt[1] * wDirt + rock[1] * wRock + sand[1] * wSand) / sum;
    out[2] = (gb * wGrass + dirt[2] * wDirt + rock[2] * wRock + sand[2] * wSand) / sum;

    // Vegetation breaks: (a) green patches on the gentler lower mountain
    // slopes, and (b) noise-patch scrub clinging to the cliff walls (slope-
    // independent) so the strata are broken up by green.
    const wVegSlope = smooth(8, 18, h) * (1 - smooth(34, 54, h))
                   * (1 - smooth(0.24, 0.46, slope))
                   * smooth(0.50, 0.78, N.fbm(x * 0.05 + 7.7, z * 0.05 - 3.3, 3)) * 0.85;
    const patch = N.fbm(x * 0.085 + 3.1, z * 0.085 - 6.2, 3);
    const wVegWall = smooth(3, 10, h) * (1 - smooth(24, 44, h))
                   * smooth(0.58, 0.80, patch)
                   * (0.70 + 0.30 * N.fbm(x * 0.30 - 1.7, z * 0.30 + 8.4, 2));
    const wVeg = Math.max(wVegSlope, wVegWall);
    out[0] = out[0] * (1 - wVeg) + 0.15 * wVeg;
    out[1] = out[1] * (1 - wVeg) + 0.38 * wVeg;
    out[2] = out[2] * (1 - wVeg) + 0.11 * wVeg;

    // ---- underwater: sand near the shore -> deep-blue seabed ---------------
    if (h < wl) {
      const depth = clamp((wl - h) / 22, 0, 1);
      out[0] = out[0] * (1 - depth) + 0.06 * depth;
      out[1] = out[1] * (1 - depth) + 0.12 * depth;
      out[2] = out[2] * (1 - depth) + 0.18 * depth;
    }
  }

  // Subtle light detail map from SMOOTH low/mid-freq fBm only. A high-frequency
  // speckle (like the shared 'concrete' surface) aliases into horizontal bands
  // at grazing view angles, so the terrain uses its own banding-free texture.
  _makeDetailTexture() {
    const T = this.ctx.three;
    const S = 256;
    const cv = document.createElement('canvas'); cv.width = cv.height = S;
    const g = cv.getContext('2d');
    const img = g.createImageData(S, S);
    const N = this._noise;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const u = x / S, v = y / S;
        const n = N.fbm(u * 6 + 3.1, v * 6 + 7.7, 4);
        const mid = N.fbm(u * 18 - 1.2, v * 18 + 2.4, 3);
        const shade = 0.74 + 0.26 * (n * 0.7 + mid * 0.3); // 0.74..1.0, light
        const i = (y * S + x) * 4;
        img.data[i] = img.data[i + 1] = img.data[i + 2] = Math.round(255 * shade);
        img.data[i + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    const tex = new T.CanvasTexture(cv);
    tex.wrapS = tex.wrapT = T.RepeatWrapping;
    tex.colorSpace = T.SRGBColorSpace;
    // anisotropy >1 produces horizontal banding in SwiftShader (the headless
    // renderer used for screenshots); keep it at 1.
    tex.anisotropy = 1;
    tex.repeat.set(6, 6);
    tex.needsUpdate = true;
    return tex;
  }

  _build() {
    const T = this.ctx.three;
    const { res, heights, normals, waterLevel } = this.world.terrain;
    const n = res + 1;
    const pos = new Float32Array(n * n * 3);
    const col = new Float32Array(n * n * 3);
    const uv = new Float32Array(n * n * 2);
    const idx = [];
    const c = [0, 0, 0];
    for (let iz = 0; iz < n; iz++) {
      for (let ix = 0; ix < n; ix++) {
        const i = iz * n + ix;
        pos[i * 3] = (ix - res / 2) * this._cell;
        pos[i * 3 + 1] = heights[i];
        pos[i * 3 + 2] = (iz - res / 2) * this._cell;
        uv[i * 2] = ix / res; uv[i * 2 + 1] = iz / res;
        const slope = normals[i * 3 + 1] < 0.985 ? 1 - normals[i * 3 + 1] : 0;
        this._tint(heights[i], slope, pos[i * 3], pos[i * 3 + 2], c);
        col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2];
      }
    }
    for (let iz = 0; iz < res; iz++) {
      for (let ix = 0; ix < res; ix++) {
        const a = iz * n + ix, b = a + 1, d = a + n, e = d + 1;
        idx.push(a, d, b, b, d, e);
      }
    }
    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.BufferAttribute(pos, 3));
    geo.setAttribute('normal', new T.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new T.BufferAttribute(uv, 2));
    geo.setAttribute('color', new T.BufferAttribute(col, 3));
    geo.setIndex(idx);
    geo.computeBoundingSphere();

    // subtle light detail map so vertex colours read as textured ground, not flat
    this._detailTex = this._makeDetailTexture();
    const mat = new T.MeshStandardMaterial({
      map: this._detailTex, roughness: 0.92, metalness: 0,
      vertexColors: true,
    });
    this._mesh = new T.Mesh(geo, mat);
    this._mesh.castShadow = true;
    this._mesh.receiveShadow = true;

    // ---- deep-sea floor with a radial depth gradient ------------------------
    // Only visible OUTSIDE the 512m heightfield (the terrain mesh covers the
    // floor inside it), so its colours continue the terrain's deep-water tint
    // at the map edge and darken with radius: the open sea reads as one body
    // that gets deeper — and hazier — away from the coast.
    const fSeg = 96;
    const fgeo = new T.PlaneGeometry(6000, 6000, fSeg, fSeg);
    fgeo.rotateX(-Math.PI / 2);
    const fCount = (fSeg + 1) * (fSeg + 1);
    const fCol = new Float32Array(fCount * 3);
    const fp = fgeo.attributes.position;
    for (let i = 0; i < fCount; i++) {
      const r = Math.hypot(fp.getX(i), fp.getZ(i));
      // deep teal at the map edge -> navy far out (matches _tint's 0.06/0.12/0.18)
      const g1 = smooth(240, 900, r);
      const g2 = smooth(900, 2000, r);
      fCol[i * 3]     = mix(0.055, 0.020, g1) * (1 - 0.5 * g2);
      fCol[i * 3 + 1] = mix(0.115, 0.065, g1) * (1 - 0.5 * g2);
      fCol[i * 3 + 2] = mix(0.175, 0.115, g1) * (1 - 0.5 * g2);
    }
    fgeo.setAttribute('color', new T.BufferAttribute(fCol, 3));
    this._floorMat = new T.MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 });
    this._floor = new T.Mesh(fgeo, this._floorMat);
    this._floor.position.y = -34;
    this._floor.receiveShadow = true;

    // ---- water: translucent physical sheet; scene IBL gives fresnel sky -----
    // reflection at grazing angles. Opacity is modest so the floor gradient
    // and shallow ring show through as a depth cue.
    const wgeo = new T.PlaneGeometry(6000, 6000, 1, 1);
    const wset = this.ctx.assets.surface('water', this.world.seed);
    this._waterMat = new T.MeshPhysicalMaterial({
      map: wset.map, roughnessMap: wset.roughnessMap,
      color: 0x9ccfe8, roughness: 0.06, metalness: 0.0,
      transparent: true, opacity: 0.68,
      envMapIntensity: 1.25, specularIntensity: 1.4,
      // push the water back in depth so the terrain always wins where they meet;
      // without this the 6000-unit plane z-fights into horizontal bands.
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    wset.map.repeat.set(24, 24);
    this._water = new T.Mesh(wgeo, this._waterMat);
    this._water.rotation.x = -Math.PI / 2;
    this._water.position.y = waterLevel;
    this._water.receiveShadow = true;

    // ---- shallow-water ring -------------------------------------------------
    // A thin turquoise sheet hugging the seabed just under the main water
    // surface. Per-vertex alpha is 1 in the shallow band (seabed near the
    // waterline) and fades to 0 on land and in deep water, giving the
    // shallow->deep tint at the coast. It sits below the reflective main
    // sheet, which blends over it.
    const sPos = new Float32Array(n * n * 3);
    const sCol = new Float32Array(n * n * 4); // RGBA -> per-vertex alpha
    for (let iz = 0; iz < n; iz++) {
      for (let ix = 0; ix < n; ix++) {
        const i = iz * n + ix;
        sPos[i * 3] = (ix - res / 2) * this._cell;
        // hug the seabed, clamped just below the waterline so it never pokes up
        sPos[i * 3 + 1] = Math.min(heights[i] + 0.3, waterLevel - 0.05);
        sPos[i * 3 + 2] = (iz - res / 2) * this._cell;
        const depth = waterLevel - heights[i];            // >0 when submerged
        const shallow = depth > 0 ? 1 - smooth(0.3, 6.0, depth) : 0;
        sCol[i * 4] = 0.30; sCol[i * 4 + 1] = 0.68; sCol[i * 4 + 2] = 0.66;
        sCol[i * 4 + 3] = shallow * 0.70;
      }
    }
    const sgeo = new T.BufferGeometry();
    sgeo.setAttribute('position', new T.BufferAttribute(sPos, 3));
    sgeo.setAttribute('color', new T.BufferAttribute(sCol, 4));
    sgeo.setIndex(idx); // same quad layout as the terrain mesh
    sgeo.computeBoundingSphere();
    this._shallowMat = new T.MeshStandardMaterial({
      vertexColors: true, transparent: true, opacity: 1,
      roughness: 0.3, metalness: 0, depthWrite: false, side: T.DoubleSide,
    });
    this._shallow = new T.Mesh(sgeo, this._shallowMat);
    this._shallow.renderOrder = 1;   // draw before the main water sheet
    this._water.renderOrder = 2;

    // ---- shore foam ----------------------------------------------------------
    // A thin white band broken along the waterline: on the sea side a patchy
    // whitewater line at the cliff foot fading out over ~2.5m of depth, on the
    // land side a faint wet-sand line. Noise modulates the band edge and
    // patchiness so the waterline jags with the coast instead of reading as a
    // hard straight ring. Drawn AFTER the water sheet (renderOrder 3).
    const N = this._noise;
    const fPos = new Float32Array(n * n * 3);
    const fColA = new Float32Array(n * n * 4);
    for (let iz = 0; iz < n; iz++) {
      for (let ix = 0; ix < n; ix++) {
        const i = iz * n + ix;
        const x = (ix - res / 2) * this._cell;
        const z = (iz - res / 2) * this._cell;
        const h = heights[i];
        let a = 0, y = waterLevel;
        if (h < waterLevel) {
          const depth = waterLevel - h;
          const n1 = N.fbm(x * 0.09 + 3.7, z * 0.09 - 2.2, 3);   // patchiness
          const n2 = N.fbm(x * 0.30 - 6.1, z * 0.30 + 9.8, 2);   // fine streaks
          let band = 1 - smooth(0.15, 2.6, depth);
          band *= 0.45 + 0.55 * smooth(0.35, 0.75, n1);
          band *= 0.75 + 0.25 * (n2 * 2 - 0.5);
          a = band;
          a = Math.max(a, (1 - smooth(0.05, 0.7, depth)) * 0.95); // bright line at the wall
          y = waterLevel + 0.06;
        } else {
          const above = h - waterLevel;
          a = (1 - smooth(0.0, 0.55, above)) * 0.38
            * (0.5 + 0.5 * N.fbm(x * 0.11 + 1.2, z * 0.11 + 7.3, 3));
          y = h + 0.05; // wet-sand line on the land side
        }
        fPos[i * 3] = x; fPos[i * 3 + 1] = y; fPos[i * 3 + 2] = z;
        fColA[i * 4] = 0.93; fColA[i * 4 + 1] = 0.97; fColA[i * 4 + 2] = 1.0; fColA[i * 4 + 3] = a;
      }
    }
    const foGeo = new T.BufferGeometry();
    foGeo.setAttribute('position', new T.BufferAttribute(fPos, 3));
    foGeo.setAttribute('color', new T.BufferAttribute(fColA, 4));
    foGeo.setIndex(idx);
    foGeo.computeBoundingSphere();
    this._foamMat = new T.MeshBasicMaterial({
      vertexColors: true, transparent: true, depthWrite: false, side: T.DoubleSide,
    });
    this._foam = new T.Mesh(foGeo, this._foamMat);
    this._foam.renderOrder = 3;

    this._root = new T.Group();
    this._root.add(this._floor, this._mesh, this._shallow, this._water, this._foam);
  }

  update(dt, world) {
    if (this._waterMat && this._waterMat.map) {
      const o = this._waterMat.map;
      o.offset.x += dt * 0.008;
      o.offset.y += dt * 0.005;
    }
  }

  // ---- public height/normal queries (bilinear over world.terrain) ----------
  getHeight(x, z) {
    const { res, heights } = this.world.terrain;
    const n = res + 1;
    const gx = clamp((x + this._half) / this._cell, 0, res - 1e-4);
    const gz = clamp((z + this._half) / this._cell, 0, res - 1e-4);
    const x0 = gx | 0, z0 = gz | 0, fx = gx - x0, fz = gz - z0;
    const i00 = z0 * n + x0, i10 = i00 + 1, i01 = i00 + n, i11 = i01 + 1;
    const a = heights[i00] + (heights[i10] - heights[i00]) * fx;
    const b = heights[i01] + (heights[i11] - heights[i01]) * fx;
    return a + (b - a) * fz;
  }
  getNormal(x, z) {
    const e = this._cell;
    const hx = this.getHeight(x + e, z) - this.getHeight(x - e, z);
    const hz = this.getHeight(x, z + e) - this.getHeight(x, z - e);
    const inv = 1 / Math.hypot(hx, 2 * e, hz);
    return new this.ctx.three.Vector3(-hx * inv, 2 * e * inv, -hz * inv);
  }
  get waterLevel() { return this.world.terrain.waterLevel; }

  showcase(scene, world, ctx) {
    // heightmap/mesh are already built in init; restage them in the fresh scene
    // with a warm sun + sky so the land reads clearly on its own.
    // Remove the harness's default flat staging ground — it would hide the
    // terrain relief and sit on top of the water.
    for (const c of [...scene.children]) {
      if (c.isMesh && !this._root.children.includes(c)) scene.remove(c);
    }
    scene.add(this._root);
    this._liveScene = scene;
    if (!scene.fog) scene.fog = new ctx.three.Fog(0x9db6d4, 300, 2400);

    // Borrow the environment module's sky-derived IBL so the land gets saturated,
    // image-based lighting (and the water gets sky reflection) instead of a
    // flat hemisphere fill.
    const env = ctx.registry && ctx.registry.get('environment');
    if (env && env._ibRT) {
      scene.environment = env._ibRT.texture;
      scene.environmentIntensity = 0.55;
    }

    for (const o of this._showExtras) { o.removeFromParent(); if (o.geometry) o.geometry.dispose(); }
    this._showExtras.length = 0;

    const sun = new ctx.three.DirectionalLight(0xfff6ec, 1.8);
    sun.position.set(260, 360, 180);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera; sc.left = -420; sc.right = 420; sc.top = 420; sc.bottom = -420; sc.near = 20; sc.far = 1400; sc.updateProjectionMatrix();
    sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.9;
    scene.add(sun, sun.target);
    this._showExtras.push(sun, sun.target);
  }

  dispose() {
    if (this._root) this._root.removeFromParent();
    for (const o of this._showExtras) { o.removeFromParent(); if (o.geometry) o.geometry.dispose(); }
    this._showExtras.length = 0;
    if (this._mesh) { this._mesh.geometry.dispose(); this._mesh.material.dispose(); }
    if (this._detailTex) this._detailTex.dispose();
    this._detailTex = null;
    if (this._water) { this._water.geometry.dispose(); this._waterMat.dispose(); }
    if (this._shallow) { this._shallow.geometry.dispose(); this._shallowMat.dispose(); }
    if (this._foam) { this._foam.geometry.dispose(); this._foamMat.dispose(); }
    if (this._floor) { this._floor.geometry.dispose(); this._floorMat.dispose(); }
    this._mesh = this._water = this._shallow = this._foam = this._floor = this._root = null;
    this._liveScene = null;
  }

  stats() {
    return { drawCalls: 5, vertices: (this._res + 1) * (this._res + 1), notes: '1 heightfield + 1 depth-gradient seafloor + 1 shallow-water ring + 1 water sheet + 1 shore foam' };
  }
}
