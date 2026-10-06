// Merged-geometry builder for the building classes.
//
// Each class (residential / commercial / industrial) accumulates into ONE
// BufferGeometry: position + normal + uv + vertex-color + index. One material
// per class renders walls AND roofs — 3 draw calls for the whole city.
//
// Facade variety: each class atlas carries four vertical "style strips"
// (brick / stucco / ribbon / dark-panel, see atlas.js). A building picks one
// style; its side faces map into that strip with per-face offsets, so
// neighbouring facades read as different materials. A per-building vertex
// tint (±~8% per channel) adds a final tonal shift on top.
//
// Rooftops are not flat slabs: every flat tier edge gets a parapet cap with a
// stone coping band, larger roofs get equipment boxes + a mechanical plant
// room and (on big roofs) an octagonal water tank, tall towers carry spires
// and antenna masts with red beacon tips (glowing at night via the shared
// emissive atlas), and commercial towers get illuminated rooftop signage.
//
// Street level is designed, not a box: the ground-floor band is RECESSED
// behind the facade plane (shadowed soffit + dark corner reveals), each
// building has a stone plinth at the pavement, corner pilasters (masonry) or
// dark frame corners (glass) run up every flat tier, a shadow line breaks
// the roofline under the parapet, and canopies project from the facades —
// full-width storefront slabs with support columns (commercial), small entry
// canopies (residential), flat loading slabs (industrial). All detail merges
// into the class geometry — no extra draw calls.
//
// Side-face UVs come from face size in whole 3.4m x 3.2m cells (see TILE), so
// windows land at a fixed real-world size and no facade shares its atlas
// pattern. Ground-floor storeys map into the atlas band at rows 60..63.
// Bases follow the terrain: each corner is lifted to the ground so nothing
// floats on slopes (walls stay vertical).

import { ATLAS, TILE, STRIP_START, STRIP_W, N_STYLES, SIGN_UV, BEACON_UV } from './atlas.js';

export class MergedBuilder {
  constructor() {
    this.pos = []; this.nrm = []; this.uv = []; this.col = []; this.idx = [];
  }
  v(x, y, z, nx, ny, nz, u, v, r, gc, b) {
    this.pos.push(x, y, z);
    this.nrm.push(nx, ny, nz);
    this.uv.push(u, v);
    this.col.push(r, gc, b);
    return this.pos.length / 3 - 1;
  }
  quad(a, b, c, d) {
    const n = this.pos.length / 3 - 4;
    this.idx.push(n, n + 1, n + 2, n, n + 2, n + 3);
  }
  tri(a, b, c) { this.idx.push(a, b, c); }
  get vertexCount() { return this.pos.length / 3; }
  finish(T) {
    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.Float32BufferAttribute(this.pos, 3));
    geo.setAttribute('normal', new T.Float32BufferAttribute(this.nrm, 3));
    geo.setAttribute('uv', new T.Float32BufferAttribute(this.uv, 2));
    geo.setAttribute('color', new T.Float32BufferAttribute(this.col, 3));
    geo.setIndex(this.idx);
    geo.computeBoundingSphere();
    this.pos = this.nrm = this.uv = this.col = this.idx = null;
    return geo;
  }
}

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

// b = { x, z, y0, w, d, h, groundH, tiers:[{w,d,y0,y1,u?}], roof:null|{type,h},
//       style:0..3, roofTint:[r,g,b], recess:set by emitBuilding, _off(px,pz) }

// One closed box (5 visible faces; the base is hidden) following the terrain
// at its footprint corners, sampled at a single atlas point (roof swatch).
function emitBox(out, cx, cz, yBase, w, d, h, off, ru, rv, cr, cg, cb) {
  const hw = w / 2, hd = d / 2;
  const x0 = cx - hw, x1 = cx + hw, z0 = cz - hd, z1 = cz + hd;
  const o00 = off(x0, z0), o10 = off(x1, z0), o11 = off(x1, z1), o01 = off(x0, z1);
  const V = (px, py, pz, nx, ny, nz) => out.v(px, py, pz, nx, ny, nz, ru, rv, cr, cg, cb);
  // +X
  let A = V(x1, yBase + o11, z1, 1, 0, 0), B = V(x1, yBase + o10, z0, 1, 0, 0),
      C = V(x1, yBase + o10 + h, z0, 1, 0, 0), D = V(x1, yBase + o11 + h, z1, 1, 0, 0);
  out.quad(A, B, C, D);
  // -X
  A = V(x0, yBase + o00, z0, -1, 0, 0); B = V(x0, yBase + o01, z1, -1, 0, 0);
  C = V(x0, yBase + o01 + h, z1, -1, 0, 0); D = V(x0, yBase + o00 + h, z0, -1, 0, 0);
  out.quad(A, B, C, D);
  // +Z
  A = V(x0, yBase + o01, z1, 0, 0, 1); B = V(x1, yBase + o11, z1, 0, 0, 1);
  C = V(x1, yBase + o11 + h, z1, 0, 0, 1); D = V(x0, yBase + o01 + h, z1, 0, 0, 1);
  out.quad(A, B, C, D);
  // -Z
  A = V(x1, yBase + o10, z0, 0, 0, -1); B = V(x0, yBase + o00, z0, 0, 0, -1);
  C = V(x0, yBase + o00 + h, z0, 0, 0, -1); D = V(x1, yBase + o10 + h, z0, 0, 0, -1);
  out.quad(A, B, C, D);
  // top
  A = V(x0, yBase + o01 + h, z1, 0, 1, 0); B = V(x1, yBase + o11 + h, z1, 0, 1, 0);
  C = V(x1, yBase + o10 + h, z0, 0, 1, 0); D = V(x0, yBase + o00 + h, z0, 0, 1, 0);
  out.quad(A, B, C, D);
}

// A closed box INCLUDING its base — canopies and slabs need the underside at
// street eye level. Same footprint/terrain logic as emitBox.
function emitBoxFull(out, cx, cz, yBase, w, d, h, off, ru, rv, cr, cg, cb) {
  emitBox(out, cx, cz, yBase, w, d, h, off, ru, rv, cr, cg, cb);
  const hw = w / 2, hd = d / 2;
  const x0 = cx - hw, x1 = cx + hw, z0 = cz - hd, z1 = cz + hd;
  const o00 = off(x0, z0), o10 = off(x1, z0), o11 = off(x1, z1), o01 = off(x0, z1);
  const V = (px, py, pz, nx, ny, nz) => out.v(px, py, pz, nx, ny, nz, ru, rv, cr, cg, cb);
  const A = V(x0, yBase + o00, z0, 0, -1, 0), B = V(x1, yBase + o10, z0, 0, -1, 0),
      C = V(x1, yBase + o11, z1, 0, -1, 0), D = V(x0, yBase + o01, z1, 0, -1, 0);
  out.quad(A, B, C, D);
}

// Street-level canopy over one facade (face 0:+X 1:-X 2:+Z 3:-Z): a slim slab
// projecting from the facade plane — top, four sides AND the underside — with
// two slim support columns when the span is wide enough. `span` runs along
// the facade, `dep` the projection, `thick` the slab depth.
function emitCanopy(out, b, face, y, span, dep, thick, rng, ru, rv, ac) {
  const t = b.tiers[0];
  const P = (su, sd) => (face === 0 ? [b.x + t.w / 2 + sd, b.z + su]
    : face === 1 ? [b.x - t.w / 2 - sd, b.z + su]
    : face === 2 ? [b.x + su, b.z + t.d / 2 + sd]
    : [b.x + su, b.z - t.d / 2 - sd]);
  const c0 = P(0, 0), c1 = P(0, dep);
  const cx = (c0[0] + c1[0]) / 2, cz = (c0[1] + c1[1]) / 2;
  emitBoxFull(out, cx, cz, y, face < 2 ? dep : span, face < 2 ? span : dep, thick,
    b._off, ru, rv, ac[0], ac[1], ac[2]);
  if (span > 3.2) {
    const ct = [ac[0] * 0.62, ac[1] * 0.62, ac[2] * 0.62];
    for (const sgn of [-1, 1]) {
      const [px, pz] = P(sgn * span * 0.34, dep - 0.16);
      emitBox(out, px, pz, b.y0, 0.13, 0.13, Math.max(0.4, y - b.y0), b._off, ru, rv, ct[0], ct[1], ct[2]);
    }
  }
}

// A dark stone plinth band around the street-level base: walls proud of the
// facade plus a lit top edge. Grounds the building and breaks the
// wall-to-pavement seam.
function emitPlinth(out, b, t, ru, rv, rc) {
  const r = 0.10, h = b.kind === 'industrial' ? 0.8 : 0.55;
  const hw = t.w / 2, hd = t.d / 2;
  const x0 = b.x - hw, x1 = b.x + hw, z0 = b.z - hd, z1 = b.z + hd;
  const off = b._off;
  const o00 = off(x0, z0), o10 = off(x1, z0), o11 = off(x1, z1), o01 = off(x0, z1);
  const yb = (oy) => t.y0 + oy, yT = (oy) => t.y0 + h + oy;
  const dk = [rc[0] * 0.5, rc[1] * 0.5, rc[2] * 0.52];
  const lt = [rc[0] * 0.85, rc[1] * 0.85, rc[2] * 0.88];
  const V = (px, py, pz, nx, ny, nz, c) => out.v(px, py, pz, nx, ny, nz, ru, rv, c[0], c[1], c[2]);
  // +X wall + top edge
  let A = V(x1 + r, yb(o11), z1, 1, 0, 0, dk), B = V(x1 + r, yb(o10), z0, 1, 0, 0, dk),
      C = V(x1 + r, yT(o10), z0, 1, 0, 0, dk), D = V(x1 + r, yT(o11), z1, 1, 0, 0, dk);
  out.quad(A, B, C, D);
  A = V(x1, yT(o11), z1, 0, 1, 0, lt); B = V(x1 + r, yT(o11), z1, 0, 1, 0, lt);
  C = V(x1 + r, yT(o10), z0, 0, 1, 0, lt); D = V(x1, yT(o10), z0, 0, 1, 0, lt);
  out.quad(A, B, C, D);
  // -X
  A = V(x0 - r, yb(o00), z0, -1, 0, 0, dk); B = V(x0 - r, yb(o01), z1, -1, 0, 0, dk);
  C = V(x0 - r, yT(o01), z1, -1, 0, 0, dk); D = V(x0 - r, yT(o00), z0, -1, 0, 0, dk);
  out.quad(A, B, C, D);
  A = V(x0 - r, yT(o01), z1, 0, 1, 0, lt); B = V(x0, yT(o01), z1, 0, 1, 0, lt);
  C = V(x0, yT(o00), z0, 0, 1, 0, lt); D = V(x0 - r, yT(o00), z0, 0, 1, 0, lt);
  out.quad(A, B, C, D);
  // +Z
  A = V(x0, yb(o01), z1 + r, 0, 0, 1, dk); B = V(x1, yb(o11), z1 + r, 0, 0, 1, dk);
  C = V(x1, yT(o11), z1 + r, 0, 0, 1, dk); D = V(x0, yT(o01), z1 + r, 0, 0, 1, dk);
  out.quad(A, B, C, D);
  A = V(x0, yT(o01), z1 + r, 0, 1, 0, lt); B = V(x1, yT(o11), z1 + r, 0, 1, 0, lt);
  C = V(x1, yT(o11), z1, 0, 1, 0, lt); D = V(x0, yT(o01), z1, 0, 1, 0, lt);
  out.quad(A, B, C, D);
  // -Z
  A = V(x1, yb(o10), z0 - r, 0, 0, -1, dk); B = V(x0, yb(o00), z0 - r, 0, 0, -1, dk);
  C = V(x0, yT(o00), z0 - r, 0, 0, -1, dk); D = V(x1, yT(o10), z0 - r, 0, 0, -1, dk);
  out.quad(A, B, C, D);
  A = V(x1, yT(o10), z0 - r, 0, 1, 0, lt); B = V(x0, yT(o00), z0 - r, 0, 1, 0, lt);
  C = V(x0, yT(o00), z0, 0, 1, 0, lt); D = V(x1, yT(o10), z0, 0, 1, 0, lt);
  out.quad(A, B, C, D);
}

// Vertical corner strips on a flat tier: a proud bay/pilaster line at each
// corner that breaks the flat box silhouette. `light` gives a bright stone
// pilaster on masonry styles; glass/panel/rib towers get a slim dark frame
// corner instead.
function emitCornerStrips(out, b, t, ru, rv, rc, light) {
  const w = light ? 0.36 : 0.22, r = light ? 0.14 : 0.10;
  const hw = t.w / 2, hd = t.d / 2;
  const x0 = b.x - hw, x1 = b.x + hw, z0 = b.z - hd, z1 = b.z + hd;
  const off = b._off;
  const yb = (oy) => t.y0 + oy, yT = (oy) => t.y1 + oy;
  const tk = light
    ? [Math.min(1.35, rc[0] * 1.22), Math.min(1.35, rc[1] * 1.22), Math.min(1.3, rc[2] * 1.18)]
    : [rc[0] * 0.42, rc[1] * 0.45, rc[2] * 0.5];
  const V = (px, py, pz, nx, ny, nz) => out.v(px, py, pz, nx, ny, nz, ru, rv, tk[0], tk[1], tk[2]);
  const corners = [[x1, z1, 1, 1], [x0, z1, -1, 1], [x0, z0, -1, -1], [x1, z0, 1, -1]];
  for (const [cx, cz, sx, sz] of corners) {
    const oy = off(cx, cz);
    // flank parallel to the Z facade, proud of the X facade
    const xlo = sx > 0 ? cx - w : cx, xhi = sx > 0 ? cx : cx + w;
    const czp = cz + sz * r;
    if (sz > 0) out.quad(
      V(xlo, yb(oy), czp, 0, 0, 1), V(xhi, yb(oy), czp, 0, 0, 1),
      V(xhi, yT(oy), czp, 0, 0, 1), V(xlo, yT(oy), czp, 0, 0, 1));
    else out.quad(
      V(xhi, yb(oy), czp, 0, 0, -1), V(xlo, yb(oy), czp, 0, 0, -1),
      V(xlo, yT(oy), czp, 0, 0, -1), V(xhi, yT(oy), czp, 0, 0, -1));
    // flank parallel to the X facade, proud of the Z facade
    const zlo = sz > 0 ? cz - w : cz, zhi = sz > 0 ? cz : cz + w;
    const cxp = cx + sx * r;
    if (sx > 0) out.quad(
      V(cxp, yb(oy), zhi, 1, 0, 0), V(cxp, yb(oy), zlo, 1, 0, 0),
      V(cxp, yT(oy), zlo, 1, 0, 0), V(cxp, yT(oy), zhi, 1, 0, 0));
    else out.quad(
      V(cxp, yb(oy), zlo, -1, 0, 0), V(cxp, yb(oy), zhi, -1, 0, 0),
      V(cxp, yT(oy), zhi, -1, 0, 0), V(cxp, yT(oy), zlo, -1, 0, 0));
  }
}

// A dark shadow line just under the parapet: a crisp roofline break that
// separates the wall from the cap.
function emitEave(out, b, t, ru, rv, rc) {
  const h = 0.26, o = 0.015;
  const hw = t.w / 2, hd = t.d / 2;
  const x0 = b.x - hw, x1 = b.x + hw, z0 = b.z - hd, z1 = b.z + hd;
  const off = b._off;
  const o00 = off(x0, z0), o10 = off(x1, z0), o11 = off(x1, z1), o01 = off(x0, z1);
  const yb = (oy) => t.y1 - h + oy, yT = (oy) => t.y1 + oy;
  const tk = [rc[0] * 0.4, rc[1] * 0.4, rc[2] * 0.44];
  const V = (px, py, pz, nx, ny, nz) => out.v(px, py, pz, nx, ny, nz, ru, rv, tk[0], tk[1], tk[2]);
  let A = V(x1 + o, yb(o11), z1, 1, 0, 0), B = V(x1 + o, yb(o10), z0, 1, 0, 0),
      C = V(x1 + o, yT(o10), z0, 1, 0, 0), D = V(x1 + o, yT(o11), z1, 1, 0, 0);
  out.quad(A, B, C, D);
  A = V(x0 - o, yb(o00), z0, -1, 0, 0); B = V(x0 - o, yb(o01), z1, -1, 0, 0);
  C = V(x0 - o, yT(o01), z1, -1, 0, 0); D = V(x0 - o, yT(o00), z0, -1, 0, 0);
  out.quad(A, B, C, D);
  A = V(x0, yb(o01), z1 + o, 0, 0, 1); B = V(x1, yb(o11), z1 + o, 0, 0, 1);
  C = V(x1, yT(o11), z1 + o, 0, 0, 1); D = V(x0, yT(o01), z1 + o, 0, 0, 1);
  out.quad(A, B, C, D);
  A = V(x1, yb(o10), z0 - o, 0, 0, -1); B = V(x0, yb(o00), z0 - o, 0, 0, -1);
  C = V(x0, yT(o00), z0 - o, 0, 0, -1); D = V(x1, yT(o10), z0 - o, 0, 0, -1);
  out.quad(A, B, C, D);
}
function emitParapet(out, b, t, rng, ru, rv, rc) {
  const hw = t.w / 2, hd = t.d / 2;
  const x0 = b.x - hw, x1 = b.x + hw, z0 = b.z - hd, z1 = b.z + hd;
  const off = b._off;
  const oy00 = off(x0, z0), oy10 = off(x1, z0), oy11 = off(x1, z1), oy01 = off(x0, z1);
  const e = 0.26 + 0.06 * rng.next();
  const ph = 0.55 + 0.55 * rng.next() + Math.min(0.5, t.w * 0.035);
  const base = (oy) => t.y1 + oy;
  const V = (px, py, pz, nx, ny, nz) => out.v(px, py, pz, nx, ny, nz, ru, rv, rc[0], rc[1], rc[2]);
  // +X outer wall + top strip
  let A = V(x1 + e, base(oy11), z1, 1, 0, 0), B = V(x1 + e, base(oy10), z0, 1, 0, 0),
      C = V(x1 + e, base(oy10) + ph, z0, 1, 0, 0), D = V(x1 + e, base(oy11) + ph, z1, 1, 0, 0);
  out.quad(A, B, C, D);
  A = V(x1 - e, base(oy11) + ph, z1, 0, 1, 0); B = V(x1 + e, base(oy11) + ph, z1, 0, 1, 0);
  C = V(x1 + e, base(oy10) + ph, z0, 0, 1, 0); D = V(x1 - e, base(oy10) + ph, z0, 0, 1, 0);
  out.quad(A, B, C, D);
  // -X
  A = V(x0 - e, base(oy00), z0, -1, 0, 0); B = V(x0 - e, base(oy01), z1, -1, 0, 0);
  C = V(x0 - e, base(oy01) + ph, z1, -1, 0, 0); D = V(x0 - e, base(oy00) + ph, z0, -1, 0, 0);
  out.quad(A, B, C, D);
  A = V(x0 - e, base(oy01) + ph, z1, 0, 1, 0); B = V(x0 + e, base(oy01) + ph, z1, 0, 1, 0);
  C = V(x0 + e, base(oy00) + ph, z0, 0, 1, 0); D = V(x0 - e, base(oy00) + ph, z0, 0, 1, 0);
  out.quad(A, B, C, D);
  // +Z
  A = V(x0, base(oy01), z1 + e, 0, 0, 1); B = V(x1, base(oy11), z1 + e, 0, 0, 1);
  C = V(x1, base(oy11) + ph, z1 + e, 0, 0, 1); D = V(x0, base(oy01) + ph, z1 + e, 0, 0, 1);
  out.quad(A, B, C, D);
  A = V(x0, base(oy01) + ph, z1 + e, 0, 1, 0); B = V(x1, base(oy11) + ph, z1 + e, 0, 1, 0);
  C = V(x1, base(oy11) + ph, z1 - e, 0, 1, 0); D = V(x0, base(oy01) + ph, z1 - e, 0, 1, 0);
  out.quad(A, B, C, D);
  // -Z
  A = V(x1, base(oy10), z0 - e, 0, 0, -1); B = V(x0, base(oy00), z0 - e, 0, 0, -1);
  C = V(x0, base(oy00) + ph, z0 - e, 0, 0, -1); D = V(x1, base(oy10) + ph, z0 - e, 0, 0, -1);
  out.quad(A, B, C, D);
  A = V(x1, base(oy10) + ph, z0 - e, 0, 1, 0); B = V(x0, base(oy00) + ph, z0 - e, 0, 1, 0);
  C = V(x0, base(oy00) + ph, z0 + e, 0, 1, 0); D = V(x1, base(oy10) + ph, z0 + e, 0, 1, 0);
  out.quad(A, B, C, D);
  // stone coping: a lighter cap band prouder than the parapet wall — the
  // roofline reads as a designed cap, not the top of a box
  const cp = 0.06 + 0.05 * rng.next(), ch = 0.16 + 0.12 * rng.next();
  const ct = [Math.min(1.4, rc[0] * 1.3), Math.min(1.4, rc[1] * 1.3), Math.min(1.35, rc[2] * 1.26)];
  const Cv = (px, py, pz, nx, ny, nz) => out.v(px, py, pz, nx, ny, nz, ru, rv, ct[0], ct[1], ct[2]);
  const yb2 = (oy) => t.y1 + oy + ph, yT2 = (oy) => t.y1 + oy + ph + ch;
  A = Cv(x1 + e + cp, yb2(oy11), z1, 1, 0, 0); B = Cv(x1 + e + cp, yb2(oy10), z0, 1, 0, 0);
  C = Cv(x1 + e + cp, yT2(oy10), z0, 1, 0, 0); D = Cv(x1 + e + cp, yT2(oy11), z1, 1, 0, 0);
  out.quad(A, B, C, D);
  A = Cv(x1 + e, yT2(oy11), z1, 0, 1, 0); B = Cv(x1 + e + cp, yT2(oy11), z1, 0, 1, 0);
  C = Cv(x1 + e + cp, yT2(oy10), z0, 0, 1, 0); D = Cv(x1 + e, yT2(oy10), z0, 0, 1, 0);
  out.quad(A, B, C, D);
  A = Cv(x0 - e - cp, yb2(oy00), z0, -1, 0, 0); B = Cv(x0 - e - cp, yb2(oy01), z1, -1, 0, 0);
  C = Cv(x0 - e - cp, yT2(oy01), z1, -1, 0, 0); D = Cv(x0 - e - cp, yT2(oy00), z0, -1, 0, 0);
  out.quad(A, B, C, D);
  A = Cv(x0 - e - cp, yT2(oy01), z1, 0, 1, 0); B = Cv(x0 - e, yT2(oy01), z1, 0, 1, 0);
  C = Cv(x0 - e, yT2(oy00), z0, 0, 1, 0); D = Cv(x0 - e - cp, yT2(oy00), z0, 0, 1, 0);
  out.quad(A, B, C, D);
  A = Cv(x0, yb2(oy01), z1 + e + cp, 0, 0, 1); B = Cv(x1, yb2(oy11), z1 + e + cp, 0, 0, 1);
  C = Cv(x1, yT2(oy11), z1 + e + cp, 0, 0, 1); D = Cv(x0, yT2(oy01), z1 + e + cp, 0, 0, 1);
  out.quad(A, B, C, D);
  A = Cv(x0, yT2(oy01), z1 + e + cp, 0, 1, 0); B = Cv(x1, yT2(oy11), z1 + e + cp, 0, 1, 0);
  C = Cv(x1, yT2(oy11), z1 + e, 0, 1, 0); D = Cv(x0, yT2(oy01), z1 + e, 0, 1, 0);
  out.quad(A, B, C, D);
  A = Cv(x1, yb2(oy10), z0 - e - cp, 0, 0, -1); B = Cv(x0, yb2(oy00), z0 - e - cp, 0, 0, -1);
  C = Cv(x0, yT2(oy00), z0 - e - cp, 0, 0, -1); D = Cv(x1, yT2(oy10), z0 - e - cp, 0, 0, -1);
  out.quad(A, B, C, D);
  A = Cv(x1, yT2(oy10), z0 - e - cp, 0, 1, 0); B = Cv(x0, yT2(oy00), z0 - e - cp, 0, 1, 0);
  C = Cv(x0, yT2(oy00), z0 - e, 0, 1, 0); D = Cv(x1, yT2(oy10), z0 - e, 0, 1, 0);
  out.quad(A, B, C, D);
}

// Roof equipment on the topmost structural tier: AC units / boxes plus a
// mechanical plant room on big commercial roofs. Boxes are kept clear of
// penthouse footprints, spires and antenna masts.
function emitRoofBoxes(out, b, t, rng, ru, rv, rc, obstacles, spire, masts) {
  const area = t.w * t.d;
  let n = 0;
  if (area > 80) n = 3 + (rng.chance(0.5) ? 1 : 0);
  else if (area > 50) n = 2 + (rng.chance(0.5) ? 1 : 0);
  else if (area > 25) n = rng.chance(0.75) ? 1 + (rng.chance(0.4) ? 1 : 0) : 0;
  else n = rng.chance(0.35) ? 1 : 0;
  const placed = obstacles.slice();
  for (const m of masts) placed.push(m);
  const tryBox = (bw, bd, bh) => {
    for (let att = 0; att < 8; att++) {
      const cx = b.x + rng.range(-t.w / 2 + 0.7 + bw / 2, t.w / 2 - 0.7 - bw / 2);
      const cz = b.z + rng.range(-t.d / 2 + 0.7 + bd / 2, t.d / 2 - 0.7 - bd / 2);
      if (spire && Math.hypot(cx - spire.x, cz - spire.z) < 1.8) continue;
      let ok = true;
      for (const p of placed) {
        if (Math.abs(p.x - cx) < (p.w + bw) / 2 + 0.3 && Math.abs(p.z - cz) < (p.d + bd) / 2 + 0.3) { ok = false; break; }
      }
      if (!ok) return false;
      placed.push({ x: cx, z: cz, w: bw, d: bd });
      emitBox(out, cx, cz, t.y1, bw, bd, bh, b._off, ru, rv, rc[0] * 0.92, rc[1] * 0.92, rc[2] * 0.92);
      return true;
    }
    return false;
  };
  // mechanical plant room on the biggest commercial roofs
  if (b.kind === 'commercial' && area > 55 && rng.chance(0.55)) {
    tryBox(
      Math.min(t.w * 0.5, rng.range(3.2, 5.0)),
      Math.min(t.d * 0.5, rng.range(2.8, 4.2)),
      rng.range(2.4, 3.6),
    );
  }
  for (let i = 0; i < n; i++) {
    tryBox(
      Math.min(t.w * 0.45, rng.range(1.5, 4.0)),
      Math.min(t.d * 0.45, rng.range(1.5, 3.4)),
      rng.range(0.9, 2.6),
    );
  }
}

// Octagonal water tank — a round silhouette-breaker on large flat roofs.
function emitTank(b, out, t, rng, ru, rv, rc) {
  if (t.w * t.d < 45) return;
  if (!rng.chance(0.4)) return;
  const r = rng.range(0.8, 1.25);
  const th = rng.range(1.8, 2.8);
  const cx = b.x + rng.range(-t.w / 2 + r + 0.9, t.w / 2 - r - 0.9);
  const cz = b.z + rng.range(-t.d / 2 + r + 0.9, t.d / 2 - r - 0.9);
  const cr = rc[0] * 0.85, cg = rc[1] * 0.85, cb = rc[2] * 0.85;
  const B = [], Tp = [];
  for (let k = 0; k < 8; k++) {
    const a = (k / 8) * Math.PI * 2 + Math.PI / 8;
    const px = cx + Math.cos(a) * r, pz = cz + Math.sin(a) * r;
    const o = b._off(px, pz);
    B.push(out.v(px, t.y1 + o, pz, Math.cos(a), 0, Math.sin(a), ru, rv, cr, cg, cb));
    Tp.push(out.v(px, t.y1 + o + th, pz, Math.cos(a), 0, Math.sin(a), ru, rv, cr, cg, cb));
  }
  for (let k = 0; k < 8; k++) {
    const k2 = (k + 1) % 8;
    out.quad(B[k2], B[k], Tp[k], Tp[k2]);
  }
  const cap = out.v(cx, t.y1 + b._off(cx, cz) + th + 0.14, cz, 0, 1, 0, ru, rv, cr, cg, cb);
  for (let k = 0; k < 8; k++) out.tri(cap, Tp[k], Tp[(k + 1) % 8]);
}

// Spires on the tallest towers, antennas on some mid-rise. The tall spire tip
// and antenna beacons sample the emissive atlas' red beacon dot, so they glow
// at night. Returns the spire footprint (so roof boxes stay clear) or null.
function emitSpire(b, out, t, rng, ru, rv, rc) {
  if (b.h < 52 && !(b.kind === 'commercial' && b.h >= 26 && rng.chance(0.35))) return null;
  const tall = b.h >= 52;
  // keep the spire clear of a penthouse unit on the same roof
  let sx = b.x, sz = b.z;
  for (const u of b.tiers) {
    if (u.u) { sx = b.x - (u.x - b.x) * 1.4; sz = b.z - (u.z - b.z) * 1.4; }
  }
  const hw = t.w / 2 - 0.9, hd = t.d / 2 - 0.9;
  sx = clamp(sx, b.x - hw, b.x + hw);
  sz = clamp(sz, b.z - hd, b.z + hd);
  const sh = tall ? rng.range(4.5, 11) : rng.range(2, 5);
  const sw = tall ? 0.4 : 0.3;
  emitBox(out, sx, sz, t.y1, sw, sw, sh, b._off, ru, rv, rc[0] * 0.9, rc[1] * 0.9, rc[2] * 0.9);
  const bu = (BEACON_UV.u0 + BEACON_UV.u1) / 2, bv = (BEACON_UV.v0 + BEACON_UV.v1) / 2;
  if (tall) {
    const th = rng.range(1.2, 2.6);
    emitBox(out, sx, sz, t.y1 + sh, 0.16, 0.16, th, b._off, ru, rv,
      rc[0] * 0.8, rc[1] * 0.8, rc[2] * 0.8);
    emitBox(out, sx, sz, t.y1 + sh + th - 0.34, 0.3, 0.3, 0.3, b._off, bu, bv, rc[0], rc[1], rc[2]);
  }
  return { x: sx, z: sz, w: sw + 1, d: sw + 1 };
}

// Antenna masts on tall towers: thin dark masts with a cross-arm and a red
// beacon cube at the tip. Returns mast footprints (obstacles for roof boxes).
function emitMasts(b, out, t, rng, ru, rv, rc, spire) {
  const obs = [];
  if (b.h < 40) return obs;
  const n = b.h >= 62 ? 2 : rng.chance(0.6) ? 1 : 0;
  const du = (BEACON_UV.u0 + BEACON_UV.u1) / 2, dv = (BEACON_UV.v0 + BEACON_UV.v1) / 2;
  for (let i = 0; i < n; i++) {
    let mx = 0, mz = 0, ok = false;
    for (let att = 0; att < 4; att++) {
      mx = b.x + rng.range(-t.w * 0.32, t.w * 0.32);
      mz = b.z + rng.range(-t.d * 0.32, t.d * 0.32);
      if (!spire || Math.hypot(mx - spire.x, mz - spire.z) > 1.6) { ok = true; break; }
    }
    if (!ok) continue;
    const mh = rng.range(3.5, b.h >= 62 ? 9 : 6.5);
    const mw = 0.24;
    const mrc = [rc[0] * 0.5, rc[1] * 0.53, rc[2] * 0.58];
    emitBox(out, mx, mz, t.y1, mw, mw, mh, b._off, ru, rv, mrc[0], mrc[1], mrc[2]);
    // cross-arm
    const armL = rng.range(0.9, 1.6);
    const alongX = rng.chance(0.5);
    emitBox(out, mx, mz, t.y1 + mh * rng.range(0.55, 0.8),
      alongX ? armL : 0.1, alongX ? 0.1 : armL, 0.08, b._off, ru, rv, mrc[0], mrc[1], mrc[2]);
    // beacon tip
    emitBox(out, mx, mz, t.y1 + mh, 0.3, 0.3, 0.3, b._off, du, dv, rc[0], rc[1], rc[2]);
    obs.push({ x: mx, z: mz, w: 1.2, d: 1.2 });
  }
  return obs;
}

// Illuminated rooftop sign on a commercial tower: a thin box mounted just
// above the parapet on one facade. Front + back faces map into the shared
// sign band (dark fascia + bright letters in the albedo atlas, a warm glow in
// the emissive atlas) so it reads by day and lights up at night.
function emitSign(b, out, t, rng, rc) {
  if (b.kind !== 'commercial' || b.h < 18) return;
  if (!rng.chance(0.65)) return;
  const face = rng.int(0, 3); // 0:+X 1:-X 2:+Z 3:-Z
  const fw = face < 2 ? t.d : t.w; // facade width
  const sw = Math.min(fw * 0.72, rng.range(3.0, fw * 0.8));
  const sh = rng.range(1.4, 2.3);
  const dep = 0.4;
  const yb = t.y1 + 0.4;
  const span = Math.max(0.25, fw / 2 - sw / 2 - 0.5);
  const offc = rng.range(-span, span);
  let cx, cz;
  if (face === 0) { cx = b.x + t.w / 2 + dep / 2; cz = b.z + offc; }
  else if (face === 1) { cx = b.x - t.w / 2 - dep / 2; cz = b.z + offc; }
  else if (face === 2) { cx = b.x + offc; cz = b.z + t.d / 2 + dep / 2; }
  else { cx = b.x + offc; cz = b.z - t.d / 2 - dep / 2; }
  const nx = face === 0 ? 1 : face === 1 ? -1 : 0;
  const nz = face === 2 ? 1 : face === 3 ? -1 : 0;
  const ux = nx === 0 ? 1 : 0, uz = nx === 0 ? 0 : 1;
  const P = (su, sv, sy) => [cx + ux * su * sw + nx * sy * dep, yb + sv * sh, cz + uz * su * sw + nz * sy * dep];
  // winding-safe quad: flips the middle two verts if the normal ends up inward
  const qn = (A, B, C, D, n) => {
    const px = (i) => out.pos[i * 3], py = (i) => out.pos[i * 3 + 1], pz = (i) => out.pos[i * 3 + 2];
    const ax = px(B) - px(A), ay = py(B) - py(A), az = pz(B) - pz(A);
    const cdx = px(C) - px(A), cdy = py(C) - py(A), cdz = pz(C) - pz(A);
    const nx2 = ay * cdz - az * cdy, ny2 = az * cdx - ax * cdz, nz2 = ax * cdy - ay * cdx;
    if (nx2 * n[0] + ny2 * n[1] + nz2 * n[2] < 0) out.quad(A, C, B, D);
    else out.quad(A, B, C, D);
  };
  const du = (u) => SIGN_UV.u0 + u * (SIGN_UV.u1 - SIGN_UV.u0);
  const dv = (v) => SIGN_UV.v0 + v * (SIGN_UV.v1 - SIGN_UV.v0);
  const nv = (p, n, u, v) => out.v(p[0], p[1], p[2], n[0], n[1], n[2], u, v, rc[0] * 0.9, rc[1] * 0.9, rc[2] * 0.9);
  const nOut = [nx, 0, nz], nIn = [-nx, 0, -nz];
  const dU = du(0.02), dV = dv(0.02); // dark fascia margin for the side/top faces
  // front (outward) + back faces: sign band
  {
    const A = nv(P(0.5, 0, 0.5), nOut, du(1), dv(0));
    const B = nv(P(-0.5, 0, 0.5), nOut, du(0), dv(0));
    const C = nv(P(-0.5, 1, 0.5), nOut, du(0), dv(1));
    const D = nv(P(0.5, 1, 0.5), nOut, du(1), dv(1));
    qn(A, B, C, D, nOut);
  }
  {
    const A = nv(P(-0.5, 0, -0.5), nIn, du(0), dv(0));
    const B = nv(P(0.5, 0, -0.5), nIn, du(1), dv(0));
    const C = nv(P(0.5, 1, -0.5), nIn, du(1), dv(1));
    const D = nv(P(-0.5, 1, -0.5), nIn, du(0), dv(1));
    qn(A, B, C, D, nIn);
  }
  // top + two short sides: dark fascia
  {
    const A = nv(P(-0.5, 1, -0.5), [0, 1, 0], dU, dV);
    const B = nv(P(0.5, 1, -0.5), [0, 1, 0], dU, dV);
    const C = nv(P(0.5, 1, 0.5), [0, 1, 0], dU, dV);
    const D = nv(P(-0.5, 1, 0.5), [0, 1, 0], dU, dV);
    qn(A, B, C, D, [0, 1, 0]);
  }
  {
    const su = 0.5, ns = [ux, 0, uz];
    const A = nv(P(su, 0, -0.5), ns, dU, dV);
    const B = nv(P(su, 0, 0.5), ns, dU, dV);
    const C = nv(P(su, 1, 0.5), ns, dU, dV);
    const D = nv(P(su, 1, -0.5), ns, dU, dV);
    qn(A, B, C, D, ns);
  }
  {
    const su = -0.5, ns = [-ux, 0, -uz];
    const A = nv(P(su, 0, -0.5), ns, dU, dV);
    const B = nv(P(su, 0, 0.5), ns, dU, dV);
    const C = nv(P(su, 1, 0.5), ns, dU, dV);
    const D = nv(P(su, 1, -0.5), ns, dU, dV);
    qn(A, B, C, D, ns);
  }
}

// One tier: 4 wall faces (with ground-floor split on the base tier) + flat
// top + parapet (suppressed under a pitched roof).
function emitTier(b, out, t, uv, tint, rc, ru, rv, rng) {
  const hw = t.w / 2, hd = t.d / 2;
  const x0 = b.x - hw, x1 = b.x + hw, z0 = b.z - hd, z1 = b.z + hd;
  const off = b._off;
  const oy00 = off(x0, z0), oy10 = off(x1, z0), oy11 = off(x1, z1), oy01 = off(x0, z1);
  const yt = (oy) => t.y1 + oy;
  const isBase = t.y0 === b.y0;
  const G = b.groundH;
  const cols = ATLAS.cols, rows = ATLAS.rows;
  const vU = (y, ground) => (ground
    ? (ATLAS.groundRow + uv.gOff + (y - b.y0) / TILE.h) / rows
    : (uv.vOff + (y - b.y0) / TILE.h) / rows);
  const tc = [tint[0], tint[1], tint[2]];

  // Ground-floor recess: on the base tier the shopfront/entry band sits a
  // little BEHIND the facade plane (a real recessed ground floor), with a
  // shadowed soffit and corner reveals filling the step.
  const rec = (isBase && (b.recess || 0) > 0 && t.y1 - b.y0 > G + 0.01) ? b.recess : 0;
  const ix0 = x0 + rec, ix1 = x1 - rec, iz0 = z0 + rec, iz1 = z1 - rec;
  const ioy00 = rec ? off(ix0, iz0) : oy00, ioy10 = rec ? off(ix1, iz0) : oy10,
        ioy11 = rec ? off(ix1, iz1) : oy11, ioy01 = rec ? off(ix0, iz1) : oy01;

  // A face = bottom-edge corners A->B (CCW seen from outside) + width along
  // its axis. The quad emitter makes (A,B,C),(A,C,D) with C/D the top of B/A.
  const faces = rec > 0 ? [
    { // +X (recessed)
      fu: uv.fuX,
      a: [ix1, iz1, ioy11], b: [ix1, iz0, ioy10],
      n: [1, 0, 0],
      s: (px, pz) => (iz1 - pz) / TILE.w,
    },
    { // -X (recessed)
      fu: uv.fuX2,
      a: [ix0, iz0, ioy00], b: [ix0, iz1, ioy01],
      n: [-1, 0, 0],
      s: (px, pz) => (pz - iz0) / TILE.w,
    },
    { // +Z (recessed)
      fu: uv.fuZ,
      a: [ix0, iz1, ioy01], b: [ix1, iz1, ioy11],
      n: [0, 0, 1],
      s: (px, pz) => (px - ix0) / TILE.w,
    },
    { // -Z (recessed)
      fu: uv.fuZ2,
      a: [ix1, iz0, ioy10], b: [ix0, iz0, ioy00],
      n: [0, 0, -1],
      s: (px, pz) => (ix1 - px) / TILE.w,
    },
  ] : [
    { // +X
      fu: uv.fuX,
      a: [x1, z1, oy11], b: [x1, z0, oy10],
      n: [1, 0, 0],
      s: (px, pz) => (z1 - pz) / TILE.w,
    },
    { // -X
      fu: uv.fuX2,
      a: [x0, z0, oy00], b: [x0, z1, oy01],
      n: [-1, 0, 0],
      s: (px, pz) => (pz - z0) / TILE.w,
    },
    { // +Z
      fu: uv.fuZ,
      a: [x0, z1, oy01], b: [x1, z1, oy11],
      n: [0, 0, 1],
      s: (px, pz) => (px - x0) / TILE.w,
    },
    { // -Z
      fu: uv.fuZ2,
      a: [x1, z0, oy10], b: [x0, z0, oy00],
      n: [0, 0, -1],
      s: (px, pz) => (x1 - px) / TILE.w,
    },
  ];

  for (const f of faces) {
    const [ax, az, aoy] = f.a;
    const [bx, bz, boy] = f.b;
    const segments = [];
    if (isBase && t.y1 - b.y0 > G + 0.01) {
      const ym = b.y0 + G;
      segments.push({ y0: t.y0, y1: ym, ground: true });
      segments.push({ y0: ym, y1: t.y1, ground: false });
    } else {
      segments.push({ y0: t.y0, y1: t.y1, ground: isBase });
    }
    for (const seg of segments) {
      const A = out.v(ax, seg.y0 + aoy, az, f.n[0], f.n[1], f.n[2],
        (f.fu + f.s(ax, az)) / cols, vU(seg.y0, seg.ground), tc[0], tc[1], tc[2]);
      const B = out.v(bx, seg.y0 + boy, bz, f.n[0], f.n[1], f.n[2],
        (f.fu + f.s(bx, bz)) / cols, vU(seg.y0, seg.ground), tc[0], tc[1], tc[2]);
      const C = out.v(bx, seg.y1 + boy, bz, f.n[0], f.n[1], f.n[2],
        (f.fu + f.s(bx, bz)) / cols, vU(seg.y1, seg.ground), tc[0], tc[1], tc[2]);
      const D = out.v(ax, seg.y1 + aoy, az, f.n[0], f.n[1], f.n[2],
        (f.fu + f.s(ax, az)) / cols, vU(seg.y1, seg.ground), tc[0], tc[1], tc[2]);
      out.quad(A, B, C, D);
    }
  }

  // Recess fill: the shadowed soffit (ceiling of the recess) under the ground
  // floor plus the dark corner reveals between facade and shopfront planes.
  if (rec > 0) {
    const ym = b.y0 + G;
    const dk = [rc[0] * 0.4, rc[1] * 0.4, rc[2] * 0.42];
    const V = (px, py, pz, nx, ny, nz) => out.v(px, py, pz, nx, ny, nz, ru, rv, dk[0], dk[1], dk[2]);
    const Y0 = (oy) => t.y0 + oy, Ym = (oy) => ym + oy;
    // soffit bands (facing down), one per facade
    let A = V(ix1, Ym(oy11), iz1, 0, -1, 0), B = V(ix1, Ym(oy10), iz0, 0, -1, 0),
        C = V(x1, Ym(oy10), iz0, 0, -1, 0), D = V(x1, Ym(oy11), iz1, 0, -1, 0);
    out.quad(A, B, C, D);
    A = V(ix0, Ym(oy01), iz1, 0, -1, 0); B = V(x0, Ym(oy01), iz1, 0, -1, 0);
    C = V(x0, Ym(oy00), iz0, 0, -1, 0); D = V(ix0, Ym(oy00), iz0, 0, -1, 0);
    out.quad(A, B, C, D);
    A = V(ix0, Ym(oy01), iz1, 0, -1, 0); B = V(ix1, Ym(oy11), iz1, 0, -1, 0);
    C = V(ix1, Ym(oy11), z1, 0, -1, 0); D = V(ix0, Ym(oy01), z1, 0, -1, 0);
    out.quad(A, B, C, D);
    A = V(ix1, Ym(oy10), iz0, 0, -1, 0); B = V(ix0, Ym(oy00), iz0, 0, -1, 0);
    C = V(ix0, Ym(oy00), z0, 0, -1, 0); D = V(ix1, Ym(oy10), z0, 0, -1, 0);
    out.quad(A, B, C, D);
    // corner reveals (vertical, at the four corners)
    A = V(ix1, Y0(oy11), iz1, 0, 0, 1); B = V(x1, Y0(oy11), iz1, 0, 0, 1);
    C = V(x1, Ym(oy11), iz1, 0, 0, 1); D = V(ix1, Ym(oy11), iz1, 0, 0, 1);
    out.quad(A, B, C, D);
    A = V(ix1, Y0(oy11), z1, 1, 0, 0); B = V(ix1, Y0(oy11), iz1, 1, 0, 0);
    C = V(ix1, Ym(oy11), iz1, 1, 0, 0); D = V(ix1, Ym(oy11), z1, 1, 0, 0);
    out.quad(A, B, C, D);
    A = V(ix0, Y0(oy00), iz0, 0, 0, -1); B = V(x0, Y0(oy00), iz0, 0, 0, -1);
    C = V(x0, Ym(oy00), iz0, 0, 0, -1); D = V(ix0, Ym(oy00), iz0, 0, 0, -1);
    out.quad(A, B, C, D);
    A = V(ix0, Y0(oy00), z0, -1, 0, 0); B = V(ix0, Y0(oy00), iz0, -1, 0, 0);
    C = V(ix0, Ym(oy00), iz0, -1, 0, 0); D = V(ix0, Ym(oy00), z0, -1, 0, 0);
    out.quad(A, B, C, D);
    A = V(x0, Y0(oy01), iz1, 0, 0, 1); B = V(ix0, Y0(oy01), iz1, 0, 0, 1);
    C = V(ix0, Ym(oy01), iz1, 0, 0, 1); D = V(x0, Ym(oy01), iz1, 0, 0, 1);
    out.quad(A, B, C, D);
    A = V(ix0, Y0(oy01), iz1, -1, 0, 0); B = V(ix0, Y0(oy01), z1, -1, 0, 0);
    C = V(ix0, Ym(oy01), z1, -1, 0, 0); D = V(ix0, Ym(oy01), iz1, -1, 0, 0);
    out.quad(A, B, C, D);
    A = V(x1, Y0(oy10), iz0, 0, 0, -1); B = V(ix1, Y0(oy10), iz0, 0, 0, -1);
    C = V(ix1, Ym(oy10), iz0, 0, 0, -1); D = V(x1, Ym(oy10), iz0, 0, 0, -1);
    out.quad(A, B, C, D);
    A = V(ix1, Y0(oy10), iz0, 1, 0, 0); B = V(ix1, Y0(oy10), z0, 1, 0, 0);
    C = V(ix1, Ym(oy10), z0, 1, 0, 0); D = V(ix1, Ym(oy10), iz0, 1, 0, 0);
    out.quad(A, B, C, D);
  }

  // top face (flat roof / setback ledge) + parapet cap
  if (!t._pitched) {
    const A = out.v(x0, yt(oy01), z1, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    const B = out.v(x1, yt(oy11), z1, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    const C = out.v(x1, yt(oy10), z0, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    const D = out.v(x0, yt(oy00), z0, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    out.quad(A, B, C, D);
    emitParapet(out, b, t, rng, ru, rv, rc);
  }
}

function emitRoof(b, out, t, uv, tint, rc, ru, rv) {
  const hw = t.w / 2, hd = t.d / 2;
  const x0 = b.x - hw, x1 = b.x + hw, z0 = b.z - hd, z1 = b.z + hd;
  const off = b._off;
  const oy00 = off(x0, z0), oy10 = off(x1, z0), oy11 = off(x1, z1), oy01 = off(x0, z1);
  const yt = (oy) => t.y1 + oy;
  // Apex / ridge must clear the HIGHEST corner: on a slope a centre-based apex
  // can sink below a raised corner and invert the roof faces.
  const topY = Math.max(t.y1 + oy00, t.y1 + oy10, t.y1 + oy11, t.y1 + oy01);
  const uvC = (y) => (uv.vOff + (y - b.y0) / TILE.h) / ATLAS.rows;
  if (b.roof.type === 'pyramid') {
    const AP = out.v(b.x, topY + b.roof.h, b.z, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    out.tri(out.v(x0, yt(oy01), z1, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]),
      out.v(x1, yt(oy11), z1, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]), AP);
    out.tri(out.v(x1, yt(oy10), z0, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]),
      out.v(x0, yt(oy00), z0, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]), AP);
    out.tri(out.v(x1, yt(oy11), z1, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]),
      out.v(x1, yt(oy10), z0, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]), AP);
    out.tri(out.v(x0, yt(oy00), z0, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]),
      out.v(x0, yt(oy01), z1, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]), AP);
  } else { // gable ridge along x
    const ridgeY = topY + b.roof.h;
    const R0x = x0 + 0.6, R1x = x1 - 0.6;
    const a = out.v(x0, yt(oy01), z1, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    const bq = out.v(x1, yt(oy11), z1, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    const c = out.v(R1x, ridgeY, b.z, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    const d = out.v(R0x, ridgeY, b.z, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    out.quad(a, bq, c, d);
    const a2 = out.v(x1, yt(oy10), z0, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    const b2 = out.v(x0, yt(oy00), z0, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    const c2 = out.v(R0x, ridgeY, b.z, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    const d2 = out.v(R1x, ridgeY, b.z, 0, 1, 0, ru, rv, rc[0], rc[1], rc[2]);
    out.quad(a2, b2, c2, d2);
    // gable end walls (wall colour, wall UVs)
    const ua = (uv.fuZ + 0.04) / ATLAS.cols;
    const tc = [tint[0], tint[1], tint[2]];
    out.tri(out.v(x0, yt(oy00), z0, -1, 0, 0, ua, uvC(yt(oy00)), tc[0], tc[1], tc[2]),
      out.v(x0, yt(oy01), z1, -1, 0, 0, ua, uvC(yt(oy01)), tc[0], tc[1], tc[2]),
      out.v(R0x, ridgeY, b.z, -1, 0, 0, ua, uvC(ridgeY), tc[0], tc[1], tc[2]));
    out.tri(out.v(x1, yt(oy11), z1, 1, 0, 0, ua, uvC(yt(oy11)), tc[0], tc[1], tc[2]),
      out.v(x1, yt(oy10), z0, 1, 0, 0, ua, uvC(yt(oy10)), tc[0], tc[1], tc[2]),
      out.v(R1x, ridgeY, b.z, 1, 0, 0, ua, uvC(ridgeY), tc[0], tc[1], tc[2]));
  }
}

// Append one building (with tiers / roof / units / rooftop detail) to the
// class builder.
export function emitBuilding(b, out, cls, rng) {
  const style = Math.min(N_STYLES - 1, (b.style || 0) | 0);
  const st = cls.styles[style];
  const rc = b.roofTint || st.roof;
  const [ru, rv] = [0.5 / ATLAS.cols, (style + 0.5) / ATLAS.rows]; // roof swatch
  const maxCellsX = Math.ceil(b.d / TILE.w) + 0.5;
  const maxCellsZ = Math.ceil(b.w / TILE.w) + 0.5;
  const r0 = Math.max(0.1, ATLAS.rows - b.h / TILE.h - 1);
  const strip = STRIP_START[style];
  const fu = (mc) => strip + rng.next() * Math.max(0.1, STRIP_W - mc);
  const uv = {
    vOff: rng.range(0, r0),
    gOff: rng.range(0, 2.4),
    fuX: fu(maxCellsX),
    fuX2: fu(maxCellsX),
    fuZ: fu(maxCellsZ),
    fuZ2: fu(maxCellsZ),
  };
  // per-building tonal tint (±~8% per channel) on top of the style palette
  const tint = [
    0.88 + 0.18 * rng.next(),
    0.88 + 0.18 * rng.next(),
    0.88 + 0.18 * rng.next(),
  ];
  const top = b.tiers[b.tiers.length - 1];
  if (b.roof) top._pitched = true;
  // street-level recess depth for the ground-floor band (shopfronts / entries)
  b.recess = st.ground === 'shop' ? 0.45 : st.ground === 'door' ? 0.30 : 0;
  for (const t of b.tiers) emitTier(b, out, t, uv, tint, rc, ru, rv, rng);

  // ---- street-level + roofline detail (merged: no extra draw calls) --------
  const t0 = b.tiers[0];
  emitPlinth(out, b, t0, ru, rv, rc);
  // corner pilasters on masonry styles, slim dark frame corners on glass
  const lightCorners = !st.panel && !st.rib;
  for (const t of b.tiers) if (!t._pitched && !t.u) emitCornerStrips(out, b, t, ru, rv, rc, lightCorners);
  // roofline shadow line under the parapet of the topmost structural tier
  let tTop = top;
  while (tTop.u && b.tiers.length > 1) tTop = b.tiers[b.tiers.length - 2];
  if (!tTop._pitched) emitEave(out, b, tTop, ru, rv, rc);
  // canopies: projecting storefront slabs (commercial), entry canopies
  // (residential), a flat loading slab (industrial)
  {
    const aw = cls.awnings && cls.awnings.length
      ? cls.awnings[rng.int(0, cls.awnings.length - 1)] : [120, 122, 128];
    const ac = [aw[0] / 165, aw[1] / 165, aw[2] / 165];
    const ym = b.y0 + b.groundH;
    if (st.ground === 'shop') {
      for (let f = 0; f < 4; f++) {
        if (!rng.chance(0.8)) continue;
        const fw = f < 2 ? t0.d : t0.w;
        emitCanopy(out, b, f, ym - 0.02, Math.max(2.0, fw - 1.2),
          rng.range(0.9, 1.35), 0.24, rng, ru, rv, ac);
      }
    } else if (st.ground === 'door') {
      for (let f = 0; f < 4; f++) {
        if (!rng.chance(0.7)) continue;
        emitCanopy(out, b, f, b.y0 + 2.45, 2.0, 0.75, 0.16, rng, ru, rv, ac);
      }
    } else if (b.kind === 'industrial' && rng.chance(0.5)) {
      const fw = rng.chance(0.5) ? t0.d : t0.w;
      emitCanopy(out, b, rng.int(0, 3), b.y0 + 2.9, Math.min(Math.max(2.5, fw - 1.0), 4.5),
        1.5, 0.22, rng, ru, rv, [0.55, 0.56, 0.58]);
    }
  }

  if (b.roof) {
    emitRoof(b, out, top, uv, tint, rc, ru, rv);
  } else {
    // rooftop detail on the topmost structural tier (penthouses come last)
    let t = top;
    while (t.u && b.tiers.length > 1) t = b.tiers[b.tiers.length - 2];
    const obstacles = b.tiers.filter((u) => u.u);
    const spire = emitSpire(b, out, t, rng, ru, rv, rc);
    const masts = emitMasts(b, out, t, rng, ru, rv, rc, spire);
    emitRoofBoxes(out, b, t, rng, ru, rv, rc, obstacles, spire, masts);
    emitTank(b, out, t, rng, ru, rv, rc);
    emitSign(b, out, t, rng, rc);
  }
}
