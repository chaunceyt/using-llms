// Facade atlas textures for the merged building geometry.
//
// Each class atlas is a 2048x2048 canvas split into a 32 x 64 grid of
// 64x32 px cells; a cell maps to 3.4 m x 3.2 m in world space. To give one
// material per class several distinct facades, the atlas is divided into four
// vertical "style strips" — one per facade material family (warm brick, pale
// stucco, ribbon glass, dark panel, ... per class):
//
//   col 0         roof swatches (rows 0..3, one neutral tone per style index)
//   cols  1..7    style A  (window shape A: standard)
//   cols  9..15   style B  (shape B: small punched)
//   cols 17..23   style C  (shape C: horizontal ribbon)
//   cols 25..31   style D  (shape D: curtain wall)
//   cols  8/16/24 one-cell neutral padding between strips
//
// Rows 60..63 are the ground-floor band (storefronts / doors / loading bays).
// Between the windows the atlas draws the architectural grid: a structural
// mullion at each cell's right edge, a two-tone string course every 4 storeys
// and shaded pilaster bays every 4 cells on smooth walls; punched windows
// vary in height and are sometimes split into pairs, so large light panels
// never read as blank. The shared emissive atlas uses the SAME strip layout
// and derives its lit areas from the same window shapes (plus a warm
// ground-floor storefront band), so the per-window night glow aligns with the
// albedo glass of every style. Buildings pick a style and offset their UVs
// inside that strip, so no two neighbouring facades share a pattern.

export const ATLAS = { size: 2048, cols: 32, rows: 64, cellW: 64, cellH: 32, groundRow: 60 };
export const TILE = { w: 3.4, h: 3.2 }; // metres per atlas cell
export const STRIP_START = [1, 9, 17, 25]; // first column of each style strip
export const STRIP_W = 7; // cells per style strip
export const N_STYLES = 4;

// Rooftop-detail regions in column 0 (facade walls live in strips starting at
// column 1, so these are never sampled by a facade). Row r occupies v in
// [r/rows, (r+1)/rows]. Rows 4..7: an illuminated sign band (dark fascia +
// bright letters; warm glow in the emissive atlas). Rows 8..9: a red beacon
// for antenna tips.
export const SIGN_UV = { u0: 0, u1: 1 / ATLAS.cols, v0: 4 / ATLAS.rows, v1: 8 / ATLAS.rows };
export const BEACON_UV = { u0: 0, u1: 1 / ATLAS.cols, v0: 8 / ATLAS.rows, v1: 10 / ATLAS.rows };

// Window shape per strip (fractions of one cell). The albedo glass and the
// emissive lit-area both derive from these so night glow matches the glazing.
const SHAPES = [
  { name: 'std',     x: 0.15, w: 0.65, y: 0.22, h: 0.60 },
  { name: 'small',   x: 0.24, w: 0.46, y: 0.30, h: 0.46 },
  { name: 'ribbon',  x: 0.07, w: 0.86, y: 0.38, h: 0.28 },
  { name: 'curtain', x: 0.06, w: 0.88, y: 0.12, h: 0.78 },
];

// Neutral light-grey roof swatches (tinted by per-building vertex colour).
const ROOF_SWATCH = [[206, 206, 208], [184, 182, 180], [154, 152, 152], [124, 122, 124]];

const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v) | 0;
const rgb = (r, g, b) => `rgb(${clamp255(r)},${clamp255(g)},${clamp255(b)})`;
const shade = (c, f) => rgb(c[0] * f, c[1] * f, c[2] * f);
const frameCol = (st) => (st.frame === 'light' ? shade(st.wall, 1.42) : shade(st.wall, 0.42));

// One upper-floor window cell for a given style + shape. Punched windows are
// occasionally split into a PAIR (a mullion between) and heights vary a few
// percent, so the grid reads as built, not stamped.
function windowCell(g, x, y, CW, CH, st, sh, rng) {
  // spandrel / balcony band across the foot of the storey
  if (st.spandrel) {
    g.fillStyle = shade(st.wall, 0.60);
    g.fillRect(x, y + CH - 6, CW, 6);
    g.fillStyle = shade(st.wall, 0.88);
    g.fillRect(x, y + CH - 7, CW, 1);
    g.fillStyle = 'rgba(20,18,16,0.35)';
    for (let rx = x + 4; rx < x + CW - 2; rx += 6) g.fillRect(rx, y + CH - 6, 1, 4);
  }
  const wx = x + Math.round(CW * sh.x), ww = Math.round(CW * sh.w);
  const wy = y + Math.round(CH * sh.y);
  const wh = Math.round(CH * sh.h * (0.90 + 0.18 * rng.next()));
  // split punched windows into a two-pane group sometimes
  const panes = [];
  if ((sh.name === 'std' || sh.name === 'small') && rng.chance(0.30)) {
    const gap = 3, pw = (ww - gap) >> 1;
    panes.push([wx, wy, pw, wh], [wx + pw + gap, wy, pw, wh]);
  } else {
    panes.push([wx, wy, ww, wh]);
  }
  for (const [px, py, pw, ph] of panes) {
    const t = 0.72 + 0.55 * rng.next(); // per-window glass tone
    g.fillStyle = shade(st.glass, t);
    g.fillRect(px, py, pw, ph);
    if (sh.name === 'curtain') {
      // fine mullion grid
      g.fillStyle = 'rgba(18,22,28,0.55)';
      for (let mx = px + 9; mx < px + pw - 2; mx += 10) g.fillRect(mx, py, 1, ph);
      for (let my = py + 8; my < py + ph - 2; my += 9) g.fillRect(px, my, pw, 1);
    } else if (st.mullion === 1 && rng.chance(0.75)) {
      g.fillStyle = frameCol(st);
      g.fillRect(px + (pw >> 1), py, 1, ph);
    } else if (st.mullion === 2) {
      g.fillStyle = frameCol(st);
      g.fillRect(px + ((pw * 0.34) | 0), py, 1, ph);
      g.fillRect(px + ((pw * 0.66) | 0), py, 1, ph);
    } else if (st.mullion === 3 && sh.name === 'ribbon') {
      g.fillStyle = frameCol(st);
      g.fillRect(px, py + ((ph * 0.5) | 0), pw, 1);
    }
    // frame border
    g.fillStyle = frameCol(st);
    g.fillRect(px - 1, py - 1, pw + 2, 2);
    g.fillRect(px - 1, py + ph - 1, pw + 2, 2);
    g.fillRect(px - 1, py, 2, ph);
    g.fillRect(px + pw - 1, py, 2, ph);
    // light sill under punched windows
    if (st.frame === 'light' && sh.name !== 'curtain' && sh.name !== 'ribbon') {
      g.fillStyle = shade(st.wall, 1.5);
      g.fillRect(px - 2, py + ph + 1, pw + 4, 2);
    }
    // faint sheen on one side of the pane
    g.fillStyle = `rgba(255,255,255,${(0.05 + 0.10 * rng.next()).toFixed(3)})`;
    g.fillRect(px + 1, py + 1, Math.max(2, (pw * 0.3) | 0), Math.max(2, ph - 2));
  }
}

// One ground-floor cell (storefront / entry / loading bay). The shopfront
// reads as a real shop: fascia, scalloped awning, mullioned glass with a
// transom line and a door leaf; the residential entry gets a framed door, a
// house-number plate and an occasional canopy.
function groundCell(g, x, y, CW, CH, cls, st, rng) {
  if (st.ground === 'shop') {
    // kick plate at the pavement line
    g.fillStyle = 'rgba(16,16,20,0.5)';
    g.fillRect(x, y + CH - 2, CW, 2);
    // fascia / sign band
    g.fillStyle = rgb(44, 46, 52);
    g.fillRect(x, y + Math.round(CH * 0.05), CW, Math.round(CH * 0.22));
    g.fillStyle = 'rgba(255,255,255,0.16)';
    g.fillRect(x, y + Math.round(CH * 0.05), CW, 1);
    // awning over most storefronts
    if (cls.awnings && cls.awnings.length && rng.chance(0.85)) {
      const ac = cls.awnings[(rng.next() * cls.awnings.length) | 0];
      const ay = y + Math.round(CH * 0.30);
      g.fillStyle = shade(ac, 0.9 + 0.25 * rng.next());
      g.fillRect(x + 2, ay, CW - 4, 4);
      g.fillStyle = 'rgba(20,18,16,0.4)';
      g.fillRect(x + 2, ay + 4, CW - 4, 1);
      // scalloped valance edge
      g.fillStyle = shade(ac, 0.72);
      for (let sx = x + 3; sx < x + CW - 5; sx += 7) g.fillRect(sx, ay + 4, 4, 2);
    }
    // storefront glass
    const wx = x + Math.round(CW * 0.06), ww = Math.round(CW * 0.88);
    const wy = y + Math.round(CH * 0.36), wh = Math.round(CH * 0.60);
    const t = 0.6 + 0.5 * rng.next();
    g.fillStyle = shade(st.glass, t);
    g.fillRect(wx, wy, ww, wh);
    g.fillStyle = 'rgba(24,28,34,0.6)';
    for (let mx = wx + 6; mx < wx + ww - 2; mx += 7) g.fillRect(mx, wy, 1, wh);
    // transom line
    g.fillStyle = 'rgba(24,28,34,0.5)';
    g.fillRect(wx, wy + Math.round(wh * 0.18), ww, 1);
    // a door leaf in most bays
    if (rng.chance(0.55)) {
      const dx = wx + 4 + ((rng.next() * (ww - 14)) | 0);
      g.fillStyle = shade(st.glass, 0.5);
      g.fillRect(dx, wy + 2, 10, wh - 4);
      g.fillStyle = rgb(212, 216, 222);
      g.fillRect(dx + 8, wy + ((wh >> 1) - 2), 1, 4);
    }
    g.fillStyle = 'rgba(255,255,255,0.10)';
    g.fillRect(wx + 1, wy + 1, Math.max(3, (ww * 0.25) | 0), wh - 2);
  } else if (st.ground === 'door') {
    const dw = Math.round(CW * 0.26), dh = Math.round(CH * 0.80);
    const dx = x + Math.round(CW * (0.14 + 0.48 * rng.next()));
    // step
    g.fillStyle = shade(st.wall, 0.68);
    g.fillRect(dx - 3, y + CH - 3, dw + 6, 3);
    // door frame (a light reveal around the leaf)
    g.fillStyle = shade(st.wall, 1.35);
    g.fillRect(dx - 2, y + CH - dh - 2, dw + 4, dh + 2);
    // door
    g.fillStyle = shade([104, 70, 48], 0.85 + 0.3 * rng.next());
    g.fillRect(dx, y + CH - dh, dw, dh);
    g.fillStyle = 'rgba(20,14,10,0.35)';
    g.fillRect(dx, y + CH - dh + ((dh * 0.5) | 0), dw, 1);
    // house-number plate above the door
    g.fillStyle = rgb(226, 224, 216);
    g.fillRect(dx + (dw >> 1) - 2, y + CH - dh - 9, 5, 4);
    // small entry window
    const t = 0.7 + 0.4 * rng.next();
    const exw = x + CW - dw - 12, ey = y + Math.round(CH * 0.28),
          ew = 8, eh = Math.round(CH * 0.46);
    g.fillStyle = shade(st.glass, t);
    g.fillRect(exw, ey, ew, eh);
    g.fillStyle = frameCol(st);
    g.fillRect(exw - 1, ey - 1, ew + 2, 1);
    g.fillRect(exw - 1, ey + eh, ew + 2, 1);
    // occasional canopy over the door
    if (st.awning && cls.awnings && cls.awnings.length && rng.chance(0.65)) {
      const ac = cls.awnings[(rng.next() * cls.awnings.length) | 0];
      const cy = y + Math.round(CH * 0.16);
      g.fillStyle = shade(ac, 0.9 + 0.25 * rng.next());
      g.fillRect(dx - 4, cy, dw + 8, 4);
      g.fillStyle = 'rgba(20,18,16,0.4)';
      g.fillRect(dx - 4, cy + 4, dw + 8, 1);
    }
  } else { // garage / loading bay
    // base course at the pavement line
    g.fillStyle = 'rgba(16,16,20,0.4)';
    g.fillRect(x, y + CH - 2, CW, 2);
    const dw = Math.round(CW * 0.62), dh = Math.round(CH * 0.74);
    const dx = x + Math.round(CW * 0.18);
    g.fillStyle = shade([94, 98, 104], 0.85 + 0.3 * rng.next());
    g.fillRect(dx, y + CH - dh, dw, dh);
    g.fillStyle = 'rgba(16,18,22,0.5)';
    for (let ly = y + CH - dh + 3; ly < y + CH - 2; ly += 4) g.fillRect(dx, ly, dw, 1);
    const t = 0.7 + 0.3 * rng.next();
    g.fillStyle = shade(st.glass, t);
    g.fillRect(x + 2, y + Math.round(CH * 0.2), Math.round(CW * 0.12), Math.round(CH * 0.34));
  }
}

// Per-class albedo atlas: four style strips, each with its own palette and
// surface treatment (brick course, panel seams, metal rib, stucco grain).
export function makeAlbedoAtlas(T, cls, rng) {
  const S = ATLAS.size, CW = ATLAS.cellW, CH = ATLAS.cellH;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const g = cv.getContext('2d');
  g.fillStyle = rgb(128, 126, 124);
  g.fillRect(0, 0, S, S);
  // roof swatches (col 0, rows 0..3)
  for (let k = 0; k < N_STYLES; k++) {
    const y = (ATLAS.rows - 1 - k) * CH;
    const c = ROOF_SWATCH[k];
    g.fillStyle = rgb(c[0], c[1], c[2]);
    g.fillRect(0, y, CW, CH);
    for (let s = 0; s < 26; s++) {
      g.fillStyle = rng.chance(0.5)
        ? `rgba(30,30,32,${(0.05 + 0.08 * rng.next()).toFixed(3)})`
        : `rgba(255,255,255,${(0.04 + 0.07 * rng.next()).toFixed(3)})`;
      g.fillRect((rng.next() * CW) | 0, y + ((rng.next() * CH) | 0), 2, 1);
    }
  }
  cls.styles.forEach((st, s) => {
    const sh = SHAPES[s];
    const c0 = STRIP_START[s];
    for (let c = c0; c < c0 + STRIP_W; c++) {
      for (let r = 0; r < ATLAS.rows; r++) {
        const x = c * CW, y = (ATLAS.rows - 1 - r) * CH;
        const tone = 0.80 + 0.34 * rng.next(); // per-cell wall tone
        g.fillStyle = shade(st.wall, tone);
        g.fillRect(x, y, CW, CH);
        if (st.brick) {
          // mortar: horizontal courses + staggered vertical joints
          g.fillStyle = 'rgba(40,26,20,0.16)';
          for (let ly = y + 3; ly < y + CH - 1; ly += 4) g.fillRect(x, ly, CW, 1);
          g.fillStyle = 'rgba(40,26,20,0.22)';
          const rows8 = Math.ceil(CH / 4);
          for (let ry = 0; ry < rows8; ry++) {
            const off = (ry % 2) * 4;
            for (let lx = x + off; lx < x + CW; lx += 8) g.fillRect(lx, y + ry * 4, 1, 4);
          }
        } else if (st.panel) {
          g.fillStyle = 'rgba(16,18,22,0.18)';
          g.fillRect(x + (CW / 3) | 0, y, 1, CH);
          g.fillRect(x + (2 * CW / 3) | 0, y, 1, CH);
        } else if (st.rib) {
          g.fillStyle = 'rgba(30,24,20,0.14)';
          for (let ly = y + 5; ly < y + CH; ly += 6) g.fillRect(x, ly, CW, 1);
        } else {
          // stucco / concrete grain
          for (let s2 = 0; s2 < 5; s2++) {
            g.fillStyle = rng.chance(0.6)
              ? `rgba(28,24,20,${(0.03 + 0.07 * rng.next()).toFixed(3)})`
              : `rgba(255,250,238,${(0.03 + 0.08 * rng.next()).toFixed(3)})`;
            g.fillRect(x + ((rng.next() * CW) | 0), y + ((rng.next() * CH) | 0), 2 + ((rng.next() * 3) | 0), 1);
          }
        }
        // structural mullion at the cell's right edge: a continuous vertical
        // line every 3.4 m that, with the slab lines, breaks the flat grid
        g.fillStyle = 'rgba(18,16,14,0.20)';
        g.fillRect(x + CW - 1, y, 1, CH);
        // string course every 4 storeys: two-tone banding on smooth walls
        if (r % 4 === 3 && !st.brick && !st.rib) {
          g.fillStyle = shade(st.wall, 0.84);
          g.fillRect(x, y + CH - 5, CW, 3);
          g.fillStyle = shade(st.wall, 1.22);
          g.fillRect(x, y + CH - 8, CW, 1);
        }
        // pilaster: a shaded bay line every 4 cells on smooth walls
        if (!st.brick && !st.panel && !st.rib && (c - c0) % 4 === 0) {
          g.fillStyle = shade(st.wall, 0.85);
          g.fillRect(x, y, 3, CH);
          g.fillStyle = shade(st.wall, 1.15);
          g.fillRect(x + 3, y, 1, CH);
        }
        // broad tonal patches keep large light panels from reading as blank
        if (st.wall[0] + st.wall[1] > 400) {
          g.fillStyle = `rgba(58,50,42,${(0.04 + 0.05 * rng.next()).toFixed(3)})`;
          const pw = 18 + ((rng.next() * 34) | 0);
          g.fillRect(x + ((rng.next() * (CW - pw)) | 0), y + ((rng.next() * (CH - 12)) | 0), pw, 10 + ((rng.next() * 12) | 0));
        }
        // floor slab line at the foot of each storey
        g.fillStyle = 'rgba(22,20,18,0.28)';
        g.fillRect(x, y + CH - 2, CW, 2);
        if (r >= ATLAS.groundRow) groundCell(g, x, y, CW, CH, cls, st, rng);
        else windowCell(g, x, y, CW, CH, st, sh, rng);
      }
    }
  });
  // --- rooftop-detail regions (col 0, below the roof swatches) ---------------
  // sign band: dark fascia with bright letter bars (rows 4..7)
  {
    const yS = (ATLAS.rows - 1 - 7) * CH, hS = 4 * CH;
    g.fillStyle = rgb(24, 26, 32);
    g.fillRect(0, yS, CW, hS);
    g.fillStyle = 'rgba(255,255,255,0.25)';
    g.fillRect(0, yS, CW, 2);
    g.fillRect(0, yS + hS - 2, CW, 2);
    let lx = 4;
    while (lx < CW - 8) {
      const lw = 3 + ((rng.next() * 4) | 0);
      const lh = (hS * (0.42 + 0.34 * rng.next())) | 0;
      const ly = (yS + (hS - lh) / 2) | 0;
      g.fillStyle = rgb(226, 232, 244);
      g.fillRect(lx, ly, lw, lh);
      lx += lw + 3 + ((rng.next() * 3) | 0);
    }
  }
  // beacon: dark panel with a red dot (rows 8..9)
  {
    const yB = (ATLAS.rows - 1 - 9) * CH, hB = 2 * CH;
    g.fillStyle = rgb(40, 36, 36);
    g.fillRect(0, yB, CW, hB);
    g.fillStyle = rgb(122, 42, 40);
    g.fillRect(CW / 2 - 8, yB + hB / 2 - 8, 16, 16);
  }
  const tex = new T.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = T.ClampToEdgeWrapping;
  tex.colorSpace = T.SRGBColorSpace;
  tex.anisotropy = 1; // >1 bands in SwiftShader
  tex.needsUpdate = true;
  return tex;
}

// Shared emissive atlas (same strip layout): a deterministic subset of
// windows lit warm per style. Curtain-wall strips light individual panes so
// the tower glow reads as a grid of rooms, not one flat slab.
export function makeEmissiveAtlas(T, rng) {
  const S = ATLAS.size, CW = ATLAS.cellW, CH = ATLAS.cellH;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const g = cv.getContext('2d');
  g.fillStyle = '#000000';
  g.fillRect(0, 0, S, S);
  for (let s = 0; s < N_STYLES; s++) {
    const sh = SHAPES[s];
    const c0 = STRIP_START[s];
    for (let c = c0; c < c0 + STRIP_W; c++) {
      for (let r = 0; r < ATLAS.rows; r++) {
        const x = c * CW, y = (ATLAS.rows - 1 - r) * CH;
        const ground = r >= ATLAS.groundRow;
        if (ground) {
          // ground-floor glow: a wide warm band across the shop glass line /
          // entry, so streets read as occupied at night
          if (rng.chance(0.62)) {
            const b = 0.55 + 0.45 * rng.next() * rng.next();
            g.fillStyle = `rgb(${clamp255(255 * b)},${clamp255(206 * b)},${clamp255(138 * b)})`;
            g.fillRect(x + Math.round(CW * 0.10), y + Math.round(CH * 0.38),
              Math.round(CW * 0.80), Math.round(CH * 0.54));
          }
          continue;
        }
        const wx = x + Math.round(CW * sh.x), ww = Math.round(CW * sh.w);
        const wy = y + Math.round(CH * sh.y), wh = Math.round(CH * sh.h);
        if (sh.name === 'curtain') {
          const pc = 3, pr = 4, pw = ww / pc, ph = wh / pr;
          for (let i = 0; i < pc; i++) {
            for (let j = 0; j < pr; j++) {
              if (!rng.chance(0.44)) continue;
              const b = 0.5 + 0.5 * rng.next() * rng.next();
              g.fillStyle = `rgb(${clamp255(255 * b)},${clamp255(214 * b)},${clamp255(150 * b)})`;
              g.fillRect(wx + ((i * pw) | 0) + 1, wy + ((j * ph) | 0) + 1, (pw | 0) - 2, (ph | 0) - 2);
            }
          }
        } else if (!rng.chance(0.42)) {
          continue;
        } else {
          const b = 0.5 + 0.5 * rng.next() * rng.next();
          g.fillStyle = `rgb(${clamp255(255 * b)},${clamp255(214 * b)},${clamp255(150 * b)})`;
          g.fillRect(wx + 1, wy + 1, ww - 2, wh - 2);
        }
      }
    }
  }
  // --- rooftop-detail regions (same layout as the albedo atlas) --------------
  // sign band: warm glowing letters (rows 4..7)
  {
    const yS = (ATLAS.rows - 1 - 7) * CH, hS = 4 * CH;
    let lx = 4;
    while (lx < CW - 8) {
      const lw = 3 + ((rng.next() * 4) | 0);
      const lh = (hS * (0.42 + 0.34 * rng.next())) | 0;
      const ly = (yS + (hS - lh) / 2) | 0;
      g.fillStyle = 'rgb(255,224,168)';
      g.fillRect(lx, ly, lw, lh);
      lx += lw + 3 + ((rng.next() * 3) | 0);
    }
  }
  // beacon: red dot (rows 8..9)
  {
    const yB = (ATLAS.rows - 1 - 9) * CH, hB = 2 * CH;
    g.fillStyle = 'rgb(255,46,38)';
    g.fillRect(CW / 2 - 8, yB + hB / 2 - 8, 16, 16);
  }
  const tex = new T.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = T.ClampToEdgeWrapping;
  tex.colorSpace = T.SRGBColorSpace;
  tex.anisotropy = 1;
  tex.needsUpdate = true;
  return tex;
}
