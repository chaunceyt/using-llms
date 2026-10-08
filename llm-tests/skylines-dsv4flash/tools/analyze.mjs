// tools/analyze.mjs — objective screenshot metrics. The inference gateway has no
// vision, so "looking at a PNG" is done programmatically: these stats catch
// all-black/empty frames and flat placeholder renders, and give a rough quality signal.
//
//   node tools/analyze.mjs docs/shots/<dir>/<name>.png [more.png...]
import fs from 'fs';
import zlib from 'zlib';

export function decodePng(b) {
  let o = 8, W = 0, H = 0, bpp = 4, idat = [];
  while (o < b.length) {
    const len = b.readUInt32BE(o);
    const t = b.toString('ascii', o + 4, o + 8);
    if (t === 'IHDR') { W = b.readUInt32BE(o + 8); H = b.readUInt32BE(o + 12); }
    if (t === 'IDAT') idat.push(b.slice(o + 8, o + 8 + len));
    o += 12 + len;
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = W * bpp, out = Buffer.alloc(raw.length);
  let prev = Buffer.alloc(stride);
  for (let y = 0; y < H; y++) {
    const f = raw[y * (stride + 1)];
    const line = raw.slice(y * (stride + 1) + 1, y * (stride + 1) + 1 + stride);
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? out[y * stride + x - bpp] : 0;
      const bb = y > 0 ? prev[x] : 0;
      const c = y > 0 && x >= bpp ? prev[x - bpp] : 0;
      let v = line[x];
      if (f === 1) v = (v + a) & 255;
      else if (f === 2) v = (v + bb) & 255;
      else if (f === 3) v = (v + ((a + bb) >> 1)) & 255;
      else if (f === 4) { const p = a + bb - c, pa = Math.abs(p - a), pb = Math.abs(p - bb), pc = Math.abs(p - c); v = (v + (pa <= pb && pa <= pc ? a : pb <= pc ? bb : c)) & 255; }
      out[y * stride + x] = v;
    }
    prev = out.slice(y * stride, y * stride + stride);
  }
  return { W, H, bpp, out };
}

export function analyzePng(buffer) {
  const { W, H, bpp, out } = decodePng(buffer);
  const N = W * H;
  // sample every pixel for luma stats, subsample for the rest
  let sumL = 0, sumL2 = 0, dark = 0, black = 0, saturated = 0;
  const colors = new Set();
  const step = Math.max(1, Math.floor(N / 200000));
  // edge energy on luma (Sobel) — subsampled
  let edgeE = 0, edges = 0;
  const luma = Buffer.alloc(N);
  for (let i = 0; i < N; i++) {
    const r = out[i * bpp], g = out[i * bpp + 1], bl = out[i * bpp + 2];
    const L = 0.299 * r + 0.587 * g + 0.114 * bl;
    luma[i] = L; sumL += L; sumL2 += L * L;
    if (L < 18) dark++; else if (L < 3) black++;
    const mx = Math.max(r, g, bl), mn = Math.min(r, g, bl);
    if (mx - mn > 60) saturated++;
    if (i % step === 0) colors.add(((r >> 4) << 8) | ((g >> 4) << 4) | (bl >> 4));
  }
  // sobel on sampled grid
  const gstep = Math.max(1, Math.floor(Math.sqrt(N / 40000))); // ~40k samples
  for (let y = gstep; y < H - gstep; y += gstep)
    for (let x = gstep; x < W - gstep; x += gstep) {
      const i = y * W + x;
      const gx = Math.abs(luma[i - 1] - luma[i + 1]) + Math.abs(luma[i - W - 1] - luma[i - W + 1]) + Math.abs(luma[i + W - 1] - luma[i + W + 1]);
      const gy = Math.abs(luma[i - W] - luma[i + W]) + Math.abs(luma[i - W - 1] - luma[i + W - 1]) + Math.abs(luma[i - W + 1] - luma[i + W + 1]);
      edgeE += gx + gy; if (gx + gy > 60) edges++;
    }
  const meanL = sumL / N;
  const stdL = Math.sqrt(sumL2 / N - meanL * meanL);
  const samples = Math.max(1, Math.floor((H / gstep) * (W / gstep)));
  return {
    W, H,
    luminance: { mean: +meanL.toFixed(1), std: +stdL.toFixed(1) },
    darkness: +(dark / N).toFixed(3),
    blackFraction: +(black / N).toFixed(4),
    saturationFraction: +(saturated / N).toFixed(3),
    distinctColors: colors.size,
    edgeEnergyPerSample: +(edgeE / samples).toFixed(2),
    edgePixels: +((edges / samples) * 100).toFixed(1), // % of samples with a strong edge
  };
}

export function verdict(a) {
  const probs = [];
  if (a.blackFraction > 0.98) probs.push('BLACK/EMPTY FRAME');
  if (a.distinctColors < 6 && a.luminance.std < 3) probs.push('FLAT/UNIFORM');
  if (a.edgePixels < 1.5 && a.luminance.std < 8) probs.push('NO STRUCTURE/EDGES');
  const ok = probs.length === 0;
  return { ok, problems: probs };
}

if (import.meta.url === new URL(process.argv[1], 'file:').href) {
  for (const p of process.argv.slice(2)) {
    try {
      const a = analyzePng(fs.readFileSync(p));
      const v = verdict(a);
      console.log(`\n${p}`);
      console.log(`  ${a.W}x${a.H}  meanL=${a.luminance.mean} std=${a.luminance.std} dark=${a.darkness} black=${a.blackFraction}`);
      console.log(`  colors=${a.distinctColors} sat=${a.saturationFraction} edgeEn=${a.edgeEnergyPerSample} edgePx=${a.edgePixels}`);
      console.log(`  ${v.ok ? 'OK' : 'FAIL'}: ${v.problems.join(', ') || 'looks like a real render'}`);
    } catch (e) { console.log(`\n${p}: ERROR ${e.message}`); }
  }
}
