#!/usr/bin/env node
// Tiny PNG pixel probe (no deps): decodes 8-bit RGB/RGBA non-interlaced PNGs
// using node:zlib and prints sampled pixel rows for visual analysis.
// Usage: node tools/pngprobe.mjs <file.png> [rows...]   (rows = fractions 0..1 of height)
import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';

const file = process.argv[2];
if (!file) { console.error('usage: pngprobe.mjs <file.png> [rowFrac ...]'); process.exit(1); }
const buf = readFileSync(file);
if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('not a png');
let off = 8, w = 0, h = 0, bitDepth = 0, colorType = 0;
const idat = [];
while (off < buf.length) {
  const len = buf.readUInt32BE(off);
  const type = buf.toString('ascii', off + 4, off + 8);
  const data = buf.subarray(off + 8, off + 8 + len);
  if (type === 'IHDR') { w = data.readUInt32BE(0); h = data.readUInt32BE(4); bitDepth = data[8]; colorType = data[9]; }
  else if (type === 'IDAT') idat.push(data);
  off += 12 + len;
}
if (bitDepth !== 8 || colorType !== 2 && colorType !== 6) throw new Error(`unsupported png (depth ${bitDepth}, type ${colorType})`);
const ch = colorType === 6 ? 4 : 3;
const raw = inflateSync(Buffer.concat(idat));
const stride = w * ch;
const px = Buffer.alloc(h * stride);
let rp = 0;
for (let y = 0; y < h; y++) {
  const ft = raw[rp++];
  for (let x = 0; x < stride; x++) {
    const cur = raw[rp++];
    const a = x >= ch ? px[y * stride + x - ch] : 0;
    const b = y > 0 ? px[(y - 1) * stride + x] : 0;
    const c = (x >= ch && y > 0) ? px[(y - 1) * stride + x - ch] : 0;
    let v;
    switch (ft) {
      case 0: v = cur; break;
      case 1: v = cur + a; break;
      case 2: v = cur + b; break;
      case 3: v = cur + ((a + b) >> 1); break;
      case 4: { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v = cur + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c); break; }
      default: throw new Error('bad filter ' + ft);
    }
    px[y * stride + x] = v & 0xff;
  }
}
const get = (fx, fy) => {
  const x = Math.min(w - 1, Math.max(0, Math.round(fx * (w - 1))));
  const y = Math.min(h - 1, Math.max(0, Math.round(fy * (h - 1))));
  const i = y * stride + x * ch;
  return [px[i], px[i + 1], px[i + 2]];
};
const rows = (process.argv.slice(3).map(Number).length ? process.argv.slice(3).map(Number)
  : [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
console.log(`${file} ${w}x${h}`);
for (const fy of rows) {
  const samples = [0.05, 0.25, 0.5, 0.75, 0.95].map((fx) => get(fx, fy).join(','));
  console.log(`y=${fy.toFixed(2)}  ${samples.join('  ')}`);
}
