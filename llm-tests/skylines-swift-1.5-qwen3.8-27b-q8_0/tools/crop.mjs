#!/usr/bin/env node
// Minimal PNG crop/resize (no deps): node tools/crop.mjs in.png out.png x y w h [scale]
import { readFileSync, writeFileSync } from 'node:fs';
import { inflateSync, deflateSync } from 'node:zlib';

const [, , inP, outP, x0, y0, w0, h0, sc] = process.argv;
const x = +x0, y = +y0, w = +w0, h = +h0, scale = sc ? +sc : 1;
const buf = readFileSync(inP);

// parse chunks
let off = 8, wdt = 0, hgt = 0, bitDepth = 0, colorType = 0;
const idat = [];
while (off < buf.length) {
  const len = buf.readUInt32BE(off);
  const type = buf.toString('ascii', off + 4, off + 8);
  const data = buf.subarray(off + 8, off + 8 + len);
  if (type === 'IHDR') { wdt = data.readUInt32BE(0); hgt = data.readUInt32BE(4); bitDepth = data[8]; colorType = data[9]; }
  else if (type === 'IDAT') idat.push(data);
  off += 12 + len;
}
if (bitDepth !== 8 || (colorType !== 6 && colorType !== 2)) throw new Error('only 8-bit RGBA/RGB supported');
const ch = colorType === 6 ? 4 : 3;
const raw = inflateSync(Buffer.concat(idat));
const stride = wdt * ch;
const px = Buffer.alloc(wdt * hgt * 4);
const line = Buffer.alloc(stride);
for (let j = 0; j < hgt; j++) {
  const rs = raw.subarray(j * (stride + 1) + 1, j * (stride + 1) + 1 + stride);
  const f = raw[j * (stride + 1)];
  line.set(rs);
  for (let i = 0; i < stride; i++) {
    const a = i >= ch ? line[i - ch] : 0;
    const b = j > 0 ? px[(j - 1) * stride + i] : 0;
    const c = j > 0 && i >= ch ? px[(j - 1) * stride + i - ch] : 0;
    let v = line[i];
    if (f === 1) v = (v + a) & 255;
    else if (f === 2) v = (v + b) & 255;
    else if (f === 3) v = (v + ((a + b) >> 1)) & 255;
    else if (f === 4) {
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const pr = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      v = (v + pr) & 255;
    }
    line[i] = v;
  }
  line.copy(px, j * stride);
}

const ow = Math.round(w * scale), oh = Math.round(h * scale);
const out = Buffer.alloc(ow * oh * 4);
for (let j = 0; j < oh; j++) {
  const sy = Math.min(hgt - 1, y + Math.floor(j / scale));
  for (let i = 0; i < ow; i++) {
    const sx = Math.min(wdt - 1, x + Math.floor(i / scale));
    const si = (sy * wdt + sx) * ch, di = (j * ow + i) * 4;
    out[di] = px[si]; out[di + 1] = px[si + 1]; out[di + 2] = px[si + 2]; out[di + 3] = ch === 4 ? px[si + 3] : 255;
  }
}

// encode PNG
const crcTable = [];
for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; crcTable[n] = c >>> 0; }
const crc32 = (b) => { let c = 0xffffffff; for (const v of b) c = crcTable[(c ^ v) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
const chunk = (type, data) => {
  const t = Buffer.from(type, 'ascii');
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([t, data])));
  return Buffer.concat([len, t, data, crc]);
};
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(ow, 0); ihdr.writeUInt32BE(oh, 4); ihdr[8] = 8; ihdr[9] = 6;
const rawOut = Buffer.alloc(oh * (ow * 4 + 1));
for (let j = 0; j < oh; j++) { rawOut[j * (ow * 4 + 1)] = 0; out.subarray(j * ow * 4, (j + 1) * ow * 4).copy(rawOut, j * (ow * 4 + 1) + 1); }
const png = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk('IHDR', ihdr),
  chunk('IDAT', deflateSync(rawOut)),
  chunk('IEND', Buffer.alloc(0)),
]);
writeFileSync(outP, png);
console.log(`wrote ${outP} (${ow}x${oh})`);
