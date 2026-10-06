#!/usr/bin/env node
// Zero-dependency static ES-module dev server (replaces Vite in this sandbox, where
// Vite's esbuild/rollup platform deps cannot be fetched). Serves the project root
// with correct MIME types; the browser import map in index.html resolves `three`
// and `three/addons/*` to /node_modules. Keeps the app loadable at all times.
import { createServer } from 'node:http';
import { stat, readFile } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const root = resolve(__dirname, '..');
const port = Number(process.env.PORT || 5173);
const host = process.env.HOST || '0.0.0.0';

const MIME = {
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.cjs': 'text/javascript; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.hdr': 'application/octet-stream',
  '.ktx2': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/plain; charset=utf-8',
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    let pathname = decodeURIComponent(url.pathname);

    if (pathname === '/healthz') {
      res.writeHead(200, { 'content-type': 'text/plain' });
      return res.end('ok');
    }
    if (pathname === '/') pathname = '/index.html';

    // Resolve safely under root (block traversal).
    const filePath = normalize(join(root, pathname));
    if (!filePath.startsWith(root + sep) && filePath !== root) {
      res.writeHead(403); return res.end('forbidden');
    }

    let st;
    try { st = await stat(filePath); } catch { st = null; }
    if (!st || !st.isFile()) {
      // ESM apps have no client routing; miss -> 404 (index.html only for '/').
      res.writeHead(404, { 'content-type': 'text/plain' });
      return res.end('not found: ' + pathname);
    }
    const body = await readFile(filePath);
    res.writeHead(200, {
      'content-type': MIME[extname(filePath).toLowerCase()] || 'application/octet-stream',
      'cache-control': 'no-cache',
      'content-length': body.length,
    });
    res.end(body);
  } catch (e) {
    res.writeHead(500, { 'content-type': 'text/plain' });
    res.end('server error: ' + (e && e.message));
  }
});

server.listen(port, host, () => {
  console.log(`[serve] Skylines dev server at http://localhost:${port}/  (root: ${root})`);
});
