// shot-ghost.mjs — stages a tools ghost preview for a screenshot.
// shot.mjs cannot dispatch pointer events, so this tiny CDP script hovers the
// pointer over the city plain (Input.dispatchMouseEvent mouseMoved), activates a
// tool, warms up, and captures a PNG to shots/<name>.png.
//
// Usage: node src/tools/shot-ghost.mjs --name tools-ghost [--tool zone-res|road|bulldoze]
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const root = join(__dirname, '../..');
const base = process.env.SHOT_BASE || 'http://localhost:5173';
const args = process.argv.slice(2);
const get = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 && args[i + 1] ? args[i + 1] : d; };
const name = get('name', 'tools-ghost');
const tool = get('tool', 'zone-res');
const W = 1280, H = 720;
const outDir = join(root, 'shots');
mkdirSync(outDir, { recursive: true });
const url = `${base}/?seed=1337`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function resolveChrome() {
  const m = await import('@sparticuz/chromium');
  const ch = m.default || m;
  const inflate = m.inflate;
  if (typeof inflate === 'function') {
    const b = 'node_modules/@sparticuz/chromium/bin/';
    await Promise.all([inflate(b + 'al2023.tar.br'), inflate(b + 'swiftshader.tar.br'), inflate(b + 'fonts.tar.br')]);
  }
  return { exec: await ch.executablePath(), args: ch.args || [], env: { LD_LIBRARY_PATH: '/tmp/al2023/lib:/tmp', FONTCONFIG_PATH: '/tmp/fonts' } };
}
const resolved = await resolveChrome();
const dbgPort = 9500 + Math.floor(Math.random() * 300);
const spawnArgs = [...resolved.args, `--window-size=${W},${H}`, `--remote-debugging-port=${dbgPort}`, '--hide-scrollbars', '--force-color-profile=srgb', '--disable-dev-shm-usage', 'about:blank']
  .map((a) => (typeof a === 'string' ? a.replace(/'/g, '') : a));
const chrome = spawn(resolved.exec, spawnArgs, { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, ...resolved.env } });
chrome.stderr.on('data', () => {});
const watchdog = setTimeout(() => { console.error('[shot-ghost] timeout'); try { chrome.kill('SIGKILL'); } catch {} process.exit(3); }, 120000);

async function httpJson(path, method) {
  const r = await fetch(`http://127.0.0.1:${dbgPort}${path}`, { method });
  return r.json();
}
let wsUrl = null;
for (let i = 0; i < 60 && !wsUrl; i++) {
  try { const t = await httpJson('/json/new?about:blank', 'PUT'); wsUrl = t.webSocketDebuggerUrl; } catch { await sleep(500); }
}
if (!wsUrl) { console.error('[shot-ghost] no DevTools endpoint'); process.exit(1); }
const ws = new WebSocket(wsUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error('ws')); });
let cdpId = 0;
const pending = new Map();
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    const { res, rej } = pending.get(msg.id); pending.delete(msg.id);
    msg.error ? rej(new Error(msg.error.message)) : res(msg.result);
  }
};
const send = (method, params = {}) => new Promise((res, rej) => {
  const id = ++cdpId; pending.set(id, { res, rej });
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expr) => {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval failed');
  return r.result?.value;
};
const warm = (n) => evaluate(`new Promise(r=>{let i=0;const t=()=>{if(++i>=${n})r(1);else requestAnimationFrame(t)};requestAnimationFrame(t)})`);

try {
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url });
  let ready = false;
  for (let i = 0; i < 120 && !ready; i++) { ready = await evaluate('window.__APP_READY__ === true').catch(() => false); if (!ready) await sleep(500); }
  if (!ready) throw new Error('app not ready');
  await evaluate(`window.__APP__.setCameraPreset('orbit')`);
  await warm(40);

  // activate the tool and hover the pointer at a world point on the plain
  const pt = tool === 'road' ? { x: -20, z: -60 } : { x: 24, z: 8 };
  const sp = await evaluate(`(() => {
    const A = window.__APP__;
    const th = A.registry.get('terrain');
    return A.registry.get('tools').projectToScreen(${pt.x}, th.getHeight(${pt.x}, ${pt.z}) + 0.3, ${pt.z});
  })()`);
  await evaluate(`window.__APP__.registry.get('tools').setTool(${JSON.stringify(tool)})`);
  if (tool === 'road') {
    // stage an anchor so the ribbon between anchor and cursor is visible
    await evaluate(`window.__APP__.registry.get('tools').anchor = { x: -60, z: -20 }`);
  }
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: sp.sx, y: sp.sy, button: 'none' });
  await warm(20);
  const st = await evaluate(`window.__APP__.registry.get('tools').getState()`);
  console.log('[shot-ghost] tools state:', JSON.stringify(st));

  const shot = await send('Page.captureScreenshot', { format: 'png' });
  const png = join(outDir, `${name}.png`);
  writeFileSync(png, Buffer.from(shot.data, 'base64'));
  console.log('[shot-ghost] wrote', png);
} catch (e) {
  console.error('[shot-ghost] error:', e.message);
  process.exitCode = 1;
} finally {
  clearTimeout(watchdog);
  try { ws.close(); } catch {}
  try { chrome.kill('SIGKILL'); } catch {}
}
process.exit(process.exitCode || 0);
