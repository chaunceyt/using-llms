// probe.mjs — INTERACTION PROOF for the tools module.
//
// shot.mjs can't dispatch pointer events, so this self-contained CDP script
// (same recipe: @sparticuz/chromium + built-in fetch/WebSocket, PUT /json/new,
// --disable-dev-shm-usage) loads the LIVE app, activates tools via
// window.__APP__.registry, dispatches REAL mouse clicks with
// Input.dispatchMouseEvent at screen coords projected from world points on the
// city plain, then reads the zoning grid / road graph back and verifies they
// changed. Exits 0 on success, 1 on failure, 3 on timeout.
//
// Usage: node src/tools/probe.mjs   (env: SHOT_BASE, SEED, PROBE_TIMEOUT)
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const root = join(__dirname, '../..');
const base = process.env.SHOT_BASE || 'http://localhost:5173';
const seed = process.env.SEED || '1337';
const W = 1280, H = 720;
const url = `${base}/?seed=${seed}`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- resolve a usable browser (mirrors tools/shot.mjs) --------------------------
async function resolveChrome() {
  if (process.env.CHROME_PATH) return { exec: process.env.CHROME_PATH, args: [], env: {} };
  try {
    const m = await import('@sparticuz/chromium');
    const ch = m.default || m;
    const inflate = m.inflate;
    if (typeof inflate === 'function') {
      const b = 'node_modules/@sparticuz/chromium/bin/';
      await Promise.all([
        inflate(b + 'al2023.tar.br'),
        inflate(b + 'swiftshader.tar.br'),
        inflate(b + 'fonts.tar.br'),
      ]);
    }
    const exec = await ch.executablePath();
    return {
      exec,
      args: ch.args || [],
      env: {
        LD_LIBRARY_PATH: '/tmp/al2023/lib:/tmp:' + (process.env.LD_LIBRARY_PATH || ''),
        FONTCONFIG_PATH: '/tmp/fonts',
      },
    };
  } catch { /* fall through */ }
  return { exec: join(root, '.chrome', 'chrome-linux64', 'chrome'), args: [], env: {} };
}
const resolved = await resolveChrome();

const dbgPort = 9400 + Math.floor(Math.random() * 400);
const baseFlags = [
  `--window-size=${W},${H}`,
  `--remote-debugging-port=${dbgPort}`,
  '--hide-scrollbars', '--force-color-profile=srgb',
  '--disable-dev-shm-usage',
];
const headlessFlags = resolved.args.length ? [] : ['--headless=new', '--no-sandbox', '--disable-gpu', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader'];
const spawnArgs = [...headlessFlags, ...resolved.args, ...baseFlags, 'about:blank']
  .map((a) => (typeof a === 'string' ? a.replace(/'/g, '') : a));
const chrome = spawn(resolved.exec, spawnArgs, { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, ...resolved.env } });
chrome.stderr.on('data', () => {});

const TIMEOUT = (Number(process.env.PROBE_TIMEOUT) || 120) * 1000;
const watchdog = setTimeout(() => {
  console.error('[probe] global timeout — forcing exit');
  try { chrome.kill('SIGKILL'); } catch {}
  process.exit(3);
}, TIMEOUT);

async function httpJson(path, method = 'GET') {
  const r = await fetch(`http://127.0.0.1:${dbgPort}${path}`, { method });
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${method} ${path}`);
  return r.json();
}

let wsUrl = null;
for (let i = 0; i < 60 && !wsUrl; i++) {
  try {
    const t = await httpJson('/json/new?about:blank', 'PUT');
    wsUrl = t.webSocketDebuggerUrl;
  } catch { await sleep(500); }
}
if (!wsUrl) { console.error('[probe] DevTools endpoint did not come up'); process.exit(1); }

const ws = new WebSocket(wsUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error('ws connect failed')); });
let cdpId = 0;
const pending = new Map();
const consoleErrors = [];
ws.onmessage = (ev) => {
  const msg = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString());
  if (msg.id && pending.has(msg.id)) {
    const { res, rej } = pending.get(msg.id); pending.delete(msg.id);
    msg.error ? rej(new Error(msg.error.message)) : res(msg.result);
  } else if (msg.method) {
    if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error')
      consoleErrors.push((msg.params.args || []).map((a) => a.value ?? a.description ?? '').join(' '));
    if (msg.method === 'Runtime.exceptionThrown')
      consoleErrors.push(msg.params.exceptionDetails?.exception?.description || 'exception');
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

// a real click: move, press, release (Chrome synthesizes pointerdown/up from this)
const click = async (x, y) => {
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y, button: 'none' });
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', buttons: 1, clickCount: 1 });
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', buttons: 0, clickCount: 1 });
  await warm(3);
};

const result = { url, seed, pass: false, consoleErrors: [], moduleErrors: [], zone: null, road: null, bulldoze: null };
let failures = 0;
const check = (label, ok, detail) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  — ' + JSON.stringify(detail) : ''}`);
  if (!ok) failures++;
};

try {
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url });

  let ready = false;
  for (let i = 0; i < 120 && !ready; i++) {
    ready = await evaluate('window.__APP_READY__ === true').catch(() => false);
    if (!ready) await sleep(500);
  }
  if (!ready) throw new Error('app did not become ready in time');
  console.log('[probe] app ready');

  await evaluate(`window.__APP__.setCameraPreset('orbit')`);
  await warm(30);

  // capture tools:placed events in-page
  await evaluate(`(() => {
    window.__PLACED__ = [];
    window.__APP__.registry.ctx.events.on('tools:placed', (p) => window.__PLACED__.push(p));
  })()`);

  // ---- BEFORE state + screen coordinates for the click points -------------------
  const before = await evaluate(`(() => {
    const A = window.__APP__;
    const z = A.registry.get('zoning');
    const r = A.registry.get('roads');
    const t = A.registry.get('tools');
    const th = A.registry.get('terrain');
    let res = 0;
    z.forEachZoned((tl) => { if (tl.use === 'residential') res++; });
    const P = (x, z) => t.projectToScreen(x, th.getHeight(x, z) + 0.3, z);
    return {
      resTiles: res,
      tileC: z.getZone({ x: 0, z: 0 }),          // grid centre tile (on a road: unzoned)
      edges: r.graph.edges.length,
      screenZone: P(0, 0),
      screenRoadA: P(-80, -40),
      screenRoadB: P(-40, -80),
      screenBulldoze: P(60, 60),
      screenDragA: P(-20, 90),
      screenDragB: P(12, 96),
    };
  })()`);
  console.log('[probe] before:', JSON.stringify(before));

  // ---- 1) ZONE PAINT: real click on the plain centre ----------------------------
  console.log('[probe] zone paint: setTool(zone-res) + click at screen', before.screenZone);
  await evaluate(`window.__APP__.registry.get('tools').setTool('zone-res')`);
  const elAt = await evaluate(`document.elementFromPoint(${before.screenZone.sx}, ${before.screenZone.sy}).tagName`);
  check('click lands on the WebGL canvas', elAt === 'CANVAS', elAt);
  await click(before.screenZone.sx, before.screenZone.sy);
  const zoneAfter = await evaluate(`(() => {
    const A = window.__APP__;
    const z = A.registry.get('zoning');
    let res = 0;
    z.forEachZoned((tl) => { if (tl.use === 'residential') res++; });
    return { resTiles: res, tileC: z.getZone({ x: 0, z: 0 }), state: A.registry.get('tools').getState() };
  })()`);
  result.zone = { before: before.tileC, after: zoneAfter.tileC, resTiles: [before.resTiles, zoneAfter.resTiles] };
  check('zoned tile at centre now residential', zoneAfter.tileC && zoneAfter.tileC.use === 'residential', zoneAfter.tileC);
  check('residential tile count grew', zoneAfter.resTiles > before.resTiles, [before.resTiles, zoneAfter.resTiles]);

  // ---- 1b) DRAG PAINT: press, move, release paints a run of tiles -----------------
  const dragBefore = await evaluate(`(() => {
    const z = window.__APP__.registry.get('zoning');
    let res = 0; z.forEachZoned((t) => { if (t.use === 'residential') res++; });
    return res;
  })()`);
  console.log('[probe] drag paint: press at', before.screenDragA, '-> release at', before.screenDragB);
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: before.screenDragA.sx, y: before.screenDragA.sy, button: 'none' });
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: before.screenDragA.sx, y: before.screenDragA.sy, button: 'left', buttons: 1, clickCount: 1 });
  for (let i = 1; i <= 4; i++) {
    const mx = before.screenDragA.sx + (before.screenDragB.sx - before.screenDragA.sx) * (i / 4);
    const my = before.screenDragA.sy + (before.screenDragB.sy - before.screenDragA.sy) * (i / 4);
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: mx, y: my, button: 'left', buttons: 1 });
    await warm(2);
  }
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: before.screenDragB.sx, y: before.screenDragB.sy, button: 'left', buttons: 0, clickCount: 1 });
  await warm(3);
  const dragAfter = await evaluate(`(() => {
    const z = window.__APP__.registry.get('zoning');
    let res = 0; z.forEachZoned((t) => { if (t.use === 'residential') res++; });
    return res;
  })()`);
  result.drag = { resTiles: [dragBefore, dragAfter] };
  check('drag painted a run of tiles (>=2)', dragAfter >= dragBefore + 2, [dragBefore, dragAfter]);

  // ---- 2) ROAD DRAW: anchor click + end click ------------------------------------
  console.log('[probe] road: setTool(road) + click A', before.screenRoadA, 'then B', before.screenRoadB);
  await evaluate(`window.__APP__.registry.get('tools').setTool('road')`);
  await click(before.screenRoadA.sx, before.screenRoadA.sy);
  const anchorState = await evaluate(`window.__APP__.registry.get('tools').getState()`);
  check('anchor set after first click', !!anchorState.anchor, anchorState.anchor);
  await click(before.screenRoadB.sx, before.screenRoadB.sy);
  const roadAfter = await evaluate(`(() => {
    const A = window.__APP__;
    return { edges: A.registry.get('roads').graph.edges.length, state: A.registry.get('tools').getState() };
  })()`);
  result.road = { edges: [before.edges, roadAfter.edges], anchorAfter: roadAfter.state.anchor };
  check('road edge count grew by 1', roadAfter.edges === before.edges + 1, [before.edges, roadAfter.edges]);
  check('anchor reset after second click', roadAfter.state.anchor === null, roadAfter.state.anchor);

  // ---- 3) BULLDOZE: click emits tools:placed --------------------------------------
  console.log('[probe] bulldoze: setTool(bulldoze) + click at', before.screenBulldoze);
  await evaluate(`window.__APP__.registry.get('tools').setTool('bulldoze')`);
  const placedBefore = await evaluate(`window.__PLACED__.length`);
  await click(before.screenBulldoze.sx, before.screenBulldoze.sy);
  const placedAfter = await evaluate(`window.__PLACED__`);
  result.bulldoze = { placedBefore, placedAfter: placedAfter.map((p) => p.tool) };
  const bulldozeEvents = placedAfter.slice(placedBefore).filter((p) => p.tool === 'bulldoze');
  check('bulldoze emitted tools:placed', bulldozeEvents.length === 1, bulldozeEvents[0]);

  // ---- summary ---------------------------------------------------------------------
  result.consoleErrors = consoleErrors;
  result.moduleErrors = await evaluate('window.__MODULE_ERRORS__ || []') || [];
  check('no console errors', consoleErrors.length === 0, consoleErrors);
  check('no module errors', result.moduleErrors.length === 0, result.moduleErrors);
  result.pass = failures === 0;

  const out = join(root, 'shots', 'probe-tools.json');
  writeFileSync(out, JSON.stringify(result, null, 2));
  console.log(`[probe] ${result.pass ? 'SUCCESS' : 'FAILURE'} (${failures} failing check(s)) — log: ${out}`);
} catch (e) {
  console.error('[probe] error:', e.message);
  failures++;
} finally {
  clearTimeout(watchdog);
  try { ws.close(); } catch {}
  try { chrome.kill('SIGKILL'); } catch {}
}
process.exit(failures === 0 && result.pass ? 0 : 1);
