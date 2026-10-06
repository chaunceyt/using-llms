#!/usr/bin/env node
// Headless screenshot + JSON log tool (the verification loop).
// Drives Chromium over the Chrome DevTools Protocol using Node's built-in fetch +
// WebSocket (NO puppeteer package, so no flaky browser-download dependency tree).
// Loads the app, waits for ready, applies module/camera/time/weather, warms up,
// writes a PNG and a JSON log {consoleErrors, moduleErrors, fps, drawCalls, triangles,
// moduleStates, budget, over}.
//
// Usage:
//   node tools/shot.mjs --name demo --module null --cam orbit --time 0.5 \
//        --weather clear --seed 1337 --width 1920 --height 1080 --warm 60
// Env: CHROME_PATH (path to a chromium binary), SHOT_BASE (default http://localhost:5173),
//      SHOT_OUT (default ./shots)
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const root = join(__dirname, '..');

function parseArgs(argv) {
  // warm defaults to 150 frames: under SwiftShader (~10fps) the city's merged
  // geometry + 2048px atlas textures take many frames to upload and settle. A
  // short warm catches partial builds (missing buildings, unlit windows) and reads
  // as a "non-deterministic" scene — it is not. 150 frames (~15s) is steady-state.
  const a = { name: 'shot', module: 'null', cam: 'orbit', time: null, weather: null, seed: '1337', width: 1920, height: 1080, warm: 150, port: 9222 + Math.floor(Math.random() * 500) };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    if (!k.startsWith('--')) continue;
    const key = k.slice(2);
    const val = argv[i + 1];
    if (val === undefined || val.startsWith('--')) { a[key] = true; continue; }
    a[key] = val; i++;
  }
  a.width = Number(a.width) || 1920;
  a.height = Number(a.height) || 1080;
  a.warm = Number(a.warm) || 60;
  a.port = Number(a.port) || 9333;
  if (a.time !== null && a.time !== true) a.time = Number(a.time);
  return a;
}

const args = parseArgs(process.argv);
const base = process.env.SHOT_BASE || 'http://localhost:5173';
const outDir = process.env.SHOT_OUT ? join(root, process.env.SHOT_OUT) : join(root, 'shots');
mkdirSync(outDir, { recursive: true });
const url = `${base}/?seed=${encodeURIComponent(args.seed)}`;
const pngPath = join(outDir, `${args.name}.png`);
const jsonPath = join(outDir, `${args.name}.json`);

const log = {
  name: args.name, url, seed: args.seed, width: args.width, height: args.height,
  module: args.module, cam: args.cam, time: args.time, weather: args.weather,
  browser: null, consoleErrors: [], moduleErrors: [], ready: false,
  fps: null, drawCalls: null, triangles: null, budget: null, over: null,
  moduleStates: null, timeOfDay: null, timestamp: new Date().toISOString(),
};

// Resolve a usable browser. Prefers @sparticuz/chromium (self-contained arm64,
// bundled in node_modules/@sparticuz/chromium/bin), inflating its brotli libs.
async function resolveChrome() {
  if (process.env.CHROME_PATH) return { exec: process.env.CHROME_PATH, args: [], env: {} };
  try {
    const m = await import('@sparticuz/chromium');
    const ch = m.default || m;
    // `inflate` is a named export in current @sparticuz/chromium (not on the
    // default class). Idempotent: skips work when the target dir already exists.
    const inflate = m.inflate;
    if (typeof inflate === 'function') {
      const base = 'node_modules/@sparticuz/chromium/bin/';
      await Promise.all([
        inflate(base + 'al2023.tar.br'),
        inflate(base + 'swiftshader.tar.br'),
        inflate(base + 'fonts.tar.br'),
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
  } catch (e) { /* fall through to local */ }
  return { exec: join(root, '.chrome', 'chrome-linux64', 'chrome'), args: [], env: {} };
}
const resolved = await resolveChrome();
const chromePath = resolved.exec;
log.browser = chromePath;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- Launch Chrome with a remote debugging port -----------------------------
const dbgPort = args.port;
const baseFlags = [
  `--window-size=${args.width},${args.height}`,
  `--remote-debugging-port=${dbgPort}`,
  '--hide-scrollbars', '--force-color-profile=srgb',
  // /dev/shm is root-owned + tiny in this sandbox; force Chrome to use /tmp.
  '--disable-dev-shm-usage',
];
// sparticuz args already carry --headless/--no-sandbox/swiftshader/single-process;
// for a bare binary add minimal headless flags.
const headlessFlags = resolved.args.length ? [] : ['--headless=new', '--no-sandbox', '--disable-gpu', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader'];
// sparticuz emits --headless='shell' with literal single quotes; Node spawn has no
// shell to strip them, so Chrome would read an invalid value. Quotes are never
// meaningful in a Chrome flag value, so strip them.
const spawnArgs = [...headlessFlags, ...resolved.args, ...baseFlags, 'about:blank']
  .map((a) => (typeof a === 'string' ? a.replace(/'/g, '') : a));
const chrome = spawn(chromePath, spawnArgs,
  { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, ...resolved.env } });
chrome.stderr.on('data', () => {}); // swallow chrome's verbose stderr

// Hard watchdog: if Chrome hangs (e.g. under memory pressure with many parallel
// renderers) the tool must self-terminate instead of stalling a caller for minutes.
const GLOBAL_TIMEOUT = (Number(process.env.SHOT_TIMEOUT) || 120) * 1000;
setTimeout(() => {
  console.error(`[shot] global timeout after ${GLOBAL_TIMEOUT / 1000}s — forcing exit`);
  try { writeFileSync(jsonPath, JSON.stringify({ ...log, timeout: true }, null, 2)); } catch {}
  try { chrome.kill('SIGKILL'); } catch {}
  process.exit(3);
}, GLOBAL_TIMEOUT);
chrome.on('error', (e) => {
  console.error('[shot] chrome spawn error: ' + (e && e.message));
  try { writeFileSync(jsonPath, JSON.stringify({ ...log, spawnError: e.message }, null, 2)); } catch {}
  process.exit(4);
});

async function httpJson(path, method = 'GET') {
  const r = await fetch(`http://127.0.0.1:${dbgPort}${path}`, { method });
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${method} ${path}`);
  return r.json();
}

// Wait for the DevTools endpoint to come up, then open a page target.
// Chrome >=111 requires PUT (not GET) to create a new target via /json/new.
let wsUrl = null;
for (let i = 0; i < 60 && !wsUrl; i++) {
  try {
    const t = await httpJson('/json/new?about:blank', 'PUT');
    wsUrl = t.webSocketDebuggerUrl;
  } catch { await sleep(500); }
}
if (!wsUrl) throw new Error('Chrome DevTools endpoint did not come up (port ' + dbgPort + ')');

// --- CDP client over built-in WebSocket -------------------------------------
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
    if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') {
      consoleErrors.push((msg.params.args || []).map((a) => a.value ?? a.description ?? '').join(' '));
    }
    if (msg.method === 'Runtime.exceptionThrown') {
      consoleErrors.push(msg.params.exceptionDetails?.exception?.description || 'exception');
    }
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

try {
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: args.width, height: args.height, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url });

  // Wait for the app to signal ready.
  let ready = false;
  for (let i = 0; i < 120 && !ready; i++) {
    ready = await evaluate('window.__APP_READY__ === true').catch(() => false);
    if (!ready) await sleep(500);
  }
  log.ready = ready;
  if (!ready) throw new Error('app did not become ready in time');

  // Apply the shot configuration (single application, after ready).
  await evaluate(`(function(){const A=window.__APP__;
    const m=${JSON.stringify(args.module)}; if(m&&m!=='null')A.setModule(m);
    A.setCameraPreset(${JSON.stringify(args.cam)});
    const t=${JSON.stringify(args.time)}; if(t!=null)A.setTimeOfDay(t);
    A.setSpeed(0); // freeze the clock so the frame is captured at EXACTLY t. The live app
                   // auto-plays the clock (tps>0); left running it would drift t across the
                   // 150-frame warm and make night/day shots land at the wrong time of day.
    const w=${JSON.stringify(args.weather)}; if(w&&w!=='null')A.setWeather(w);
  })()`);

  const warm = (n) => evaluate(`new Promise(r=>{let i=0;const t=()=>{if(++i>=${n})r(1);else requestAnimationFrame(t)};requestAnimationFrame(t)})`);
  await warm(args.warm);
  await warm(8);

  const shot = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(pngPath, Buffer.from(shot.data, 'base64'));

  const snap = await evaluate(`(function(){const A=window.__APP__;const s=A.stats();return {
    drawCalls:A.renderer.info.render.calls, triangles:A.renderer.info.render.triangles,
    budget:s.budget, over:s.over, moduleStates:s.moduleStates, timeOfDay:A.world.time.t, fps:s.fps
  }})()`);
  log.fps = snap.fps; log.drawCalls = snap.drawCalls; log.triangles = snap.triangles;
  log.budget = snap.budget; log.over = snap.over; log.moduleStates = snap.moduleStates;
  log.timeOfDay = snap.timeOfDay;
  log.moduleErrors = await evaluate('window.__MODULE_ERRORS__ || []') || [];
} finally {
  log.consoleErrors = consoleErrors;
  writeFileSync(jsonPath, JSON.stringify(log, null, 2));
  try { ws.close(); } catch {}
  try { chrome.kill('SIGKILL'); } catch {}
}

const pass = log.ready && consoleErrors.length === 0 && log.moduleErrors.length === 0 && !log.over;
console.log(JSON.stringify({
  name: log.name, ready: log.ready, pass,
  fps: log.fps, drawCalls: log.drawCalls, triangles: log.triangles,
  consoleErrors: consoleErrors.length, moduleErrors: log.moduleErrors.length,
  over: log.over, browser: log.browser, png: pngPath, json: jsonPath,
}));
// Exit promptly: the SIGKILLed chrome + closed ws can otherwise keep the event
// loop alive and make the caller wait for the global watchdog.
process.exit(pass ? 0 : 2);
