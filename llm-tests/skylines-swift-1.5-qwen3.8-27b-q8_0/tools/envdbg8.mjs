#!/usr/bin/env node
// Blending-mode isolation on the known-good plane at cloud0 world pos.
// A  AdditiveBlending + cloudTex, red
// B  NormalBlending + 1x1 white DataTexture
// C  CustomBlending (ONE, ONE_MINUS_SRC_ALPHA) + cloudTex   [premultiplied normal]
// D  premultipliedAlpha:true (NormalBlending path) + cloudTex
// E  control: no map, NormalBlending, red
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
const root = '/sandbox/skylines';
async function resolveChrome() {
  const m = await import('@sparticuz/chromium');
  const ch = m.default || m;
  const inflate = m.inflate;
  if (typeof inflate === 'function') {
    const b = 'node_modules/@sparticuz/chromium/bin/';
    await Promise.all([inflate(b + 'al2023.tar.br'), inflate(b + 'swiftshader.tar.br'), inflate(b + 'fonts.tar.br')]);
  }
  return { exec: await ch.executablePath(), args: ch.args || [], env: { LD_LIBRARY_PATH: '/tmp/al2023/lib:/tmp:' + (process.env.LD_LIBRARY_PATH || ''), FONTCONFIG_PATH: '/tmp/fonts' } };
}
const resolved = await resolveChrome();
const dbgPort = 9600 + Math.floor(Math.random() * 400);
const spawnArgs = [...(resolved.args.length ? [] : ['--headless=new', '--no-sandbox', '--disable-gpu', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader']),
  ...resolved.args, '--window-size=1280,720', `--remote-debugging-port=${dbgPort}`, '--hide-scrollbars', '--force-color-profile=srgb', '--disable-dev-shm-usage', 'about:blank']
  .map((x) => String(x).replace(/'/g, ''));
const chrome = spawn(resolved.exec, spawnArgs, { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, ...resolved.env } });
chrome.stderr.on('data', () => {});
setTimeout(() => { try { chrome.kill('SIGKILL'); } catch {} process.exit(3); }, 170000);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let wsUrl = null;
for (let i = 0; i < 60 && !wsUrl; i++) {
  try { const r = await fetch(`http://127.0.0.1:${dbgPort}/json/new?about:blank`, { method: 'PUT' }); wsUrl = (await r.json()).webSocketDebuggerUrl; } catch { await sleep(500); }
}
const ws = new WebSocket(wsUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error('ws fail')); });
let id = 0; const pending = new Map();
ws.onmessage = (ev) => { const msg = JSON.parse(ev.data); if (msg.id && pending.has(msg.id)) { const { res, rej } = pending.get(msg.id); pending.delete(msg.id); msg.error ? rej(new Error(msg.error.message)) : res(msg.result); } };
const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expr) => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description); return r.result?.value; };
const warm = (n) => evaluate(`new Promise(r=>{let i=0;const t=()=>{if(++i>=${n})r(1);else requestAnimationFrame(t)};requestAnimationFrame(t)})`);
const snap = async (name) => { const s = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(join(root, 'shots', name), Buffer.from(s.data, 'base64')); };

try {
  await send('Page.enable'); await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url: 'http://localhost:5173/?seed=1337' });
  let ready = false;
  for (let i = 0; i < 120 && !ready; i++) { ready = await evaluate('window.__APP_READY__ === true').catch(() => false); if (!ready) await sleep(500); }
  if (!ready) throw new Error('not ready');
  await evaluate(`window.__APP__.setModule('environment'); window.__APP__.setCameraPreset('orbit'); window.__APP__.setTimeOfDay(0.78);`);
  await warm(60);

  await evaluate(`(() => {
    const A = window.__APP__; const T = A.registry.ctx.three;
    const p = new T.Mesh(new T.PlaneGeometry(500, 220),
      new T.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 1, side: T.DoubleSide, depthWrite: false }));
    p.position.set(-743, 134, -46); p.lookAt(A.camera.position);
    A.loop.scene.add(p); window.__P__ = p;
  })()`);
  await warm(5);
  const tex = `window.__APP__.registry.get('environment')._cloudTex`;

  // A: additive
  await evaluate(`(() => { const T = window.__APP__.registry.ctx.three; const m = window.__P__.material; m.map = ${tex}; m.blending = T.AdditiveBlending; m.needsUpdate = true; })()`);
  await warm(15); await snap('dbg8-a.png');
  // B: normal + 1x1 white datatexture
  await evaluate(`(() => { const T = window.__APP__.registry.ctx.three; const m = window.__P__.material; m.blending = T.NormalBlending; m.map = new T.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1); m.needsUpdate = true; })()`);
  await warm(15); await snap('dbg8-b.png');
  // C: custom premultiplied (ONE, ONE_MINUS_SRC_ALPHA)
  await evaluate(`(() => { const T = window.__APP__.registry.ctx.three; const m = window.__P__.material; m.blending = T.CustomBlending; m.blendSrc = T.OneFactor; m.blendDst = T.OneMinusSrcAlphaFactor; m.blendEquation = T.AddEquation; m.map = ${tex}; m.needsUpdate = true; })()`);
  await warm(15); await snap('dbg8-c.png');
  // D: premultipliedAlpha flag + cloudTex
  await evaluate(`(() => { const T = window.__APP__.registry.ctx.three; const m = window.__P__.material; m.blending = T.NormalBlending; m.premultipliedAlpha = true; m.map = ${tex}; m.needsUpdate = true; })()`);
  await warm(15); await snap('dbg8-d.png');
  // E: control no map
  await evaluate(`(() => { const m = window.__P__.material; m.map = null; m.premultipliedAlpha = false; m.needsUpdate = true; })()`);
  await warm(15); await snap('dbg8-e.png');
  console.log('done');
} catch (e) { console.error('ERR', e.message); }
finally { try { ws.close(); } catch {} try { chrome.kill('SIGKILL'); } catch {} }
process.exit(0);
