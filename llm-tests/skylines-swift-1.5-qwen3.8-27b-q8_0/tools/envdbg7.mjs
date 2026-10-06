#!/usr/bin/env node
// Texture-variant matrix on one known-good plane (cloud0 world pos).
// T1 baseline cloudTex (expected: invisible)
// T2 clone -> RepeatWrapping
// T3 clone -> NoColorSpace
// T4 DataTexture from same pixels (linear)
// T5 clone -> LinearFilter + no mipmaps
// T6 128 radial-gradient canvas + SRGB + ClampToEdge (pool recipe w/ my settings)
// T7 128 radial-gradient canvas, all defaults (pool recipe)
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
let id = 0; const pending = new Map(); const consoleMsgs = [];
ws.onmessage = (ev) => { const msg = JSON.parse(ev.data); if (msg.id && pending.has(msg.id)) { const { res, rej } = pending.get(msg.id); pending.delete(msg.id); msg.error ? rej(new Error(msg.error.message)) : res(msg.result); }
  else if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type !== 'log') { consoleMsgs.push(`[${msg.params.type}] ` + (msg.params.args || []).map(a => a.value ?? a.description ?? '').join(' ')); }
  else if (msg.method === 'Runtime.exceptionThrown') { consoleMsgs.push('[exception] ' + (msg.params.exceptionDetails?.exception?.description || 'exception')); } };
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

  // the plane (transparent red, opacity 1) at cloud0 world position
  await evaluate(`(() => {
    const A = window.__APP__; const T = A.registry.ctx.three;
    const p = new T.Mesh(new T.PlaneGeometry(500, 220),
      new T.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 1, side: T.DoubleSide, depthWrite: false }));
    p.position.set(-743, 134, -46); p.lookAt(A.camera.position);
    A.loop.scene.add(p); window.__P__ = p;
  })()`);
  await warm(5);

  const setMap = (expr) => evaluate(`(() => { const m = window.__P__.material; m.map = ${expr}; m.needsUpdate = true; return true; })()`);

  // T1 baseline
  await setMap(`window.__APP__.registry.get('environment')._cloudTex`);
  await warm(15); await snap('dbg7-t1.png');
  // T2 repeat wrap
  await setMap(`(() => { const t = window.__APP__.registry.get('environment')._cloudTex.clone(); t.wrapS = t.wrapT = 1000; t.needsUpdate = true; return t; })()`);
  await warm(15); await snap('dbg7-t2.png');
  // T3 no colorspace
  await setMap(`(() => { const t = window.__APP__.registry.get('environment')._cloudTex.clone(); t.colorSpace = ''; t.needsUpdate = true; return t; })()`);
  await warm(15); await snap('dbg7-t3.png');
  // T4 DataTexture from the same pixels
  await setMap(`(() => {
    const T = window.__APP__.registry.ctx.three;
    const env = window.__APP__.registry.get('environment');
    const cv = env._cloudTex.image;
    const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
    const t = new T.DataTexture(d, cv.width, cv.height, T.RGBAFormat, T.UnsignedByteType);
    t.colorSpace = T.SRGBColorSpace; t.flipY = false; t.needsUpdate = true; return t;
  })()`);
  await warm(15); await snap('dbg7-t4.png');
  // T5 linear min filter, no mipmaps
  await setMap(`(() => { const T = window.__APP__.registry.ctx.three; const t = window.__APP__.registry.get('environment')._cloudTex.clone(); t.minFilter = T.LinearFilter; t.generateMipmaps = false; t.needsUpdate = true; return t; })()`);
  await warm(15); await snap('dbg7-t5.png');
  // T6 pool recipe + SRGB + ClampToEdge
  await setMap(`(() => {
    const T = window.__APP__.registry.ctx.three; const S = 128;
    const cv = document.createElement('canvas'); cv.width = cv.height = S;
    const g = cv.getContext('2d');
    const grad = g.createRadialGradient(S/2, S/2, 0, S/2, S/2, S/2);
    grad.addColorStop(0, 'rgba(255,60,60,1)'); grad.addColorStop(0.5, 'rgba(255,60,60,0.5)'); grad.addColorStop(1, 'rgba(255,60,60,0)');
    g.fillStyle = grad; g.fillRect(0, 0, S, S);
    const t = new T.CanvasTexture(cv); t.colorSpace = T.SRGBColorSpace; t.wrapS = t.wrapT = T.ClampToEdgeWrapping; t.needsUpdate = true; return t;
  })()`);
  await warm(15); await snap('dbg7-t6.png');
  // T7 pool recipe, all defaults
  await setMap(`(() => {
    const T = window.__APP__.registry.ctx.three; const S = 128;
    const cv = document.createElement('canvas'); cv.width = cv.height = S;
    const g = cv.getContext('2d');
    const grad = g.createRadialGradient(S/2, S/2, 0, S/2, S/2, S/2);
    grad.addColorStop(0, 'rgba(255,60,60,1)'); grad.addColorStop(0.5, 'rgba(255,60,60,0.5)'); grad.addColorStop(1, 'rgba(255,60,60,0)');
    g.fillStyle = grad; g.fillRect(0, 0, S, S);
    const t = new T.CanvasTexture(cv); t.needsUpdate = true; return t;
  })()`);
  await warm(15); await snap('dbg7-t7.png');

  console.log('CONSOLE', JSON.stringify(consoleMsgs, null, 1));
  console.log('done');
} catch (e) { console.error('ERR', e.message); }
finally { try { ws.close(); } catch {} try { chrome.kill('SIGKILL'); } catch {} }
process.exit(0);
