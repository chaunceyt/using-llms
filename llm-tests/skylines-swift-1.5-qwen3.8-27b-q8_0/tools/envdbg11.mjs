#!/usr/bin/env node
// State dump on the REAL showcase: staged cloud world positions, NDC from the
// orbit camera, material state; then a forced-red test.
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

  const state = await evaluate(`(() => {
    const A = window.__APP__;
    const T = A.registry.ctx.three;
    const env = A.registry.get('environment');
    const cam = A.camera;
    cam.updateMatrixWorld();
    const fwd = new T.Vector3(); cam.getWorldDirection(fwd);
    const rgt = new T.Vector3().crossVectors(fwd, cam.up).normalize();
    const upv = new T.Vector3().crossVectors(rgt, fwd).normalize();
    const tanH = Math.tan(T.MathUtils.degToRad(cam.fov / 2));
    const tanV = tanH * cam.aspect;
    const out = env._clouds.map((c, i) => {
      c.updateWorldMatrix(true, false);
      const wp = new T.Vector3(); c.getWorldPosition(wp);
      const v = wp.sub(cam.position);
      const x = v.dot(rgt) / (v.dot(fwd) * tanV);
      const y = v.dot(upv) / (v.dot(fwd) * tanH);
      return { i, wx: Math.round(wp.x), wy: Math.round(wp.y), wz: Math.round(wp.z),
               d: Math.round(v.dot(fwd)), ndc: [ +x.toFixed(2), +y.toFixed(2) ],
               in: Math.abs(x) < 1 && Math.abs(y) < 1 && v.dot(fwd) > 0,
               vis: c.visible, s: [ Math.round(c.scale.x), Math.round(c.scale.y) ] };
    });
    return {
      fov: cam.fov, aspect: +cam.aspect.toFixed(2),
      camPos: cam.position.toArray().map(v => Math.round(v)),
      skyGroupPos: env._skyGroup.position.toArray().map(v => Math.round(v)),
      mat: { fog: env._cloudMat.fog, opacity: +env._cloudMat.opacity.toFixed(3),
             color: [env._cloudMat.color.r, env._cloudMat.color.g, env._cloudMat.color.b].map(v => +v.toFixed(2)),
             map: !!env._cloudMat.map, transparent: env._cloudMat.transparent },
      clouds: out,
    };
  })()`);
  console.log('STATE', JSON.stringify(state, null, 1));
  await snap('dbg11-base.png');

  // force: all clouds red, opaque-ish, fog off
  await evaluate(`(() => { const env = window.__APP__.registry.get('environment'); const m = env._cloudMat; m.fog = false; m.opacity = 1; m.color.setRGB(1, 0, 0); m.needsUpdate = true; })()`);
  await warm(15);
  await snap('dbg11-forced.png');
  console.log('done');
} catch (e) { console.error('ERR', e.message); }
finally { try { ws.close(); } catch {} try { chrome.kill('SIGKILL'); } catch {} }
process.exit(0);
