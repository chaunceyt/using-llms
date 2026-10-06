#!/usr/bin/env node
// Decisive cloud-isolation probe:
//  S0 baseline (state + drawCalls + per-cloud world pos / frustum visibility)
//  S1 all clouds hidden  -> drawCalls delta + shot
//  S2 one cloud: transparent, opacity 1, RED, map=null, fog=false
//  S3 same: opaque (transparent=false)
//  S4 same: transparent again, map restored
//  + texture canvas pixel dump (center + a few samples) and GL capability info
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
const dbgPort = 9800 + Math.floor(Math.random() * 400);
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

  const state0 = await evaluate(`(() => {
    const A = window.__APP__;
    const env = A.registry.get('environment');
    const T = A.registry.ctx.three;
    const cam = A.camera;
    cam.updateMatrixWorld();
    const fr = new T.Frustum(); fr.setFromProjectionMatrix(new T.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse));
    const out = env._clouds.map((c, i) => {
      c.updateWorldMatrix(true, false);
      const wp = new T.Vector3(); c.getWorldPosition(wp);
      return { i, wp: wp.toArray().map(v => Math.round(v)), d: Math.round(wp.distanceTo(cam.position)),
               inFrust: fr.intersectsObject(c), vis: c.visible,
               scale: c.scale.toArray().map(v => Math.round(v)) };
    });
    const gl = A.renderer.getContext();
    return {
      drawCalls: A.renderer.info.render.calls,
      webgl: gl.getParameter(gl.VERSION),
      clouds: out,
      mat: { transparent: env._cloudMat.transparent, opacity: env._cloudMat.opacity, fog: env._cloudMat.fog,
             depthWrite: env._cloudMat.depthWrite, map: !!env._cloudMat.map, blending: env._cloudMat.blending },
    };
  })()`);
  console.log('S0', JSON.stringify(state0, null, 1));
  await snap('dbg5-s0.png');

  // S1: hide all clouds
  await evaluate(`window.__APP__.registry.get('environment')._clouds.forEach(c => c.visible = false);`);
  await warm(10);
  const s1 = await evaluate(`window.__APP__.renderer.info.render.calls`);
  console.log('S1 draws(hidden)=', s1);
  await snap('dbg5-s1.png');

  // pick the best candidate: visible in frustum, highest, nearest
  const pick = state0.clouds.filter(c => c.inFrust).sort((a, b) => (b.wp[1] - a.wp[1]) || (a.d - b.d))[0] || state0.clouds[0];
  console.log('pick cloud', pick.i, JSON.stringify(pick));

  // S2: transparent, red, opacity 1, NO map, no fog
  await evaluate(`(() => {
    const env = window.__APP__.registry.get('environment');
    env._clouds.forEach(c => c.visible = true);
    const m = env._cloudMat;
    m.map = null; m.transparent = true; m.opacity = 1; m.fog = false;
    m.color.setRGB(1, 0, 0); m.needsUpdate = true;
  })()`);
  await warm(20);
  const s2 = await evaluate(`window.__APP__.renderer.info.render.calls`);
  console.log('S2 draws(transparent,noMap)=', s2);
  await snap('dbg5-s2.png');

  // S3: opaque control
  await evaluate(`(() => { const m = window.__APP__.registry.get('environment')._cloudMat; m.transparent = false; m.needsUpdate = true; })()`);
  await warm(20);
  const s3 = await evaluate(`window.__APP__.renderer.info.render.calls`);
  console.log('S3 draws(opaque,noMap)=', s3);
  await snap('dbg5-s3.png');

  // S4: transparent + map restored
  await evaluate(`(() => { const env = window.__APP__.registry.get('environment'); const m = env._cloudMat; m.map = env._cloudTex; m.transparent = true; m.needsUpdate = true; })()`);
  await warm(20);
  const s4 = await evaluate(`window.__APP__.renderer.info.render.calls`);
  console.log('S4 draws(transparent,map)=', s4);
  await snap('dbg5-s4.png');

  // texture + GL info
  const tex = await evaluate(`(() => {
    const A = window.__APP__;
    const env = A.registry.get('environment');
    const tex = env._cloudTex;
    const cv = tex.image;
    const g = cv.getContext('2d');
    const px = (x, y) => { const d = g.getImageData(x, y, 1, 1).data; return [d[0], d[1], d[2], d[3]]; };
    const gl = A.renderer.getContext();
    return {
      canvas: [cv.width, cv.height],
      center: px(128, 128), q1: px(64, 64), corner: px(4, 4),
      version: tex.version, colorSpace: tex.colorSpace,
      maxTex: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      webgl2: !!gl.getExtension('WEBGL_multi_draw'),
    };
  })()`);
  console.log('TEX', JSON.stringify(tex));
  console.log('done');
} catch (e) { console.error('ERR', e.message); }
finally { try { ws.close(); } catch {} try { chrome.kill('SIGKILL'); } catch {} }
process.exit(0);
