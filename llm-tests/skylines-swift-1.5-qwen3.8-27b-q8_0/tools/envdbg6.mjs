#!/usr/bin/env node
// Material x parent matrix probe with console capture.
// P1 fresh plane, loop.scene, transparent, no map, red            (control)
// P2 fresh plane, loop.scene, transparent, cloudTex, red
// P3 fresh plane, loop.scene, transparent, cloudTex, white .55    (production values)
// P4 cloud0 reparented to loop.scene, production material
// P5 cloud0 in group, fresh transparent red no-map material
// P6 cloud0 in group, fresh transparent red WITH cloudTex
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
  else if (msg.method === 'Runtime.consoleAPICalled') { consoleMsgs.push(`[${msg.params.type}] ` + (msg.params.args || []).map(a => a.value ?? a.description ?? '').join(' ')); }
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
  await snap('dbg6-base.png');

  // P1 control: fresh transparent red plane, no map, at cloud0 world pos
  await evaluate(`(() => {
    const A = window.__APP__; const T = A.registry.ctx.three;
    const p = new T.Mesh(new T.PlaneGeometry(500, 220),
      new T.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 1, side: T.DoubleSide, depthWrite: false }));
    p.position.set(-743, 134, -46); p.lookAt(A.camera.position);
    A.loop.scene.add(p); window.__P__ = p;
  })()`);
  await warm(15); await snap('dbg6-p1.png');

  // P2: + cloud texture
  await evaluate(`window.__P__.material.map = window.__APP__.registry.get('environment')._cloudTex; window.__P__.material.needsUpdate = true;`);
  await warm(15); await snap('dbg6-p2.png');

  // P3: production values (white tint, opacity .55)
  await evaluate(`(() => { const m = window.__P__.material; m.color.setRGB(1,1,1); m.opacity = 0.55; })()`);
  await warm(15); await snap('dbg6-p3.png');

  // cleanup fresh plane
  await evaluate(`window.__P__.removeFromParent();`);

  // P4: reparent cloud0 to loop.scene (world pos preserved), production material
  await evaluate(`(() => {
    const A = window.__APP__; const env = A.registry.get('environment');
    const c = env._clouds[0];
    const wp = new A.registry.ctx.three.Vector3(); c.getWorldPosition(wp);
    c.removeFromParent(); A.loop.scene.add(c); c.position.copy(wp);
  })()`);
  await warm(15); await snap('dbg6-p4.png');

  // P5: back in group; fresh transparent red no-map material
  await evaluate(`(() => {
    const A = window.__APP__; const T = A.registry.ctx.three; const env = A.registry.get('environment');
    const c = env._clouds[0];
    const wp = new T.Vector3(); c.getWorldPosition(wp);
    c.removeFromParent(); env._skyGroup.add(c);
    c.position.copy(wp).sub(env._skyGroup.position);
    c.material = new T.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 1, side: T.DoubleSide, depthWrite: false });
  })()`);
  await warm(15); await snap('dbg6-p5.png');

  // P6: same + cloudTex
  await evaluate(`(() => { const m = window.__APP__.registry.get('environment')._clouds[0].material; m.map = window.__APP__.registry.get('environment')._cloudTex; m.needsUpdate = true; })()`);
  await warm(15); await snap('dbg6-p6.png');

  console.log('CONSOLE', JSON.stringify(consoleMsgs, null, 1));
  console.log('done');
} catch (e) { console.error('ERR', e.message); }
finally { try { ws.close(); } catch {} try { chrome.kill('SIGKILL'); } catch {} }
process.exit(0);
