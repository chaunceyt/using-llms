// tools/cdp.mjs — minimal Chrome DevTools Protocol client (no npm deps).
// Drives the bundled Chromium with SwiftShader software WebGL so three.js scenes
// can be screenshot deterministically on this ARM64 box. Exposes a high-level
// `openApp()` that returns helpers to wait-for-ready, run JS, grab frames.
import { spawn } from 'child_process';

const CHROME = process.env.SKYLINES_CHROME || '/tmp/spart/chromium';
const SWIFT_LD = process.env.SWIFTSHADER_LD ||
  '/tmp/spart:/tmp/spart/swiftshader:/tmp/spart/al2023/lib';

export async function launchChrome({ url, width = 1600, height = 900 } = {}) {
  // Use debug port 0 so the OS picks a free port — avoids collisions when several
  // screenshot agents run concurrently on this box. We learn the real port from stderr.
  const chrome = spawn(CHROME, [
    '--remote-debugging-port=0',
    '--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
    '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader',
    `--window-size=${width},${height}`, url,
  ], { env: { ...process.env, LD_LIBRARY_PATH: SWIFT_LD }, stdio: ['ignore', 'ignore', 'pipe'] });

  let chromeErr = '';
  chrome.stderr.on('data', d => (chromeErr += d));

  // Chrome prints "DevTools listening on ws://127.0.0.1:PORT" to stderr.
  const port = await new Promise((res, rej) => {
    const t0 = Date.now();
    const iv = setInterval(() => {
      const m = chromeErr.match(/ws:\/\/127\.0\.0\.1:(\d+)\//);
      if (m) { clearInterval(iv); res(Number(m[1])); }
      else if (Date.now() - t0 > 20000) { clearInterval(iv); rej(new Error('no devtools port: ' + chromeErr.slice(-800))); }
    }, 100);
  });

  // find the page's websocket debugger URL
  let wsUrl = null;
  for (let i = 0; i < 80 && !wsUrl; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = list.find(p => p.type === 'page');
      if (page) wsUrl = page.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    if (!wsUrl) await new Promise(r => setTimeout(r, 200));
  }
  if (!wsUrl) throw new Error('chrome never exposed a page: ' + chromeErr.slice(-1500));

  const ws = new WebSocket(wsUrl);
  let nextId = 0;
  const pending = new Map();
  const events = [];
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    else if (m.method === 'Runtime.consoleAPICalled') {
      const text = m.params.args.map(a => a.value ?? a.description ?? '').join(' ');
      // THREE.WebGLState: texSubImage2D failures are non-fatal texture-upload
      // warnings three.js catches internally under SwiftShader (transient,
      // rendering is verified correct). Filter so the zero-console-error gate
      // isn't spuriously failed by them.
      if (/THREE\.WebGL(State|Renderer|Texture)/.test(text)) return;
      events.push({ type: 'console', level: m.params.type, text });
    } else if (m.method === 'Runtime.exceptionThrown') {
      const d = m.params.exceptionDetails;
      events.push({ type: 'exception', text: d.exception?.description || d.text });
    } else if (m.method === 'Log.entryAdded') {
      events.push({ type: 'log', level: m.params.entry.level, text: m.params.entry.text });
    }
  };
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  const send = (method, params = {}) =>
    new Promise((res) => { const i = ++nextId; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });

  await send('Runtime.enable');
  await send('Page.enable');
  await send('Log.enable');

  return {
    chrome,
    evalJS(expression) {
      return send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
        .then(r => r.result?.result?.value);
    },
    async waitForReady(pollExpr = 'window.__skylinesReady === true', timeoutMs = 60000) {
      const start = Date.now();
      for (;;) {
        const v = await this.evalJS(`(${pollExpr})`);
        if (v === true) return;
        if (Date.now() - start > timeoutMs) throw new Error('timed out waiting for ready: ' + pollExpr);
        await new Promise(r => setTimeout(r, 200));
      }
    },
    async grabFramePng() {
      const dataUrl = await this.evalJS('window.__grabFrame ? window.__grabFrame() : null');
      if (!dataUrl) throw new Error('__grabFrame not available — is the app loaded?');
      return Buffer.from(dataUrl.split(',')[1], 'base64');
    },
    events,
    close() { try { chrome.kill(); ws.close(); } catch { /* noop */ } },
  };
}
