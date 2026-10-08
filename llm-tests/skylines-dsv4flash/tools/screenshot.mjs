// tools/screenshot.mjs — verification-loop capture tool.
// Loads the app headlessly (SwiftShader software WebGL), waits for ready, then
// captures one or more named shots. Each shot sets a camera preset + time of day.
//
//   node tools/screenshot.mjs                    # all presets -> docs/shots/
//   node tools/screenshot.mjs --preset night     # just one
//   node tools/screenshot.mjs --url http://...   # custom app URL
//
// Writes, per shot:  docs/shots/<name>.png  and a combined JSON log of fps,
// draw calls, console errors/exceptions (used by the critic gauntlet).
import { launchChrome } from './cdp.mjs';
import fs from 'fs';
import path from 'path';

const PRESETS = {
  overview: {
    camera: { pos: [900, 700, 1150], target: [0, 0, 0] },
    timeOfDaySec: 15 * 3600 + 20 * 60,       // mid-afternoon
  },
  street: {
    camera: { pos: [70, 9, 40], target: [260, 5, 0] },
    timeOfDaySec: 10 * 3600,
  },
  night: {
    // Skyline framing — frames the illuminated downtown (lit windows + street
    // lights). A low street-level night view reads near-black with no structure.
    camera: { pos: [430, 190, 310], target: [0, 40, 0] },
    timeOfDaySec: 21.5 * 3600,
  },
  sunset: {
    camera: { pos: [520, 220, 640], target: [-200, 0, -120] },
    timeOfDaySec: 18.2 * 3600,
  },
};

function parseArgs(argv) {
  const out = { preset: null, url: 'http://127.0.0.1:5173/', dir: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--preset') out.preset = argv[++i];
    else if (argv[i] === '--url') out.url = argv[++i];
    else if (argv[i] === '--dir') out.dir = argv[++i];
  }
  return out;
}

export async function captureShots({ preset, url, dir } = {}) {
  const shotsDir = dir || path.resolve(process.cwd(), 'docs/shots');
  fs.mkdirSync(shotsDir, { recursive: true });
  const names = preset ? [preset] : Object.keys(PRESETS);
  const report = { app: url, shotAt: new Date().toISOString(), shots: [], errors: [] };

  const client = await launchChrome({ url });
  try {
    await client.waitForReady();
    // Pause the clock immediately so presets are exact & deterministic (the ui
    // module advances meta.timeOfDaySec each frame unless paused) and to avoid
    // transient texture-upload churn during unpaused settle.
    await client.evalJS('window.__skylines.world.meta.paused = true');
    // Let the module graph + progressive terrain LOD / texture uploads settle.
    // A cold-start first frame can otherwise catch geometry mid-stream and read
    // as an empty/low-structure shot (determinism fix).
    await new Promise(r => setTimeout(r, 2400));

    for (const name of names) {
      const p = PRESETS[name];
      if (!p) throw new Error(`unknown preset "${name}"`);
      // set camera from preset
      await client.evalJS(`window.__skylines.setCamera(${JSON.stringify(p.camera.pos)}, ${JSON.stringify(p.camera.target)})`);
      // set time of day via the bus so sky/lighting modules react
      await client.evalJS(`window.__skylines.setTimeOfDay(${p.timeOfDaySec})`);
      // let lighting/sky update + a couple frames bake (generous: first shot is
      // the coldest, give LOD/textures time to finish)
      await new Promise(r => setTimeout(r, 900));
      const png = await client.grabFramePng();
      fs.writeFileSync(path.join(shotsDir, `${name}.png`), png);

      const stats = await client.evalJS('window.__skylines.getStats()') || {};
      const shotErrors = client.events.filter(e => e.level === 'error' || e.type === 'exception');
      report.shots.push({
        preset: name, timeOfDaySec: p.timeOfDaySec,
        fps: stats.fps ?? null, drawCalls: stats.drawCalls ?? null,
        bytes: png.length, errors: shotErrors.length,
      });
    }
  } finally {
    client.close();
  }
  report.errors = client.events.filter(e => e.level === 'error' || e.type === 'exception').map(e => e.text);
  fs.writeFileSync(path.join(shotsDir, 'report.json'), JSON.stringify(report, null, 2));
  console.log(`[screenshot] captured ${report.shots.length} shots to ${shotsDir}`);
  for (const s of report.shots) console.log(`  ${s.preset}: ${s.bytes}B fps=${s.fps} drawCalls=${s.drawCalls} errors=${s.errors}`);
  if (report.errors.length) console.log('[screenshot] page errors:', report.errors);
  return report;
}

// allow `node tools/screenshot.mjs` to run directly
if (import.meta.url === new URL(process.argv[1], 'file:').href) {
  const args = parseArgs(process.argv.slice(2));
  captureShots(args).catch(e => { console.error('[screenshot] failed:', e.message); process.exit(1); });
}
