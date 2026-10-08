// ---------------------------------------------------------------------------
// audio — procedural ambient + city ambience (WebAudio), no sample files.
//
// Everything is synthesized at runtime: low wind/air noise, a distant low city
// hum, and a light traffic wash that can scale with population/traffic later,
// plus a gentle ambient music bed. It never throws, stays silent until a real
// user gesture (autoplay policy), and exposes simple controls on world.audio.
// ---------------------------------------------------------------------------

export const id = 'audio';

// Module-private state (kept out of world except the public control surface).
const S = {
  ctx: null,
  started: false,
  ready: false,
  masterGain: null,
  trafficGain: null,
  windGain: null,
  humGain: null,
  musicGain: null,
  boundResume: null,
};

// ---------------------------------------------------------------------------
// Noise buffer synthesis (Math.random here is fine — audio is deterministic-free
// and this never affects world layout/city/materials).
// ---------------------------------------------------------------------------
function fillNoise(ctx, type = 'white', dur = 2) {
  const len = Math.floor(ctx.sampleRate * dur);
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const d = buf.getChannelData(0);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  let last = 0;
  for (let i = 0; i < len; i++) {
    const w = Math.random() * 2 - 1;
    if (type === 'white') {
      d[i] = w * 0.55;
    } else { // brown noise via leaky integrator
      last = (last + 0.02 * w) / 1.02;
      d[i] = last * 3.5;
      void b0; void b1; void b2; void b3; void b4; void b5; void b6;
    }
  }
  return buf;
}

// ---------------------------------------------------------------------------
// Build the whole node graph once, when first allowed to make sound.
// ---------------------------------------------------------------------------
function buildGraph(ctx) {
  // master bus -> destination
  const master = ctx.createGain();
  master.gain.value = 0; // filled by update() with enabled/volume
  master.connect(ctx.destination);

  // --- Wind / air noise -----------------------------------------------------
  const windSrc = ctx.createBufferSource();
  windSrc.buffer = fillNoise(ctx, 'white', 2);
  windSrc.loop = true;
  const windLP = ctx.createBiquadFilter();
  windLP.type = 'lowpass';
  windLP.frequency.value = 320;
  windLP.Q.value = 0.4;
  const windGain = ctx.createGain();
  windGain.gain.value = 0.1;
  // slow gust LFO on the wind amplitude
  const gustLfo = ctx.createOscillator();
  gustLfo.frequency.value = 0.07;
  const gustAmt = ctx.createGain();
  gustAmt.gain.value = 0.05;
  gustLfo.connect(gustAmt);
  gustAmt.connect(windGain.gain);
  windSrc.connect(windLP).connect(windGain).connect(master);

  // --- Distant city hum (low industrial/electric drone) ----------------------
  const humBus = ctx.createGain();
  humBus.gain.value = 0.055;
  const humLp = ctx.createBiquadFilter();
  humLp.type = 'lowpass';
  humLp.frequency.value = 160;
  humLp.connect(humBus).connect(master);
  // a few detuned low sines — electric substation / distant machinery
  const humFreqs = [50, 60, 100, 120];
  for (const f of humFreqs) {
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.value = f;
    o.detune.value = (f * 0.002);
    const g = ctx.createGain();
    g.gain.value = (f <= 60 ? 0.6 : 0.3) / humFreqs.length;
    o.connect(g).connect(humLp);
    o.start(0);
  }

  // --- Light traffic wash (scales with agents later) --------------------------
  const trSrc = ctx.createBufferSource();
  trSrc.buffer = fillNoise(ctx, 'brown', 2);
  trSrc.loop = true;
  const trBand = ctx.createBiquadFilter();
  trBand.type = 'bandpass';
  trBand.frequency.value = 650;
  trBand.Q.value = 0.7;
  const trLP = ctx.createBiquadFilter();
  trLP.type = 'lowpass';
  trLP.frequency.value = 1400;
  const trafficGain = ctx.createGain();
  trafficGain.gain.value = 0.03; // faint idle presence even when empty
  trSrc.connect(trBand).connect(trLP).connect(trafficGain).connect(master);

  // --- Gentle ambient music bed (soft pad) -----------------------------------
  const musicOut = ctx.createGain();
  musicOut.gain.value = 0.05;
  const musicLP = ctx.createBiquadFilter();
  musicLP.type = 'lowpass';
  musicLP.frequency.value = 850;
  // A-major-ish open stack, slow independent tremolos for a living pad
  const chord = [110, 164.81, 220, 261.63, 329.63];
  for (const f of chord) {
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.value = f;
    o.detune.value = ((Math.random() - 0.5) * 9);
    const g = ctx.createGain();
    g.gain.value = 1 / chord.length * 0.8;
    // very slow amplitude LFO for gentle movement
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 0.04 + Math.random() * 0.06;
    const lfoAmt = ctx.createGain();
    lfoAmt.gain.value = g.gain.value * 0.45;
    lfo.connect(lfoAmt).connect(g.gain);
    o.connect(g).connect(musicOut);
    o.start(0); lfo.start(0);
  }
  musicOut.connect(musicLP).connect(master);

  // start looping noise sources
  windSrc.start(0);
  trSrc.start(0);

  return { master, trafficGain, windGain, humGain: humBus, musicGain: musicOut };
}

// ---------------------------------------------------------------------------
// Resume / create the context on the first real user gesture (autoplay policy).
// Never throws; stays silent when not allowed.
// ---------------------------------------------------------------------------
function resume() {
  try {
    if (!S.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      S.ctx = new AC();
      const g = buildGraph(S.ctx);
      S.masterGain = g.master;
      S.trafficGain = g.trafficGain;
      S.windGain = g.windGain;
      S.humGain = g.humGain;
      S.musicGain = g.musicGain;
      S.ready = true;
    }
    if (S.ctx.state === 'suspended') {
      const p = S.ctx.resume();
      if (p && typeof p.catch === 'function') p.catch(() => {});
    }
    S.started = true;
    return true;
  } catch (e) {
    // blocked / unsupported — remain silent, never log noisy errors
    S.started = false;
    return false;
  }
}

// ---------------------------------------------------------------------------
// Public contract
// ---------------------------------------------------------------------------
export function init(world) {
  if (!world || typeof window === 'undefined') return;

  // Controls live on world so UI/demo can drive them later.
  if (!world.audio) {
    const api = {
      enabled: true,
      volume: 1.0,
      contextStarted: false,
      setVolume(v) {
        api.volume = Math.max(0, Math.min(1, Number(v) || 0));
        return api.volume;
      },
    };
    world.audio = api;
  }

  // Lazy start on first pointer/key interaction (once), then keep listening so
  // we can recover if the context gets suspended (e.g. after tab backgrounding).
  const engage = () => {
    if (!S.started) {
      const ok = resume();
      if (world.audio) world.audio.contextStarted = !!ok && S.ctx && S.ctx.state === 'running';
    } else if (S.ctx && S.ctx.state === 'suspended') {
      resume();
    }
  };
  S.boundResume = engage;
  window.addEventListener('pointerdown', engage, { once: false });
  window.addEventListener('keydown', engage, { once: false });
  window.addEventListener('mousedown', engage, { once: false });
}

export function update(dtSec, world) {
  if (!world || !world.audio) return;
  if (!S.ctx || !S.masterGain) return; // still pre-gesture / blocked — silent

  try {
    const a = world.audio;

    // apply enable + volume smoothly
    const target = a.enabled ? a.volume : 0;
    const now = S.ctx.currentTime;
    if (S.masterGain.gain.cancelScheduledValues) {
      S.masterGain.gain.cancelScheduledValues(now);
      S.masterGain.gain.setTargetAtTime(target, now, 0.3);
    } else {
      S.masterGain.gain.value = target;
    }

    // Traffic wash scales with live agents (population/traffic) when present.
    let ag = 0;
    if (Array.isArray(world.agents)) ag = world.agents.length;
    const factor = Math.min(ag / 150, 1);
    const trafficTarget = 0.03 + factor * 0.16;
    if (S.trafficGain.gain.cancelScheduledValues) {
      S.trafficGain.gain.cancelScheduledValues(now);
      S.trafficGain.gain.setTargetAtTime(trafficTarget, now, 1.2);
    } else {
      S.trafficGain.gain.value = trafficTarget;
    }
  } catch (e) {
    // never take the app down
  }
}

export function showcase(container) {
  if (!container || typeof document === 'undefined') return;
  const root = container.appendChild(document.createElement('div'));
  root.style.cssText =
    'font-family:ui-monospace,monospace;color:#cfe3ff;background:#0d1421;' +
    'border-radius:10px;padding:16px 18px;max-width:340px;';
  const title = document.createElement('div');
  title.textContent = 'Audio — procedural ambience (WebAudio)';
  title.style.cssText = 'font-weight:600;margin-bottom:10px;color:#8fe3ff;';
  root.appendChild(title);

  const lines = [
    ['wind', 'low air noise through a low-pass filter'],
    ['city hum', 'detuned 50/60/100/120 Hz drone'],
    ['traffic wash', 'band-passed brown noise, scales with agents'],
    ['music bed', 'slow A-major sine pad'],
  ];
  for (const [n, d] of lines) {
    const row = document.createElement('div');
    row.textContent = `• ${n} — ${d}`;
    row.style.cssText = 'font-size:12px;margin:3px 0;opacity:.9;';
    root.appendChild(row);
  }

  // Volume control bound to world.audio.setVolume (if reachable via the module).
  const wrap = root.appendChild(document.createElement('div'));
  wrap.style.cssText = 'margin-top:12px;display:flex;align-items:center;gap:8px;';
  const label = document.createElement('span');
  label.textContent = 'volume';
  label.style.cssText = 'font-size:12px;opacity:.85;';
  const slider = document.createElement('input');
  slider.type = 'range'; slider.min = '0'; slider.max = '1'; slider.step = '0.01';
  slider.value = '1';
  slider.style.cssText = 'flex:1;';
  wrap.appendChild(label);
  wrap.appendChild(slider);

  const note = document.createElement('div');
  note.textContent = 'Sound begins on first click / keypress (browser autoplay policy).';
  note.style.cssText = 'font-size:11px;margin-top:10px;opacity:.7;';
  root.appendChild(note);

  slider.addEventListener('input', () => {
    // Reach into the loaded world if it is exposed by the demo/bootstrap.
    const w = (typeof window !== 'undefined' && window.__skylines) ? window.__skylines.world : null;
    if (w && w.audio && typeof w.audio.setVolume === 'function') w.audio.setVolume(Number(slider.value));
  });
}
