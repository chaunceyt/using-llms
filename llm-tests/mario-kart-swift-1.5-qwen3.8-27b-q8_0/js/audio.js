export function createAudio() {
  let ctx = null;
  let master = null;
  let engineGain = null;
  let engineFilter = null;
  let engineOsc = null;
  let engineOsc2 = null;

  function init() {
    if (ctx) return;
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      master = ctx.createGain();
      master.gain.value = 0.5;
      master.connect(ctx.destination);

      engineGain = ctx.createGain();
      engineGain.gain.value = 0;
      engineFilter = ctx.createBiquadFilter();
      engineFilter.type = 'lowpass';
      engineFilter.frequency.value = 700;
      engineOsc = ctx.createOscillator();
      engineOsc.type = 'sawtooth';
      engineOsc.frequency.value = 60;
      engineOsc2 = ctx.createOscillator();
      engineOsc2.type = 'square';
      engineOsc2.frequency.value = 30;
      engineOsc.connect(engineFilter);
      engineOsc2.connect(engineFilter);
      engineFilter.connect(engineGain);
      engineGain.connect(master);
      engineOsc.start();
      engineOsc2.start();
    } catch (e) {
      ctx = null;
    }
  }

  function ready() { return !!ctx; }

  function setEngine(ratio) {
    if (!ready()) return;
    const now = ctx.currentTime;
    const f = 60 + ratio * 170;
    engineOsc.frequency.setTargetAtTime(f, now, 0.06);
    engineOsc2.frequency.setTargetAtTime(f * 0.5, now, 0.06);
    engineGain.gain.setTargetAtTime(0.04 + ratio * 0.09, now, 0.09);
  }

  function blip(freq, dur, type, vol, when) {
    if (!ready()) return;
    const t = ctx.currentTime + (when || 0);
    const o = ctx.createOscillator();
    o.type = type || 'square';
    o.frequency.value = freq;
    const g = ctx.createGain();
    g.gain.setValueAtTime(vol || 0.3, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + dur + 0.03);
  }

  function sweep(f0, f1, dur, type, vol, when) {
    if (!ready()) return;
    const t = ctx.currentTime + (when || 0);
    const o = ctx.createOscillator();
    o.type = type || 'sawtooth';
    o.frequency.setValueAtTime(f0, t);
    o.frequency.exponentialRampToValueAtTime(Math.max(1, f1), t + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(vol || 0.25, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + dur + 0.03);
  }

  function noise(dur, vol) {
    if (!ready()) return;
    const t = ctx.currentTime;
    const buf = ctx.createBuffer(1, Math.floor(ctx.sampleRate * dur), ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const f = ctx.createBiquadFilter();
    f.type = 'highpass';
    f.frequency.value = 400;
    const g = ctx.createGain();
    g.gain.value = vol || 0.3;
    src.connect(f); f.connect(g); g.connect(master);
    src.start(t);
  }

  return {
    init,
    setEngine,
    beep(final) { if (final) blip(880, 0.4, 'square', 0.35); else blip(440, 0.15, 'square', 0.3); },
    item() { blip(660, 0.08, 'square', 0.25); blip(990, 0.1, 'square', 0.25, 0.08); },
    boost() { noise(0.3, 0.3); sweep(200, 900, 0.32, 'sawtooth', 0.25); },
    spin() { sweep(420, 80, 0.42, 'sawtooth', 0.25); },
    hit() { blip(90, 0.2, 'sine', 0.4); noise(0.15, 0.25); },
    star(on) {
      if (on) { blip(523, 0.08, 'square', 0.25); blip(659, 0.08, 'square', 0.25, 0.08); blip(784, 0.14, 'square', 0.25, 0.16); }
      else { blip(784, 0.08, 'square', 0.2); blip(523, 0.14, 'square', 0.2, 0.08); }
    },
    lap() { blip(660, 0.1, 'triangle', 0.3); blip(880, 0.16, 'triangle', 0.3, 0.1); },
    finish() { [523, 659, 784, 1047].forEach((f, i) => blip(f, 0.16, 'square', 0.3, i * 0.12)); },
  };
}
