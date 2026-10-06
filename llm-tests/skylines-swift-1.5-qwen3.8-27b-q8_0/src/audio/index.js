// audio — a procedural Web-Audio ambience bed for the city. No samples, no
// copyrighted audio: just filtered noise + a couple of oscillators shaped into
// a low city rumble, weather (rain/wind/fog), and a day/night mood swing. UI
// clicks are short synth blips.
//
// Browsers block autoplay, so the context starts suspended and `resume()` is
// called on the first user gesture (wired from core). Everything degrades
// gracefully if AudioContext is unavailable (headless) — it just stays silent
// and never throws.
//
// Public API: setMood(m), setMuted(b), resume(). Emits audio:ready. 0 draw calls.
// a looping white-noise buffer (2s), shared by all noise sources
function makeNoiseBuffer(ctx) {
  const len = ctx.sampleRate * 2;
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  return buf;
}

export default class Audio {
  name = 'audio';
  constructor() {
    this._ac = null; this._ready = false; this._muted = false;
    this._master = null; this._rain = null; this._wind = null; this._rumble = null;
    this._mood = 'day';
  }

  async init(world, ctx) {
    this.ctx = ctx;
    let AC = null;
    try { AC = window.AudioContext || window.webkitAudioContext; } catch { AC = null; }
    if (!AC) { ctx.events.emit('audio:ready', { available: false }); return; }

    this._ac = new AC();
    this._master = this._ac.createGain();
    this._master.gain.value = 0.0; // fade in on resume
    this._master.connect(this._ac.destination);

    const noise = makeNoiseBuffer(this._ac);
    this._buildBed(noise);
    this._ready = true;
    // self-unlock on the first user gesture (browsers block autoplay)
    const unlock = () => this.resume();
    window.addEventListener('pointerdown', unlock, { passive: true });
    window.addEventListener('keydown', unlock);
    ctx.events.emit('audio:ready', { available: true });
  }

  // city rumble (low noise) + wind + rain buses, all fed from one noise buffer
  _buildBed(noise) {
    const ac = this._ac;
    // --- city rumble: very low-passed noise, slow LFO for traffic swell ---
    const rumbleSrc = ac.createBufferSource();
    rumbleSrc.buffer = noise; rumbleSrc.loop = true;
    const rumbleFilt = ac.createBiquadFilter();
    rumbleFilt.type = 'lowpass'; rumbleFilt.frequency.value = 180; rumbleFilt.Q.value = 0.6;
    this._rumble = ac.createGain(); this._rumble.gain.value = 0.5;
    const lfo = ac.createOscillator(); lfo.frequency.value = 0.07;
    const lfoGain = ac.createGain(); lfoGain.gain.value = 0.18;
    lfo.connect(lfoGain).connect(this._rumble.gain);
    rumbleSrc.connect(rumbleFilt).connect(this._rumble).connect(this._master);
    rumbleSrc.start(); lfo.start();

    // --- wind: band-passed noise, slow wandering ---
    const windSrc = ac.createBufferSource();
    windSrc.buffer = noise; windSrc.loop = true; windSrc.playbackRate.value = 0.7;
    const windFilt = ac.createBiquadFilter();
    windFilt.type = 'bandpass'; windFilt.frequency.value = 500; windFilt.Q.value = 0.5;
    this._wind = ac.createGain(); this._wind.gain.value = 0.12;
    const wlfo = ac.createOscillator(); wlfo.frequency.value = 0.05;
    const wlfoGain = ac.createGain(); wlfoGain.gain.value = 0.08;
    wlfo.connect(wlfoGain).connect(this._wind.gain);
    windSrc.connect(windFilt).connect(this._wind).connect(this._master);
    windSrc.start(); wlfo.start();

    // --- rain: high-ish filtered noise, off by default ---
    const rainSrc = ac.createBufferSource();
    rainSrc.buffer = noise; rainSrc.loop = true; rainSrc.playbackRate.value = 1.3;
    const rainFilt = ac.createBiquadFilter();
    rainFilt.type = 'highpass'; rainFilt.frequency.value = 1200;
    const rainFilt2 = ac.createBiquadFilter();
    rainFilt2.type = 'lowpass'; rainFilt2.frequency.value = 7000;
    this._rain = ac.createGain(); this._rain.gain.value = 0.0;
    rainSrc.connect(rainFilt).connect(rainFilt2).connect(this._rain).connect(this._master);
    rainSrc.start();
  }

  // Start/unlock the context on a user gesture and fade the bed in.
  resume() {
    if (!this._ac) return;
    const doResume = () => {
      if (this._ac.state === 'suspended') this._ac.resume().catch(() => {});
      if (this._master && !this._muted) {
        const t = this._ac.currentTime;
        this._master.gain.cancelScheduledValues(t);
        this._master.gain.setValueAtTime(this._master.gain.value, t);
        this._master.gain.linearRampToValueAtTime(0.6, t + 1.2);
      }
    };
    doResume();
    if (!this._wired) {
      this._wired = true;
      const unlock = () => { doResume(); };
      window.addEventListener('pointerdown', unlock, { passive: true });
      window.addEventListener('keydown', unlock);
    }
  }

  setMuted(b) {
    this._muted = !!b;
    if (this._master && this._ac) {
      const t = this._ac.currentTime;
      this._master.gain.cancelScheduledValues(t);
      this._master.gain.linearRampToValueAtTime(b ? 0.0 : 0.6, t + 0.4);
    }
  }

  // mood: 'day' | 'dusk' | 'night' | weather name. Shapes the bed.
  setMood(m) {
    this._mood = m;
    if (!this._ac) return;
    const t = this._ac.currentTime;
    const ramp = (g, v) => { g.gain.cancelScheduledValues(t); g.gain.linearRampToValueAtTime(v, t + 1.5); };
    const rainy = m === 'rain';
    const foggy = m === 'fog';
    const night = m === 'night';
    ramp(this._rain, rainy ? 0.5 : 0.0);
    ramp(this._wind, foggy ? 0.28 : (m === 'cloudy' ? 0.18 : 0.1));
    ramp(this._rumble, night ? 0.32 : 0.5); // quieter city at night
  }

  // A short UI blip (procedural).
  click() {
    if (!this._ac || this._muted || this._ac.state !== 'running') return;
    const ac = this._ac, t = ac.currentTime;
    const o = ac.createOscillator(), g = ac.createGain();
    o.type = 'triangle'; o.frequency.setValueAtTime(660, t);
    o.frequency.exponentialRampToValueAtTime(440, t + 0.06);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.18, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.09);
    o.connect(g).connect(this._master);
    o.start(t); o.stop(t + 0.1);
  }

  update(dt, world) {
    // track the sim weather so the bed follows it
    const w = world && world.time ? world.time.weather : null;
    if (w && w !== this._lastWeather) { this._lastWeather = w; this.setMood(w); }
  }

  showcase(scene, world, ctx) { /* audio is scene-agnostic */ }

  dispose() {
    if (this._ac) { this._ac.close().catch(() => {}); this._ac = null; }
    this._master = this._rain = this._wind = this._rumble = null;
    this._ready = false;
  }

  stats() {
    return { drawCalls: 0, notes: this._ready ? 'procedural ambience (rumble/wind/rain)' : 'audio unavailable' };
  }
}
