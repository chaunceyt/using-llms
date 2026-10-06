// Main animation loop. rAF -> dt -> clock -> modules -> render -> perf sample.
// Rendering goes through ctx.render (swappable) so the effects module can take over
// the pass with an EffectComposer without the loop knowing.
export class Loop {
  constructor({ renderer, registry, clock, perf, scene, camera, ctx }) {
    this.renderer = renderer;
    this.registry = registry;
    this.clock = clock;
    this.perf = perf;
    this.scene = scene;
    this.camera = camera;
    this.ctx = ctx;
    this._raf = 0;
    this._last = 0;
    this._running = false;
  }
  _render() {
    const fn = (this.ctx && this.ctx.render) || ((s, c) => this.renderer.render(s, c));
    fn(this.scene, this.camera);
  }
  // Keep the shared world.time in lockstep with the clock so the UI, sim, and the
  // screenshot log (which reads world.time.t) all report the true time of day.
  _syncTime() {
    const w = this.ctx && this.ctx.world;
    if (!w) return;
    w.time.t = this.clock.t;
    w.time.day = this.clock.day;
    w.time.weather = this.clock.weather;
    w.time.tps = this.clock.tps;
  }
  start() {
    if (this._running) return;
    this._running = true;
    this._last = performance.now();
    const frame = (now) => {
      if (!this._running) return;
      const dt = Math.min(0.1, (now - this._last) / 1000);
      this._last = now;
      this.clock.advance(dt);
      this._syncTime();
      this.registry.update(dt);
      this._render();
      this.perf.sample(this.renderer, dt);
      this._raf = requestAnimationFrame(frame);
    };
    this._raf = requestAnimationFrame(frame);
  }
  // Render a single frame on demand (stable captures for the screenshot tool).
  renderOnce() {
    this.clock.advance(1 / 60);
    this._syncTime();
    this.registry.update(1 / 60);
    this._render();
    this.perf.sample(this.renderer, 1 / 60);
  }
  stop() { this._running = false; cancelAnimationFrame(this._raf); }
}
