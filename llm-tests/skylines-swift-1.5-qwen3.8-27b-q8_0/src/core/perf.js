// Performance budget tracker. Enforces the arch budget:
// >=50 fps @1080p and <=1500 draw calls/frame.
export class Perf {
  constructor() {
    this.budget = { maxDrawCalls: 1500, minFps: 50, maxTris: 2_000_000 };
    this.fps = 0;
    this.drawCalls = 0;
    this.triangles = 0;
    this._frames = 0;
    this._t = 0;
  }
  sample(renderer, dt) {
    this._frames++;
    this._t += dt;
    if (this._t >= 0.5) {
      this.fps = Math.round(this._frames / this._t);
      this._frames = 0;
      this._t = 0;
    }
    const r = renderer?.info?.render;
    if (r) {
      this.drawCalls = r.calls;
      this.triangles = r.triangles;
    }
    return { fps: this.fps, drawCalls: this.drawCalls, triangles: this.triangles };
  }
  overBudget() {
    return (
      this.drawCalls > this.budget.maxDrawCalls ||
      (this.fps > 0 && this.fps < this.budget.minFps)
    );
  }
  snapshot() {
    return {
      fps: this.fps,
      drawCalls: this.drawCalls,
      triangles: this.triangles,
      budget: this.budget,
      over: this.overBudget(),
    };
  }
}
