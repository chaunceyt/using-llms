// Game clock: time of day + weather + sun path. Advances only by accumulated dt.
// t in [0,1): 0 = midnight, 0.25 = sunrise, 0.5 = noon, 0.75 = sunset.
export class Clock {
  constructor() {
    this.t = 0.5;
    this.day = 1;
    this.weather = 'clear'; // clear|cloudy|rain|fog
    this.tps = 0;           // simulated hours per real second (0 = paused)
    this._acc = 0;
  }
  advance(dt) {
    if (this.tps <= 0 || dt <= 0) return;
    this._acc += (dt * this.tps) / 24;
    const d = Math.floor(this._acc);
    if (d > 0) { this._acc -= d; this.day += d; }
    this.t = (this.t + this._acc) % 1;
    this._acc = 0;
  }
  setTimeOfDay(t) { this.t = ((t % 1) + 1) % 1; }
  setWeather(w) { this.weather = w; }
  setTps(tps) { this.tps = tps; }
  // Unit vector pointing toward the sun (world space, +Y up).
  sunDir() {
    const a = (this.t - 0.25) * Math.PI * 2;
    const elev = Math.sin(a);
    const x = Math.cos(a);      // east at sunrise -> west at sunset
    const z = -0.35;            // slight constant southern tilt
    const len = Math.hypot(x, elev, z) || 1;
    return { x: x / len, y: elev / len, z: z / len, elev };
  }
  isDay() { return this.sunDir().elev > 0.02; }
  // 0 at deep night, 1 at full day; smooth around horizon.
  daylight() {
    const e = this.sunDir().elev;
    return Math.max(0, Math.min(1, (e + 0.12) / 0.35));
  }
}
