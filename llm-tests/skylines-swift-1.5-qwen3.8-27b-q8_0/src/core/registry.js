// Module registry + failure isolation. Wraps init/update in try/catch so one
// broken module never takes the game down. Failures are recorded to
// window.__MODULE_ERRORS__ (read by the screenshot tool) and the module is skipped.
export class Registry {
  constructor(ctx) {
    this.ctx = ctx;
    this.mods = new Map(); // name -> { mod, state: 'init'|'live'|'failed', stats }
    this.errors = [];
  }
  register(mod) {
    const entry = { mod, state: 'init', stats: null };
    this.mods.set(mod.name, entry);
    return entry;
  }
  async initAll() {
    for (const entry of this.mods.values()) {
      try {
        await entry.mod.init(this.ctx.world, this.ctx);
        entry.state = 'live';
      } catch (e) {
        entry.state = 'failed';
        this._fail(entry, 'init', e);
      }
    }
  }
  update(dt) {
    for (const entry of this.mods.values()) {
      if (entry.state !== 'live') continue;
      try {
        entry.mod.update(dt, this.ctx.world);
      } catch (e) {
        entry.state = 'failed';
        this._fail(entry, 'update', e);
      }
    }
  }
  emit(name, payload) {
    // route world events to live modules that implement onEvent
    for (const entry of this.mods.values()) {
      if (entry.state !== 'live' || typeof entry.mod.onEvent !== 'function') continue;
      try { entry.mod.onEvent(name, payload); } catch (e) { this._fail(entry, 'onEvent', e); }
    }
  }
  _fail(entry, at, e) {
    const rec = {
      module: entry.mod.name,
      at,
      message: String((e && e.message) || e),
      stack: (e && e.stack) || '',
    };
    this.errors.push(rec);
    if (typeof window !== 'undefined') {
      window.__MODULE_ERRORS__ = window.__MODULE_ERRORS__ || [];
      window.__MODULE_ERRORS__.push(rec);
    }
    console.error(`[module:${entry.mod.name}] ${at} failed:`, e);
  }
  get(name) { return this.mods.get(name)?.mod; }
  states() {
    const o = {};
    for (const e of this.mods.values()) o[e.mod.name] = e.state;
    return o;
  }
  disposeAll() {
    for (const e of this.mods.values()) {
      try { e.mod.dispose(); } catch { /* best effort */ }
    }
  }
}
