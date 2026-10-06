// Tiny pub/sub. A module only emits its own namespace ("terrain:*").
// Listener errors are contained so one bad subscriber can't break the bus.
export class EventBus {
  constructor() { this._m = new Map(); }
  on(name, cb) {
    if (!this._m.has(name)) this._m.set(name, new Set());
    this._m.get(name).add(cb);
    return () => this.off(name, cb);
  }
  once(name, cb) {
    const wrap = (p) => { this.off(name, wrap); cb(p); };
    return this.on(name, wrap);
  }
  off(name, cb) { this._m.get(name)?.delete(cb); }
  emit(name, payload) {
    const s = this._m.get(name);
    if (!s) return;
    for (const cb of [...s]) {
      try { cb(payload); } catch (e) { console.error(`[event:${name}]`, e); }
    }
  }
  clear() { this._m.clear(); }
}
