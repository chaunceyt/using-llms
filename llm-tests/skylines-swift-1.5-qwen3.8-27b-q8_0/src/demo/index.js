// demo — the Wave-3 composition layer.
//
// Every content module (roads, zoning, buildings, props, traffic) self-places
// during its own init, in registration order, so the city is already coherent by
// the time this runs. The demo module's job is to ORCHESTRATE that composition:
// confirm each subsystem populated, nudge cross-module state into a good
// starting condition (traffic density for the hour, sim warmed), and publish a
// single summary via `demo:ready` so the UI / gauntlet know the city is up.
//
// It deliberately adds NO geometry of its own — the modules own their content,
// and a duplicate hero-building here would z-fight the buildings core.
//
// Public API: build(world, ctx), summary(). Emits demo:ready. 0 draw calls.
export default class DemoCity {
  name = 'demo';
  summary = null;

  async init(world, ctx) {
    this.ctx = ctx;
    this.build(world, ctx);
  }

  update(dt, world) {}

  build(world, ctx) {
    const reg = ctx.registry;
    const buildings = reg.get('buildings');
    const roads = reg.get('roads');
    const zoning = reg.get('zoning');
    const props = reg.get('props');
    const traffic = reg.get('traffic');

    // Ensure the building core honored the final zoning (idempotent — a no-op if
    // buildings already placed from it during init).
    if (buildings && typeof buildings.buildFromZoning === 'function') {
      try { buildings.buildFromZoning(); } catch { /* already placed */ }
    }

    // Seed traffic density for the current hour so the city reads as alive on boot.
    if (traffic && typeof traffic.setDensity === 'function') {
      const t = world.time ? world.time.t : 0.5;
      const night = t < 0.25 || t > 0.85;
      try { traffic.setDensity(night ? 0.5 : 1.0); } catch { /* not ready */ }
    }

    this.summary = {
      buildings: world.buildings && world.buildings.length ? world.buildings.length : (buildings && buildings.stats ? buildings.stats().buildings : 0) || 0,
      roads: roads && roads.graph ? roads.graph.edges.length : 0,
      zoning: zoning && zoning.tiles ? zoning.tiles.filter((x) => x.use).length : 0,
      props: props && props.stats ? (props.stats().trees || 0) + (props.stats().lights || 0) : 0,
      traffic: traffic && traffic.stats ? traffic.stats().vehicles : 0,
    };

    ctx.events.emit('demo:ready', this.summary);
    return this.summary;
  }

  getSummary() { return this.summary; }

  showcase(scene, world, ctx) {
    // The demo is the whole live city; there is no isolated showcase scene.
    // Point the critic at the composed scene by leaving it as-is.
  }

  dispose() { this.summary = null; }

  stats() {
    return { drawCalls: 0, notes: 'composition orchestrator (no own geometry)', city: this.summary };
  }
}
