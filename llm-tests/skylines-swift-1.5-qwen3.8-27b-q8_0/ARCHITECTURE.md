# ARCHITECTURE — "Skylines" (CS2-class city builder)

Three.js (latest) + Vite, **plain ES modules**, no TypeScript, no build-time codegen.
This document is the **binding contract** for every module builder and every critic.
If a module does not match its contract here, the critic fails it.

## 0. Ground rules

- **Units:** metres. **+Y is up.** X/Z are the ground plane. World origin (0,0,0) is the
  map centre. One zoning/road **tile = 16 m** (matches CS). Terrain is a heightfield.
- **Determinism:** a single **seeded RNG** (`core/rng.js`). No `Math.random()`, no
  `Date.now()`/`performance.now()` for any *logic* value. Every module derives its own
  stream via `rng.fork("<module>")` from `world.seed`, so a given seed reproduces the
  exact same city. Time advances only by accumulated `dt` from the clock.
- **Performance budget:** **≥ 50 fps at 1920×1080**, **≤ 1500 draw calls/frame**,
  ≤ ~2.0 M triangles in view, ≤ 4 GB GPU memory. Per-module draw-call allocation in §6.
- **Asset policy:** **CC0 only** — Poly Haven, ambientCG, or **procedural**. All surfaces
  are PBR (`MeshStandardMaterial`/`MeshPhysicalMaterial`). Albedo textures are sRGB;
  data/metalness/roughness/normal are linear. Photographic assets load through the same
  material factory as procedural ones, so the two are swappable. **Never programmer art**
  for anything the critic can see.
- **Failure isolation:** one broken module must **never** take the game down (see §4).
- **Ownership:** a builder edits **only** its own `src/<module>/` folder. Core changes are
  requested, not made (§8).

## 1. Folder layout (one folder per subsystem)

```
skylines/
  package.json          core (integrator-only)
  vite.config.js        core
  index.html            core
  src/
    main.js             core — boot, module wiring, app handle
    core/               core — integrator-only
      world.js          shared world data model
      event.js          event bus (pub/sub)
      rng.js            seeded PRNG + forks
      clock.js          game time + weather
      perf.js           draw-call/fps budget tracker
      assets.js         CC0 asset + PBR material factory
      registry.js       module registry + failure isolation
      loop.js           requestAnimationFrame main loop
    terrain/            heightfield + water
    environment/        sun/sky/shadows/atmosphere/fog/IBL
    roads/              road network geometry + graph
    zoning/             land-use + density + overlays
    buildings/          PBR building footprints + night lighting
    props/              instanced props (trees, streetlights, signs)
    traffic/            vehicles on the road graph
    effects/            post-processing (bloom, tone-map, AA, grade)
    simulation/         economy / population / tick
    tools/              build & interaction tools (zone/road/bulldoze)
    ui/                 DOM HUD, menus, minimap, time controls
    audio/              Web-Audio ambient + UI (procedural)
    demo/               the composed demo city (scripted scene)
  tools/                core — shot.mjs (screenshots), dev helpers
  public/assets/        CC0 assets (downloaded) + procedural gen output
  docs/                 TOOLING.md, STATUS.json, per-round critic notes
```

## 2. Shared world data model (`core/world.js`)

`world` is a single plain object of typed collections. **Each module owns exactly one key**
and writes only to it. Reading other keys is allowed; writing is not.

```js
world = {
  seed: <int>,                 // master determinism seed
  tileSize: 16,                // metres
  size: { x: 1024, z: 1024 },  // map extent in metres
  terrain:   { res, size, heights: Float32Array, normals: Float32Array, waterLevel },
  roads:     { nodes: [], edges: [] },       // edge: {a,b,lanes,type,elev,offset}
  zoning:    { grid: Map<tileKey,{use,density}>, },
  buildings: { list: [] },                   // {id,use,density,pos,rot,footprint,height,mesh}
  props:     { byType: { name: InstancedMesh } },
  traffic:   { vehicles: [] },               // {edge,t,offset,speed,kind,light}
  sim:       { funds, population, jobs, services, satisfaction, demand:{} },
  time:      { t: 0.5, day: 1, weather: 'clear', tps: 0 },   // t∈[0,1) 0=midnight
}
```

`tileKey = `${ix},${iz}``. Position is always a `{x,y,z}` in metres (y from terrain).

## 3. Module lifecycle contract

Every module default-exports a **class** (or factory returning an object) implementing:

```js
class MyModule {
  name = 'terrain';             // must match folder
  async init(world, ctx) {}     // build scene objects; register into world.<name>
  update(dt, world) {}          // per-frame; dt in seconds
  onEvent(name, payload) {}     // optional; subscribe to world events
  dispose() {}                  // remove objects, dispose GPU resources
  showcase(scene, world, ctx) {}// stage a scene of ONLY this module for the gauntlet
  stats() { return { drawCalls: 0, notes: '' }; }  // optional
}
```

**`ctx`** (provided by core, read-only to modules) exposes:
`{ three, scene, camera, renderer, events, rng, clock, assets, perf, world, canvas }`.

**Showcase** is mandatory: it must render a representative, lit scene using only that
module (plus a neutral ground + one light from core) so the critic can score the module
in isolation at several times of day and zooms.

## 4. Failure isolation (`core/registry.js`)

- The registry wraps `init` and every `update` in `try/catch`.
- On throw: the module is marked `state='failed'`, removed from the update set, its
  message + stack pushed to `window.__MODULE_ERRORS__ = [{module, at, message, stack}]`,
  and a red tag is drawn in the UI. **The game keeps running with the rest.**
- `init` failures are non-fatal too (module is skipped). Boot never throws.
- `dispose` is best-effort.

## 5. Events (namespaced, via `core/event.js`)

`events.on('a:b', cb)`, `events.emit('a:b', payload)`, `events.off(...)`. A module only
emits its own namespace. Key events:

| Module | Emits |
|---|---|
| terrain | `terrain:ready`, `terrain:surface` |
| environment | `env:sun {dir,intensity}`, `env:weather {w}` |
| roads | `roads:ready`, `roads:edge {edge}` |
| zoning | `zoning:changed {tile,use,density}` |
| buildings | `buildings:placed {b}`, `buildings:ready` |
| props | `props:ready` |
| traffic | `traffic:ready`, `traffic:density {d}` |
| effects | `effects:ready` |
| simulation | `sim:tick {stats}`, `sim:funds {f}` |
| tools | `tools:activated {name}`, `tools:placed {kind,tile}` |
| ui | `ui:action {name,payload}` |
| audio | `audio:ready` |

Consumers subscribe in `init`. No module reaches into another module's internals.

## 6. Per-module public API + budget

Budgets are draw-call ceilings (merged/instanced). The `perf` tracker reports actuals.

| Module | Must expose (public) | Emits | Draw-call ceiling |
|---|---|---|---|
| **terrain** | `getHeight(x,z)`, `getNormal(x,z)`, `world.terrain`, water mesh | `terrain:ready` | 40 |
| **environment** | `setTimeOfDay(t)`, `setWeather(w)`, `sun` (DirectionalLight w/ shadows), `sky` | `env:sun`,`env:weather` | 10 |
| **roads** | `addEdge(e)`, `graph {nodes,edges}`, `samplePoint(edge,t)`, `nearestEdge(x,z)` | `roads:ready` | 300 |
| **zoning** | `setZone(tile,use,density)`, `getZone(tile)`, `overlay` (color mesh) | `zoning:changed` | 20 |
| **buildings** | `buildFromZoning()`, `addBuilding(b)`, `setLit(on)`, `world.buildings` | `buildings:placed`,`buildings:ready` | 600 |
| **props** | `addProp(type,pos)`, `setCount(type,n)`, `world.props` | `props:ready` | 400 |
| **traffic** | `setDensity(d)`, `spawn(kind)`, `world.traffic` | `traffic:ready` | 300 |
| **effects** | `composer`, `setPreset('day'|'dusk'|'night')` | `effects:ready` | 0 (post) |
| **simulation** | `tick(dt)`, `world.sim`, `getFunds()` | `sim:tick` | 0 |
| **tools** | `setTool(name)`, `getState()` | `tools:activated`,`tools:placed` | 30 |
| **ui** | `showPanel(name)`, `setStats(s)`, `setTimeControls(t)` | `ui:action` | 0 (DOM) |
| **audio** | `setMood(m)`, `setMuted(b)`, `resume()` | `audio:ready` | 0 (audio) |
| **demo** | `build(world, ctx)` — composes the full city | — | (sum) |

Total ceiling ≈ 1700 nominal; the running demo city must land **≤ 1500** in practice
(most modules run well under ceiling; `perf` enforces it).

## 7. The main loop & app handle (`core/loop.js`, `src/main.js`)

Loop: `rAF → dt → clock.advance(dt) → for each live module: module.update(dt, world) →
renderer.render`. FPS + `renderer.info.render.calls` are sampled every frame into `perf`.

`main.js` must expose the **app handle** used by the screenshot tool and UI:

```js
window.__APP__ = {
  world, modules, renderer, camera, scene,
  setCameraPreset(name),   // 'aerial'|'street'|'skyline'|'close'|'orbit'
  setTimeOfDay(t),         // 0..1
  setWeather(w),
  setModule(name),         // switch to a module's showcase scene
  stats(),                 // {fps, drawCalls, triangles, moduleStates, errors}
};
window.__APP_READY__ = true;   // set after first rendered frame + all init settled
```

Camera presets are fixed, reproducible positions (no RNG) so screenshots are comparable.

## 8. Integration rules (who may touch what)

- **Builder** (one per module): edits **only** `src/<module>/`. To change core, add a
  request to `src/<module>/INTEGRATION.md` (what + why + proposed diff). Never edit core.
- **Integrator** (single role, between waves): the **only** actor that edits
  `src/core/**`, `src/main.js`, `index.html`, `vite.config.js`, `tools/**`, `package.json`.
  Applies builders' core-change requests, fixes seams, keeps the app loadable.
- **Critic** (per module, then whole-game): **read-only + screenshots**. Writes a score
  (0–10) and a ranked issue list to `docs/critics/<module>-r<n>.md`. Writes **no code**.
- **Dev server stays up** at all times (`npm run dev`, port 5173); the app must stay
  loadable because critics are screenshotting it continuously.

## 9. Verification loop

`tools/shot.mjs` (Puppeteer) → load `http://localhost:5173/` → wait for
`window.__APP_READY__` → optionally `setModule(name)` / `setCameraPreset(p)` /
`setTimeOfDay(t)` → warm up N frames → **`Page.screenshot` PNG** + **JSON log**
`{consoleErrors, moduleErrors: __MODULE_ERRORS__, fps, drawCalls, triangles,
moduleStates}`. Output to `shots/<name>.png` + `shots/<name>.json`. Every module ships a
showcase so the gauntlet can score it in isolation. **No claim without a screenshot.**

## 10. Scoring rubric (critic)

10 = indistinguishable from CS2 in this frame · 8.5 = AAA with nits · 7 = good indie ·
5 = programmer art. **Pass = ≥ 8.5 with zero console/module errors.** Below → ranked
issue list back to the builder (≤ 4 rounds). Scores + open issues persist to
`docs/STATUS.json`; the loop always resumes from the **weakest** module.
