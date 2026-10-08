# Skylines — Architecture

A Cities: Skylines II–class city builder in **Three.js + Vite**, plain ES modules.
The bar is AAA: photographic PBR materials, physically plausible sun/sky/shadows,
atmospheric depth, a living city at night, believable roads and traffic. Never programmer art.

## Non-negotiable principles

1. **Architecture before features.** This document is the contract. Every module
   implements exactly the public API below. No module depends on another module's internals.
2. **One folder per subsystem.** A builder agent owns only its folder and nothing else.
   Core changes go through a single integrator agent.
3. **Failure isolation.** If any module throws, the game keeps running. Each module is
   loaded behind a boundary (`loadModule` + try/catch); a broken module logs an error,
   shows its showcase fallback or nothing, and never takes down the app.
4. **Isolated RNG.** All randomness comes from seeded PRNGs (see Units & Determinism).
5. **Verification before claims.** Nothing is "done" until it is screenshotted at several
   times of day and the screenshots are actually looked at.

## Stack

- Three.js latest release (`three`, r160+) — plain ES modules, no React.
- Vite (dev server + build). All game code in `src/`, plain `.js` ESM, no TS.
- No physics engine, no nav-mesh library. Everything hand-rolled (small, deterministic).

## Module layout

```
src/
  core/        # shared world data model, event bus, clock, RNG, registry, module loader
  terrain/     # heightfield mesh, chunking/LOD, ground PBR material, elevation paint
  environment/ # sun/sky dome (physically based), clouds, atmosphere, weather, time-of-day
  roads/       # road spline network, intersections, surface mesh, road markings/lights
  zoning/      # zone grid, density levels, growth rules driving building placement
  buildings/   # procedural + instanced building meshes, windows/roofs, day-night lighting
  props/       # trees, street furniture, fences, decorative instancing (CC0)
  traffic/     # agent cars/peds on the road network, lanes, simple flow control
  effects/     # particles (rain/snow/dust), bloom/SSAO/post pipeline, water reflections
  simulation/  # fixed-tick economic/citizen sim feeding zoning+traffic demand
  tools/       # camera controls, build/zone/demolish brushes, UI state hooks
  ui/          # HTML overlay: budgets, zone tool palette, time controls, info toasts
  audio/       # procedural ambient + traffic audio (WebAudio), CC0 samples
  demo/        # the showcase city built on all modules; also per-module showcases
```

`scripts/` — non-game tooling (screenshot runner, status aggregator). `tools/` — dev helpers.

## Shared world data model (`src/core/`)

Everything talks through `world` — a plain serializable object owned by core.

```js
// src/core/world.js  — the single source of truth
export const world = {
  meta:     { name:'', seed: 0, tick: 0, timeOfDaySec: 9*3600, paused:false },
  clock:    null,            // Clock-like { dtSec, simDtSec }
  rng:      null,            // seeded PRNG (mulberry32), see determinism
  terrain:  { heightAt(x,z):number, normalAt(x,z):THREE.Vector3, size:2048 },
  zones:    [],              // grid cells -> {x,z,density,type,growth}
  buildings:[],              // {id,x,z,type,w,h,density}
  roads:    [],              // network segments
  props:    [],
  agents:   [],
  budget:   {},
  stats:    { fps, drawCalls, ... },
};
```

Core exposes (public API):

| export | purpose |
|---|---|
| `createWorld({seed})` | build empty world + seeded RNG |
| `bus` (`on/off/emit`) | global typed event bus |
| `tick(now)` | advance clock; emit `sim-tick`, `frame` |
| `registerModule(mod)` | register a module (see isolation) |
| `startModules()` / `stopModules()` | lifecycle |
| `setTimeOfDay(sec)` / `setWeather(id)` | environment controls (used by screenshot tool) |

**Event bus contract** (all events flow through core `bus`; namespaced):
- `sim-tick` — each simulation tick, payload `{tick, dt}`.
- `frame` — each rendered frame `{dtSec, fps, drawCalls}`.
- `timeofday:changed` `{sec}` · `weather:changed` `{id}`.
- `road:add/remove` `{segment}` · `building:add/remove` `{b}` · `zone:changed` `{cell}`.
- `tool:mode` `{mode}` · `ui:toast` `{text,level}`.
- `screenshot-ready` — fired once per frame the app is idle enough to capture.
- `module:error` `{name, error}` — emitted when any module fails (isolation).

## Public API per subsystem

Each module's folder exports exactly these names. Missing/extra exports are a contract breach.

### terrain
```js
export const id = 'terrain';
export function init(world, renderer)            // build heightfield mesh + material
export function generate(world, opts)            // (re)generate terrain from seed; emits none
export function heightAt(x, z)                   // metres (world.heightAt delegates here)
export function normalAt(x, z)
export function update(dt, world)                // chunk/LOD streaming; no-op per frame if static
export function showcase(container)              // staged scene for the screenshot tool
```

### environment
```js
export const id = 'environment';
export function init(world, renderer)
export function setTimeOfDay(sec)                // 0..86400; drives sun/sky/lighting
export function setWeather(id)                   // clear|overcast|rain|snow|fog
export function update(dt, world)
export function showcase(container)
```
Owns: `THREE.HemisphereLight`, key directional light (shadow-caster), sky dome shader,
cloud layer, fog, atmosphere scattering.

### roads
```js
export const id = 'roads';
export function init(world, renderer)
export function buildNetwork(world, layout)      // deterministic network from seed/layout
export function addRoad(from, to, opts) / removeRoad(id)
export function surfaceAt(x,z) → {onRoad, lane, dir}
export function update(dt, world)
export function showcase(container)
```

### zoning · buildings · props · traffic · simulation · tools · ui · audio

Each exports `id`, `init(world,renderer)`, `update(dt,world)` and a module-specific
`showcase(container)` plus its own build/generate entry points. Exact signatures are
defined inline in each builder's first commit; the **contract to hold is:** every module
must (a) be loadable standalone via `loadModule('x')`, (b) emit only core-bus events,
(c) never write outside its folder except through integrator-approved core changes.

### demo
```js
export const id = 'demo';
export function init(world, renderer)            // compose every module into the showcase city
export function showcase(container)
```
Runs all modules with a curated seed + camera presets for whole-game shots.

## Units

- **Metres**, **+Y up**. World XZ is ground plane. `terrain.size` default 2048 m.
- One world unit = one metre. Building heights, road widths, agent speeds in m and m/s.
- Time of day in seconds since midnight; default start 09:00 (morning).
- Angle convention: degrees for user-facing UI; radians in math.

## Determinism

- **Only seeded RNG.** `world.rng = mulberry32(seed)` lives in core. Any module needing
  random numbers must draw from `world.rng` — never `Math.random`.
- The demo city, terrain, road layout, zoning pattern and all procedural generation are a
  pure function of the seed. Screenshots at a fixed time-of-day are reproducible run to run.
- Simulation ticks are fixed-step (`simDtSec`, default 1/20) so results don't depend on fps.

## Performance budget

- **≥50 fps @ 1080p** on the target machine, **≤1500 draw calls** typical frame.
- Budgeted shares: terrain ~250 calls, environment ~20, roads ~300, buildings ~500,
  props ~200 (instanced), traffic ~80, effects ~100, UI negligible.
- Rules to stay inside the budget:
  - Instancing for anything repeated (windows, trees, cars, streetlights).
  - Frustum + distance culling; LOD for terrain and big buildings.
  - Reuse materials; avoid per-object uniforms where a shared texture atlas works.
  - Shadows only from the key light; shadow-map size capped; objects beyond a radius
    are excluded from the shadow pass.
- The screenshot runner records `fps` and `drawCalls` (via `world.stats`, updated in core)
  into each JSON log. **A frame with fps < 40 or drawCalls > 1500 fails the perf gate.**

## Asset policy (CC0 only)

- Textures/HDRIs: **Poly Haven** or **ambientCG**, both CC0. Or fully procedural (generated
  at runtime) — preferred for reproducibility and zero network dependency at load.
- No copyrighted/model-scanned assets, no non-CC0 downloads.
- Any downloaded asset is cached in `assets/` with a `SOURCES.md` noting author + licence.
- Procedural fallback is the default: PBR maps (albedo/normal/R/AO) generated at runtime.

## Module isolation & loading

Core loads modules through:

```js
export async function loadModule(id){
  try { const m = await import(`../${id}/index.js`); return m; }
  catch(e){ bus.emit('module:error',{id, error:e}); return null; }
}
```

If a module fails to load or throws in `update`, core catches it, emits `module:error`,
marks the module dead this frame, and continues. One broken module never stops the loop.

## Verification loop (the gauntlet)

`scripts/screenshot.mjs` runs **headless Chromium**:
1. Loads the app at `http://localhost:5173/`.
2. Waits for `document.readyState==='complete'` and the app's ready signal.
3. Sets a camera preset + time-of-day via query params or injected API (`?cam=overview&tod=17000`).
4. Writes `PNG` (1080p) + `JSON` log: console errors, fps, draw calls, module error events.
5. Each module's `showcase(container)` stages a representative scene for isolated shots.

Presets: `overview` (high, far), `street` (ground level on a road), `night` (same, 02:00),
`sunset` (18:45). Critic agents screenshot at several presets/times and zoom levels.

## STATUS persistence

Critics write real scores + ranked open issues to `docs/STATUS.json`. The orchestrator
resumes each iteration from the **weakest scoring module**, never from scratch. Scores are
real; failed rounds and missing features are recorded, never inflated.

## Agent roles (the ultracode orchestration)

- **Builder agents**: one per module, own only their folder, implement its public API.
- **Integrator agent**: the only agent allowed to touch `src/core/`; applies builders'
  core-change requests and fixes seams between modules each wave.
- **Critic agents**: brutal AAA art directors. Write no code. Screenshot at several
  times/zooms, check API contract + console errors + perf, score 0–10 vs Cities: Skylines II:
  `10` indistinguishable · `8.5` AAA with nits · `7` good indie · `5` programmer art.
  **Pass = ≥8.5 and zero console errors.** Below that the builder gets the ranked issue
  list and iterates, up to 4 rounds.
- **Whole-game critic**: final gate over the demo city. Then blind A/B judges compare our
  screenshots against CS2 references (labels shuffled) and say which looks better.

## How a module ships

1. Builder writes `src/<mod>/index.js` + any support files, exporting its public API.
2. Builder adds a `showcase(container)` staging a representative scene.
3. Screenshot the showcase at ≥3 times of day; **look** at each PNG; iterate until clean.
4. Critic scores it. Integrator merges core changes between waves.
5. Record result + issues in `docs/STATUS.json`.
