# Module Builder Briefing — read this before writing any code

You are a **module builder** on Skylines, a Cities: Skylines II–class city
builder in Three.js (r186) + Vite. The bar is AAA: photographic PBR materials,
physically plausible sun/sky/shadows, atmospheric depth, believable roads and
traffic. Never programmer art.

`ARCHITECTURE.md` is the binding contract — read it. This file gives the
universal rules that apply to every module. Your specific task is in your prompt.

## Hard rules

1. **Own only your folder** (`src/<your-module>/`). Never edit `src/core/` or any
   other module's folder, nor `index.html`, `vite.config.js`, `package.json`,
   `src/main.js`. If you need a change to core or the bootstrap, DO NOT make it —
   note it in your report and the integrator will apply it.
2. **Keep the app loadable.** The dev server is running at
   `http://127.0.0.1:5173/` and other agents are screenshotting it continuously.
   Never leave a syntax error or broken import behind.
3. **Failure isolation.** Your module must never throw on boot even if its
   dependencies are absent — core wraps each module in try/catch. Still, guard
   against missing pieces of `world`.
4. **Determinism.** Use the seeded RNG: `world.rng()` (mulberry32). NEVER use
   `Math.random()` for anything that affects world layout, city, or materials.
   The same seed must produce the same city every time.
5. **Units.** Metres, +Y up. Time of day is seconds since midnight (`meta.timeOfDaySec`,
   default 09:00). Tick is `world.meta.tick`; sim dt ≈ 1/20 s via `clock.simDtSec`.
6. **Performance budget** (see ARCHITECTURE.md for your module's draw-call share;
   whole game target ≥50 fps / ≤1500 draw calls at 1080p). Prefer instancing
   (`THREE.InstancedMesh`) over thousands of separate meshes.
7. **Asset policy.** CC0 only (Poly Haven / ambientCG) or fully **procedural**
   PBR textures generated at runtime from the seed. Procedural is preferred —
   it needs no network fetches and stays deterministic. Do not block boot on a
   network fetch; generate fallbacks.
8. **Report real numbers.** Never inflate quality, FPS, or draw-call counts.

## Public API every module exports (replace the stub)

```js
export const id = 'your-module-id';          // matches folder name

// called once after registration. world.scene/camera/renderer are ready.
export function init(world) { ... }

// called every frame with sim dt. Keep it cheap; do heavy work rarely.
export function update(dtSec, world) { ... }

// optional: render your module's showcase into a DOM container (used by demo).
export function showcase(container) { ... }
```

Access scene/camera via `world.scene`, `world.camera`, `world.renderer`.
Listen/emit bus events: `import { bus } from '../core/index.js'`. See
ARCHITECTURE.md "Events" for names.

## How to verify your work (you MUST do this)

IMPORTANT: The inference gateway in this environment has NO vision (`mmproj` is
missing). **Do NOT use the Read tool on image/PNG files — it will 500 and kill
your task.** "Looking" at renders is done programmatically instead.

1. Edit `src/<your-module>/index.js`.
2. Screenshot the app:
   ```
   cd /sandbox/skylines-dsv4flash
   node tools/screenshot.mjs --preset overview --dir docs/shots/<your-module>
   ```
   (or `--preset night`, `--preset sunset`). If your module needs a custom camera
   angle to be seen, note it in your report so the demo/integrator can add a preset.
3. **Analyze the PNGs objectively** (this is how you "look"):
   ```
   node tools/analyze.mjs docs/shots/<your-module>/*.png
   ```
   It prints luminance mean/std, darkness, black-fraction, distinct-color count,
   edge energy and a verdict (`OK` vs `BLACK/EMPTY FRAME` / `FLAT/UNIFORM` /
   `NO STRUCTURE/EDGES`). A shot must be `OK`, not all-black, not flat. Use the
   numbers to reason about your render: e.g. a night preset SHOULD be dark but
   not 98% black; an overview should have structure and edge energy.
4. Confirm `docs/shots/<your-module>/report.json` has zero console errors and
   sane drawCalls (whole game ≤1500; your module's share is in ARCHITECTURE.md).
5. Iterate until it passes metrics and the code is obviously correct.

If your module has no visible output yet (e.g. pure simulation/audio), at least
confirm zero console errors in `docs/shots/<your-module>/report.json` and that
the app still boots.

## Report back

End with: what you built, the visual/sim result, screenshot paths,
real fps/drawCall numbers from report.json, any core/bootstrap changes you
need (for the integrator), and anything still missing.
