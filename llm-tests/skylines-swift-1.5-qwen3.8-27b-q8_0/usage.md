# Skylines — Usage

A city-builder rendered with **Three.js** as plain ES modules. It generates a
deterministic procedural city (terrain → roads → zoning → buildings → props →
traffic) and runs a live simulation (economy, day/night, weather) on top of it,
with a DOM HUD (stats, minimap, time controls, notifications) layered over the
canvas.

Everything is driven by a **single seed** (default `1337`): the same seed always
produces the same city, so links are shareable and reproducible.

---

## 1. Prerequisites

- **Node.js ≥ 20.19** (the tools use built-in `fetch` + `WebSocket`).
- A web browser (any modern Chromium/Firefox/Safari).
- No build step. There is no Vite/Webpack — a tiny dependency-free static server
  serves the ES modules and a browser import map resolves `three` from
  `node_modules`.

---

## 2. Starting the project

From the project root:

```bash
npm install        # once — installs `three`
npm run dev        # start the server  (== `node tools/serve.mjs`)
```

The server listens on **http://localhost:5173/** by default.

- `PORT` and `HOST` env vars override the bind address, e.g.
  `PORT=8080 npm run dev`.
- Health check: `curl http://localhost:5173/healthz` → `ok`.
- Keep the server running while you use the app — it is a plain static server
  and needs no watch/rebuild; refresh the browser after editing source.

Open the app:

```
http://localhost:5173/
```

You land on the composed city at the **orbit** camera, mid-morning, in **clear**
weather, with the game clock already playing (1×).

---

## 3. URL parameters

All configuration is via query string, so any view is a shareable link.

| Param       | Values                                   | Default | Meaning |
|-------------|------------------------------------------|---------|---------|
| `seed`      | any integer                              | `1337`  | Procedural seed. Same seed ⇒ same city. |
| `cam`       | `aerial` `orbit` `skyline` `street` `close` | `orbit` | Camera preset (see §4). |
| `time`      | `0.0`–`1.0`                              | —       | Time of day. `0`=midnight, `0.25`=sunrise, `0.5`=noon, `0.75`=sunset. |
| `weather`   | `clear` `cloudy` `rain` `fog`            | `clear` | Sky/atmosphere preset. |
| `module`    | a module name (see §6)                   | —       | Isolated single-module showcase view (used for QA). |

Examples:

```
http://localhost:5173/                                  # default day city, orbit
http://localhost:5173/?cam=aerial&time=0.5              # noon, top-down
http://localhost:5173/?cam=orbit&time=0.93              # night city, orbit
http://localhost:5173/?cam=street&time=0.78&weather=fog # foggy dusk street level
http://localhost:5173/?seed=42&cam=skyline              # a different city, skyline
```

---

## 4. Camera presets

The camera has **no mouse orbit/pan/zoom** — it is placed by a preset. Use the
`cam` URL param, or drive it at runtime with `setCameraPreset(name)` (see §7).

| Preset    | Position            | Look-at     | View |
|-----------|---------------------|-------------|------|
| `aerial`  | `[0, 430, 470]`     | city center | Top-down overview of the whole landmass. |
| `orbit`   | `[150, 95, 150]`    | city center | Classic 3/4 orbit — the default hero view. |
| `skyline` | `[200, 42, 300]`    | skyline     | Low, looking up at the tower cluster. |
| `street`  | `[8, 9, 42]`        | down a road | Eye-level street view. |
| `close`   | `[26, 15, 28]`      | a block     | Tight close-up of facades. |

---

## 5. In-app controls

### Time & weather bar (bottom center)
- **Play / pause** button — starts playing; toggles the clock.
- **1× / 2× / 4×** — simulation speed (1× = one in-game day per 60 real seconds).
- **Clock** — live `HH:MM` readout of the current time of day.
- **Weather** button — cycles `clear → cloudy → rain → fog`, with an icon.

### City stats panel (top left)
Live economy: **Funds, Population, Jobs, Happiness, Services, Day**, each with a
trend arrow (▲/▼), plus a treasury **sparkline** and a **Power / Water /
Sanitation** coverage breakdown. Hover any row for a detail tooltip.

### City Map / minimap (top right)
Height-shaded terrain, the road network, building footprints by class, a north
marker, and a live **camera position + view-frustum wedge**.

### Notifications (top right, below the minimap)
Toasts for the initial **city charter**, **daily reports** on each new day,
**population milestones**, and simulation **alerts** (budget / services / unrest).

---

## 6. Building & editing tools

A toolbar sits in the bottom-left. Select a tool by clicking it or pressing its
hotkey, then use the mouse on the canvas.

| Tool | Hotkey | How to use |
|------|--------|------------|
| Select     | `1` / `Esc` | Default. No placement. |
| Residential | `2`       | Click to zone one tile; **drag** to paint a run of tiles. |
| Commercial  | `3`       | Same as residential. |
| Industrial  | `4`       | Same as residential. |
| Road        | `R`       | **Click** to set the start point, **click** again to set the end point — a road segment is laid between them. |
| Bulldoze    | `B`       | Click a building to remove it. |

- Zoning paints onto a fixed **8 m grid** (the zone overlay is a 35×35 cell
  field). Dragging interpolates between tiles so fast drags leave no gaps.
- Every action emits a `tools:placed` event and updates the world live.
- Keyboard shortcuts are ignored when `Ctrl`/`Meta`/`Alt` is held, so browser
  shortcuts never clash with the game.

---

## 7. Programmatic API

The app exposes `window.__APP__` for scripting / the screenshot tool. A frame is
ready when `window.__APP_READY__ === true`.

```js
const A = window.__APP__;
A.setCameraPreset('aerial');   // one of: aerial | orbit | skyline | street | close
A.setTimeOfDay(0.93);          // 0..1  (0.93 ≈ 22:19, night)
A.setWeather('fog');           // clear | cloudy | rain | fog
A.setSpeed(0);                 // simulation speed in sim-hours/real-second (0 = paused)
A.setModule('terrain');        // switch to an isolated module showcase (or null for the full city)
A.stats();                     // { fps, drawCalls, triangles, budget, over, seed,
                               //   timeOfDay, weather, moduleStates, moduleErrors }
```

`window.__MODULE_ERRORS__` (an array) holds any per-module init/update errors —
an empty array means every module is healthy.

---

## 8. Modules (the feature set)

The game is composed of 13 modules, registered in dependency order. Each is
independently loadable via `?module=<name>` for QA/showcase use:

| Module       | What it provides |
|--------------|------------------|
| `terrain`     | Procedural valley city: heightfield, irregular coastline, water with depth gradient, cliffs. |
| `environment` | Atmospheric sky (sun/moon, scattering), clouds, fog, day/night cycle, weather. |
| `roads`       | Road network: grid + curved boulevards, intersections, crosswalks. |
| `zoning`      | Residential / commercial / industrial zone field driving where buildings go. |
| `buildings`   | ~300 varied buildings with facades, windows, and rooftop detail. |
| `props`       | Street furniture: trees, streetlights, and other props. |
| `traffic`     | Animated vehicle traffic and sidewalk pedestrians. |
| `simulation`  | The living economy: funds, population, jobs, happiness, services, day counter, alerts. |
| `tools`       | The build/zone/road/bulldoze interaction layer + ghost previews. |
| `effects`     | Post-processing: bloom, color grade, vignette, atmospheric depth. |
| `ui`          | The DOM HUD: stats, minimap, time/weather bar, toasts, tooltips. |
| `audio`       | Ambient audio (unlocks on first interaction). |
| `demo`        | The composed default city scene. |

> Note: `ui` and `audio` have no meaningful isolated 3D showcase; their
> `?module=` views are minimal by design.

---

## 9. Screenshot / verification tool

`tools/shot.mjs` drives a headless browser over the Chrome DevTools Protocol,
loads the app, applies a module/camera/time/weather, lets it settle, and writes
a **PNG** plus a **JSON** log (console errors, module errors, fps, draw calls,
triangle count, per-module states, budget).

```bash
# A night-city orbit shot at 1280x720
node tools/shot.mjs --name demo-night --module null --cam orbit --time 0.93 \
      --width 1280 --height 720 --seed 1337

# An isolated terrain showcase
node tools/shot.mjs --name terrain --module terrain --cam orbit
```

Useful flags: `--name`, `--module` (module name or `null`), `--cam`, `--time`,
`--weather`, `--seed`, `--width`, `--height`, `--warm` (settle frames; default
150). Env: `SHOT_BASE` (default `http://localhost:5173`), `SHOT_OUT` (default
`./shots`), `CHROME_PATH`, `SHOT_TIMEOUT` (seconds, default 120).

Output: `shots/<name>.png` and `shots/<name>.json`. A clean run exits `0` and
logs `{"ready":true,"consoleErrors":0,"moduleErrors":0,...}`.

> The tool freezes the game clock (`setSpeed(0)`) right after setting the time,
> so a `--time 0.93` shot is captured at exactly `0.93` and does not drift during
> the settle frames.

---

## 10. Example workflows

### A. Just look at the city
1. `npm run dev`
2. Open `http://localhost:5173/`.
3. Try the camera presets: `?cam=aerial`, `?cam=skyline`, `?cam=street`.
4. Scrub the time of day: `?time=0.5` (noon) → `?time=0.78` (dusk) →
   `?time=0.93` (night).
5. Change the weather with the button in the bottom bar (or `?weather=fog`).

### B. Build on the map
1. Open `http://localhost:5173/?cam=aerial&time=0.5`.
2. Press **2** (Residential) and drag across empty ground to paint a neighborhood.
3. Press **3** or **4** to add commercial / industrial zones nearby.
4. Press **R**, click a start point and an end point to lay a road between them.
5. Press **B** and click a building to bulldoze it.
6. Press **1** or **Esc** to return to Select.

### C. Capture a marketing / QA frame
1. `npm run dev` (server must be up).
2. Run:
   ```bash
   node tools/shot.mjs --name hero-night --module null --cam orbit \
         --time 0.93 --width 1280 --height 720 --seed 1337
   ```
3. Open `shots/hero-night.png`; check `shots/hero-night.json` for
   `consoleErrors: []`, `moduleErrors: []`, and `timeOfDay ≈ 0.93`.

### D. Drive it from the browser console
1. Open `http://localhost:5173/` and the DevTools console.
2. Run:
   ```js
   const A = window.__APP__;
   A.setTimeOfDay(0.93);  A.setWeather('clear');
   A.setCameraPreset('skyline');
   A.stats();             // inspect fps / draw calls / module states
   ```

---

## 11. Notes & limits

- **Deterministic**: layout comes from a seeded PRNG (`ctx.rng.fork(name)`), not
  `Math.random`, so a given `seed` always yields the same city.
- **Performance** depends on the renderer. Under software rendering (SwiftShader)
  expect low single-digit-to-low-double-digit fps; the screenshot tool's
  `over`/low-fps fields reflect that and are not real-world GPU numbers.
- **No camera mouse controls** are wired — the camera is preset-driven (URL
  `?cam` or `setCameraPreset`).
- A module that throws during init/update is isolated (it won't crash the app);
  the failure is recorded in `window.__MODULE_ERRORS__`.
