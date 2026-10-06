# Turbo Kart — Module Contracts

You are building ONE module of a Three.js kart-racing game (a Mario Kart-style clone).
Five modules are built in parallel by different agents. **Your code MUST match the
interfaces below exactly**, because `js/main.js` (already written, do NOT modify it)
wires everything together.

## Ground rules

- Vanilla JS ES modules. Import three via `import * as THREE from 'three'`
  (an importmap resolves it to `./js/lib/three.module.js` — the file exists, do not
  re-download it). Import shared constants via `import { CONFIG } from './config.js'`.
- **No external assets**: no texture images, no fonts, no audio files, no CDN requests.
  Everything procedural (geometry, canvas-generated textures, WebAudio synthesis).
- Write **only the files assigned to you**. Do not create or edit any other file.
- All world units are meters. Y is up. The track is flat at y=0.
- Karts face +Z when `heading === 0`; heading rotates counter-clockwise viewed from
  above. Forward vector = `(sin(heading), 0, cos(heading))`.
- Code must be clean and run at 60fps: merge/instance repeated geometry (trees,
  rocks, boxes), keep draw calls low, no per-frame allocations in hot loops where
  avoidable.
- No `console.log` spam. No comments except where a WHY is non-obvious.
- **Verification (required before you finish)**: every `.js` file you write must
  pass `node --check` as an ES module:
  ```bash
  for f in <your files>; do cp "$f" /tmp/check.mjs && node --check /tmp/check.mjs && echo "OK $f"; done
  ```

## The world at a glance

- 8 karts (index 0 = player), 3 laps, closed circuit, flat ground, procedural world.
- Game states: `menu` → `racing` (includes 3-2-1-GO countdown) → `results`.
- `js/main.js` calls, per frame while racing:
  1. `aiControllers[i].update(dt, karts, race)`
  2. `karts[i].update(dt, track, particles, canDrive)` for every kart
  3. `race.update(dt)`
  4. `camRig.update(dt, karts[0], track, gameState, race)`
  5. `particles.update(dt)`
  6. `hud.update({...})`
- While in `menu`: `karts[i].updateIdle(dt)` instead of `update(...)`, camera in
  'menu' mode, HUD shows menu.

---

## Module A — Scene & Environment
**Files: `js/scene.js`, `js/environment.js`**

### `js/scene.js`
```js
export function createScene(containerEl)
// Returns: { scene, camera, renderer, render() }
```
- `renderer`: `THREE.WebGLRenderer({ antialias: true })`, `shadowMap.enabled = true`
  (PCFSoft), sRGB output, pixel ratio capped at 2, sized to container, added to it.
- `camera`: `new THREE.PerspectiveCamera(60, aspect, 0.5, 800)` (position is managed
  by the camera rig, not by you).
- `scene`: background gradient sky. Build a large inverted sphere (radius ~700) with
  a vertical gradient shader (top `CONFIG.COLORS.SKY_TOP` → bottom `CONFIG.COLORS.SKY_BOTTOM`),
  `side: THREE.BackSide`, `fog: false` on the material. Add `scene.fog =
  new THREE.Fog(CONFIG.COLORS.FOG, 120, 650)`.
- Lighting: one directional "sun" (`0xfff3d6`, intensity ~1.6) casting shadows
  (shadow camera must cover ~±180 of the track area, 2048 map), one hemisphere light
  (sky/ground, ~0.7), subtle ambient.
- `render()`: handles window resize (updates renderer size + camera aspect) and calls
  `renderer.render(scene, camera)`. The resize listener attaches itself.

### `js/environment.js`
```js
export function buildEnvironment(scene, track)
```
`track` is the object returned by `buildTrack` (see Module C) — use
`track.centerlineXZ` (Float32Array of densely sampled x,z pairs, length
`track.sampleCount`) and `track.TRACK_WIDTH` to keep decorations OUT of the road
corridor. A point is "road" if within `TRACK_WIDTH/2 + CURB_WIDTH + 6` of the
centerline.
- Ground: large plane (±600) in `CONFIG.COLORS.GRASS` with a subtle procedural
  canvas texture (two green tones, grass noise), repeat it.
- Trees: ~140 (trunk cylinder + 2-3 cone/sphere foliage), placed randomly inside
  ±280 of origin but never in the road corridor. Use `THREE.InstancedMesh` for
  trunks and foliage (2 instanced meshes total). Varied scale/rotation.
- Rocks: ~40 small dodecahedrons, instanced. Flowers: ~80 tiny colored
  (pink/white/yellow) crosses or spheres, instanced, in grass only.
- Clouds: ~12 soft white flattened sphere clusters floating at y 60-120, slow
  drift is NOT required (static is fine).
- Mountains: 8-12 large cones in the far distance (radius 200-450 from origin),
  bluish-grey, half-fogged — sell the horizon.
- A couple of lakes (flat blue circles) in the far field, optional.
- Everything added to `scene`. No return value needed (return void).

---

## Module B — Kart model & physics
**File: `js/kart.js`**

```js
export function createKart(index, isPlayer)
// Returns a kart object:
{
  group: THREE.Group,      // the whole kart, added to scene by main.js
  isPlayer: boolean,
  index: number,
  state: {
    x: number, y: number, z: number,
    heading: number,       // radians
    speed: number,         // m/s, signed (negative = reverse)
    drifting: boolean,
    driftDir: 1 | -1,
    driftCharge: number,   // seconds of current drift
    boostTime: number,     // seconds of boost remaining (mushroom/mini-turbo)
    starTime: number,      // seconds of star remaining
    spinTime: number,      // seconds of spin-out remaining
  },
  item: null | 'mushroom' | 'shell' | 'banana' | 'star',  // owned, unused item
  wantUseItem: boolean,   // set by AI (or player) to fire item
  t: number,              // 0..1 progress along track, owned by race.js (read by AI)
  lap: number,            // owned by race.js
  input: { throttle: 0|1, brake: 0|1, steer: -1|0|1, drift: 0|1, useItem: 0|1 },
  reset(pos: THREE.Vector3, heading: number, t: number),
  setColor(hex: number),
  update(dt, track, particles, canDrive),
  updateIdle(dt),
  boost(duration, extraSpeed),   // item hooks (race.js calls these)
  spinOut(),
  setStar(on: boolean),
}
```

### Model (procedural, no assets)
- Chassis: rounded box (BoxGeometry is fine, slightly beveled look via scale),
  color from `setColor` (default `CONFIG.COLORS.KARTS[index]`).
- Driver: helmet sphere (white) + torso, seated, small visor.
- 4 wheels: cylinders (dark grey, white hubcap), front pair visually steers with
  `input.steer`, all wheels spin with speed. Store wheel refs in `kart.wheels`.
- Small rear spoiler. Total kart ~2.2m long, 1.8m wide, 1.2m tall. y=0 is the
  ground; wheel bottoms at y≈0.
- `group` position = `(state.x, 0, state.z)`, `group.rotation.y = heading`.

### Physics (in `update(dt, track, particles, canDrive)`)
- If `spinTime > 0`: decelerate hard, ignore input, spin the group
  (`rotation.y += 12*dt` visual wobble is fine), count down. No steering.
- Acceleration: `throttle` adds `PHYS.ACCEL*dt` (scaled by `1 - speed/MAX_SPEED`
  near top speed); `brake` decelerates (and reverses to `REVERSE_MAX` if speed≈0);
  friction always applies. Clamp to effective max speed:
  - base `PHYS.MAX_SPEED`
  - if off-road (see below): `PHYS.OFFROAD_MAX_SPEED` (and extra drag)
  - if `boostTime > 0`: `+ PHYS.BOOST_SPEED`
  - if `starTime > 0`: `+ PHYS.STAR_SPEED`
- Steering: only when `|speed| > PHYS.STEER_MIN_SPEED`.
  `heading += steer * PHYS.STEER_RATE * dt * speedFactor * sign(speed)`,
  where `speedFactor = clamp(|speed| / 12, 0, 1)`.
- **Drift / mini-turbo**: when `input.drift` and `|speed| > PHYS.DRIFT_MIN_SPEED`
  and steering: `drifting = true`, `driftDir = sign(steer)`, visual yaw offset
  (group rotation + driftDir * 0.35 rad * clamp(speed/30,0,1)), `driftCharge += dt`.
  On drift end (drift key released or straightening): if `driftCharge` crosses the
  thresholds in `PHYS.DRIFT_TURBO_TIME`, call
  `this.boost(PHYS.DRIFT_TURBO_DURATION, PHYS.DRIFT_TURBO_BOOST[tier])` and emit
  particles via `particles.drift(x, z, color)` (blue `0x4cc9f0` / orange
  `0xff9e00` / red `0xff3b3b` by tier). While drifting, emit `particles.drift`
  every ~0.06s at the rear wheels.
- **Off-road detection**: `track.offRoad(state.x, state.z)` (see Module C) — when
  true, apply `OFFROAD_DRAG` and cap speed at `OFFROAD_MAX_SPEED`, emit
  `particles.dust(x, z)` every ~0.05s while moving.
- **Walls**: `track.wallPush(state.x, state.z, state)` — Module C provides this
  helper; it nudges `state.x/z` (and kills some speed) when the kart goes beyond
  `TRACK.WIDTH/2 + PHYS`-defined margin. Call it once per frame and let it mutate
  `state`.
- `canDrive === false` (countdown): force `speed` toward 0, ignore throttle.
- `boost(duration, extraSpeed)`: sets `boostTime = max(boostTime, duration)`;
  store `boostSpeed = extraSpeed` (default `PHYS.BOOST_SPEED`) used in the max
  speed clamp above. Emit `particles.boost(x, z, heading)`.
- `spinOut()`: `spinTime = PHYS.SPIN_DURATION`, `speed *= 0.3`.
- `setStar(on)`: `starTime = on ? PHYS.STAR_DURATION : 0`. While starred, the
  kart body material emits a pulsing emissive glow (star color) — store mat ref.
- `updateIdle(dt)`: wheels stop, gentle hover bob is NOT needed; just zero speed
  and let it sit (used on menu screen while camera orbits).
- `setColor(hex)`: updates chassis material color.
- `reset(pos, heading, t)`: place kart, zero all timers/item/speed.

### `setupPlayerInput(playerKart)`
Attach `window` keydown/keyup listeners (with `e.preventDefault()` for the game
keys to stop page scroll):
- `W`/`ArrowUp` = throttle, `S`/`ArrowDown` = brake, `A`/`ArrowLeft` &
  `D`/`ArrowRight` = steer, `Shift` or `Space` = drift, `E` = `useItem`.
- Map to `playerKart.input`. `useItem` is a one-shot: set to 1 on keydown (the
  race module consumes and resets it).

---

## Module C — Track & Race
**Files: `js/track.js`, `js/race.js`**

### `js/track.js`
```js
export function buildTrack(scene)
// Returns:
{
  curve: THREE.CatmullRomCurve3,   // closed centerline
  startT: number,                  // curve parameter of the start/finish line (use 0)
  sampleCount: number,             // = CONFIG.TRACK.SAMPLES
  centerlineXZ: Float32Array,      // [x0,z0, x1,z1, ...] length sampleCount*2
  TRACK_WIDTH: number,             // = CONFIG.TRACK.WIDTH
  offRoad(x, z): boolean,          // true when outside road+curb
  wallPush(x, z, state): void,     // nudge back if past wall; mutates state.x/state.z,
                                   // damps state.speed; no-op otherwise.
  nearestT(x, z, hintT): number,   // progress 0..1 of the nearest centerline sample
                                   // (scans a ±4% window around hintT; falls back to
                                   // a coarse full scan if nothing close)
  trackLength: number,             // meters, total circuit length
  // For AI/HUD:
  pointAt(t): THREE.Vector3,       // curve.getPointAt(t)
  tangentAt(t): THREE.Vector3,
}
```
- **Track shape**: a hand-authored closed circuit (NOT a random circle). Use a
  `THREE.CatmullRomCurve3` through ~12-14 control points in the XZ plane (y=0)
  forming an interesting layout: a long straight, an S-bend, a hairpin, a sweeping
  curve. Keep the whole track inside roughly ±160 of the origin. Make the start
  straight point in a pleasing direction. All control points y=0.
- **Road mesh**: build a ribbon along the curve (sample ~400 segments), width
  `TRACK_WIDTH`, dark asphalt color, with a procedural canvas texture: faint
  asphalt noise + a yellow dashed center line + solid white edge lines. UVs so
  the texture repeats along the road (length-based repeat).
- **Curbs**: red/white striped strips (width `CURB_WIDTH`) on both edges. Classic
  checkered red/white pattern via canvas texture, alternating.
- **Start/finish line**: checkerboard strip across the road at `startT`, plus a
  start gantry/arch (two poles + banner) spanning the road — banner can be a
  simple box with a canvas texture reading "START" (canvas text is allowed).
- `offRoad`: distance from (x,z) to nearest centerline sample > `TRACK_WIDTH/2 + CURB_WIDTH`.
  For speed, keep the dense `centerlineXZ` array and do a full linear scan
  (1200 pts) ONLY if needed — better: expose `nearest(x, z, hintT)` that scans a
  ±4% window around `hintT` (AI/karts pass their own `t`), and have `offRoad`
  use a coarse scan (every 8th point) which is cheap enough at 8 karts.
- `wallPush`: if distance to centerline > `TRACK_WIDTH/2 + CURB_WIDTH + WALL_MARGIN`,
  pull the kart back toward the road edge and `state.speed *= 0.85`.

### `js/race.js`
```js
export function createRace(karts, track, events)
// Returns:
{
  racing: boolean,      // false during countdown, true after GO
  time: number,         // race clock (starts at GO)
  positions: number[],  // kart indices sorted best→worst (by lap + t)
  results: null | { index, name, time, place }[],  // filled at race over
  reset(),
  update(dt),
}
// events (all optional to fire; guard with if):
// onCountdown(n)  n = 3,2,1 then 0 (0 = "GO" moment)
// onGo(), onLap(kart, lap), onItem(kart, item), onBoost(kart), onSpin(kart),
// onHit(kart), onStar(kart, on), onPlace(kart, place), onRaceOver(results)
```
- **Countdown**: on `reset()`, `racing=false`, internal countdown 3.9s. Emit
  `onCountdown(3/2/1/0)` at the right beats, then `onGo()`, set `racing=true`,
  start `time`.
- **Progress tracking** (per kart, each frame while racing):
  - `kart.t = track.nearestT(kart.state.x, kart.state.z, kart.t)` (the track
    module provides `nearestT` with hint-window search). Store result in `kart.t`.
  - **Lap counting** with checkpoints: checkpoints at t ≈ 0.25, 0.5, 0.75. A kart
    must pass them in order (track a `nextCheckpoint` index per kart). When `t`
    wraps from >0.9 to <0.1 AND all checkpoints were hit, `kart.lap++`,
    reset checkpoint progress, fire `onLap(kart, kart.lap)`. If `kart.lap >
    CONFIG.RACE.LAPS` the kart finished: record its finish time/place, fire
    `onPlace`. When the PLAYER finishes, keep racing for a few seconds or until
    all finish, then `onRaceOver(results)` (results sorted by finish time;
    unfinished karts ranked after, by progress). Set a flag so `onRaceOver`
    fires once.
- **Rankings**: `positions` = indices sorted by (`lap` desc, then `t` desc).
  Recompute each frame (8 karts — cheap).
- **Item boxes**: 4 zones (t ≈ 0.12, 0.4, 0.68, 0.9), 4 boxes each (placed in a
  2x2 grid across the road, bobbing up/down). Each box is a `THREE.Group`
  (translucent magenta box + white "?", canvas texture) added to the `scene`
  that `createRace` receives as its 3rd argument: `createRace(karts, track, scene, events)`.
  - Pickup: kart within 2.2m of an active box → box hides, respawns after
    `ITEMS.BOX_RESPAWN`s. Kart gets an item (if it has none): pick from a
    rank-based pool — ranks 1-2: [shell, shell, mushroom, banana]; 3-5:
    [shell, mushroom, banana, banana]; 6-8: [mushroom, mushroom, banana,
    mushroom]. Fire `onItem(kart, item)`.
- **Item use**: each frame, if `kart.wantUseItem` and `kart.item`:
  - `mushroom` → `kart.boost(PHYS.BOOST_DURATION, PHYS.BOOST_SPEED)`,
    fire `onBoost(kart)`.
  - `star` → `kart.setStar(true)`, fire `onStar(kart, true)`.
  - `banana` → drop a banana (yellow split sphere, half-buried) 2.5m behind the
    kart; remove after `ITEMS.BANANA_LIFETIME`.
  - `shell` → spawn a shell entity (green sphere w/ fins, or just a colored
    sphere + trail) at the kart, `speed = ITEMS.SHELL_SPEED`, life
    `ITEMS.SHELL_LIFETIME`.
  - consume: `kart.item = null; kart.wantUseItem = false`.
  - When starTime reaches 0, fire `onStar(kart, false)` (race watches
    `kart.state.starTime`).
- **Projectiles & hazards**:
  - Shell: moves forward along its heading; homing: steer toward the nearest
    other kart within `ITEMS.SHELL_RANGE` ahead (simple proportional turn).
    On contact (dist < 2) with a non-starred kart: target `spinOut()`,
    fire `onSpin(target)` and `onHit(target)`, destroy shell. Starred karts are
    immune.
  - Banana: static. Any non-starred kart within 1.5m → `spinOut()`, fire
    `onSpin`/`onHit`, remove banana.
- **Kart-kart collision**: each frame, for pairs within
  `2 * PHYS.KART_RADIUS`: push apart along the connecting vector
  (`PHYS.KART_PUSH * overlap * dt`), and damp relative speed slightly
  (`*0.98`). Skip if either is starred (starred kart passes through and
  `spinOut()` the other).

---

## Module D — AI & Camera
**Files: `js/ai.js`, `js/camera.js`**

### `js/ai.js`
```js
export function createAIController(kart, track, index)
// Returns: { update(dt, karts, race) }
```
- Each AI kart has a fixed lateral lane offset (deterministic from `index`,
  range `±AI.LANE_SPREAD`) and a skill multiplier `(1 ± AI.SKILL_JITTER)` from a
  seeded per-index value.
- Steering: aim point = `track.pointAt((kart.t + AI.LOOKAHEAD / trackLength) % 1)`
  offset laterally by the lane. Compute desired heading to the aim point, steer
  proportionally (clamp to -1..1, dead zone 0.15).
- Throttle: full, except braking slightly into sharp corners — approximate
  corner sharpness by the angle between the current tangent and the tangent 30m
  ahead; if sharp, throttle to 0.6.
- **Drift**: in sharp corners at speed, occasionally set `input.drift = 1` for
  the corner (gives AI mini-turbos — makes them feel alive).
- **Rubber-banding**: compute kart's gap to leader (leader = karts[positions[0]]
  — you have `race.positions` and `race` passed in): if more than 40m behind,
  multiply effective throttle toward 1 and let it ride `RUBBERBAND_BEHIND`
  speed bonus; if more than 25m ahead, ease off (`RUBBERBAND_AHEAD`).
  Implement by scaling the kart's top speed feel via throttle duty — do NOT
  directly mutate kart.speed (physics owns that).
- **Avoidance**: if another kart is within 6m ahead in roughly the same lane,
  bias the lateral aim offset away from it for a few seconds.
- **Item usage**: when `kart.item` exists: use `mushroom`/`star` after a short
  random delay; `shell` only when within 30m of a kart ahead; `banana` only
  when no kart is within 15m behind. Set `kart.wantUseItem = true` to fire.
- On lap/position changes nothing special is needed.

### `js/camera.js`
```js
export function createCameraRig(camera)
// Returns: { update(dt, playerKart, track, gameState, race), setMode(m) }
// m: 'menu' | 'chase' | 'results'
```
- **Chase**: target position = kart pos − forward * `CHASE_DISTANCE` + up *
  `CHASE_HEIGHT`. Smooth with exponential lerp (`CHASE_LERP` / `LOOK_LERP`).
  Look-at = kart pos + forward * 6 + up * 1.5.
  - FOV: lerp between `FOV_MIN` and `FOV_MAX` by `|speed| / (MAX_SPEED + BOOST)`.
    Update `camera.fov` + `updateProjectionMatrix()`.
  - **Shake**: on player boost (compare previous/ current `boostTime`) add a
    small decaying positional jitter; on spin-out a bigger one.
- **Menu**: slow orbit around the track center at radius ~90, height ~35,
  looking at the start grid. Smooth, cinematic.
- **Results**: orbit slowly around the player kart (radius 14, height 6),
  looking at it — celebratory.
- `setMode` switches the mode (main.js calls it on transitions); `update` also
  receives `gameState` so it can self-correct.
- Keep a persistent internal `camPos`/`lookAt` Vector3 that lerps (no
  per-frame allocation — reuse temp vectors).

---

## Module E — HUD, Audio & Particles
**Files: `js/hud.js`, `js/audio.js`, `js/particles.js`**

### `js/hud.js`
```js
export function createHUD(containerEl, track)
// Returns:
{
  update(data),            // per-frame, data = { mode, time, lap, laps, position,
                           //   totalKarts, speed, item, karts }
  showMenu(), hideMenu(),
  setSelectedColor(i),
  setCountdown(n),         // 3 | 2 | 1 | 'GO' | null (hide)
  flashLap(lap),
  showResults(results),    // results = [{ index, name, time, place }]
}
```
Build the DOM inside `containerEl` (the `#hud` div; it is `pointer-events:none`,
so no click handling needed — all input is keyboard, handled by main.js).
Style with inline CSS / a `<style>` tag you inject. Fonts: system sans-serif,
bold, with text-shadow for legibility.
- **In-race HUD**:
  - Top-left: big position ("3rd" with ordinal suffix) + "LAP 2/3".
  - Top-center: race time `M:SS.mmm`; once a lap completes, show last-lap time
    small underneath.
  - Bottom-right: speedometer — big number (km/h = m/s * 3.6, rounded) + "km/h".
  - Bottom-right above speed: item box slot — shows the owned item as a colored
    icon/label (mushroom=red cap circle, shell=green, banana=yellow, star=gold).
  - **Minimap** top-right: a `<canvas>` ~180x180. Draw once at init the track
    outline (scale `track.centerlineXZ` to fit with padding, stroke thick grey),
    then each frame clear + redraw road + a dot per kart (white for player,
    `CONFIG.COLORS.KARTS[i]` for others, player dot bigger).
  - Countdown: giant centered number / "GO!" that scales+fades (CSS animation or
    JS timer). `setCountdown(null)` hides it.
  - Lap flash: brief "LAP 2/3" center popup on `flashLap`.
- **Menu** (`showMenu`): centered title "TURBO KART" (big, styled), subtitle
  "Press ENTER to race", controls list (WASD/arrows drive, Shift/Space drift,
  E item), and a row of 8 color swatches with the selected one highlighted
  (`setSelectedColor` toggles the highlight).
- **Results** (`showResults`): "RACE COMPLETE" + table of all 8 (place, name
  "You" for index 0 else "Kart #n", time or "—" for DNF) + "Press R for menu".
- All overlays are absolutely positioned divs you create/destroy. Keep one
  update pass per frame cheap (cache DOM refs, only touch text when changed).

### `js/audio.js`
```js
export function createAudio()
// Returns:
{
  init(),            // create AudioContext on first user gesture; call safely
  setEngine(ratio),  // 0..1 — continuous engine sound
  beep(final),       // countdown: higher pitch for final "GO"
  item(), boost(), spin(), hit(), star(on), lap(), finish(),
}
```
- 100% procedural WebAudio. `init()` creates the `AudioContext` (must be called
  from a user gesture — main.js calls it on Enter). Guard all methods if the
  context doesn't exist yet (no-op).
- **Engine**: 2 oscillators (sawtooth ~55-110Hz base + square sub) through a
  lowpass filter + gain. `setEngine(ratio)` maps ratio → frequency (e.g.
  `60 + ratio*160`) and gain (0 → 0.12). Smooth with linearRampToValueAtTime
  (avoid zipper noise; update at most a few times/sec is fine — call it with
  smoothing internally).
- **SFX** (short synthesized blips/chords):
  - `beep(final)`: square 440Hz 0.15s (final: 880Hz 0.4s).
  - `item()`: quick rising two-tone.
  - `boost()`: noise burst + rising sawtooth sweep.
  - `spin()`: descending wobble.
  - `hit()`: thud (low sine burst + noise).
  - `star(on)`: rising arpeggio (on) / falling (off).
  - `lap()`: cheerful two-note.
  - `finish()`: 4-note fanfare arpeggio.
- Keep a master gain (~0.5) so nothing clips.

### `js/particles.js`
```js
export function createParticles(scene)
// Returns:
{
  update(dt),
  dust(x, z),        // brown-grey puffs (off-road)
  drift(x, z, color),// mini-turbo sparks (blue/orange/red)
  boost(x, z, heading), // exhaust burst behind kart
  hit(x, z),         // impact burst (shells/bananas)
  confetti(x, z),    // victory (fire from above the player at finish)
}
```
- A pool of ~250 tiny `THREE.Mesh` (small box/plane, `MeshBasicMaterial`,
  per-particle color, `depthWrite: false`). Inactive particles are
  `visible = false` and parked far away. No per-frame allocation: recycle.
- Each particle: position, velocity, gravity (variable per type), lifetime,
  scale fade. `update(dt)` advances actives.
- `dust`: 3-5 low-velocity grey-brown puffs spreading outward at y≈0.3, life 0.5s.
- `drift`: 4-6 bright sparks at y≈0.25, slight upward + random spread, life 0.3s,
  additive blending for the charged tiers (you get the color; use
  `blending: THREE.AdditiveBlending` on all spark materials).
- `boost`: 8-10 particles in a cone behind the kart (opposite of `heading`),
  fast, orange→white, life 0.4s.
- `hit`: 10 particles radial burst, white/yellow, life 0.5s.
- `confetti`: 30 colorful pieces from y≈8 above (x,z), random velocities, slow
  gravity, life 2.5s, varied colors.

---

## Reporting

When done, report: (1) files written, (2) any place you had to deviate from this
contract (and why), (3) the exact exported signatures you implemented. Keep it
under 200 words.
