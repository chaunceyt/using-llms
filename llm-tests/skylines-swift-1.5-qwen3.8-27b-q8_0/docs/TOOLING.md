# Tooling & Environment Report (ACTION ITEM 1)

Generated: 2026-10-01. Grounded in live probes of this sandbox, not assumptions.

## TL;DR

The sandbox has a **policy-filtered egress proxy** that blocks the npm registry, every
JS CDN, git-clone/tarball endpoints, the GitHub API, apt mirrors, asset CDNs, and the
Puppeteer browser host. **Only `github.com` (web) and `raw.githubusercontent.com`
(individual files) are reachable.** Consequences:

- **Three.js: vendorable** (fetch the prebuilt `r186` build + addons file-by-file from raw).
- **Vite: not installable** (needs native `esbuild`/`rollup` binaries from npm). → use a
  zero-dependency static ES-module dev server (Node built-in `http`). Swappable to real
  Vite if `registry.npmjs.org` is whitelisted.
- **Headless Chrome: NOT installable here** (no browser on disk, no `sudo`, apt 403,
  googleapis blocked). This is the **#1 blocker** for the screenshot / gauntlet loop.
  → a Chromium binary must be **provisioned** into the sandbox (or its download host
  whitelisted). The screenshot tool is written to need **only a `CHROME_PATH` binary**
  (driven over CDP with Node's built-in `WebSocket` — no puppeteer package).
- **CC0 assets (Poly Haven / ambientCG): not reachable.** → build on **procedural PBR**
  now (explicitly allowed by the asset policy); swap in photographic textures once the
  asset CDNs are reachable.

## 1. Already present (nothing to install)

| Tool        | Version    | Path               |
|-------------|-----------|--------------------|
| node        | v22.22.1  | /usr/bin/node      |
| npm         | 11.11.0   | /usr/bin/npm       |
| git         | 2.43.0    | /usr/bin/git       |
| python3     | 3.13.12   | /sandbox/.venv     |

Hardware: 16 cores, 7.7 GiB RAM (≈4.5 GiB available), 829 GiB free disk. Plenty.

Node 22 notes that help us stay dependency-free: global `fetch` and global `WebSocket`
are stable — we can drive the Chrome DevTools Protocol with **no npm package at all**.

## 2. Egress map (observed)

| Host                          | Status | Notes |
|-------------------------------|:------:|-------|
| `github.com`                  | 200    | web UI reachable |
| `raw.githubusercontent.com`  | 200    | **individual file fetches work** (follows 301) |
| `registry.npmjs.org`          | 403    | `policy_denied` — npm install impossible |
| `cdn.jsdelivr.net`            | 000    | blocked |
| `unpkg.com`                   | 000    | blocked |
| `registry.npmmirror.com`      | 000    | blocked |
| `cdnjs.cloudflare.com`        | 000    | blocked |
| `codeload.github.com`         | 000    | git clone / tarball blocked |
| `objects.githubusercontent.com` | 000  | release assets blocked |
| `api.github.com`              | 000    | REST API blocked |
| `deb.debian.org` (apt)        | 403    | apt install impossible |
| `storage.googleapis.com`      | 000    | Puppeteer/Playwright browser host blocked |
| `polyhaven.org`               | 000    | blocked |
| `ambientcg.com`               | 000    | blocked |

System: no `sudo`; `apt-get` present but its mirror is 403. No Chrome/Chromium/
headless_shell binary anywhere on disk. No npm/puppeteer/playwright caches.

## 3. Install list (what's needed, status, mitigation)

### 3a. Runtime / system tools
| Item | Needed for | Status | Mitigation |
|------|-----------|:------:|------------|
| Node 22 | app + tooling | ✅ present | — |
| git | versioning | ✅ present | — |
| **Headless Chromium / Chrome binary** | screenshots, gauntlet, blind judge | ❌ **BLOCKED** | **Provision a binary** to a known path (e.g. `/usr/local/bin/chromium` or set `CHROME_PATH`). Alt: whitelist `storage.googleapis.com` / `edgedl.me.gvt1.com` and let a tool fetch it. This is the single hard blocker. |

### 3b. npm packages (all blocked via registry — see mitigations)
| Package | Purpose | npm status | Mitigation |
|---------|---------|:----------:|------------|
| `three@0.186.0` | 3D engine | ❌ 403 | **Vendor from raw** `mrdoob/three.js` tag `r186` (verified 200). See 3d. |
| `vite` | dev server / bundler | ❌ 403 (native deps `esbuild`,`rollup` also blocked) | **Zero-dep static ESM server** in `tools/serve.mjs` (Node `http`). Drop-in replace with real Vite if registry whitelisted. |
| `puppeteer` / `playwright` | screenshots | ❌ (pkg + browser host both blocked) | **Tiny CDP driver** using Node built-in `WebSocket` against a provisioned Chromium. No package needed. |

### 3c. Zero-dependency tooling we will write in-repo (no install)
- `tools/serve.mjs` — static ES-module dev server (correct MIME types, SPA fallback,
  `/healthz`, keeps app loadable for screenshotting).
- `tools/shot.mjs` — headless screenshot + JSON log (console errors, fps, draw calls)
  via CDP over built-in `WebSocket`. Reads `CHROME_PATH`.
- `tools/vendor-three.mjs` — fetches the pinned three.js files (below) from raw into
  `vendor/three/` with a lockfile of URLs + byte lengths.

### 3d. Three.js vendor manifest (all verified HTTP 200 on tag `r186`)
Base: `https://raw.githubusercontent.com/mrdoob/three.js/r186/`

Core:
- `build/three.module.js` (662,772 bytes)

Addons (`examples/jsm/...`) to fetch:
- `controls/OrbitControls.js`
- `postprocessing/EffectComposer.js`
- `postprocessing/RenderPass.js`
- `postprocessing/OutputPass.js`
- `postprocessing/ShaderPass.js`
- `postprocessing/UnrealBloomPass.js`
- `postprocessing/SMAAPass.js`
- `postprocessing/Pass.js` (dep of the above)
- `environments/RoomEnvironment.js`
- `utils/BufferGeometryUtils.js`
- `math/SimplexNoise.js`
- `shaders/CopyShader.js`
- `shaders/SMAAShader.js`
- `shaders/LuminosityHighPassShader.js`
- `objects/Water.js`
- `misc/Stats.js`

(Exact set can grow; every `examples/jsm/*` path on `r186` returns 200.)

## 4. What I need from the operator (to unblock)

1. **A Chromium/Chrome binary in the sandbox** (path exposed as `CHROME_PATH`), OR
   whitelist its download host. → unblocks the verification loop (highest priority).
2. **Whitelist `registry.npmjs.org`** (optional) → enables real `three`/`vite`/tooling
   via npm and removes the need to vendor.
3. **Whitelist `polyhaven.org` + `ambientcg.com` asset CDNs** (optional) → enables
   photographic PBR textures; until then we render with procedural PBR.

None of 2/3 are required to *start*: Three.js is vendorable and procedural PBR is
policy-approved. Only item 1 (Chrome) is a true gate on the gauntlet.

## 5. Stated decisions / assumptions (per "make routine decisions yourself")

- **Plain ES modules, no TypeScript, no bundler** — served statically. Matches the brief
  ("plain ES modules") and the no-npm reality.
- **Three.js pinned to `r186`**, vendored under `vendor/three/` (immutable, lockfile-pinned).
- **Dev server = zero-dep Node static server**, not Vite, for the same reason.
- **Screenshot driver = hand-rolled CDP client** (Node `WebSocket`), not puppeteer.
- **Art = procedural PBR first**; photographic assets slot in behind the same material
  interface when CDNs become reachable.
- Units: metres, +Y up. Seeded RNG only (determinism). These are re-asserted in
  ARCHITECTURE.md.
