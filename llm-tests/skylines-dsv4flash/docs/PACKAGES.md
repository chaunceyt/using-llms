# Packages & Tools — ACTION ITEM 1 (definitive)

The dev environment is aarch64 Ubuntu with no sudo, an HTTP proxy that resets
burst connections, and no native GPU. This documents exactly what is needed to
build and *screenshot* the game, and how each piece was obtained.

## Runtime dependencies (npm — already installed via curl bootstrap)

| Package      | Version   | Why                                                     |
|--------------|-----------|---------------------------------------------------------|
| `three`      | 0.186.1   | WebGL renderer + math                                   |
| `vite`       | ^5.4.0    | dev server / build (target es2022)                      |

That is the entire dependency set. Everything else is hand-rolled on purpose:
no physics engine, no nav-mesh lib, no post-processing suite (per ARCHITECTURE.md).

`node_modules` is a symlink to `/tmp/pp-test/node_modules` (populated by
`tools/install-deps.mjs`, the curl-based resolver — npm itself is broken through
the proxy for multi-package installs). The `npm run dev` script can't find vite
because `.bin` shims were never generated, so the dev server is launched via:
`node /tmp/pp-test/node_modules/vite/bin/vite.js --host --port 5173`.

## Screenshot / verification stack (ZERO npm packages — deliberately)

The critic gauntlet needs deterministic renders of the three.js scene. On this
box there is no GPU and Chrome's bundled SwiftShader was broken; instead:

- **Headless Chromium** (`/tmp/spart/chromium`, ~199 MB) from the
  `@sparticuz/chromium` GitHub release, placed next to an **external v153
  SwiftShader** layer at `/tmp/spart/swiftshader/`.
- Software WebGL is enabled with:
  `--headless=new --no-sandbox --disable-dev-shm-usage --enable-unsafe-swiftshader
  --use-gl=angle --use-angle=swiftshader`
  plus `LD_LIBRARY_PATH=/tmp/spart:/tmp/spart/swiftshader:/tmp/spart/al2023/lib`.
- **CDP over Node's built-in WebSocket/fetch** (`tools/cdp.mjs`) drives it. No
  puppeteer/playwright npm package needed.
- Frames are captured by **in-page readback** (`renderer.domElement.toDataURL`)
  with `preserveDrawingBuffer: true` — the browser compositor presents black
  under SwiftShader, but the GL drawing buffer has real content.

Verified end-to-end: geometry rasterizes, per-object color works, and
MeshStandardMaterial lighting produces per-face shading. `tools/screenshot.mjs`
captures overview/street/night/sunset presets and writes PNG + JSON (fps,
drawCalls, console errors).

## Assets

CC0 only: Poly Haven / ambientCG textures, or fully **procedural** PBR textures
generated at runtime (deterministic from the seed). Procedural is preferred —
it needs no network fetches (the proxy whitelist is fragile) and stays
deterministic.

## Tooling files

- `tools/cdp.mjs` — minimal CDP client (launch chrome, eval JS, grab frame).
- `tools/screenshot.mjs` — preset screenshot capture + JSON report.
- `tools/install-deps.mjs` — curl-based dependency bootstrap for future installs.
