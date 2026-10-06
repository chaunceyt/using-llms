// effects — the photographic finish: the post-processing chain that turns the
// physically-lit scene into a graded, AAA-camera frame.
//
//   RenderPass -> BloomCapPass (per-light HDR soft-clip) -> UnrealBloomPass ->
//   GradePass (S-curve / saturation / split-tone / shadow-lift / vignette /
//   grain, all in linear HDR) -> OutputPass (ACES + sRGB) -> FXAA (final, on
//   LDR) -> screen.
//
// The composer buffers are HalfFloat and UNtone-mapped (three only applies
// tone mapping when rendering straight to the canvas), so bloom thresholds
// operate on true linear radiance: night windows (emissive ~1.6), headlight
// quads (~7) and the sun clear the bar; flat albedo never does. BloomCapPass
// sits in front of the bloom and soft-clips anything above the knee
// (2.0 linear) down to the cap (3.4): the brightest emitters keep a strong,
// tasteful halo without blowing into a huge flat disc. OutputPass picks up
// the renderer's ACES + sRGB settings, and FXAA runs after it because it
// wants sRGB input.
//
// Time-of-day aware: bloom strength ~0.28 by day -> ~1.08 at night, threshold
// 0.86 -> 0.77, with a weather nudge (fog dims the glow). Grade params drift
// gently (more contrast/saturation/lift at night) and everything is
// exponentially smoothed so time jumps never pop.
//
// The loop calls ctx.render(scene, camera) each frame; we override it to drive
// the composer, re-pointing the RenderPass at whatever scene is current so the
// chain works for both the live scene and any module's showcase scene. The
// composer self-resizes (core never calls module onResize).
//
// Public API: setPreset() (compat shim), stats(). Emits effects:ready.
// FOG IS NOT OURS in the LIVE scene: the environment module owns ctx.scene.fog;
// we never touch it. The showcase diorama is a separate harness scene, and
// there we set our own night fog so the atmospheric depth is demonstrable.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { FXAAShader } from 'three/addons/shaders/FXAAShader.js';

// ---- bloom cap: per-light intensity cap (linear HDR soft-clip) -------------
// Identity below the knee; above it a linear rolloff (uSoft) that flattens
// into a hard cap. A 4.5-intensity hero light lands at ~2.7-3.4 instead of
// 4.5+, so its halo has bounded energy: a glow, not a blown disc. Windows
// (~1.6), streetlights (~3) and everything else sit under the knee untouched.
const CAP = { knee: 2.0, cap: 3.4, soft: 0.55 };
const CapShader = {
  name: 'FxBloomCap',
  uniforms: {
    tDiffuse: { value: null },
    uKnee: { value: CAP.knee },
    uCap: { value: CAP.cap },
    uSoft: { value: CAP.soft },
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
  `,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float uKnee;
    uniform float uCap;
    uniform float uSoft;
    varying vec2 vUv;
    void main() {
      const vec3 LUM = vec3(0.2126, 0.7152, 0.0722);
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      float luma = dot(c, LUM);
      float over = max(luma - uKnee, 0.0);
      c -= over * uSoft;          // rolloff above the knee
      c = min(c, vec3(uCap));     // hard safety cap
      gl_FragColor = vec4(max(c, 0.0), 1.0);
    }
  `,
};

// ---- final grade (runs in linear HDR, BEFORE ACES) -------------------------
const GradeShader = {
  name: 'FxGrade',
  uniforms: {
    tDiffuse: { value: null },
    uVig: { value: 0.26 },        // vignette strength
    uSat: { value: 1.08 },        // saturation lift
    uGain: { value: 1.07 },       // base linear contrast
    uCurve: { value: 0.14 },      // luma-dependent S
    uLift: { value: 0.02 },       // shadow-lift amount
    uLiftColor: { value: new THREE.Vector3(0.35, 0.50, 0.80) }, // cool floor
    uSplit: { value: 0.75 },      // split-tone amount (teal sh / orange hi)
    uGrain: { value: 0.020 },     // film grain amplitude
    uTime: { value: 0 },
    uAspect: { value: 16 / 9 },
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
  `,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float uVig;
    uniform float uSat;
    uniform float uGain;
    uniform float uCurve;
    uniform float uLift;
    uniform vec3 uLiftColor;
    uniform float uSplit;
    uniform float uGrain;
    uniform float uTime;
    uniform float uAspect;
    varying vec2 vUv;
    void main() {
      const vec3 LUM = vec3(0.2126, 0.7152, 0.0722);
      vec3 col = texture2D(tDiffuse, vUv).rgb;
      float luma = dot(col, LUM);

      // 1. shadow lift — no crushed pure blacks, floor tinted cool
      float sh = 1.0 - smoothstep(0.0, 0.5, luma);
      col += uLift * sh * uLiftColor;

      // 2. gentle S-curve: linear gain + luma-dependent punch about 0.18
      col = max(col, 0.0);
      col *= uGain;
      float s = dot(col, LUM) - 0.18;
      col *= 1.0 + uCurve * s;

      // 3. saturation lift
      float l2 = dot(max(col, 0.0), LUM);
      col = mix(vec3(l2), col, uSat);

      // 4. split tone: cool shadows, warm highlights (very subtle)
      l2 = dot(max(col, 0.0), LUM);
      vec3 tint = mix(vec3(0.95, 0.99, 1.07), vec3(1.055, 1.0, 0.94),
                      smoothstep(0.12, 0.8, l2));
      col *= mix(vec3(1.0), tint, uSplit);

      // 5. vignette
      vec2 q = (vUv - 0.5) * vec2(uAspect, 1.0);
      float d = length(q) * 1.41421;
      col *= 1.0 - uVig * smoothstep(0.52, 1.08, d);

      // 6. film grain (cheap hash, animated by sim time so it is deterministic)
      float n = fract(sin(dot(vUv * vec2(1620.0, 920.0) +
                  vec2(uTime * 0.61, uTime * 0.73), vec2(12.9898, 78.233))) * 43758.5453);
      col += (n - 0.5) * uGrain;

      gl_FragColor = vec4(max(col, 0.0), 1.0);
    }
  `,
};

// ---- showcase night-sky gradient (fog-proof, BackSide dome) ----------------
const SkyDomeShader = {
  name: 'FxShowSky',
  uniforms: {
    uTop: { value: new THREE.Vector3(0.010, 0.016, 0.038) },
    uMid: { value: new THREE.Vector3(0.018, 0.030, 0.062) },
    uHorizon: { value: new THREE.Vector3(0.040, 0.058, 0.100) },
    uGlow: { value: new THREE.Vector3(0.55, 0.30, 0.14) },
    uGlowDir: { value: new THREE.Vector3(-0.62, 0.05, -0.78).normalize() },
    uGlowAmt: { value: 0.30 },
  },
  vertexShader: /* glsl */`
    varying vec3 vDir;
    void main() {
      vDir = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */`
    uniform vec3 uTop; uniform vec3 uMid; uniform vec3 uHorizon;
    uniform vec3 uGlow; uniform vec3 uGlowDir; uniform float uGlowAmt;
    varying vec3 vDir;
    void main() {
      vec3 d = normalize(vDir);
      float h = clamp(d.y, 0.0, 1.0);
      vec3 col = mix(uHorizon, uMid, smoothstep(0.0, 0.25, h));
      col = mix(col, uTop, smoothstep(0.18, 0.65, h));
      // faint warm light-pollution glow toward the far-city direction
      float g = pow(max(dot(d, uGlowDir), 0.0), 6.0) * (1.0 - smoothstep(0.0, 0.35, h));
      col += uGlow * g * uGlowAmt;
      gl_FragColor = vec4(col, 1.0);
    }
  `,
};

// Weather nudge for the glow: haze soaks up bloom.
const WX = {
  clear:  { bloom: 1.00, thr: 0.00 },
  cloudy: { bloom: 0.90, thr: 0.02 },
  rain:   { bloom: 0.82, thr: 0.04 },
  fog:    { bloom: 0.72, thr: 0.07 },
};

export default class Effects {
  name = 'effects';
  constructor() {
    this.ctx = null;
    this._composer = null; this._renderPass = null; this._cap = null;
    this._bloom = null;
    this._grade = null; this._fxaa = null;
    this._size = { w: 0, h: 0 };
    this._mode = 'day';
    // smoothed params (live values written to passes each frame)
    this._s = { strength: 0.3, threshold: 0.85, vig: 0.24, sat: 1.07, gain: 1.06, curve: 0.12, lift: 0.012, split: 0.7 };
    // showcase bookkeeping
    this._showObjs = [];
    this._showTexs = [];
  }

  async init(world, ctx) {
    this.ctx = ctx;
    const T = THREE;
    const renderer = ctx.renderer;
    const size = renderer.getSize(new T.Vector2());

    this._composer = new EffectComposer(renderer); // HalfFloat HDR buffers
    this._renderPass = new RenderPass(ctx.scene, ctx.camera);
    this._composer.addPass(this._renderPass);

    // Per-light cap BEFORE the bloom: bounds the radiance any single emitter
    // can contribute, so strong lights glow without blowing out (see CAP).
    this._cap = new ShaderPass(CapShader);
    this._composer.addPass(this._cap);

    // Bloom: internal mips run at half of the given resolution; the pass is
    // fed the full buffer size so internals land at W/2 x H/2 (modest for
    // SwiftShader). Threshold operates on LINEAR radiance (buffers are
    // un-tone-mapped), so 0.77-0.86 picks out emissives, not the scene.
    this._bloom = new UnrealBloomPass(new T.Vector2(size.x, size.y), 0.6, 0.5, 0.8);
    this._composer.addPass(this._bloom);

    this._grade = new ShaderPass(GradeShader);
    this._grade.uniforms.uAspect.value = size.x / Math.max(1, size.y);
    this._composer.addPass(this._grade);

    // ACES + sRGB exactly as the renderer is configured (main.js).
    this._composer.addPass(new OutputPass());

    // FXAA last: it wants LDR sRGB input, so it follows the OutputPass.
    this._fxaa = new ShaderPass(FXAAShader);
    const pr = renderer.getPixelRatio();
    this._fxaa.material.uniforms.resolution.value.set(1 / (size.x * pr), 1 / (size.y * pr));
    this._composer.addPass(this._fxaa);
    this._size = { w: size.x, h: size.y };

    // Take over the render pass. The loop passes the CURRENT scene+camera, so
    // the chain follows live/showcase scene switches automatically.
    ctx.render = (s, c) => {
      this._renderPass.scene = s;
      this._renderPass.camera = c;
      this._syncSize();
      this._composer.render();
    };

    // Snap to the current hour (the clock may already be at night).
    this._apply(world, 1);
    ctx.events.emit('effects:ready', { mode: this._mode });
  }

  // Core never calls module onResize, so keep the composer in lockstep with
  // the canvas by checking once per frame (cheap int compare).
  _syncSize() {
    const r = this.ctx.renderer;
    const v = this._tmp || (this._tmp = new THREE.Vector2());
    r.getSize(v);
    if (v.x === this._size.w && v.y === this._size.h) return;
    this._size.w = v.x; this._size.h = v.y;
    this._composer.setSize(v.x, v.y);
    this._bloom.setSize(v.x, v.y);
    const pr = r.getPixelRatio();
    this._fxaa.material.uniforms.resolution.value.set(1 / (v.x * pr), 1 / (v.y * pr));
    this._grade.uniforms.uAspect.value = v.x / Math.max(1, v.y);
  }

  // Time-of-day + weather targets, exponentially smoothed (no popping).
  update(dt, world) {
    if (!this.ctx || !this._composer) return;
    this._apply(world, dt > 0 ? 1 - Math.exp(-dt * 3.0) : 1);
  }

  _apply(world, k) {
    const clock = this.ctx.clock;
    const t = clock.t;
    const night = 1 - clock.daylight();
    const wx = WX[clock.weather] || WX.clear;
    const s = this._s;

    const tStrength = (0.28 + 0.80 * Math.pow(night, 1.6)) * wx.bloom;
    const tThreshold = (0.86 - 0.09 * night) + wx.thr;
    const tVig = 0.22 + 0.08 * night;
    const tSat = 1.06 + 0.04 * night;
    const tGain = 1.045 + 0.035 * night;
    const tCurve = 0.10 + 0.06 * night;
    const tLift = 0.008 + 0.018 * night;
    const tSplit = 0.55 + 0.35 * night;

    s.strength += (tStrength - s.strength) * k;
    s.threshold += (tThreshold - s.threshold) * k;
    s.vig += (tVig - s.vig) * k;
    s.sat += (tSat - s.sat) * k;
    s.gain += (tGain - s.gain) * k;
    s.curve += (tCurve - s.curve) * k;
    s.lift += (tLift - s.lift) * k;
    s.split += (tSplit - s.split) * k;

    this._bloom.strength = s.strength;
    this._bloom.threshold = s.threshold;
    const u = this._grade.uniforms;
    u.uVig.value = s.vig;
    u.uSat.value = s.sat;
    u.uGain.value = s.gain;
    u.uCurve.value = s.curve;
    u.uLift.value = s.lift;
    u.uSplit.value = s.split;
    u.uTime.value = (world.time.day - 1) * 24 + t * 24;

    this._mode = night > 0.55 ? 'night' : (clock.daylight() > 0.45 ? 'day' : 'dusk');
  }

  // Compat shim (round-1 API). The chain is now continuously driven by the
  // clock, so presets only give an instant snap for the matching hour.
  setPreset(p) {
    if (!this.ctx) return;
    const t = p === 'night' ? 0.93 : p === 'dusk' ? 0.79 : 0.5;
    this.ctx.clock.setTimeOfDay(t);
    this._apply(this.ctx.world, 1);
  }

  // ---- showcase: a grounded night city diorama that shows the whole chain --
  // Ground + roads + terrain + buildings + REAL FOG, so bloom, depth and the
  // grade are judgeable in a scene, not a void. The harness stages this scene
  // with a daytime fog/background for other modules; for our night diorama we
  // replace them (this is the showcase scene, NOT ctx.scene — the live scene's
  // fog stays the environment module's business).
  showcase(scene, world, ctx) {
    const T = ctx.three;
    const rng = ctx.rng.fork('effects-show');

    // Clear the harness staging (flat grass + its light wash).
    for (const c of [...scene.children]) if (c.isMesh) scene.remove(c);
    this._clearShow();

    // Night atmosphere for the diorama: FogExp2 matched to the sky horizon.
    // At the 230 m orbit camera the diorama center sits at ~22% haze, the far
    // towers at ~50%, the ridge line at 80%+ — a clean depth ladder.
    const nightFog = new T.Color().setRGB(0.043, 0.062, 0.105); // linear
    scene.fog = new T.FogExp2(nightFog, 0.0022);
    scene.background = new T.Color().setRGB(0.010, 0.016, 0.038);

    const add = (o) => { scene.add(o); this._showObjs.push(o); return o; };

    // --- sky dome (gradient night + warm pollution glow toward the far city)
    const sky = new T.Mesh(new T.SphereGeometry(1000, 24, 16),
      new T.ShaderMaterial({ ...SkyDomeShader, side: T.BackSide, depthWrite: false, fog: false }));
    sky.renderOrder = -10;
    add(sky);

    // --- ground disc (dark urban plain; large enough that its rim melts
    // into the fog at ~90% haze before the sky horizon)
    const ground = new T.Mesh(new T.CircleGeometry(700, 56),
      new T.MeshStandardMaterial({ color: 0x14181f, roughness: 0.95, metalness: 0 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = 0.02;
    ground.receiveShadow = true;
    add(ground);

    // --- roads: ground the streetlight row and give the city a plan --------
    const roadMat = new T.MeshStandardMaterial({ color: 0x0e1116, roughness: 0.92, metalness: 0 });
    const roadMain = new T.Mesh(new T.PlaneGeometry(300, 10), roadMat);
    roadMain.rotation.x = -Math.PI / 2;
    roadMain.position.set(0, 0.06, 38);
    roadMain.receiveShadow = true;
    add(roadMain);
    const roadCross = new T.Mesh(new T.PlaneGeometry(10, 260), roadMat);
    roadCross.rotation.x = -Math.PI / 2;
    roadCross.position.set(-40, 0.06, 0);
    roadCross.receiveShadow = true;
    add(roadCross);

    // --- moonlight (cool shape light, one small shadow map)
    const moon = new T.DirectionalLight(0x93a9d8, 0.55);
    moon.position.set(-140, 130, -90);
    moon.castShadow = true;
    moon.shadow.mapSize.set(1024, 1024);
    const sc = moon.shadow.camera;
    sc.left = -120; sc.right = 120; sc.top = 120; sc.bottom = -120; sc.near = 20; sc.far = 500;
    sc.updateProjectionMatrix();
    moon.shadow.bias = -0.0004;
    moon.shadow.normalBias = 0.8;
    scene.add(moon, moon.target);
    this._showObjs.push(moon, moon.target);

    // --- lit-window material from a tiny procedural emissive atlas ----------
    const winTex = this._makeWindowTexture(rng);
    this._showTexs.push(winTex);
    const facade = new T.MeshStandardMaterial({
      color: 0x1d232c, roughness: 0.7, metalness: 0.1,
      emissive: 0xffffff, emissiveMap: winTex, emissiveIntensity: 1.35,
    });
    const tower = (w, h, d, x, z, mat = facade) => {
      const m = new T.Mesh(new T.BoxGeometry(w, h, d), mat);
      m.position.set(x, h / 2, z);
      m.castShadow = true; m.receiveShadow = true;
      return add(m);
    };
    // downtown cluster: several heights, a few styles of window density
    // (scaled up to read clearly from the 230 m orbit camera the harness sets)
    tower(12, 56, 12, -8, -6);
    tower(14, 36, 14, 8, -14);
    tower(10, 72, 10, 2, 4);
    tower(13, 26, 13, -20, 8);
    tower(9, 42, 9, 18, 10);
    tower(11, 20, 11, -4, 20);

    // --- far suburbs: mid-distance towers that the fog eats progressively --
    // (the depth ladder that makes the atmosphere judgeable)
    tower(14, 44, 14, -96, -160);
    tower(12, 30, 12, 70, -185);
    tower(16, 56, 16, -175, -70);
    tower(11, 24, 11, -35, -225);

    // --- terrain: a ring of low ridge hills at the horizon, haze-blue ------
    const hillGeo = new T.SphereGeometry(1, 20, 10);
    const hillMat = new T.MeshStandardMaterial({ color: 0x141a24, roughness: 1.0, metalness: 0 });
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * Math.PI * 2 + 0.4 + (rng.next() - 0.5) * 0.3;
      const dist = 300 + rng.next() * 160;        // 300-460 m out
      const rad = 95 + rng.next() * 70;           // 95-165 m across
      const h = rad * (0.15 + rng.next() * 0.12); // low, rolling
      const hill = new T.Mesh(hillGeo, hillMat);
      hill.position.set(Math.cos(a) * dist, -h * 0.35, Math.sin(a) * dist);
      hill.scale.set(rad, h, rad * (0.7 + rng.next() * 0.5));
      add(hill);
    }

    // --- streetlight row: strong point glows (well over the bloom bar,
    // bounded by the cap so they glow instead of blow out)
    const poleGeo = new T.CylinderGeometry(0.10, 0.14, 5.6, 6);
    const poleMat = new T.MeshStandardMaterial({ color: 0x2a2e35, roughness: 0.6, metalness: 0.5 });
    const headGeo = new T.SphereGeometry(0.62, 12, 10);
    const headMat = new T.MeshStandardMaterial({ color: 0x000000, emissive: 0xffc07a, emissiveIntensity: 3.0 });
    for (let i = 0; i < 5; i++) {
      const x = -42 + i * 21 + (rng.next() - 0.5) * 2;
      const z = 38 + (rng.next() - 0.5) * 3;
      const pole = new T.Mesh(poleGeo, poleMat);
      pole.position.set(x, 2.8, z);
      add(pole);
      const head = new T.Mesh(headGeo, headMat);
      head.position.set(x, 5.8, z);
      add(head);
    }

    // --- hero orb: the loudest emitter in the diorama. Emissive kept at 3.6
    // (was 4.5) so with the cap its halo reads as a tasteful bloom, not a
    // blown disc.
    const orb = new T.Mesh(new T.SphereGeometry(3.6, 20, 14),
      new T.MeshStandardMaterial({ color: 0x000000, emissive: 0xff9a3c, emissiveIntensity: 3.6 }));
    orb.position.set(42, 11, -6);
    add(orb);

    // --- mid-brightness ring (cool, clearly bloomed) -------------------------
    const ring = new T.Mesh(new T.TorusGeometry(8, 0.5, 12, 48),
      new T.MeshStandardMaterial({ color: 0x000000, emissive: 0x7ab8ff, emissiveIntensity: 1.8 }));
    ring.position.set(-42, 15, 26);
    ring.rotation.set(0.9, 0.4, 0);
    add(ring);

    // --- dim emitter: BELOW the threshold — proves the bar is real ----------
    const dim = new T.Mesh(new T.BoxGeometry(9, 9, 9),
      new T.MeshStandardMaterial({ color: 0x11141a, emissive: 0x223344, emissiveIntensity: 0.5 }));
    dim.position.set(12, 4.5, -26);
    add(dim);

    // --- matte near object: shape + moonlight, no glow (depth reader) -------
    const mono = new T.Mesh(new T.BoxGeometry(8, 14, 8),
      new T.MeshStandardMaterial({ color: 0x555c66, roughness: 0.55, metalness: 0.15 }));
    mono.position.set(22, 7, 30);
    mono.castShadow = true; mono.receiveShadow = true;
    add(mono);

    // --- far tower: mid distance, lit (atmospheric depth vs the near mono) --
    tower(18, 80, 18, -120, -72);

    // --- far-city glow billboard: soft additive haze on the horizon ---------
    // (kept out of fog: additive light haze should not be pre-attenuated)
    const glowTex = this._makeGlowTexture();
    this._showTexs.push(glowTex);
    const glow = new T.Mesh(new T.PlaneGeometry(180, 76),
      new T.MeshBasicMaterial({ map: glowTex, transparent: true, blending: T.AdditiveBlending, depthWrite: false, fog: false }));
    glow.position.set(-180, 40, -140);
    glow.lookAt(60, 26, 60);
    add(glow);

    // Snap the chain to night so even frame 1 is graded + bloomed.
    this._apply(world, 1);
  }

  // 128x256 canvas: a grid of lit/unlit windows (sRGB, anisotropy 1).
  _makeWindowTexture(rng) {
    const T = THREE;
    const cv = document.createElement('canvas');
    cv.width = 128; cv.height = 256;
    const g = cv.getContext('2d');
    g.fillStyle = '#0a0d12';
    g.fillRect(0, 0, 128, 256);
    const cols = 8, rows = 16, cw = 128 / cols, ch = 256 / rows;
    const lit = ['255,214,150', '255,230,190', '205,228,255', '255,170,90', '255,196,130'];
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        if (rng.chance(0.52)) {
          const a = 0.45 + 0.55 * rng.next();
          g.fillStyle = `rgba(${lit[(rng.next() * lit.length) | 0]},${a.toFixed(3)})`;
        } else {
          g.fillStyle = 'rgba(24,30,40,0.9)';
        }
        g.fillRect(x * cw + (cw - 10) / 2, y * ch + (ch - 12) / 2, 10, 12);
      }
    }
    const tex = new T.CanvasTexture(cv);
    tex.colorSpace = T.SRGBColorSpace;
    tex.anisotropy = 1; // SwiftShader aniso is broken -> keep it at 1
    tex.wrapS = tex.wrapT = T.RepeatWrapping;
    return tex;
  }

  // Radial warm glow for the far-city billboard.
  _makeGlowTexture() {
    const T = THREE;
    const cv = document.createElement('canvas');
    cv.width = 256; cv.height = 128;
    const g = cv.getContext('2d');
    const grad = g.createRadialGradient(128, 74, 4, 128, 74, 96);
    grad.addColorStop(0, 'rgba(255,176,110,0.85)');
    grad.addColorStop(0.4, 'rgba(255,150,90,0.35)');
    grad.addColorStop(1, 'rgba(120,90,70,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 256, 128);
    const tex = new T.CanvasTexture(cv);
    tex.colorSpace = T.SRGBColorSpace;
    tex.anisotropy = 1;
    return tex;
  }

  _clearShow() {
    for (const o of this._showObjs) {
      o.removeFromParent();
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        for (const m of Array.isArray(o.material) ? o.material : [o.material]) m.dispose();
      }
    }
    this._showObjs.length = 0;
    for (const t of this._showTexs) t.dispose();
    this._showTexs.length = 0;
  }

  dispose() {
    this._clearShow();
    if (this._composer) { this._composer.dispose(); this._composer = null; }
    this._renderPass = this._cap = this._bloom = this._grade = this._fxaa = null;
    this.ctx = null;
  }

  stats() {
    const s = this._s;
    return {
      drawCalls: 0,
      mode: this._mode,
      bloom: {
        strength: +s.strength.toFixed(2),
        threshold: +s.threshold.toFixed(2),
        radius: 0.5,
        cap: { knee: CAP.knee, cap: CAP.cap, soft: CAP.soft },
      },
      notes: 'render -> bloomCap(soft-clip 2.0->3.4) -> unrealBloom(TOD+weather) -> grade(S/sat/split-tone/lift/vignette/grain) -> ACES+sRGB -> FXAA',
    };
  }
}
