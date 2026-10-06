// environment — the photographic backbone: a custom atmospheric-scattering
// sky dome (day azure -> dusk orange/pink/purple band -> deep night blue),
// in-shader sun + moon discs, procedural drifting cloud billboards, FogExp2
// aerial perspective whose colour tracks the sky's horizon haze, a
// shadow-casting sun, hemisphere + moon fill, a starfield, and a rate-
// limited PMREM IBL refreshed a few times per simulated day.
//
// Determinism: everything is a pure function of ctx.clock (t + weather);
// RNG use is one-time layout only (starfield, cloud field, showcase peaks).
// Performance: ~19 draw calls live (dome + stars + 16 cloud billboards);
// update is allocation-light (module-scope scratch vectors/colours reused).
//
// Public API: setTimeOfDay(t), setWeather(w), sun (DirectionalLight+shadow),
// sky (dome mesh), fog (FogExp2), hemi, moon. Emits 'env:sun' {dir,color,
// intensity,elevation} every frame, 'env:weather' {w} on change.
import * as THREE from 'three';

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// ---- sky palette (LINEAR rgb; the ACES+sRGB output pass grades on top) ------
const PAL = {
  zenDay:   [0.075, 0.245, 0.630],
  zenDusk:  [0.095, 0.120, 0.400],
  zenNight: [0.006, 0.010, 0.024],
  zenGrey:  [0.420, 0.460, 0.520],
  horDay:   [0.430, 0.575, 0.790],
  horDusk:  [0.950, 0.440, 0.200],
  horNight: [0.017, 0.023, 0.042],
  horGrey:  [0.500, 0.530, 0.570],
  sunLow:  [1.00, 0.42, 0.16],
  sunHigh: [1.00, 0.96, 0.88],
  sunGrey: [0.85, 0.87, 0.90],
  hemiTopDay:   [0.52, 0.70, 0.95],
  hemiTopDusk:  [0.58, 0.40, 0.36],
  hemiTopNight: [0.10, 0.14, 0.24],
  hemiBotDay:   [0.34, 0.31, 0.25],
  hemiBotNight: [0.05, 0.06, 0.08],
  moon: [0.58, 0.66, 0.92],
  cloudDay:   [0.95, 0.97, 1.00],
  cloudDusk:  [1.45, 0.80, 0.68],
  cloudNight: [0.20, 0.22, 0.32],
};
const C = {};
for (const k in PAL) C[k] = new THREE.Color(PAL[k][0], PAL[k][1], PAL[k][2]);

// ---- per-weather atmosphere targets (exponentially smoothed each frame) -----
const WX = {
  clear:  { dim: 1.00, sun: 1.00, hemi: 1.00, moon: 1.00, star: 1.00, grey: 0.00, cov: 0.65, fogDen: 0.00100 },
  cloudy: { dim: 0.72, sun: 0.55, hemi: 0.80, moon: 0.35, star: 0.35, grey: 0.45, cov: 0.95, fogDen: 0.00150 },
  rain:   { dim: 0.55, sun: 0.40, hemi: 0.65, moon: 0.15, star: 0.10, grey: 0.62, cov: 0.90, fogDen: 0.00240 },
  fog:    { dim: 0.82, sun: 0.68, hemi: 0.82, moon: 0.10, star: 0.05, grey: 0.85, cov: 0.40, fogDen: 0.00520 },
};
const WX_KEYS = Object.keys(WX.clear);

// ---- sky dome shaders --------------------------------------------------------
// gl_Position.z pinned just inside the far plane: the dome is never
// depth-clipped, always renders behind the city, and stays PMREM-safe at
// any cube-camera far.
const SKY_VERT = /* glsl */`
  varying vec3 vPos;
  void main() {
    vPos = position;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    gl_Position.z = gl_Position.w * 0.9999;
  }
`;

const SKY_FRAG = /* glsl */`
  uniform vec3 uSunDir;    // unit vector toward the sun
  uniform vec3 uMoonDir;   // unit vector toward the moon
  uniform vec3 uZenith;    // time/weather-blended zenith colour (linear)
  uniform vec3 uHorizon;   // time/weather-blended horizon haze colour (linear)
  uniform vec3 uSunCol;    // sun disc/glow colour (linear)
  uniform float uDusk;     // 0..1 warm-band strength (peaks at the horizon)
  uniform float uNight;    // 0..1
  uniform float uHaze;     // 0..1 weather haze: widens + flattens the horizon band
  uniform float uDisc;     // 0 hides the sun/moon discs (IBL pass)
  varying vec3 vPos;

  void main() {
    vec3 dir = normalize(vPos);
    float h = clamp(dir.y, 0.0, 1.0);

    // Base gradient: hazy band at the horizon rising to a clean zenith.
    // Day: tight band, so the sky reads crisp azure above a defined haze
    // line. Dusk: the band widens back out for the warm shelf. Weather haze
    // widens it further on top.
    float crisp = 1.0 - clamp(uDusk * 1.5, 0.0, 1.0);
    float expo = mix(0.30, 0.52, crisp);
    expo = mix(expo, expo * 0.45, uHaze);
    float band = 1.0 - pow(h, expo);
    vec3 col = mix(uZenith, uHorizon, band);

    // Dusk: a warm band hugs the horizon toward the sun's azimuth,
    // with a cooler purple shelf above it.
    vec2 sx = uSunDir.xz; float sl = length(sx); sx = sl > 1e-5 ? sx / sl : vec2(1.0, 0.0);
    vec2 dx = dir.xz;     float dl = length(dx); dx = dl > 1e-5 ? dx / dl : vec2(1.0, 0.0);
    float az = pow(max(dot(dx, sx), 0.0), 1.7);
    float low = 1.0 - h;
    col = mix(col, vec3(1.35, 0.45, 0.15), uDusk * az * pow(low, 3.2) * 1.00); // ember orange
    col = mix(col, vec3(0.42, 0.19, 0.48), uDusk * az * pow(low, 1.7) * 0.45); // purple shelf

    // Sun: broad halo, tight glow, then a hard disc.
    float sd = max(dot(dir, uSunDir), 0.0);
    float sunVis = smoothstep(-0.14, 0.03, uSunDir.y);
    col += uSunCol * (pow(sd, 5.0) * 0.10 + pow(sd, 48.0) * 0.30 + pow(sd, 380.0) * 0.55) * sunVis;
    col += uSunCol * smoothstep(0.99968, 0.99993, sd) * 2.8 * sunVis * uDisc;

    // Moon: dim cool disc with a faint halo, opposite the sun, night only.
    float md = max(dot(dir, uMoonDir), 0.0);
    col += vec3(0.50, 0.58, 0.80)
         * (pow(md, 60.0) * 0.06 + smoothstep(0.99940, 0.99968, md) * 0.42)
         * uNight * uDisc;

    // Below the horizon: melt into the (dimmed) haze so ground seams disappear.
    col = mix(col, uHorizon * 0.9, smoothstep(0.0, -0.10, dir.y));

    gl_FragColor = vec4(col, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

// Scratch (never allocated in update)
const _sunV = new THREE.Vector3();
const _moonV = new THREE.Vector3();
const _cA = new THREE.Color();
const _cB = new THREE.Color();

export default class Environment {
  name = 'environment';

  constructor() {
    this._weather = 'clear';
    this._w = { ...WX.clear };          // smoothed weather params (current state)
    this._sunInfo = { dir: { x: 0, y: 1, z: 0 }, color: 0xffffff, intensity: 0, elevation: 1 };
    this._ibAcc = 0; this._ibT = -1; this._ibDirty = true;
    this._ibRT = null; this._pmrem = null;
    this._shx = NaN; this._shy = NaN; this._shz = NaN; this._shvis = false;
    this._liveScene = null;
    this._parked = [];                  // pre-existing hemis we dimmed (baseline lights)
    this._showExtras = [];
    this._showMtnMats = [];
    this._clouds = [];
  }

  async init(world, ctx) {
    this.ctx = ctx;
    const T = ctx.three;
    this._rng = ctx.rng.fork('environment');

    // --- sky dome: large sphere, BackSide, custom scattering-ish shader -------
    // R=3800 (< camera.far 4000); the vertex shader pins depth to the far
    // plane, so the dome is never clipped and PMREM can render it at any far.
    // The whole distant group (dome+stars+clouds) re-centers on the camera
    // each frame so no camera preset can push the dome past `far`.
    const domeGeo = new T.SphereGeometry(3800, 32, 16);
    this._skyMat = new T.ShaderMaterial({
      uniforms: {
        uSunDir:  { value: new T.Vector3(0, 1, 0) },
        uMoonDir: { value: new T.Vector3(0, 1, 0) },
        uZenith:  { value: new T.Color(...PAL.zenDay) },
        uHorizon: { value: new T.Color(...PAL.horDay) },
        uSunCol:  { value: new T.Color(...PAL.sunHigh) },
        uDusk: { value: 0 }, uNight: { value: 0 },
        uHaze: { value: 0.15 }, uDisc: { value: 1 },
      },
      vertexShader: SKY_VERT,
      fragmentShader: SKY_FRAG,
      side: T.BackSide,
      depthWrite: false,
      fog: false,
    });
    this.sky = new T.Mesh(domeGeo, this._skyMat);
    this.sky.renderOrder = -10;
    this.sky.frustumCulled = false;

    // IBL twin: shares geometry + material (uniforms) with the visible dome,
    // in a private scene so PMREM renders only the sky, never the city.
    this._ibScene = new T.Scene();
    this._ibSky = new T.Mesh(domeGeo, this._skyMat);
    this._ibSky.scale.setScalar(10);
    this._ibSky.frustumCulled = false;
    this._ibScene.add(this._ibSky);
    this._pmrem = new T.PMREMGenerator(ctx.renderer);

    // --- sun (the only shadow caster) -----------------------------------------
    this.sun = new T.DirectionalLight(0xffffff, 3.0);
    this.sun.castShadow = true;
    const sc = this.sun.shadow.camera;
    sc.left = -550; sc.right = 550; sc.top = 550; sc.bottom = -550;
    sc.near = 10; sc.far = 3200;
    sc.updateProjectionMatrix();
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0003;
    this.sun.shadow.normalBias = 0.8;
    this.sun.shadow.autoUpdate = false; // re-render only when the sun moves (see update)
    this.sun.shadow.needsUpdate = true;
    this.sun.target.position.set(0, 0, 0);

    // --- hemisphere fill + moon ------------------------------------------------
    this.hemi = new T.HemisphereLight(0xffffff, 0x444433, 0.8);
    this.moon = new T.DirectionalLight(1, 1, 1);
    this.moon.color.copy(C.moon);
    this.moon.castShadow = false;

    // --- stars: one Points cloud on a dome, faded in at night ------------------
    // R=3300: inside the dome (3800) and, from the furthest camera preset
    // (aerial, ~470 m off-center), 3300+470 < camera.far (4000) so no star
    // is ever far-clipped.
    const N = 1600, R = 3300;
    const pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const z = this._rng.range(0.04, 1.0);           // upper-hemisphere bias
      const phi = this._rng.range(0, Math.PI * 2);
      const rr = Math.sqrt(Math.max(0, 1 - z * z));
      pos[i * 3] = Math.cos(phi) * rr * R;
      pos[i * 3 + 1] = z * R;
      pos[i * 3 + 2] = Math.sin(phi) * rr * R;
      const b = 0.35 + 0.65 * this._rng.next() * this._rng.next();
      const warm = this._rng.chance(0.12);
      col[i * 3] = b * (warm ? 1.0 : 0.80 + 0.2 * this._rng.next());
      col[i * 3 + 1] = b * (warm ? 0.86 : 0.88 + 0.12 * this._rng.next());
      col[i * 3 + 2] = b * (warm ? 0.70 : 1.0);
    }
    const sg = new T.BufferGeometry();
    sg.setAttribute('position', new T.BufferAttribute(pos, 3));
    sg.setAttribute('color', new T.BufferAttribute(col, 3));
    this._stars = new T.Points(sg, new T.PointsMaterial({
      size: 4.0, sizeAttenuation: false, vertexColors: true,
      transparent: true, opacity: 0, blending: T.AdditiveBlending,
      depthWrite: false, fog: false,
    }));
    this._stars.renderOrder = -9;

    // --- clouds: a dozen large soft billboards, generated alpha texture --------
    // fog: false is deliberate. Clouds live at 550-1500 m; FogExp2 at city
    // densities (0.001+) would wash 30-90% of horizon-coloured fog into them —
    // i.e. the exact colour of the sky behind them — erasing them. Clouds are
    // sky objects (like the dome and stars): their atmospheric fade is carried
    // by the tint/opacity ramp below, not by scene fog.
    this._cloudTex = this._makeCloudTexture();
    this._cloudMat = new T.MeshBasicMaterial({
      map: this._cloudTex, transparent: true, opacity: 0.55,
      depthWrite: false, side: T.DoubleSide, fog: false,
    });
    this._cloudGeo = new T.PlaneGeometry(1, 1);
    const rc = this._rng.fork('envclouds');
    // Two strata: a low, far "horizon bank" (the clouds that read in the
    // orbit/aerial views, hugging the haze line) and higher feature clouds
    // (visible from street/skyline views). Fog fades the far ends naturally.
    for (let i = 0; i < 16; i++) {
      const m = new T.Mesh(this._cloudGeo, this._cloudMat);
      const bank = i < 8;
      const ang = rc.range(0, Math.PI * 2);
      const rad = bank ? rc.range(550, 1150) : rc.range(550, 1500);
      const y = bank ? rc.range(85, 145) : rc.range(220, 420);
      m.position.set(Math.cos(ang) * rad, y, Math.sin(ang) * rad);
      const w = bank ? rc.range(480, 980) : rc.range(350, 760);
      m.scale.set(w, w * rc.range(0.40, 0.58), 1);
      m.renderOrder = 5;
      m.userData.speed = rc.range(2, 6);   // m/s drift (+x)
      m.userData.wrap = 2300;
      this._clouds.push(m);
    }

    // --- assemble: lights stay at world origin; the distant group follows cam --
    this.root = new T.Group();
    this.root.add(this.sun, this.sun.target, this.hemi, this.moon);
    this._skyGroup = new T.Group();
    this._skyGroup.add(this.sky, this._stars, ...this._clouds);
    ctx.scene.add(this.root, this._skyGroup);
    this._liveScene = ctx.scene;

    // --- fog: FogExp2, colour synced to the sky horizon each frame --------------
    this.fog = new T.FogExp2(0x9db8d0, 0.00105);
    ctx.scene.fog = this.fog;

    // baseline hemis (core's placeholder + showcase staging) yield to ours
    this._park(ctx.scene);

    // register our key
    world.environment = { sun: this.sun, sky: this.sky, fog: this.fog, hemi: this.hemi, moon: this.moon };

    ctx.events.on('env:set-time', () => { this._ibDirty = true; });
    ctx.events.on('env:set-weather', () => { this._ibDirty = true; });

    this.update(0, world);
    this._regenIBL(true);
  }

  // Dim pre-existing hemisphere lights that aren't ours (core baseline /
  // showcase staging add one); remembered so dispose() can restore them.
  _park(scene) {
    scene.traverse((o) => {
      if (o.isHemisphereLight && o !== this.hemi && o.intensity > 0) {
        o.userData.envSavedI = o.userData.envSavedI ?? o.intensity;
        o.intensity = 0;
        if (!this._parked.includes(o)) this._parked.push(o);
      }
    });
  }

  // Soft cloud puff: fBm alpha with a radial falloff. Pure function of a
  // constant seed (deterministic; does not consume the game RNG stream).
  _makeCloudTexture() {
    const T = this.ctx.three;
    const S = 256;
    const cv = document.createElement('canvas'); cv.width = cv.height = S;
    const g = cv.getContext('2d');
    const img = g.createImageData(S, S);
    const hash = (x, y) => {
      let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + 0x5eed) >>> 0;
      h = (h ^ (h >>> 13)) >>> 0; h = Math.imul(h, 0x5bd1e995) >>> 0; h ^= h >>> 15;
      return (h >>> 0) / 4294967296;
    };
    const noise2 = (x, y) => {
      const ix = Math.floor(x), iy = Math.floor(y);
      const fx = x - ix, fy = y - iy;
      const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
      const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
      return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
    };
    const fbm = (x, y, oct) => {
      let sum = 0, amp = 0.5, f = 1, norm = 0;
      for (let i = 0; i < oct; i++) { sum += amp * noise2(x * f, y * f); norm += amp; amp *= 0.5; f *= 2.03; }
      return sum / norm;
    };
    const ss = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const u = x / S, v = y / S;
        const n = fbm(u * 3.4 + 0.7, v * 3.4 + 2.1, 5);
        const f = fbm(u * 9.0 - 4.2, v * 9.0 + 6.3, 3);
        let a = ss(0.38, 0.64, n * 0.72 + f * 0.28);
        const r = Math.hypot(u - 0.5, v - 0.5) * 2;      // 0 centre -> 1.41 corner
        a *= 1 - ss(0.62, 1.05, r);                     // soften the edges
        // lit-from-below gradient: the lower edge reads brighter (sunlit base,
        // cool shadowed top) once the material tint is applied
        const shade = (0.70 + 0.30 * (1 - v)) * (0.92 + 0.08 * f);
        const i = (y * S + x) * 4;
        img.data[i] = Math.round(248 * shade);
        img.data[i + 1] = Math.round(247 * shade);
        img.data[i + 2] = 255;
        img.data[i + 3] = Math.round(255 * a);
      }
    }
    g.putImageData(img, 0, 0);
    const tex = new T.CanvasTexture(cv);
    tex.colorSpace = T.SRGBColorSpace;
    tex.anisotropy = 1; // >1 banding on the SwiftShader screenshot renderer
    tex.wrapS = tex.wrapT = T.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return tex;
  }

  update(dt, world) {
    if (!this.ctx) return;
    const { three: T, clock, events, camera } = this.ctx;
    const t = clock.t;
    const w = clock.weather || 'clear';
    const sd = clock.sunDir();
    const e = sd.elev;                        // raw sin of sun path, -1..1
    const day = clock.daylight();
    const night = 1 - day;
    const sunUp = smooth(-0.05, 0.14, e);     // 0 below horizon -> 1 by ~8 deg
    const skyDay = smooth(-0.30, 0.10, e);    // sky lags the lights: twilight lift
    const dusk = Math.exp(-(((e - 0.02) / 0.24) ** 2)); // wide warm band at horizon

    // weather change detection + smoothing toward targets
    if (w !== this._weather) {
      this._weather = w;
      events.emit('env:weather', { w });
      this._ibDirty = true;
    }
    const tgt = WX[w] || WX.clear;
    const k = dt > 0 ? 1 - Math.exp(-dt * 1.4) : 1;
    const wp = this._w;
    for (const key of WX_KEYS) wp[key] = lerp(wp[key], tgt[key], k);

    // --- sky dome: blend the palette in JS so the fog can reuse the exact ------
    // --- horizon colour (single source of truth) --------------------------------
    _cA.copy(C.zenDay).lerp(C.zenNight, 1 - skyDay).lerp(C.zenDusk, dusk * 0.85);
    _cA.lerp(C.zenGrey, wp.grey);
    _cB.copy(C.horDay).lerp(C.horNight, 1 - skyDay).lerp(C.horDusk, dusk);
    _cB.lerp(C.horGrey, wp.grey);
    _cA.multiplyScalar(wp.dim);
    _cB.multiplyScalar(wp.dim);
    const u = this._skyMat.uniforms;
    u.uZenith.value.copy(_cA);
    u.uHorizon.value.copy(_cB);
    u.uSunDir.value.set(sd.x, sd.y, sd.z);
    _moonV.set(-sd.x, 0.5, -sd.z).normalize();
    u.uMoonDir.value.copy(_moonV);
    u.uDusk.value = dusk;
    u.uNight.value = night;
    u.uHaze.value = 0.15 + wp.grey * 0.75;
    _cA.copy(C.sunLow).lerp(C.sunHigh, smooth(0.0, 0.65, e));
    if (wp.grey > 0) _cA.lerp(C.sunGrey, wp.grey * 0.5);
    u.uSunCol.value.copy(_cA);

    // --- sun --------------------------------------------------------------------
    this.sun.color.copy(_cA);
    this.sun.intensity = 2.3 * sunUp * wp.sun;
    this.sun.visible = sunUp > 0.001; // below the horizon: no light, no shadow pass
    _sunV.set(sd.x, sd.y, sd.z).multiplyScalar(1200);
    this.sun.position.copy(_sunV);
    // shadow map only re-renders when the sun actually moves (or visibility flips)
    const sp = this.sun.position;
    if (sp.x !== this._shx || sp.y !== this._shy || sp.z !== this._shz || this.sun.visible !== this._shvis) {
      this._shx = sp.x; this._shy = sp.y; this._shz = sp.z; this._shvis = this.sun.visible;
      this.sun.shadow.needsUpdate = true;
    }

    // --- hemisphere fill ----------------------------------------------------------
    _cA.copy(C.hemiTopDay).lerp(C.hemiTopNight, night).lerp(C.hemiTopDusk, dusk * 0.7);
    this.hemi.color.copy(_cA);
    _cA.copy(C.hemiBotDay).lerp(C.hemiBotNight, night);
    this.hemi.groundColor.copy(_cA);
    this.hemi.intensity = (0.65 * day + 0.28 * night) * wp.hemi;

    // --- moon (cool, opposite the sun's azimuth, never shadow-casting) ------------
    this.moon.position.copy(_moonV).multiplyScalar(1200);
    this.moon.intensity = 0.7 * (1 - sunUp) * wp.moon;

    // --- stars ---------------------------------------------------------------------
    this._stars.material.opacity = clamp((1 - sunUp) * wp.star, 0, 1) * 0.9;

    // --- clouds: drift, billboard toward the camera, tint by the sun --------------
    const camPos = camera.position;
    if (dt > 0) {
      for (const c of this._clouds) {
        c.position.x += c.userData.speed * dt;
        if (c.position.x > c.userData.wrap) c.position.x = -c.userData.wrap;
      }
    }
    for (const c of this._clouds) {
      c.rotation.y = Math.atan2(camPos.x - c.position.x, camPos.z - c.position.z);
    }
    _cA.copy(C.cloudDay).lerp(C.cloudNight, 1 - skyDay).lerp(C.cloudDusk, dusk * 0.8);
    if (wp.grey > 0) _cA.lerp(C.horGrey, wp.grey * 0.5);
    _cA.multiplyScalar(wp.dim);
    this._cloudMat.color.copy(_cA);
    this._cloudMat.opacity = wp.cov * (0.65 + 0.35 * Math.max(skyDay, dusk));

    // --- fog: colour MUST match the sky's horizon band ------------------------------
    this.fog.color.copy(_cB);
    this.fog.density = wp.fogDen * (0.85 + 0.35 * dusk);

    // --- keep the distant group centered on the camera (dome < far) -----------------
    this._skyGroup.position.set(camPos.x, 0, camPos.z);

    // --- IBL: refresh a few times per simulated day, rate-limited -------------------
    this._ibAcc += dt;
    if (this._ibDirty && this._ibAcc >= 1.5) this._regenIBL(false);
    else if (this._ibAcc >= 3.0 && Math.abs(t - this._ibT) > 0.004) this._regenIBL(false);

    // --- env:sun ---------------------------------------------------------------------
    const si = this._sunInfo;
    si.dir.x = sd.x; si.dir.y = sd.y; si.dir.z = sd.z;
    si.color = this.sun.color.getHex();
    si.intensity = this.sun.intensity;
    si.elevation = e;
    events.emit('env:sun', si);
  }

  _regenIBL(force) {
    if (!this._pmrem) return;
    try {
      const u = this._skyMat.uniforms;
      u.uDisc.value = 0; // hide discs in the env map (avoids hotspot artifacts)
      const rt = this._pmrem.fromScene(this._ibScene, 0, 0.1, 100);
      u.uDisc.value = 1;
      if (this._ibRT) this._ibRT.dispose();
      this._ibRT = rt;
      this._ibT = this.ctx.clock.t;
      this._ibAcc = 0;
      if (force || this._ibDirty) this._ibDirty = false;
      const s = this._liveScene;
      if (s) { s.environment = rt.texture; s.environmentIntensity = 0.35; }
    } catch (err) {
      console.warn('[environment] IBL refresh skipped:', err && err.message);
    }
  }

  setTimeOfDay(t) { if (this.ctx) this.ctx.clock.setTimeOfDay(t); }
  setWeather(w) { if (this.ctx) this.ctx.clock.setWeather(w); }

  // Dispose showcase geometry we built (shared cached materials are kept).
  _disposeExtra(o) {
    o.traverse((n) => { if (n.geometry) n.geometry.dispose(); });
  }

  // Showcase cloud staging: the live field is seeded randomly around the map,
  // so a given seed can leave a view nearly cloudless. For the showcase we
  // restage the same 16 billboards deterministically.
  //
  // The orbit preset (150,95,150 -> origin, 50 deg fov) pitches ~24 deg down,
  // so the frame's sky is only the horizon band: roughly elev -3..+1 deg,
  // azimuth 225 +/- 35 deg. The arc clouds are staged as a low bank hugging
  // that haze line, behind the (lowered, orbit-facing) mountain ring; the
  // rest form a high scattered ring for the street/skyline/aerial presets.
  _stageShowcaseClouds(T) {
    const rc = this._rng.fork('envshowclouds');
    const D2R = Math.PI / 180;
    for (let i = 0; i < this._clouds.length; i++) {
      const c = this._clouds[i];
      if (i < 10) {
        // horizon bank facing the orbit camera (elevation -2..+3 deg)
        const az = (225 + rc.range(-33, 33)) * D2R;
        const dist = rc.range(900, 1600);
        const el = rc.range(-1, 4) * D2R;
        const horiz = dist * Math.cos(el);
        // _skyGroup already sits at the camera's x/z, so local == camera-relative
        c.position.set(
          horiz * Math.cos(az),
          95 + dist * Math.sin(el),
          horiz * Math.sin(az),
        );
      } else {
        // scattered high ring (street/skyline/aerial presets)
        const ang = rc.range(0, Math.PI * 2);
        const rad = rc.range(600, 1500);
        c.position.set(Math.cos(ang) * rad, rc.range(180, 430), Math.sin(ang) * rad);
      }
      const w = rc.range(900, 1400);
      c.scale.set(w, w * rc.range(0.35, 0.50), 1);
      c.userData.speed = rc.range(2, 6);
      c.userData.wrap = 2300;
    }
  }

  // A low-poly mountain ring with strata bands + vegetation base + snowline
  // caps for the showcase; fog:true (default) melts the base into the
  // horizon haze.
  //
  // The rim must READ as mountains, not silhouettes, at 430-560 m:
  //  - 18 height segments so the strata band EDGES land on their own faces
  //    (fewer segments interpolate the banding away into a flat gradient);
  //  - three vertical zones in vertex colour: a dark vegetation skirt at the
  //    foot, DISCRETE high-contrast rock strata layers (each layer is a flat
  //    colour so the bands read as hard horizontal stripes, not a ramp), and
  //    a wavy snowline cap whose height varies per peak and ripples around
  //    each cone's azimuth;
  //  - flat shading gives the low-poly facets hard light/shadow breaks that
  //    echo the banding under the raking dusk sun.
  _makeMountains(scene, T) {
    const rm = this._rng.fork('envshowmtn');
    const group = new T.Group();
    const N = 14;
    for (let i = 0; i < N; i++) {
      const ang = (i / N) * Math.PI * 2 + rm.range(-0.14, 0.14);
      const rad = rm.range(430, 560);
      // The orbit camera's only visible sky is the horizon band behind the
      // ring's 225-deg-facing arc; keep those peaks low (below the 95 m
      // camera) so the staged cloud bank reads over them. The rest of the
      // ring stays tall for silhouette interest in the other presets.
      const deg = ((ang * 180 / Math.PI) % 360 + 360) % 360;
      const facing = deg > 180 && deg < 270;
      const h = facing ? rm.range(35, 85) : rm.range(70, 175);
      const r = facing ? rm.range(95, 150) : rm.range(95, 195);
      const geo = new T.ConeGeometry(r, h, 9 + (i % 3), 18);
      const p = geo.attributes.position;
      const col = new Float32Array(p.count * 3);
      // per-peak terrain params (deterministic: drawn from the forked stream
      // in a fixed order)
      const tint  = rm.range(0.92, 1.10);         // overall rock lightness
      const layers = 6 + (i % 3);                 // 6..8 discrete strata
      const vegTop = rm.range(0.20, 0.32);        // vegetation belt top (0..1)
      const vegSoft = rm.range(0.05, 0.09);       // veg -> rock blend width
      const snBase = rm.range(0.58, 0.74);        // snowline height (0..1)
      const ph1 = rm.range(0, 6.283);
      const ph2 = rm.range(0, 6.283);
      for (let v = 0; v < p.count; v++) {
        const px = p.getX(v), py = p.getY(v), pz = p.getZ(v);
        const hy = py + h / 2;                    // 0..h from the base
        const fy = hy / h;
        const az = Math.atan2(pz, px);
        // --- rock strata: DISCRETE horizontal layers (crisp stripe edges) ---
        const layer = Math.min(layers - 1, Math.floor(fy * layers));
        const L = 0.5 + 0.5 * Math.sin(layer * 1.9 + ph1); // per-layer light
        const fine = 0.92 + 0.08 * Math.sin(hy * 0.7 + ph2); // subtle texture
        const s = (0.70 + 0.55 * L) * tint * fine;          // strong contrast
        let cr = 0.40 * s, cg = 0.355 * s, cb = 0.315 * s;
        // --- vegetation skirt: dark green at the foot, sharp break to rock --
        const veg = 1 - smooth(vegTop, vegTop + vegSoft, fy);
        const canopy = 0.82 + 0.30 * L;           // crown shade follows layer
        cr = cr * (1 - veg) + 0.075 * canopy * veg;
        cg = cg * (1 - veg) + 0.150 * canopy * veg;
        cb = cb * (1 - veg) + 0.058 * canopy * veg;
        // --- snowline: wavy cap, irregular around the azimuth ---------------
        const sn = snBase + 0.05 * Math.sin(az * 3 + ph1) + 0.028 * Math.sin(az * 7 + ph2);
        const snow = smooth(sn - 0.03, sn + 0.05, fy);
        cr = cr * (1 - snow) + 0.88 * snow;
        cg = cg * (1 - snow) + 0.92 * snow;
        cb = cb * (1 - snow) + 0.98 * snow;
        col[v * 3] = cr; col[v * 3 + 1] = cg; col[v * 3 + 2] = cb;
      }
      geo.setAttribute('color', new T.BufferAttribute(col, 3));
      const mat = new T.MeshStandardMaterial({
        vertexColors: true, flatShading: true, roughness: 0.95, metalness: 0,
      });
      this._showMtnMats.push(mat);
      const m = new T.Mesh(geo, mat);
      m.position.set(Math.cos(ang) * rad, h / 2 - 4, Math.sin(ang) * rad);
      m.rotation.y = rm.range(0, Math.PI);
      m.receiveShadow = true;
      group.add(m);
    }
    scene.add(group);
    this._showExtras.push(group);
  }

  // Stage a FRESH scene: full sky system — dome + sun + clouds + fog + stars,
  // a ground plane, a mountain ring, and a few PBR masses for scale/shadows.
  showcase(scene, world, ctx) {
    const T = ctx.three;
    scene.fog = this.fog;
    if (!scene.children.includes(this.root)) scene.add(this.root);
    if (!scene.children.includes(this._skyGroup)) scene.add(this._skyGroup);
    this._liveScene = scene;
    this._park(scene);
    if (this._ibRT) { scene.environment = this._ibRT.texture; scene.environmentIntensity = 0.5; }

    // reset previous staging extras
    for (const o of this._showExtras) { o.removeFromParent(); this._disposeExtra(o); }
    this._showExtras.length = 0;
    for (const m of this._showMtnMats) m.dispose();
    this._showMtnMats.length = 0;

    // drop the harness's small staging ground; the showcase brings its own
    for (const c of [...scene.children]) {
      if (c.isMesh && c !== this.sky && !this._skyGroup.children.includes(c)) scene.remove(c);
    }

    const add = (mesh) => { scene.add(mesh); this._showExtras.push(mesh); return mesh; };

    const g = new T.Mesh(new T.PlaneGeometry(1600, 1600), ctx.assets.material('grass', {}));
    g.rotation.x = -Math.PI / 2;
    g.receiveShadow = true;
    add(g);

    this._makeMountains(scene, T);
    this._stageShowcaseClouds(T);

    // a few PBR masses: one tower, a mid block, a low slab
    const box = (name, w, h, d, x, y, z) => {
      const m = new T.Mesh(new T.BoxGeometry(w, h, d), ctx.assets.material(name, {}));
      m.position.set(x, y, z);
      m.castShadow = true;
      m.receiveShadow = true;
      add(m);
    };
    box('facade', 26, 96, 26, -70, 48, -50);
    box('concrete', 44, 44, 44, 85, 22, 25);
    box('roof', 90, 14, 60, 10, 7, 120);

    this.update(0, world);
  }

  dispose() {
    for (const o of this._showExtras) { o.removeFromParent(); this._disposeExtra(o); }
    this._showExtras.length = 0;
    for (const m of this._showMtnMats) m.dispose();
    this._showMtnMats.length = 0;
    for (const o of this._parked) { if (o.userData && 'envSavedI' in o.userData) o.intensity = o.userData.envSavedI; }
    this._parked.length = 0;
    if (this.root) this.root.removeFromParent();
    if (this._skyGroup) this._skyGroup.removeFromParent();
    if (this.sky) { this.sky.geometry.dispose(); this._skyMat.dispose(); }
    if (this._stars) { this._stars.geometry.dispose(); this._stars.material.dispose(); }
    if (this._cloudGeo) this._cloudGeo.dispose();
    if (this._cloudMat) this._cloudMat.dispose();
    if (this._cloudTex) { this._cloudTex.dispose(); this._cloudTex = null; }
    this._clouds.length = 0;
    if (this._ibRT) { this._ibRT.dispose(); this._ibRT = null; }
    if (this._pmrem) { this._pmrem.dispose(); this._pmrem = null; }
    if (this.ctx && this.ctx.scene) {
      if (this.ctx.scene.fog === this.fog) this.ctx.scene.fog = null;
      this.ctx.scene.environment = null;
      this.ctx.scene.environmentIntensity = 1;
    }
    this._liveScene = null;
    this.ctx = null;
  }

  stats() {
    return {
      drawCalls: 18,
      notes: '1 sky dome + 1 stars Points + 16 cloud billboards; sun/hemi/moon/fog cost 0 draws (shadow pass aside); PMREM IBL refreshed a few times per sim day',
    };
  }
}
