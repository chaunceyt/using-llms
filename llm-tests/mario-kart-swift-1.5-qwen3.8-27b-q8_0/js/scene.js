import * as THREE from 'three';
import { CONFIG } from './config.js';

export function createScene(containerEl) {
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(CONFIG.COLORS.FOG, 120, 650);

  const w = containerEl.clientWidth || window.innerWidth;
  const h = containerEl.clientHeight || window.innerHeight;
  const camera = new THREE.PerspectiveCamera(60, w / h, 0.5, 800);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  containerEl.appendChild(renderer.domElement);

  // Gradient sky dome
  const skyGeo = new THREE.SphereGeometry(700, 32, 16);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    fog: false,
    depthWrite: false,
    uniforms: {
      top: { value: new THREE.Color(CONFIG.COLORS.SKY_TOP) },
      bottom: { value: new THREE.Color(CONFIG.COLORS.SKY_BOTTOM) },
    },
    vertexShader: `
      varying vec3 vWorld;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorld = wp.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 top;
      uniform vec3 bottom;
      varying vec3 vWorld;
      void main() {
        float hgt = clamp(normalize(vWorld).y, 0.0, 1.0);
        gl_FragColor = vec4(mix(bottom, top, hgt), 1.0);
      }
    `,
  });
  scene.add(new THREE.Mesh(skyGeo, skyMat));

  // Sun (directional, shadows)
  const sun = new THREE.DirectionalLight(0xfff3d6, 1.7);
  sun.position.set(90, 140, 70);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 420;
  sun.shadow.camera.left = -190;
  sun.shadow.camera.right = 190;
  sun.shadow.camera.top = 190;
  sun.shadow.camera.bottom = -190;
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  scene.add(sun.target);

  const hemi = new THREE.HemisphereLight(CONFIG.COLORS.SKY_TOP, CONFIG.COLORS.GRASS, 0.75);
  scene.add(hemi);
  scene.add(new THREE.AmbientLight(0xffffff, 0.22));

  function resize() {
    const nw = containerEl.clientWidth || window.innerWidth;
    const nh = containerEl.clientHeight || window.innerHeight;
    renderer.setSize(nw, nh);
    camera.aspect = nw / nh;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);

  function render() {
    renderer.render(scene, camera);
  }

  return { scene, camera, renderer, render };
}
