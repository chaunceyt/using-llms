import { CONFIG } from './config.js';

const LAPS = CONFIG.RACE.LAPS;

function ordinal(n) {
  const s = ['th', 'st', 'nd', 'rd'], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}
function fmtTime(t) {
  if (t == null) return '—';
  const m = Math.floor(t / 60);
  const sec = t - m * 60;
  return m + ':' + sec.toFixed(3).padStart(6, '0');
}
function hex(c) { return '#' + c.toString(16).padStart(6, '0'); }
function itemInfo(item) {
  switch (item) {
    case 'mushroom': return { label: 'MUSHROOM', color: '#e63946' };
    case 'shell': return { label: 'SHELL', color: '#2fa84f' };
    case 'banana': return { label: 'BANANA', color: '#f7d02b' };
    case 'star': return { label: 'STAR', color: '#ffd800' };
    default: return { label: '?', color: 'rgba(255,255,255,0.25)' };
  }
}

export function createHUD(containerEl, track) {
  const style = document.createElement('style');
  style.textContent = `
    .tk-root{position:absolute;inset:0;pointer-events:none;font-family:"Segoe UI",system-ui,sans-serif;}
    .tk-shadow{text-shadow:0 2px 6px rgba(0,0,0,.6);}
    .tk-tl{position:absolute;top:18px;left:22px;}
    .tk-pos{font-size:64px;font-weight:800;line-height:1;}
    .tk-lap{font-size:22px;font-weight:700;opacity:.9;margin-top:2px;}
    .tk-tc{position:absolute;top:18px;left:50%;transform:translateX(-50%);text-align:center;}
    .tk-time{font-size:34px;font-weight:800;}
    .tk-lastlap{font-size:16px;opacity:.8;}
    .tk-br{position:absolute;bottom:22px;right:24px;text-align:right;}
    .tk-speed{font-size:56px;font-weight:800;line-height:1;}
    .tk-kmh{font-size:16px;font-weight:700;opacity:.8;}
    .tk-item{position:absolute;bottom:96px;right:24px;width:74px;height:74px;border-radius:14px;
      background:rgba(255,255,255,.12);border:2px solid rgba(255,255,255,.35);
      display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px;}
    .tk-map{position:absolute;top:18px;right:20px;background:rgba(10,16,30,.45);
      border-radius:12px;padding:6px;border:1px solid rgba(255,255,255,.25);}
    .tk-count{position:absolute;top:34%;left:50%;transform:translate(-50%,-50%);
      font-size:150px;font-weight:900;}
    .tk-lapflash{position:absolute;top:42%;left:50%;transform:translate(-50%,-50%);
      font-size:64px;font-weight:900;}
    .tk-hidden{display:none !important;}
    @keyframes tk-pop{0%{transform:translate(-50%,-50%) scale(1.6);opacity:0;}
      25%{opacity:1;}100%{transform:translate(-50%,-50%) scale(1);opacity:.9;}}
    @keyframes tk-lap{0%{transform:translate(-50%,-50%) scale(.6);opacity:0;}
      20%{opacity:1;}80%{opacity:1;}100%{transform:translate(-50%,-50%) scale(1);opacity:0;}}
    .tk-center{position:absolute;inset:0;display:flex;flex-direction:column;
      align-items:center;justify-content:center;background:rgba(8,12,24,.55);}
    .tk-title{font-size:84px;font-weight:900;letter-spacing:2px;
      background:linear-gradient(#ffd800,#ff7b00);-webkit-background-clip:text;background-clip:text;color:transparent;}
    .tk-sub{font-size:26px;font-weight:700;margin-top:6px;}
    .tk-controls{font-size:16px;opacity:.85;margin-top:22px;line-height:1.7;text-align:center;}
    .tk-swatches{display:flex;gap:14px;margin-top:28px;}
    .tk-sw{width:44px;height:44px;border-radius:50%;transition:transform .12s;
      border:3px solid rgba(255,255,255,.4);}
    .tk-res{background:rgba(8,12,24,.7);border-radius:16px;padding:28px 40px;}
    .tk-res h2{margin:0 0 16px;font-size:40px;font-weight:900;}
    .tk-row{display:flex;justify-content:space-between;gap:40px;font-size:20px;
      font-weight:700;padding:5px 0;}
    .tk-row.you{color:#ffd800;}
  `;
  containerEl.appendChild(style);

  const root = document.createElement('div');
  root.className = 'tk-root';
  containerEl.appendChild(root);

  // ---- in-race HUD ----
  const raceHud = document.createElement('div');
  raceHud.className = 'tk-hidden';
  root.appendChild(raceHud);

  const tl = document.createElement('div');
  tl.className = 'tk-tl tk-shadow';
  tl.innerHTML = '<div class="tk-pos">8th</div><div class="tk-lap">LAP 1/3</div>';
  raceHud.appendChild(tl);
  const posEl = tl.querySelector('.tk-pos');
  const lapEl = tl.querySelector('.tk-lap');

  const tc = document.createElement('div');
  tc.className = 'tk-tc tk-shadow';
  tc.innerHTML = '<div class="tk-time">0:00.000</div><div class="tk-lastlap"></div>';
  raceHud.appendChild(tc);
  const timeEl = tc.querySelector('.tk-time');
  const lastLapEl = tc.querySelector('.tk-lastlap');

  const br = document.createElement('div');
  br.className = 'tk-br tk-shadow';
  br.innerHTML = '<div class="tk-speed">0</div><div class="tk-kmh">km/h</div>';
  raceHud.appendChild(br);
  const speedEl = br.querySelector('.tk-speed');

  const itemEl = document.createElement('div');
  itemEl.className = 'tk-item';
  itemEl.textContent = '?';
  raceHud.appendChild(itemEl);

  const mapWrap = document.createElement('div');
  mapWrap.className = 'tk-map';
  const minimap = document.createElement('canvas');
  minimap.width = 180; minimap.height = 180;
  mapWrap.appendChild(minimap);
  raceHud.appendChild(mapWrap);

  const countEl = document.createElement('div');
  countEl.className = 'tk-count tk-shadow tk-hidden';
  countEl.style.color = '#ffd800';
  root.appendChild(countEl);

  const lapFlash = document.createElement('div');
  lapFlash.className = 'tk-lapflash tk-shadow tk-hidden';
  root.appendChild(lapFlash);

  // ---- minimap road (drawn once) ----
  const mm = minimap.getContext('2d');
  const cl = track.centerlineXZ, n = track.sampleCount;
  let minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9;
  for (let i = 0; i < n; i++) {
    const x = cl[i * 2], z = cl[i * 2 + 1];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const pad = 16;
  const scale = Math.min((180 - 2 * pad) / (maxX - minX), (180 - 2 * pad) / (maxZ - minZ));
  const offX = (180 - (maxX - minX) * scale) / 2;
  const offY = (180 - (maxZ - minZ) * scale) / 2;
  function toMap(x, z) { return [(x - minX) * scale + offX, (z - minZ) * scale + offY]; }
  function drawRoad() {
    mm.clearRect(0, 0, 180, 180);
    mm.strokeStyle = 'rgba(255,255,255,0.8)';
    mm.lineWidth = 6; mm.lineJoin = 'round'; mm.lineCap = 'round';
    mm.beginPath();
    for (let i = 0; i <= n; i++) {
      const ii = i % n;
      const [mx, my] = toMap(cl[ii * 2], cl[ii * 2 + 1]);
      if (i === 0) mm.moveTo(mx, my); else mm.lineTo(mx, my);
    }
    mm.stroke();
  }
  function drawKarts(karts) {
    drawRoad();
    for (const k of karts) {
      const [mx, my] = toMap(k.state.x, k.state.z);
      mm.fillStyle = k.isPlayer ? '#ffffff' : hex(CONFIG.COLORS.KARTS[k.index]);
      mm.beginPath();
      mm.arc(mx, my, k.isPlayer ? 5 : 3.5, 0, Math.PI * 2);
      mm.fill();
    }
  }

  // ---- menu ----
  const menuEl = document.createElement('div');
  menuEl.className = 'tk-center';
  menuEl.innerHTML =
    '<div class="tk-title">TURBO KART</div>' +
    '<div class="tk-sub">Press ENTER to race</div>' +
    '<div class="tk-controls">WASD / Arrows — drive&nbsp;&nbsp;·&nbsp;&nbsp;Shift / Space — drift<br>' +
    'E — use item&nbsp;&nbsp;·&nbsp;&nbsp;← → — pick kart color</div>' +
    '<div class="tk-swatches"></div>';
  root.appendChild(menuEl);
  const swWrap = menuEl.querySelector('.tk-swatches');
  const swatches = [];
  for (let i = 0; i < CONFIG.COLORS.KARTS.length; i++) {
    const sw = document.createElement('div');
    sw.className = 'tk-sw';
    sw.style.background = hex(CONFIG.COLORS.KARTS[i]);
    swWrap.appendChild(sw);
    swatches.push(sw);
  }

  // ---- results ----
  const resultsEl = document.createElement('div');
  resultsEl.className = 'tk-center tk-hidden';
  resultsEl.innerHTML = '<div class="tk-res"><h2>RACE COMPLETE</h2><div class="tk-rows"></div>' +
    '<div class="tk-sub" style="margin-top:18px;font-size:18px;">Press R for menu</div></div>';
  root.appendChild(resultsEl);
  const rowsEl = resultsEl.querySelector('.tk-rows');

  // ---- state ----
  let lastPos = -1, lastLap = -1, lastSpeed = -1, lastItem = undefined, lastTimeStr = '';
  let lastLapStart = 0, lastLapShown = false;
  let selColor = -1;

  function update(data) {
    if (data.mode !== 'racing' && data.mode !== 'results') { raceHud.className = 'tk-hidden'; return; }
    if (raceHud.className === 'tk-hidden') raceHud.className = '';

    if (data.position !== lastPos) { posEl.textContent = ordinal(data.position); lastPos = data.position; }
    if (data.lap !== lastLap) {
      if (lastLapShown && data.time > 0) lastLapEl.textContent = 'LAST ' + fmtTime(data.time - lastLapStart);
      lastLapStart = data.time;
      lapEl.textContent = 'LAP ' + Math.min(data.lap, LAPS) + '/' + LAPS;
      lastLap = data.lap; lastLapShown = true;
    }
    const ts = fmtTime(data.time);
    if (ts !== lastTimeStr) { timeEl.textContent = ts; lastTimeStr = ts; }
    const sp = Math.max(0, Math.round(data.speed * 3.6));
    if (sp !== lastSpeed) { speedEl.textContent = sp; lastSpeed = sp; }
    if (data.item !== lastItem) {
      const info = itemInfo(data.item);
      itemEl.textContent = info.label;
      itemEl.style.color = info.color;
      lastItem = data.item;
    }
    drawKarts(data.karts);
  }

  function showMenu() {
    menuEl.className = 'tk-center';
    raceHud.className = 'tk-hidden';
    resultsEl.className = 'tk-center tk-hidden';
  }
  function hideMenu() { menuEl.className = 'tk-center tk-hidden'; }

  function setSelectedColor(i) {
    if (i === selColor) return;
    selColor = i;
    swatches.forEach((sw, ix) => {
      sw.style.transform = ix === i ? 'scale(1.25)' : 'scale(1)';
      sw.style.borderColor = ix === i ? '#ffffff' : 'rgba(255,255,255,.4)';
    });
  }

  function setCountdown(v) {
    if (v === null) { countEl.className = 'tk-count tk-shadow tk-hidden'; return; }
    countEl.className = 'tk-count tk-shadow';
    countEl.style.color = v === 0 ? '#7CFC00' : '#ffd800';
    countEl.textContent = v === 0 ? 'GO!' : String(v);
    countEl.style.animation = 'none';
    void countEl.offsetWidth;
    countEl.style.animation = 'tk-pop .9s ease-out';
  }

  function flashLap(lap) {
    lapFlash.className = 'tk-lapflash tk-shadow';
    lapFlash.textContent = 'LAP ' + lap + '/' + LAPS;
    lapFlash.style.animation = 'none';
    void lapFlash.offsetWidth;
    lapFlash.style.animation = 'tk-lap 1.4s ease-out forwards';
  }

  function showResults(results) {
    rowsEl.innerHTML = '';
    for (const r of results) {
      const row = document.createElement('div');
      row.className = 'tk-row' + (r.index === 0 ? ' you' : '');
      row.innerHTML =
        '<span>' + (r.index === 0 ? 'You' : 'Kart #' + r.index) + '</span>' +
        '<span>' + ordinal(r.place) + '</span>' +
        '<span>' + fmtTime(r.time) + '</span>';
      rowsEl.appendChild(row);
    }
    resultsEl.className = 'tk-center';
  }

  return { update, showMenu, hideMenu, setSelectedColor, setCountdown, flashLap, showResults };
}
