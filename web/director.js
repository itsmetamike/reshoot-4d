// Reshoot 4D Virtual Camera: a Blender-style virtual camera for 4D Gaussian captures, opened from the "Reshoot 4D: Virtual Camera" node.
// The capture is previewed as GPU-animated points (FreeTimeGS motion: p = mean + v (t - time), opacity * exp(-0.5 ((t - time)/dur)^2)).
// Keyframes hold eye, target, lens (mm, 36 mm tall sensor), roll and scene time; sampling matches director.py exactly.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import * as THREE from "./lib/three.module.min.mjs";
import { TransformControls } from "./lib/TransformControls.mjs";

const SENSOR_MM = 36;
const ASPECTS = { "9:16": 9 / 16, "4:5": 4 / 5, "1:1": 1, "16:9": 16 / 9 };   // shot aspects (w/h); Shot Prep sizes each to the same pixel budget
const nearestAspect = (a) => Object.keys(ASPECTS).reduce((b, k) => (Math.abs(Math.log(ASPECTS[k] / a)) < Math.abs(Math.log(ASPECTS[b] / a)) ? k : b), "9:16");
let ASPECT = 9 / 16;   // shot aspect (w/h): ASPECTS[state.aspect]
const lensToFov = (mm) => (2 * Math.atan(SENSOR_MM / (2 * mm)) * 180) / Math.PI;
const fovToLens = (deg) => SENSOR_MM / (2 * Math.tan((deg * Math.PI) / 360));

// ---------- sampler (keep in step with director.py) ----------
const ease = (kind, u) => (kind === "linear" ? u : kind === "hold" ? (u < 1 ? 0 : 1) : u * u * (3 - 2 * u));
const cr = (p0, p1, p2, p3, u) => p1.map((_, i) => {
  const a = p0[i], b = p1[i], c = p2[i], d = p3[i], u2 = u * u, u3 = u2 * u;
  return 0.5 * (2 * b + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u2 + (-a + 3 * b - 3 * c + d) * u3);
});
function shake(t, amp) {
  if (amp <= 0) return [0, 0, 0];
  const w = 2 * Math.PI, s = Math.sin;
  return [amp * (0.5 * s(w * 1.3 * t) + 0.3 * s(w * 2.9 * t + 1.7) + 0.2 * s(w * 5.1 * t + 0.4)),
          amp * (0.5 * s(w * 1.1 * t + 2.1) + 0.3 * s(w * 3.3 * t + 0.9) + 0.2 * s(w * 4.7 * t + 2.6)),
          amp * (0.5 * s(w * 0.9 * t + 4.2) + 0.3 * s(w * 2.3 * t + 3.1) + 0.2 * s(w * 6.1 * t + 1.2))];
}
let BASIS = null;   // [up, front, side] arrays from the capture's rig; set when a capture loads
const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
function sph(eye, tgt) {
  const [up, front, side] = BASIS, rel = eye.map((v, i) => v - tgt[i]), r = Math.hypot(...rel) || 1e-9;
  return [Math.atan2(dot(rel, side), dot(rel, front)), Math.asin(Math.max(-1, Math.min(1, dot(rel, up) / r))), r];
}
function unwrapped(keys) {
  let prev = null;
  return keys.map((k) => { const s = sph(k.eye, k.target); if (prev !== null) s[0] += 2 * Math.PI * Math.round((prev - s[0]) / (2 * Math.PI)); prev = s[0]; return s; });
}
function sceneAuto(state, keys, t) {
  let frozen = 0;
  for (let i = 0; i < keys.length - 1; i++) if (keys[i].freeze && t > keys[i].t) frozen += Math.max(0, Math.min(t, keys[i + 1].t) - keys[i].t);
  let x = Math.max(0, ((state.speed ?? 1) * (t - frozen)) / Math.max(state.capture_seconds || 1, 1e-6));
  const end = state.scene_end || "pingpong";
  if (end === "loop") return x % 1;
  if (end === "pingpong") { x = x % 2; return x <= 1 ? x : 2 - x; }
  return Math.min(1, x);
}
let TRACK = null;   // {head: [[x,y,z]...], body: [...]} per capture frame, from the capture's body mesh
function trackAt(name, scene) {
  const pts = TRACK[name], x = Math.min(1, Math.max(0, scene)) * (pts.length - 1), i = Math.min(Math.floor(x), pts.length - 2), f = x - i;
  return pts[i].map((a, k) => a + (pts[i + 1][k] - a) * f);
}
function sceneAt(state, keys, t, i, j, e) {
  return (state.scene_mode || "keys") === "auto" ? sceneAuto(state, keys, t) : Math.min(1, Math.max(0, keys[i].scene + (keys[j].scene - keys[i].scene) * e));
}
function followDelta(scene) {   // her travel through the setting at scene time, from the middle of the clip
  const a = trackAt("follow", scene), b = trackAt("follow", 0.5); return a.map((v, k) => v - b[k]);
}
const EASE_LABEL = { smooth: "ease in/out", linear: "linear", hold: "hold" };   // stored values stay smooth / linear / hold
function shotTime(state, keys, t) {   // camera time under the whole-shot ease (director.py shot_time); scene time stays t
  if (state.shot_ease !== "smooth" || keys.length < 2) return t;
  const t0 = keys[0].t, t1 = keys[keys.length - 1].t;
  if (t <= t0 || t >= t1 || t1 - t0 < 1e-6) return t;
  const x = (t - t0) / (t1 - t0); return t0 + (t1 - t0) * x * x * x * (x * (6 * x - 15) + 10);
}
function sample(state, t, dist0) {
  const keys = [...state.keys].sort((a, b) => a.t - b.t);
  if (!keys.length) return null;
  const tr = t; t = shotTime(state, keys, t);   // tr: real time (scene, shake); t: camera time
  let i = 0, u = 0;
  if (keys.length === 1 || t <= keys[0].t) { i = 0; u = 0; }
  else if (t >= keys[keys.length - 1].t) { i = keys.length - 2; u = 1; }
  else { for (let j = 0; j < keys.length - 1; j++) if (keys[j].t <= t) i = j; u = (t - keys[i].t) / Math.max(keys[i + 1].t - keys[i].t, 1e-6); }
  const kind = keys[i].ease || "smooth", j = Math.min(i + 1, keys.length - 1);
  const e = ease(state.shot_ease === "smooth" && kind !== "hold" ? "linear" : kind, u);
  const scene = sceneAt(state, keys, tr, i, j, e);
  const K = (k) => keys[Math.max(0, Math.min(keys.length - 1, k))];
  const aim = (k, sc) => (TRACK && TRACK[k.aim] ? trackAt(k.aim, sc) : k.target);
  const keyScene = (k) => { const n = keys.indexOf(k); return sceneAt(state, keys, k.t, n, n, 0); };
  let tgt = cr(...[i - 1, i, j, j + 1].map((n) => aim(K(n), scene)), e), eye;
  if ((state.path || "orbit") === "orbit" && BASIS) {
    const s = [i - 1, i, j, j + 1].map((n) => { const k = K(n); return sph(k.eye, aim(k, keyScene(k))); });
    for (let n = 1; n < 4; n++) s[n][0] += 2 * Math.PI * Math.round((s[n - 1][0] - s[n][0]) / (2 * Math.PI));
    const [az, el, r] = cr(s[0], s[1], s[2], s[3], e), [up, front, side] = BASIS;
    eye = tgt.map((v, k) => v + r * (Math.cos(el) * (Math.cos(az) * front[k] + Math.sin(az) * side[k]) + Math.sin(el) * up[k]));
  } else eye = cr(K(i - 1).eye, K(i).eye, K(j).eye, K(j + 1).eye, e);
  const lerp = (n) => keys[i][n] + (keys[j][n] - keys[i][n]) * e;
  const sh = shake(tr, (state.shake || 0) * 0.012 * dist0);
  eye = eye.map((v, k) => v + sh[k]); tgt = tgt.map((v, k) => v + 0.5 * sh[k]);
  if (TRACK && TRACK.follow && state.follow !== false && !TRACK[keys[i].aim]) {
    // follow: a fixed-target camera travels with her through the setting (as the source camera did); keys are relative to the middle of the
    // clip, where the capture's centre is
    const d = followDelta(scene);
    eye = eye.map((v, k) => v + d[k]); tgt = tgt.map((v, k) => v + d[k]);
  }
  return { scene, eye, target: tgt, roll: lerp("roll"), lens: lerp("lens") };
}

// ---------- shaders ----------
const VERT = `
attribute vec3 aVel; attribute float aTime; attribute float aDur; attribute float aOpa;
uniform float uScene; uniform float uUseVel; uniform float uPx; uniform float uSize;
varying vec3 vColor; varying float vAlpha;
void main() {
  vec3 p = position + (uUseVel > 0.5 ? aVel * (uScene - aTime) : vec3(0.0));
  float d = (uScene - aTime) / aDur;
  vAlpha = aOpa * exp(-0.5 * d * d);
  vColor = color;
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp(uPx * uSize / -mv.z, 1.0, 18.0);
}`;
const FRAG = `
varying vec3 vColor; varying float vAlpha;
void main() {
  vec2 c = gl_PointCoord - 0.5; if (dot(c, c) > 0.25 || vAlpha < 0.25) discard;
  gl_FragColor = vec4(vColor, 1.0);
}`;

// ---------- styles ----------
// brutus brand: lime #C8FF00 on black, wide rounded display type (Unbounded, SIL OFL, vendored in lib/fonts)
const FONT_URL = new URL("./lib/fonts/Unbounded-latin.woff2", import.meta.url).href;
const BODY_FONT_URL = new URL("./lib/fonts/SpaceGrotesk-latin.woff2", import.meta.url).href;   // UI text (SIL OFL)
const CSS = `
@font-face{font-family:"Unbounded";src:url("${FONT_URL}") format("woff2");font-weight:200 900;font-display:swap}
@font-face{font-family:"Space Grotesk";src:url("${BODY_FONT_URL}") format("woff2");font-weight:300 700;font-display:swap}
.s3d-dir{position:fixed;inset:0;z-index:100000;background:#000;color:#e6e6e6;font:13px/1.4 "Space Grotesk",system-ui,sans-serif;display:grid;grid-template-rows:52px minmax(0,1fr) 196px;grid-template-columns:minmax(0,1fr) 290px}
.s3d-dir *{box-sizing:border-box}
.s3d-dir header{grid-column:1/3;display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:12px;align-items:center;padding:0 12px;border-bottom:1px solid #222;white-space:nowrap}
.s3d-dir .zone{display:flex;align-items:center;gap:6px;min-width:0}
.s3d-dir .zone.mid{justify-content:center}
.s3d-dir .sep{width:1px;height:22px;background:#2a2a2a;margin:0 4px}
.s3d-dir header b{background:#c8ff00;color:#000;font:800 12px/1 "Unbounded",system-ui,sans-serif;letter-spacing:.06em;padding:7px 11px;border-radius:7px;margin-right:6px}
.s3d-dir button,.s3d-dir select,.s3d-dir input{background:#151515;color:#e6e6e6;border:1px solid #2a2a2a;border-radius:6px;padding:5px 8px;font:inherit}
.s3d-dir button{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
.s3d-dir button:hover{border-color:#c8ff00}
.s3d-dir button:focus-visible,.s3d-dir summary:focus-visible{outline:2px solid #c8ff00;outline-offset:2px}
.s3d-dir button.on{background:#c8ff00;color:#000;border-color:#c8ff00}
.s3d-dir button.primary{background:#c8ff00;color:#000;font:700 11px/1 "Unbounded",system-ui,sans-serif;letter-spacing:.04em;border-color:#c8ff00;border-radius:7px;padding:8px 12px}
.s3d-dir button.ib{width:32px;height:32px;padding:6px;justify-content:center}
.s3d-dir button.quiet{background:transparent;border-color:transparent;color:#bdbdbd}
.s3d-dir button.quiet:hover{border-color:#333;color:#e6e6e6}
.s3d-dir button.quiet.on{background:#26301a;color:#c8ff00;border-color:#3a4a1f}
.s3d-dir svg{width:18px;height:18px;flex:none}
.s3d-dir .sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.s3d-dir .presets{width:150px;min-width:0}
.s3d-dir .popwrap{position:relative;display:inline-flex}
.s3d-dir .pop{position:absolute;top:calc(100% + 8px);right:0;z-index:5;min-width:270px;background:#101010;border:1px solid #2a2a2a;border-radius:8px;padding:12px;box-shadow:0 12px 32px #000a;white-space:normal}
.s3d-dir .pop.left{left:0;right:auto}
.s3d-dir [hidden]{display:none !important}
.s3d-dir .grp{margin:0 0 12px}.s3d-dir .grp:last-child{margin-bottom:0}
.s3d-dir .grp h4{margin:0 0 8px;font:600 10px/1.2 "Unbounded",system-ui,sans-serif;letter-spacing:.12em;color:#8a8a8a;text-transform:uppercase}
.s3d-dir .row{display:grid;grid-template-columns:110px 1fr;gap:8px;align-items:center;margin:5px 0;color:#9a9a9a}
.s3d-dir .row input,.s3d-dir .row select{width:100%}
.s3d-dir .menu{display:flex;flex-direction:column;gap:2px;min-width:220px}
.s3d-dir .menu button{background:transparent;border-color:transparent;justify-content:flex-start;width:100%}
.s3d-dir .menu button:hover{background:#1a1a1a;border-color:#2a2a2a}
.s3d-dir .menu kbd,.s3d-dir .keys kbd{margin-left:auto;color:#777;font:11px system-ui,sans-serif}
.s3d-dir .keys{width:380px}.s3d-dir .keys .k{display:flex;gap:10px;margin:3px 0;color:#aaa}.s3d-dir .keys kbd{margin-left:0;min-width:130px;flex:none;color:#c8ff00}
.s3d-dir .vp{position:relative;overflow:hidden;background:#000}
.s3d-dir .vp canvas{display:block;width:100%;height:100%}
.s3d-dir .vp canvas.navg{position:absolute;right:12px;top:12px;width:110px;height:110px;cursor:pointer;display:none;border-radius:50%}
.s3d-dir .vp canvas.navg:hover{background:rgba(255,255,255,.06)}
.s3d-dir .hud{position:absolute;left:10px;top:8px;color:#9a9a9a;font-size:12px;pointer-events:none;white-space:pre}
.s3d-dir .hud .live{color:#c8ff00}
.s3d-dir aside{border-left:1px solid #222;padding:12px;overflow:auto;display:flex;flex-direction:column}
.s3d-dir aside .khead{display:flex;align-items:center;justify-content:space-between;margin:0 0 8px}
.s3d-dir aside h3{margin:0;font:600 10px/1.2 "Unbounded",system-ui,sans-serif;letter-spacing:.12em;color:#9a9a9a;text-transform:uppercase}
.s3d-dir aside label{display:grid;grid-template-columns:90px 1fr;gap:6px;align-items:center;margin:5px 0;color:#9a9a9a}
.s3d-dir aside input,.s3d-dir aside select{width:100%}
.s3d-dir aside details{margin-top:10px;border-top:1px solid #1d1d1d;padding-top:8px}
.s3d-dir aside summary{cursor:pointer;color:#8a8a8a;font-size:12px;user-select:none}
.s3d-dir aside .kfoot{display:flex;gap:6px;margin-top:12px;align-items:center}
.s3d-dir aside .kfoot .grow{flex:1}
.s3d-dir aside .hint{color:#777;font-size:12px;margin-top:4px}
.s3d-dir .tl{grid-column:1/3;border-top:1px solid #222;position:relative;display:grid;grid-template-rows:42px minmax(0,1fr);min-height:0}
.s3d-dir .tl canvas{width:100%;height:100%;display:block;cursor:default;outline:none}
.s3d-dir .tbar{display:flex;align-items:center;gap:3px;padding:0 10px;border-bottom:1px solid #1c1c1c;white-space:nowrap;overflow:hidden;min-width:0}
.s3d-dir .tbar .grow{flex:1}
.s3d-dir .tbar input,.s3d-dir .tbar select{padding:3px 5px}
.s3d-dir .tbar input[type=number]{width:62px}
.s3d-dir .tbar .lab{color:#8a8a8a;font-size:12px;margin:0 2px 0 6px}
.s3d-dir .tbar .tc{font:700 14px "Unbounded",system-ui,sans-serif;font-variant-numeric:tabular-nums;color:#c8ff00;padding:0 8px;min-width:96px;text-align:right}
.s3d-dir .tbar .rate{font:12px ui-monospace,Menlo,monospace;color:#ff3d6e;min-width:34px;text-align:center}
.s3d-dir .help{color:#777;font-size:12px;margin-top:12px;line-height:1.6}
.s3d-dir .kbd{border:1px solid #333;border-radius:4px;padding:0 4px;color:#bbb}
.s3d-dir .msg{position:absolute;inset:0;display:grid;place-items:center;color:#aaa;font-size:15px}
.s3d-dir .pip{position:absolute;right:12px;bottom:12px;border:1px solid #c8ff00;border-radius:4px;cursor:pointer;display:none}
.s3d-dir .pip span{position:absolute;left:6px;top:4px;font-size:11px;color:#c8ff00;text-shadow:0 0 3px #000}`;

function el(tag, attrs = {}, kids = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;   // absent boolean attributes (disabled, selected)
    k === "style" ? (e.style.cssText = v) : k.startsWith("on") ? (e[k] = v) : e.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) e.append(c instanceof Node ? c : document.createTextNode(c));
  return e;
}

// Icons: hand-drawn 24px stroke paths, no external icon set.
const ICONS = {
  start: '<path d="M6 5v14"/><path d="M18 5L9 12l9 7z"/>', end: '<path d="M18 5v14"/><path d="M6 5l9 7-9 7z"/>',
  prevkey: '<path d="M4 12l4-4 4 4-4 4z"/><path d="M20 6l-6 6 6 6"/>', nextkey: '<path d="M20 12l-4-4-4 4 4 4z"/><path d="M4 6l6 6-6 6"/>',
  stepb: '<path d="M15 6l-6 6 6 6"/>', stepf: '<path d="M9 6l6 6-6 6"/>',
  rew: '<path d="M11 6l-7 6 7 6z"/><path d="M20 6l-7 6 7 6z"/>', ff: '<path d="M13 6l7 6-7 6z"/><path d="M4 6l7 6-7 6z"/>',
  loop: '<path d="M4 11V9a3 3 0 013-3h11"/><path d="M15 3l3 3-3 3"/><path d="M20 13v2a3 3 0 01-3 3H6"/><path d="M9 21l-3-3 3-3"/>',
  play: '<path d="M7 4.5v15l12-7.5z"/>', pause: '<path d="M8 5v14M16 5v14"/>',
  key: '<path d="M10 3.5l6 6-6 6-6-6z"/><path d="M19 15v6M16 18h6"/>',
  camera: '<path d="M3 8h4l2-3h6l2 3h4v11H3z"/><circle cx="12" cy="13" r="3.5"/>',
  editor: '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>',
  undo: '<path d="M9 14l-5-5 5-5"/><path d="M4 9h10a6 6 0 010 12h-3"/>', redo: '<path d="M15 14l5-5-5-5"/><path d="M20 9H10a6 6 0 000 12h3"/>',
  shot: '<path d="M4 7h9M17 7h3M4 12h3M11 12h9M4 17h11M19 17h1"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="17" r="2"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 015 0c0 1.7-2.5 2-2.5 3.8"/><path d="M12 17h.01"/>',
  import: '<path d="M12 4v11M7.5 10.5L12 15l4.5-4.5"/><path d="M4 16v4h16v-4"/>', export: '<path d="M12 15V4M7.5 8.5L12 4l4.5 4.5"/><path d="M4 16v4h16v-4"/>',
  min: '<path d="M5 12h14"/>', close: '<path d="M6 6l12 12M18 6L6 18"/>', check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  more: '<circle cx="5" cy="12" r="1.2"/><circle cx="12" cy="12" r="1.2"/><circle cx="19" cy="12" r="1.2"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6"/><path d="M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  update: '<path d="M3 8h4l2-3h6l2 3h4v11H3z"/><path d="M15 13a3 3 0 11-1-2.2M15 9.5v1.8h-1.8"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/>', paste: '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4V2.5h6V4"/>',
  dup: '<path d="M9 4l5 5-5 5-5-5z"/><path d="M15 10l5 5-5 5-5-5"/>', move: '<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>',
  rotate: '<path d="M20 12a8 8 0 11-2.3-5.7"/><path d="M20 4v5h-5"/>', frame: '<path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4"/><circle cx="12" cy="12" r="3"/>',
};
function icon(name) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  for (const [k, v] of Object.entries({ viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "1.8", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", focusable: "false" })) s.setAttribute(k, v);
  s.innerHTML = ICONS[name] || ""; return s;
}

// Camera-relative portable keys ("cam" in an exported path): each key as an offset from the clip's real camera at frame 0, so a move
// replays from wherever the source camera stood on any capture. Eye: azimuth / elevation change (deg) and distance ratio around the key's
// look-at point; look-at: offset from the real camera's aim point in its own frame (forward-flat, side, up), in units of the real
// camera's distance. Roll as is, lens as a ratio of the real lens. Needs a real camera solve; "locked" keys live in the setting, keys
// that travel with her live relative to the middle of the clip (meta.source is stored that way).
function camRel(meta, follow) {
  if (!meta.source) return null;
  const add = (a, b, k) => a.map((x, i) => x + b[i] * k), sub = (a, b) => a.map((x, i) => x - b[i]), dot = (a, b) => a.reduce((t, x, i) => t + x * b[i], 0);
  const up = meta.up, fl = meta.track && meta.track.follow;
  let E = [...meta.source.eye], A = [...meta.source.target];
  if (fl && follow === false) {   // the real camera in the setting itself
    const at = (x) => { const n = fl.length - 1, y = Math.min(1, Math.max(0, x)) * n, i = Math.min(Math.floor(y), n - 1), f = y - i; return fl[i].map((v, k) => v + (fl[i + 1][k] - v) * f); };
    const d = sub(at(0), at(0.5)); E = add(E, d, 1); A = add(A, d, 1);
  }
  const rel = sub(E, A), r0 = Math.hypot(...rel) || 1e-9;
  const fwd = add(rel, up, -dot(rel, up)), fn = Math.hypot(...fwd) || 1e-9, F = fwd.map((x) => x / fn);
  const S = [up[1] * F[2] - up[2] * F[1], up[2] * F[0] - up[0] * F[2], up[0] * F[1] - up[1] * F[0]];
  const sphOf = (v) => { const r = Math.hypot(...v) || 1e-9; return [Math.atan2(dot(v, S), dot(v, F)), Math.asin(Math.max(-1, Math.min(1, dot(v, up) / r))), r]; };
  const [az0, el0] = sphOf(rel), D = 180 / Math.PI;
  return {
    to: (k) => { const o = sub(k.target, A), [az, el, r] = sphOf(sub(k.eye, k.target));
      let da = (az - az0) * D; da -= 360 * Math.round(da / 360);
      return { az: da, el: (el - el0) * D, dist: r / r0, aim: [dot(o, F) / r0, dot(o, S) / r0, dot(o, up) / r0], roll: k.roll || 0, lens: (k.lens || meta.source.lens) / meta.source.lens,
               source: Math.hypot(...sub(k.eye, E)) < r0 * 1e-3 || undefined }; },
    from: (c, k) => {
      if (c.source) return Object.assign(k, { eye: [...E], target: [...A], lens: meta.source.lens, roll: c.roll || 0 });
      const T = add(add(add(A, F, c.aim[0] * r0), S, c.aim[1] * r0), up, c.aim[2] * r0), az = az0 + c.az / D, el = el0 + c.el / D, r = r0 * c.dist;
      const eye = add(T, add(add(F.map((x) => x * Math.cos(el) * Math.cos(az)), S, Math.cos(el) * Math.sin(az)), up, Math.sin(el)), r);
      return Object.assign(k, { eye, target: T, lens: meta.source.lens * (c.lens || 1), roll: c.roll || 0 }); },
  };
}

// Import a camera path onto a Director node without opening it. Same rules as the Director's Import: the same capture takes the keys
// as they are; another capture takes the portable copy (subject frame, real camera at 0 degrees, capture-distance units), keys on the
// real camera take this clip's real camera, and the shot aspect follows this clip.
// The capture wired into a Virtual Camera node wins over its dropdown, as it does when the graph runs.
function syncLinkedCapture(node) {
  const id = node.inputs?.find((i) => i.name === "train_dir")?.link, l = id != null && app.graph.links[id];
  const up = l && app.graph.getNodeById(l.origin_id), w = node.widgets.find((x) => x.name === "capture");
  if (up?.__capture && w) { if (w.options?.values && !w.options.values.includes(up.__capture)) w.options.values.unshift(up.__capture); w.value = up.__capture; }
}

async function importToNode(node, text) {
  syncLinkedCapture(node);
  const d = JSON.parse(text), st = d.state || d;
  if (!Array.isArray(st.keys) || !st.keys.length) throw new Error("no camera keys in this file");
  const W = (n) => node.widgets.find((w) => w.name === n);
  const rr = await (await fetch(`/sam3dbx/director/resolve?train_dir=${encodeURIComponent(W("capture").value)}&mode=${encodeURIComponent(W("mode")?.value || "")}`)).json();
  if (rr.error) throw new Error(rr.error);
  const meta = await (await fetch(`/sam3dbx/director/meta?train_dir=${encodeURIComponent(rr.dir)}`)).json();
  if (meta.error) throw new Error(meta.error);
  const keys = JSON.parse(JSON.stringify(st.keys));
  const cr = Array.isArray(d.cam) && d.cam.length === keys.length && camRel(meta, st.follow);
  if (st.space_dir !== rr.dir && cr) keys.forEach((k, i) => cr.from(d.cam[i], k));   // camera-relative: from this clip's real camera
  else if (st.space_dir !== rr.dir) {
    if (!Array.isArray(d.rel) || d.rel.length !== keys.length) throw new Error("this path was made on another capture and has no portable copy: export it again from the Director");
    const add = (a, b, k) => a.map((x, i) => x + b[i] * k), dot = (a, b) => a.reduce((t, x, i) => t + x * b[i], 0);
    const c = meta.centre, up = meta.up; let f = meta.front, sd = meta.side;
    if (meta.source) {
      const o = meta.source.eye.map((x, i) => x - c[i]), a = Math.atan2(dot(o, meta.side), dot(o, meta.front));
      f = add(meta.front.map((x) => x * Math.cos(a)), meta.side, Math.sin(a)); sd = add(meta.front.map((x) => -x * Math.sin(a)), meta.side, Math.cos(a));
    }
    const from = (r) => add(add(add(c, f, r[0] * meta.dist0), sd, r[1] * meta.dist0), up, r[2] * meta.dist0);
    keys.forEach((k, i) => {
      if (d.rel[i].source && meta.source) Object.assign(k, { eye: [...meta.source.eye], target: [...meta.source.target], lens: meta.source.lens });
      else { k.eye = from(d.rel[i].eye); k.target = from(d.rel[i].target); }
    });
  }
  const state = { ...st, keys, space_dir: rr.dir, aspect: nearestAspect(meta.source && meta.source.aspect ? meta.source.aspect : 9 / 16) };
  W("director_state").value = JSON.stringify(state); node.setDirtyCanvas?.(true, true);
}

async function openDirector(node) {
  if (node.__s3dRestore) return node.__s3dRestore();   // minimised: bring that window back rather than open a second one
  syncLinkedCapture(node);
  const wState = node.widgets.find((w) => w.name === "director_state");
  const wCap = node.widgets.find((w) => w.name === "capture");
  const wMode = node.widgets.find((w) => w.name === "mode");
  const modeBadge = el("span", { title: "Change on the node: mode. Keys convert when you switch.", style: "padding:4px 9px;border-radius:7px;border:1.5px solid #c8ff00;color:#c8ff00;font:600 10px/1 \"Unbounded\",system-ui,sans-serif;letter-spacing:.06em;text-transform:uppercase" }, "…");
  let trainDir = wCap.value;
  let state;
  try { state = wState.value ? JSON.parse(wState.value) : null; } catch { state = null; }
  state = state || { duration: 8, fps: 24, shake: 0, bpm: 0, first_beat: 0, keys: [], scene_mode: "auto", speed: 1, scene_end: "pingpong" };
  if (!state.keys.length) state.scene_mode = "auto";

  if (!document.getElementById("s3d-dir-css")) document.head.append(el("style", { id: "s3d-dir-css" }, CSS));
  const root = el("div", { class: "s3d-dir" });
  const vp = el("div", { class: "vp" }); const hud = el("div", { class: "hud" });
  const msg = el("div", { class: "msg" }, "building the preview of this capture…");
  const aside = el("aside"); const tl = el("div", { class: "tl" }); const tbar = el("div", { class: "tbar" }); const tlc = el("canvas", { tabindex: "0" }); tl.append(tbar, tlc);
  const B = (txt, title, fn) => el("button", { title, onclick: fn }, txt);
  // icon button: label shows when visible, else stays as hidden text (screen readers, tests); blur after a click so Space stays transport
  const IB = (name, title, fn, label = "", { cls = "ib quiet", show = false } = {}) => {
    const b = el("button", { class: show ? cls.replace("ib", "").trim() : cls, title, "aria-label": title.split(" (")[0],
      onclick: (ev) => { ev.currentTarget.blur(); fn(ev); } }, [icon(name)]);
    if (label) b.append(el("span", show ? {} : { class: "sr" }, label));
    return b;
  };
  let pop = null;   // the one open popover {trigger, panel}
  const setPop = (n) => {
    if (pop) { pop.panel.hidden = true; pop.trigger.setAttribute("aria-expanded", "false"); pop.trigger.classList.remove("on"); }
    pop = n;
    if (n) { n.panel.hidden = false; n.trigger.setAttribute("aria-expanded", "true"); n.trigger.classList.add("on"); }
  };
  const popover = (trigger, kids, left = false) => {
    const panel = el("div", { class: "pop" + (left ? " left" : ""), role: "dialog" }, kids); panel.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
    trigger.onclick = (ev) => { ev.currentTarget.blur(); setPop(pop && pop.panel === panel ? null : { trigger, panel }); };
    return el("div", { class: "popwrap" }, [trigger, panel]);
  };
  const onOutside = (ev) => { if (pop && !pop.panel.contains(ev.target) && !pop.trigger.contains(ev.target)) setPop(null); };
  document.addEventListener("pointerdown", onOutside, true);
  const row = (label, ctl, title) => el("label", { class: "row", title }, [label, ctl]);
  const grp = (title, kids, attrs = {}) => el("div", { class: "grp", ...attrs }, [el("h4", {}, title), ...kids]);

  const bView = IB("camera", "Camera view. Switch to editor (C)", () => setView(view === "camera" ? "editor" : "camera"), "Camera view", { show: true, cls: "quiet" });
  const bPlay = IB("play", "Play (Space)", () => togglePlay(), "", { cls: "ib" });
  const bKey = IB("key", "Add key at the playhead (I)", () => addKey(), "Add key", { show: true, cls: "" });
  const bUpdKey = IB("update", "Update the key at the playhead with the current camera (keeps its timing, ease, aim and freeze)", () => updateKeyAtPlayhead(), "Update", { show: true, cls: "quiet" });
  const bDelKey = IB("trash", "Delete the selected key(s) (Delete)", () => deleteKey(), "Delete", { show: true, cls: "quiet" });
  const bLoop = IB("loop", "Loop playback (the preview only; the capture's own end is in Shot settings)", () => { loopOn = !loopOn; bLoop.classList.toggle("on", loopOn); }, "", { cls: "ib quiet" });
  const tcOut = el("span", { class: "tc", title: "Playhead: seconds:frames and the output frame (the last frame is one before the length)" });
  const rateOut = el("span", { class: "rate", title: "Shuttle speed (J / L)" });
  const inFrame = el("input", { type: "number", step: "1", min: "0", title: "Go to frame" });
  const prevSel = el("select", { title: "Preview playback speed (the shot itself is unchanged)" }, ["0.25x", "0.5x", "1x", "2x"].map((x) => el("option", {}, x)));
  const presetSel = el("select", { class: "presets", title: "Replace the keys with a preset move" },
    ["Presets…", "Arc left", "Arc right", "Push in", "Crane up", "Bullet time (arc)", "Orbit left", "Orbit right", "Zoom in", "Zoom out", "Face: push in, orbit L/R, pull out"].map((p) => el("option", {}, p)));
  presetSel.onchange = () => { if (presetSel.selectedIndex > 0) applyPreset(presetSel.value); presetSel.selectedIndex = 0; };
  const inDur = el("input", { type: "number", step: "0.1", min: "0.5", value: state.duration, title: "Shot length (s)" });
  const inFps = el("input", { type: "number", step: "1", min: "8", value: state.fps, title: "Frames per second of the shot" });
  const inSpeed = el("input", { type: "number", step: "0.05", min: "0", value: state.speed ?? 1,
    title: "How fast the capture plays (1 = real time)" });
  const aspectSel = el("select", { title: "Shot aspect (a new shot takes the input clip's)" }, Object.keys(ASPECTS).map((x) => el("option", {}, x)));
  aspectSel.onchange = () => { state.aspect = aspectSel.value; ASPECT = ASPECTS[state.aspect];
    for (const c of [shot, shotProxy]) { c.aspect = ASPECT; c.updateProjectionMatrix(); } rebuild(); };
  const endSel = el("select", { title: "When the shot is longer than the capture" }, ["pingpong", "loop", "hold"].map((x) => el("option", {}, x)));
  endSel.value = state.scene_end || "pingpong"; endSel.onchange = () => rebuild();
  const pathSel = el("select", { title: "orbit: swings round the look-at point (round orbits from few keys). spline: a straight 3D curve through the keys" },
    ["orbit", "spline"].map((x) => el("option", {}, x)));
  pathSel.value = state.path || "orbit"; pathSel.onchange = () => rebuild();
  const easeSel = el("select", { title: "per key: each key's own ease (smooth stops the camera at every key). whole shot: one ease in and one ease out over the whole move, keys in between passed at speed" },
    [["per key", "keys"], ["whole shot", "smooth"]].map(([t, v]) => el("option", { value: v }, t)));
  easeSel.value = state.shot_ease === "smooth" ? "smooth" : "keys"; easeSel.onchange = () => rebuild();
  const settingSel = el("select", { title: "The clip's real setting around the person: in the editor view only (the shot stays clean), everywhere, or hidden", disabled: "" },
    [["editor view", "editor"], ["editor + shot", "both"], ["off", "off"]].map(([t, v]) => el("option", { value: v }, t)));
  settingSel.value = ["editor", "both", "off"].includes(state.show_setting) ? state.show_setting : "editor";
  settingSel.onchange = () => { state.show_setting = settingSel.value; settingSel.blur(); };
  const followSel = el("select", { title: "with her: the camera travels with her through the setting, as the source camera did. locked: the camera and its look-at point stay put in the setting" },
    [["with her", "on"], ["locked", "off"]].map(([t, v]) => el("option", { value: v }, t)));
  followSel.value = state.follow === false ? "off" : "on"; followSel.onchange = () => { state.follow = followSel.value === "on"; rebuild(); };
  const bSnap = B("Off", "Gizmo moves in 5 % steps of the capture distance and turns in 15° steps (hold Ctrl to flip while dragging)", () => { snapOn = !snapOn; applySnap(); });
  const bSave = IB("check", "Save & close", () => close(true), "Save & close", { show: true, cls: "primary" });
  const bCancel = IB("close", "Cancel (Esc)", () => close(false), "Cancel");
  const fileIn = el("input", { type: "file", accept: ".json,application/json", style: "display:none" });
  fileIn.onchange = async () => { const f = fileIn.files[0]; fileIn.value = ""; if (f) importPath(await f.text()); };
  const bImport = IB("import", "Import camera path (.json)", () => fileIn.click(), "Import");
  const bExport = IB("export", "Export camera path (.json)", () => exportPath(), "Export");
  const bMin = IB("min", "Minimise and save to the node", () => minimise(true), "–");
  const bUndo = IB("undo", "Undo (Cmd/Ctrl+Z)", () => undo()), bRedo = IB("redo", "Redo (Shift+Cmd/Ctrl+Z)", () => redo());
  const shotPop = popover(IB("shot", "Shot settings", () => {}, "Shot", { show: true, cls: "quiet" }), [
    grp("Output", [row("Aspect", aspectSel)]),
    grp("Capture", [row("Playback speed", inSpeed), row("At end", endSel)]),
    grp("Camera path", [row("Interpolation", pathSel), row("Camera", followSel), row("Gizmo snap", bSnap)]),
    grp("Setting", [row("Show", settingSel)]),
    grp("Preview", [], { class: "grp shot-preview-group", hidden: "" })]);
  const K = (k, t) => el("div", { class: "k" }, [el("kbd", {}, k), el("span", {}, t)]);
  const helpPop = popover(IB("help", "Controls and shortcuts", () => {}), [el("div", { class: "keys" }, [
    grp("Camera view", [K("Drag", "look around"), K("W A S D / Q E", "fly (Shift = fast)"), K("Wheel", "lens"), K("R / T", "roll"), K("I", "key the camera at the playhead")]),
    grp("Editor view", [K("Click a key camera", "select it"), K("Double-click", "look through it"), K("G / R", "move / rotate with the gizmo"),
      K("Lime dot", "drag where a key looks"), K("V", "key what the editor sees"), K("X", "delete key"), K("Ctrl", "hold to flip gizmo snap")]),
    grp("Timeline", [K("Click ruler / lanes", "seek"), K("Drag a key", "move it in time (a selection moves together)"), K("Shift+click key", "add to the selection"), K("Alt", "drag without frame snap"),
      K("Space", "play / pause"), K("J / K / L", "backwards / stop / forwards (again = faster)"), K("Left / Right", "one frame"), K("Shift+Left / Right", "previous / next key"),
      K("Home / End", "first / last frame"), K("I", "add key"), K("Delete", "delete selected keys")]),
    grp("Editing", [K("Space", "play / pause"), K("Shift+click key", "add to the selection: G / R move / rotate them together"), K("Shift+drag", "pan the view"), K("C", "camera / editor view"), K("Cmd+Z", "undo, Shift for redo"), K("Cmd+C / Cmd+V", "copy / paste key"),
      K("Shift+D", "duplicate key"), K("Esc", "cancel")]),
    el("div", { class: "hint" }, "Freeze time on a key for bullet time: the capture stops until the next key while the camera keeps moving.")])]);
  root.append(el("header", {}, [
    el("div", { class: "zone" }, [el("b", {}, "VIRTUAL CAMERA"), modeBadge]),
    el("div", { class: "zone mid" }, [presetSel, el("span", { class: "sep" }), bView]),
    el("div", { class: "zone" }, [bUndo, bRedo, el("span", { class: "sep" }), shotPop, helpPop, bImport, bExport, fileIn, el("span", { class: "sep" }), bMin, bCancel, bSave])]),
    vp, aside, tl);
  const sep = () => el("span", { class: "sep" }), lab = (t) => el("span", { class: "lab" }, t);
  tbar.append(
    IB("start", "Start (Home)", () => seek(0)), IB("prevkey", "Previous key (Shift+Left)", () => jumpKey(-1)), IB("stepb", "Back one frame (Left)", () => stepFrame(-1)),
    IB("rew", "Play backwards; again for faster (J)", () => shuttle(-1)), bPlay, IB("ff", "Play forwards; again for faster (L)", () => shuttle(1)),
    IB("stepf", "Forward one frame (Right)", () => stepFrame(1)), IB("nextkey", "Next key (Shift+Right)", () => jumpKey(1)), IB("end", "Last frame (End)", () => seek(lastT())),
    rateOut, bLoop, prevSel, tcOut, inFrame, sep(), bKey, bUpdKey, bDelKey, el("span", { class: "grow" }),
    lab("Easing"), easeSel, lab("Length"), inDur, lab("s"), lab("FPS"), inFps);
  const pip = el("div", { class: "pip", title: "Live shot camera. Click to look through it (Camera view)" }, [el("span", {}, "SHOT")]);
  const navg = el("canvas", { class: "navg", width: 220, height: 220,
    title: "View gizmo: click a dot for top / bottom / front / back / left / right, drag to orbit. In the viewport: drag orbits, Shift or middle drag pans, wheel zooms, . recentres. Numpad 1 / 3 / 7 = front / right / top (Ctrl for the opposite side)" });
  vp.append(msg, hud, pip, navg);
  document.body.append(root);

  // ---------- load preview ----------
  let meta;
  try {   // the mode toggle (fast mannequin | 4DGS splat) picks which capture of this clip to preview and render
    const rr = await (await fetch(`/sam3dbx/director/resolve?train_dir=${encodeURIComponent(trainDir)}&mode=${encodeURIComponent(wMode?.value || "4DGS (splat)")}`)).json();
    if (rr.error) throw new Error(rr.error);
    if (state.space_dir && state.space_dir !== rr.dir && state.keys.length) {   // keys were designed in the other capture's space
      state = await (await fetch("/sam3dbx/director/convert", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, from: state.space_dir, to: rr.dir }) })).json();
    }
    trainDir = rr.dir; state.space_dir = rr.dir; modeBadge.textContent = rr.kind === "splat" ? "4DGS splat" : "fast · mannequin";
  } catch (e) { msg.textContent = "Could not open this capture: " + e.message; return; }
  try {
    const r = await fetch(`/sam3dbx/director/meta?train_dir=${encodeURIComponent(trainDir)}`);
    meta = await r.json(); if (meta.error) throw new Error(meta.error);
  } catch (e) { msg.textContent = "Could not build the preview: " + e.message; return; }
  const buf = await (await fetch(`/sam3dbx/director/points?train_dir=${encodeURIComponent(trainDir)}`)).arrayBuffer();
  msg.remove();
  const n = meta.n, f32 = new Float32Array(buf, 0, n * 9), u8 = new Uint8Array(buf, n * 9 * 4, n * 3);
  const up = new THREE.Vector3(...meta.up), centre = new THREE.Vector3(...meta.centre);
  const front = new THREE.Vector3(...meta.front), side = new THREE.Vector3(...meta.side), dist0 = meta.dist0;
  BASIS = [meta.up, meta.front, meta.side]; TRACK = meta.track || null;
  // a new shot lasts as long as the captured clip: presets then run the performance in real time, frame for frame with the input
  if (!state.keys.length && meta.seconds) {
    // a trimmed clip spans real_seconds: the shot lasts that long and the capture plays at the speed that makes it real time
    const real = meta.real_seconds || meta.seconds;
    state.duration = +real.toFixed(2); inDur.value = state.duration;
    state.speed = +(meta.seconds / real).toFixed(4); inSpeed.value = state.speed;
  }
  state.aspect = ASPECTS[state.aspect] ? state.aspect : nearestAspect(meta.source && meta.source.aspect ? meta.source.aspect : 9 / 16);
  ASPECT = ASPECTS[state.aspect]; aspectSel.value = state.aspect;

  // ---------- three scene ----------
  const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(window.devicePixelRatio);
  vp.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(n * 3), vel = new Float32Array(n * 3), tim = new Float32Array(n), dur = new Float32Array(n), opa = new Float32Array(n), col = new Float32Array(n * 3);
  let floor = Infinity;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) { pos[i * 3 + k] = f32[i * 9 + k]; vel[i * 3 + k] = f32[i * 9 + 3 + k]; col[i * 3 + k] = u8[i * 3 + k] / 255; }
    tim[i] = f32[i * 9 + 6]; dur[i] = f32[i * 9 + 7]; opa[i] = f32[i * 9 + 8];
  }
  const hs = []; for (let i = 0; i < n; i += 37) hs.push(new THREE.Vector3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).sub(centre).dot(up));
  hs.sort((a, b) => a - b); floor = hs[Math.floor(hs.length * 0.01)];
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geo.setAttribute("aVel", new THREE.BufferAttribute(vel, 3));
  geo.setAttribute("aTime", new THREE.BufferAttribute(tim, 1)); geo.setAttribute("aDur", new THREE.BufferAttribute(dur, 1));
  geo.setAttribute("aOpa", new THREE.BufferAttribute(opa, 1)); geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, vertexColors: true,
    uniforms: { uScene: { value: 0 }, uUseVel: { value: meta.use_velocity ? 1 : 0 }, uPx: { value: 800 },
      uSize: { value: meta.fast_mode ? 0.013 : 0.0065 } } });   // a mannequin's points are sparser than a splat's: bigger dots close the gaps
  const points = new THREE.Points(geo, mat); points.frustumCulled = false; scene.add(points);
  let settingPts = null;   // the setting layer, once loaded
  if (meta.setting_n) {   // the clip's real setting (DA3 depth of the real frames, person masked out): static points around her
    fetch(`/sam3dbx/director/setting?train_dir=${encodeURIComponent(trainDir)}`).then((r) => (r.ok ? r.arrayBuffer() : null)).then((b) => {
      if (!b) return;
      const m = meta.setting_n, sg = new THREE.BufferGeometry(), sc = new Uint8Array(b, m * 12, m * 3), c = new Float32Array(m * 3);
      for (let i = 0; i < m * 3; i++) c[i] = sc[i] / 255;
      sg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(b, 0, m * 3).slice(), 3)); sg.setAttribute("color", new THREE.BufferAttribute(c, 3));
      const sp = new THREE.Points(sg, new THREE.PointsMaterial({ size: dist0 * 0.008, vertexColors: true, sizeAttenuation: true }));
      sp.frustumCulled = false; scene.add(sp); settingPts = sp; settingSel.disabled = false;
    });
  }

  const ground = centre.clone().addScaledVector(up, floor);
  const grid = new THREE.GridHelper(dist0 * 6, 24, 0x3a4a1a, 0x1c1c1c);
  grid.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), up); grid.position.copy(ground); scene.add(grid);
  const rigDots = new THREE.Group(); scene.add(rigDots);   // training cameras: editor-only reference, hidden in shot views
  for (const c of meta.cam_pos) { const m = new THREE.Mesh(new THREE.SphereGeometry(dist0 * 0.015), new THREE.MeshBasicMaterial({ color: 0x444444 })); m.position.set(...c); rigDots.add(m); }

  const shot = new THREE.PerspectiveCamera(lensToFov(50), ASPECT, 0.01, 100); shot.up.copy(up);
  const edit = new THREE.PerspectiveCamera(45, 1, 0.01, 200); edit.up.copy(up);
  // compact stand-in for the live shot camera in the editor (the real one's far plane would draw 100-unit lines)
  const shotProxy = new THREE.PerspectiveCamera(50, ASPECT, dist0 * 0.02, dist0 * 0.3); shotProxy.up.copy(up);
  const helper = new THREE.CameraHelper(shotProxy); helper.visible = false; scene.add(helper);
  helper.setColors?.(new THREE.Color(0xff3d6e), new THREE.Color(0xff3d6e), new THREE.Color(0xff3d6e), new THREE.Color(0xff3d6e), new THREE.Color(0x662233));
  const pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xc8ff00 })); scene.add(pathLine);
  const keyGroup = new THREE.Group(); scene.add(keyGroup);
  // XYZ gizmo on the selected key camera (editor view): G = move, R = rotate, like Blender
  const tc = new TransformControls(edit, renderer.domElement); tc.setSize(0.8);
  const tcHelper = tc.getHelper(); scene.add(tcHelper);
  let keyObjs = [], keyHelpers = [], pickers = [], targetObj = null, gizmoOn = "camera", snapOn = false, ctrlHeld = false;
  // multi-select (Shift+click keys): G / R move or rotate the whole group about its centre through a pivot the gizmo drives;
  // each key's eye and look-at point take the same rigid transform, roll is kept
  const multi = new Set(), pivotObj = new THREE.Object3D(); scene.add(pivotObj);
  let groupDrag = null;
  const grouped = () => multi.size > 1;
  function applySnap() {
    const on = snapOn !== ctrlHeld;
    tc.setTranslationSnap(on ? dist0 * 0.05 : null); tc.setRotationSnap(on ? THREE.MathUtils.degToRad(15) : null);
    bSnap.classList.toggle("on", snapOn); bSnap.textContent = snapOn ? "On" : "Off";
  }
  tc.addEventListener("objectChange", () => {
    if (grouped()) { applyGroup(); return; }
    syncKeyFromGizmo(); rebuildPath();
    if (gizmoOn === "target" && keyObjs[sel]) {   // re-aim the key camera live while its target moves
      const k = state.keys[sel], c = keyObjs[sel]; c.lookAt(new THREE.Vector3(...k.target)); c.rotateZ((-k.roll * Math.PI) / 180); c.updateMatrixWorld();
    }
    keyHelpers[sel]?.update();
  });
  tc.addEventListener("dragging-changed", (e) => {
    if (grouped()) {
      if (e.value) groupDrag = { inv: pivotObj.matrixWorld.clone().invert(), keys: [...multi].map((i) => ({ i, eye: new THREE.Vector3(...state.keys[i].eye), target: new THREE.Vector3(...state.keys[i].target) })) };
      else { applyGroup(); groupDrag = null; rebuild(); }
      return;
    }
    if (!e.value) { syncKeyFromGizmo(); rebuild(); }
  });
  function applyGroup() {
    if (!groupDrag) return;
    pivotObj.updateMatrixWorld(); const M = pivotObj.matrixWorld.clone().multiply(groupDrag.inv);
    for (const g of groupDrag.keys) {
      const k = state.keys[g.i], c = keyObjs[g.i]; if (!k) continue;
      k.eye = g.eye.clone().applyMatrix4(M).toArray(); k.target = g.target.clone().applyMatrix4(M).toArray();
      if (c) { c.position.set(...k.eye); c.lookAt(new THREE.Vector3(...k.target)); c.rotateZ((-k.roll * Math.PI) / 180); c.updateMatrixWorld(); keyHelpers[g.i]?.update(); pickers[g.i]?.position.copy(c.position); }
    }
    rebuildPath();
  }
  function setGizmo(mode) { tc.setMode(mode); tc.setSpace(mode === "rotate" && !grouped() ? "local" : "world"); }
  setGizmo("translate");
  function syncKeyFromGizmo() {
    const k = state.keys[sel]; if (!k) return;
    if (gizmoOn === "target") { if (targetObj) k.target = targetObj.position.toArray(); return; }
    const o = keyObjs[sel]; if (!o) return;
    const d = Math.max(new THREE.Vector3(...k.eye).distanceTo(new THREE.Vector3(...k.target)), dist0 * 0.2);
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(o.quaternion);
    const tgt = o.position.clone().addScaledVector(fwd, d);
    const ref = new THREE.PerspectiveCamera(); ref.up.copy(up); ref.position.copy(o.position); ref.lookAt(tgt);   // same eye/target, no roll
    const rel = ref.quaternion.clone().invert().multiply(o.quaternion);
    const roll = -new THREE.Euler().setFromQuaternion(rel, "XYZ").z * 180 / Math.PI;
    k.eye = o.position.toArray(); k.target = tgt.toArray(); k.roll = Math.round(roll * 10) / 10;
  }
  function attachGizmo() {
    if (view === "editor" && grouped()) {   // the group's pivot: centre of the selected cameras, world axes
      const cpos = new THREE.Vector3(); multi.forEach((i) => cpos.add(new THREE.Vector3(...state.keys[i].eye))); cpos.multiplyScalar(1 / multi.size);
      pivotObj.position.copy(cpos); pivotObj.quaternion.identity(); pivotObj.scale.set(1, 1, 1); pivotObj.updateMatrixWorld();
      gizmoOn = "camera"; tc.setSpace("world"); tc.attach(pivotObj); return;
    }
    if (view !== "editor" || sel < 0 || !keyObjs[sel]) { tc.detach(); return; }
    if (gizmoOn === "target" && targetObj) { tc.setMode("translate"); tc.setSpace("world"); tc.attach(targetObj); } else { gizmoOn = "camera"; tc.attach(keyObjs[sel]); }
  }

  // shot camera state (what the user flies / what the path evaluates to)
  const cam = { eye: centre.clone().addScaledVector(front, dist0 * 1.3).addScaledVector(up, dist0 * 0.25), yaw: 0, pitch: 0, lens: 50, roll: 0 };
  function aimAt(target) {
    const d = target.clone().sub(cam.eye).normalize();
    cam.pitch = Math.asin(THREE.MathUtils.clamp(d.dot(up), -1, 1));
    const h = d.clone().addScaledVector(up, -d.dot(up)).normalize();
    cam.yaw = Math.atan2(h.dot(side), h.dot(front.clone().negate()));
  }
  const camDir = () => front.clone().negate().multiplyScalar(Math.cos(cam.yaw)).addScaledVector(side, Math.sin(cam.yaw)).normalize()
    .multiplyScalar(Math.cos(cam.pitch)).addScaledVector(up, Math.sin(cam.pitch)).normalize();
  aimAt(centre);
  if (meta.source) { cam.eye.set(...meta.source.eye); cam.lens = meta.source.lens; aimAt(new THREE.Vector3(...meta.source.target)); }   // a new move starts where the source camera stood
  let live = false, view = "camera", playing = false, playhead = 0, sel = -1, sceneT = 0;
  // ---------- transport (preview only: none of this changes the saved shot) ----------
  let rate = 1, loopOn = true, previewMul = 1;
  const fpsV = () => Math.max(1, +inFps.value || 24), nFrames = () => Math.max(1, Math.round((+inDur.value || 1) * fpsV()));
  const lastT = () => (nFrames() - 1) / fpsV(), frameOf = (t) => Math.round(t * fpsV());
  function seek(t) { playing = false; playhead = Math.min(Math.max(0, t), +inDur.value); live = false; drawTimeline(); }
  const stepFrame = (d) => seek(Math.min(nFrames() - 1, Math.max(0, frameOf(playhead) + d)) / fpsV());
  function togglePlay() {
    if (playing) { playing = false; return; }
    rate = 1; if (!loopOn && playhead >= lastT() - 1e-6) playhead = 0; playing = true;
  }
  function shuttle(dir) {   // J / L: start that way at 1x, each press again doubles up to 4x, the other way reverses
    if (playing && Math.sign(rate) === dir) rate = Math.max(-4, Math.min(4, rate * 2)); else rate = dir;
    if (!loopOn && dir > 0 && playhead >= lastT() - 1e-6) playhead = 0;
    if (!loopOn && dir < 0 && playhead <= 1e-6) playhead = lastT();
    playing = true;
  }
  function arrival(k) {   // output time at which the camera reaches key k (differs from k.t under the whole-shot ease)
    const keys = [...state.keys].sort((a, b) => a.t - b.t);
    if (easeSel.value !== "smooth" || keys.length < 2) return k.t;
    const t0 = keys[0].t, t1 = keys[keys.length - 1].t; if (k.t <= t0 || k.t >= t1) return k.t;
    let lo = t0, hi = t1; for (let n = 0; n < 40; n++) { const m = (lo + hi) / 2; shotTime({ shot_ease: "smooth" }, keys, m) < k.t ? (lo = m) : (hi = m); }
    return (lo + hi) / 2;
  }
  function jumpKey(d) {
    if (!state.keys.length) return;
    const ts = state.keys.map((k, i) => [arrival(k), i]).sort((a, b) => a[0] - b[0]), eps = 0.5 / fpsV();
    const hit = d > 0 ? ts.find(([t]) => t > playhead + eps) : [...ts].reverse().find(([t]) => t < playhead - eps);
    if (!hit) return;
    multi.clear(); sel = hit[1]; seek(hit[0]); rebuild();
  }
  function updateKeyAtPlayhead() {
    const i = state.keys.findIndex((q) => Math.abs(arrival(q) - playhead) < 0.5 / fpsV());
    if (i < 0) return addKey();
    const nk = keyFromCam(state.keys[i].t); Object.assign(state.keys[i], { eye: nk.eye, target: nk.target, lens: nk.lens, roll: nk.roll });
    sel = i; live = false; rebuild();
  }
  const editOrbit = { r: dist0 * 3.2, az: 0.7, el: 0.35 };
  const editPan = new THREE.Vector3(), POLE = Math.PI / 2 - 1e-3;   // editor pivot = capture centre + pan; elevation stops just short of the poles
  let editTween = null;
  function snapView(az, el) {   // eased swing of the editor camera to a named view (the shorter way round)
    if (view !== "editor") setView("editor");
    const a = az + 2 * Math.PI * Math.round((editOrbit.az - az) / (2 * Math.PI));
    editTween = { az0: editOrbit.az, el0: editOrbit.el, az1: el === POLE || el === -POLE ? editOrbit.az : a, el1: el, t0: performance.now() };
  }
  const VIEWS = { front: [0, 0], back: [Math.PI, 0], left: [Math.PI / 2, 0], right: [-Math.PI / 2, 0], top: [null, POLE], bottom: [null, -POLE] };
  const goView = (n) => { const [a, e] = VIEWS[n]; snapView(a === null ? editOrbit.az : a, e); };

  // ---------- helpers ----------
  const S = () => ({ ...state, duration: +inDur.value, fps: +inFps.value, bpm: 0, first_beat: 0, shake: 0, path: pathSel.value, shot_ease: easeSel.value === "smooth" ? "smooth" : undefined, speed: +inSpeed.value, scene_end: endSel.value,
    scene_mode: state.scene_mode || "keys", capture_seconds: meta.seconds });
  const snap = (t) => Math.round(t * +inFps.value) / +inFps.value;   // keys land on frames
  function curScene(t) { const s = state.keys.length ? sample(S(), t, dist0) : null; return s ? s.scene : sceneT; }
  function keyFromCam(t) {
    const target = cam.eye.clone().addScaledVector(camDir(), Math.max(dist0 * 0.5, cam.eye.distanceTo(centre)));
    const sc = curScene(t), d = TRACK && TRACK.follow && state.follow !== false ? followDelta(sc) : [0, 0, 0];   // stored relative to her
    return { t, eye: cam.eye.toArray().map((v, k) => v - d[k]), target: target.toArray().map((v, k) => v - d[k]), lens: cam.lens, roll: cam.roll,
             scene: sc, ease: "smooth" };
  }
  function addKey() {
    multi.clear();
    const t = Math.min(snap(playhead), +inDur.value), nk = keyFromCam(t);
    const i = state.keys.findIndex((q) => Math.abs(q.t - t) < 0.5 / +inFps.value);
    const k = i >= 0 ? Object.assign(state.keys[i], { eye: nk.eye, target: nk.target, lens: nk.lens, roll: nk.roll }) : nk;
    if (i < 0) state.keys.push(k);
    state.keys.sort((a, b) => a.t - b.t); sel = state.keys.indexOf(k); live = false; rebuild();
  }
  const history = { past: [], future: [], last: null, mute: false };
  function record() {
    if (history.mute) return;
    const snap = JSON.stringify(S());
    if (snap === history.last) return;
    if (history.last) history.past.push(history.last);
    history.last = snap; history.future = []; if (history.past.length > 300) history.past.shift();
  }
  function restore(snapStr) {
    multi.clear();
    const st = JSON.parse(snapStr);
    state.keys = st.keys; state.scene_mode = st.scene_mode;
    inDur.value = st.duration; inFps.value = st.fps; inSpeed.value = st.speed;
    endSel.value = st.scene_end; pathSel.value = st.path; easeSel.value = st.shot_ease === "smooth" ? "smooth" : "keys";
    state.follow = st.follow; followSel.value = st.follow === false ? "off" : "on";
    history.last = snapStr; sel = Math.min(sel, state.keys.length - 1); live = false;
    history.mute = true; rebuild(); history.mute = false;
  }
  const undo = () => { if (history.past.length) { history.future.push(history.last); restore(history.past.pop()); } };
  const redo = () => { if (history.future.length) { history.past.push(history.last); restore(history.future.pop()); } };
  let clip = null;
  const copyKey = () => { if (state.keys[sel]) clip = JSON.parse(JSON.stringify(state.keys[sel])); };
  function placeKey(k) {
    multi.clear();
    const i = state.keys.findIndex((q) => Math.abs(q.t - k.t) < 0.5 / +inFps.value);
    if (i >= 0) { k = Object.assign(state.keys[i], { eye: k.eye, target: k.target, lens: k.lens, roll: k.roll }); } else state.keys.push(k);
    state.keys.sort((a, b) => a.t - b.t); sel = state.keys.indexOf(k); playhead = k.t; live = false; rebuild();
  }
  const pasteKey = () => { if (clip) placeKey({ ...JSON.parse(JSON.stringify(clip)), t: snap(playhead) }); };
  function duplicateKey() {
    const k = state.keys[sel]; if (!k) return;
    let t = snap(playhead); if (Math.abs(t - k.t) < 1e-6) t = Math.min(+inDur.value, k.t + 1);
    placeKey({ ...JSON.parse(JSON.stringify(k)), t });
  }
  function keyEditorView() {   // Blender's "camera to view": key what the editor is looking at
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(edit.quaternion);
    placeKey({ t: snap(playhead), eye: edit.position.toArray(), target: edit.position.clone().addScaledVector(fwd, edit.position.distanceTo(centre)).toArray(),
      lens: Math.round(fovToLens(edit.fov)), roll: 0, scene: curScene(playhead), ease: "smooth" });
  }
  function deleteKey() {
    if (grouped()) { state.keys = state.keys.filter((_, i) => !multi.has(i)); multi.clear(); sel = Math.min(sel, state.keys.length - 1); rebuild(); return; }
    if (state.keys[sel]) { state.keys.splice(sel, 1); sel = Math.min(sel, state.keys.length - 1); rebuild(); }
  }
  function rebuildPath() {
    const s = S(), pts = [];
    if (state.keys.length) for (let t = 0; t <= s.duration + 1e-6; t += 1 / s.fps) { const q = sample(s, t, dist0); pts.push(new THREE.Vector3(...q.eye)); }
    pathLine.geometry.dispose(); pathLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  }
  function rebuild() {
    // path line + key frustums (+ invisible pick spheres for clicking keys in the editor)
    rebuildPath();
    tc.detach(); keyGroup.clear(); keyObjs = []; keyHelpers = []; pickers = [];
    state.keys.forEach((k, i) => {
      const c = new THREE.PerspectiveCamera(lensToFov(k.lens), ASPECT, dist0 * 0.02, dist0 * 0.18); c.up.copy(up);
      c.position.set(...k.eye); c.lookAt(new THREE.Vector3(...k.target)); c.rotateZ((-k.roll * Math.PI) / 180); c.updateMatrixWorld(); c.updateProjectionMatrix();
      const h = new THREE.CameraHelper(c); h.setColors?.(new THREE.Color(i === sel || (grouped() && multi.has(i)) ? 0xc8ff00 : 0x888888), new THREE.Color(0x555555), new THREE.Color(0x555555), new THREE.Color(0x888888), new THREE.Color(0x333333));
      const pick = new THREE.Mesh(new THREE.SphereGeometry(dist0 * 0.07), new THREE.MeshBasicMaterial({ visible: false }));
      pick.position.copy(c.position); pick.userData.key = i;
      keyGroup.add(c, h, pick); keyObjs.push(c); keyHelpers.push(h); pickers.push(pick);
    });
    targetObj = null;
    const k = state.keys[sel];
    if (k) {   // the selected key's look-at point: lime = draggable fixed target, pink = follows the subject (read-only)
      const tracked = TRACK && TRACK[k.aim];
      const p = tracked ? trackAt(k.aim, sample(S(), k.t, dist0).scene) : k.target;
      targetObj = new THREE.Mesh(new THREE.SphereGeometry(dist0 * 0.03), new THREE.MeshBasicMaterial({ color: tracked ? 0xff3d6e : 0xc8ff00, depthTest: false }));
      targetObj.position.set(...p); targetObj.renderOrder = 10; targetObj.userData = { target: !tracked };
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...k.eye), targetObj.position.clone()]), new THREE.LineDashedMaterial({ color: 0x6b8a22, dashSize: dist0 * 0.03, gapSize: dist0 * 0.02 }));
      line.computeLineDistances(); keyGroup.add(targetObj, line);
      if (!targetObj.userData.target && gizmoOn === "target") gizmoOn = "camera";
    }
    attachGizmo(); panel(); drawTimeline(); record();
  }
  function applyPreset(name) {
    multi.clear();
    // R = the generated cameras' own distance (with the 50 mm lens ~ their field of view): Shot Prep's identity anchors then cover
    // the whole frame; further out they fall under its coverage cutoff and the render loses the person
    const D = +inDur.value, R = dist0, H = dist0 * 0.2;
    const at = (deg, r = R, h = H) => { const a = (deg * Math.PI) / 180; return centre.clone().addScaledVector(front, Math.cos(a) * r).addScaledVector(side, Math.sin(a) * r).addScaledVector(up, h).toArray(); };
    const C = centre.toArray(), K = (t, eye, scene, extra = {}) => ({ t, eye, target: C, lens: 50, roll: 0, scene, ease: "smooth", ...extra });
    // basic moves around the capture; scene time runs 0 -> 1 across the shot
    if (name.startsWith("Orbit") && meta.source) {
      // a true circle: start and end on the real camera, one look-at point (the real camera's aim) and the real camera's own distance
      // and height from it all the way round; a key every 45 degrees keeps the spherical spline round
      const src = meta.source, d = TRACK && TRACK.follow && state.follow === false ? followDelta(0) : [0, 0, 0];   // locked keys live in the setting
      const A = new THREE.Vector3(...src.target).add(new THREE.Vector3(...d)), E = new THREE.Vector3(...src.eye).add(new THREE.Vector3(...d));
      const o = E.clone().sub(A), h = o.dot(up), fl = o.clone().addScaledVector(up, -h), r = fl.length();
      const a0 = Math.atan2(fl.dot(side), fl.dot(front)), dir = name === "Orbit left" ? 1 : -1;
      const lens = src.lens * Math.max(1, ASPECT / (src.aspect || ASPECT)) * 1.02;
      state.keys = Array.from({ length: 9 }, (_, n) => {
        const f = n / 8, a = a0 + dir * f * 2 * Math.PI;
        const eye = n % 8 === 0 ? E.toArray() : A.clone().addScaledVector(front, Math.cos(a) * r).addScaledVector(side, Math.sin(a) * r).addScaledVector(up, h).toArray();
        return K(f * D, eye, f, { target: A.toArray(), lens, ease: "linear" });
      });
      pathSel.value = "orbit";
    } else {
      if (name === "Orbit left") state.keys = [0, 0.25, 0.5, 0.75, 1].map((f) => K(f * D, at(f * 360), f, { ease: "linear" }));
      if (name === "Orbit right") state.keys = [0, 0.25, 0.5, 0.75, 1].map((f) => K(f * D, at(-f * 360), f, { ease: "linear" }));
    }
    if (meta.source && /^(Arc|Push in|Crane|Bullet)/.test(name)) {
      // near-source moves: start on the real input camera and stay within ~60 degrees of it. There Shot Prep's anchors are the footage
      // itself or generated views of the side the camera saw; the far side is invented by the view model and identity drifts there
      const src = meta.source, o = new THREE.Vector3(...src.eye).sub(centre), h = o.dot(up), fl = o.clone().addScaledVector(up, -h);
      const a0 = Math.atan2(fl.dot(side), fl.dot(front)) * 180 / Math.PI, r0 = fl.length();
      const SHOT_ASPECT = ASPECT, lens = src.lens * Math.max(1, SHOT_ASPECT / (src.aspect || SHOT_ASPECT)) * 1.02;
      const near = (deg, r = r0, hh = h) => at(a0 + deg, r, hh), aim = TRACK && TRACK.body ? { aim: "body" } : {};
      const head = TRACK && TRACK.head ? { aim: "head" } : {}, k0 = K(0, src.eye, 0, { target: src.target, lens });
      if (name === "Arc left" || name === "Arc right") state.keys = [k0, K(D, near(name === "Arc left" ? 50 : -50), 1, { lens, ...aim })];
      if (name === "Push in") state.keys = [k0, K(D, near(0, r0 * 0.45), 1, { lens, ...head })];
      if (name === "Crane up") state.keys = [k0, K(D, near(0, r0 * 0.9, h + r0 * 0.6), 1, { lens, ...aim })];
      if (name === "Bullet time (arc)")   // real time, then the moment freezes while the camera swings 60 degrees, then time runs on
        state.keys = [k0, K(0.3 * D, near(0), 0.3, { lens, ...aim, freeze: true }), K(0.75 * D, near(60), 0.3, { lens, ...aim }), K(D, near(70), 0.55, { lens, ...aim })];
      pathSel.value = "orbit";
    }
    if (name === "Zoom in") state.keys = [K(0, at(0), 0, { lens: 35 }), K(D, at(0), 1, { lens: 85 })];
    if (name === "Zoom out") state.keys = [K(0, at(0), 0, { lens: 85 }), K(D, at(0), 1, { lens: 35 })];
    if (name.startsWith("Face") && TRACK && TRACK.head) {
      // wide on the body -> push in to head and shoulders -> orbit left -> orbit right -> pull back out; aims follow the subject
      const hd = TRACK.head.reduce((a, p) => a.add(new THREE.Vector3(...p)), new THREE.Vector3()).multiplyScalar(1 / TRACK.head.length);
      const dc = dist0 * 0.42, face = (deg) => { const a = (deg * Math.PI) / 180;
        return hd.clone().addScaledVector(front, Math.cos(a) * dc).addScaledVector(side, Math.sin(a) * dc).addScaledVector(up, dc * 0.08).toArray(); };
      state.keys = [K(0, at(0), 0, { aim: "body", lens: 35 }), K(0.25 * D, face(0), 0, { aim: "head", lens: 50 }),
        K(0.42 * D, face(-40), 0, { aim: "head", lens: 50 }), K(0.72 * D, face(40), 0, { aim: "head", lens: 50 }), K(D, at(0), 0, { aim: "body", lens: 35 })];
      pathSel.value = "orbit";
    }
    state.scene_mode = "auto";
    sel = 0; live = false; rebuild();
  }
  function setView(v) {
    view = v; const cam = v === "camera";
    bView.replaceChildren(icon(cam ? "camera" : "editor"), el("span", {}, cam ? "Camera view" : "Editor view"));
    bView.title = cam ? "Camera view. Switch to editor (C)" : "Editor view. Switch to camera (C)"; bView.classList.toggle("on", !cam); helper.visible = v === "editor"; keyGroup.visible = pathLine.visible = tcHelper.visible = rigDots.visible = v === "editor"; attachGizmo(); }
  setView("camera");

  // ---------- side panel ----------
  let advancedOpen = false;   // survives panel() rebuilding the inspector
  function panel() {
    if (pop && aside.contains(pop.panel)) setPop(null);
    aside.innerHTML = "";
    const k = sel >= 0 ? state.keys[sel] : null;
    const MI = (name, label, title, fn, kbd = "") => el("button", { title, onclick: (ev) => { ev.currentTarget.blur(); setPop(null); fn(); } },
      [icon(name), label, kbd ? el("kbd", {}, kbd) : ""]);
    const toEditor = (mode) => () => { setGizmo(mode); if (view !== "editor") setView("editor"); };
    const actions = popover(IB("more", "Key actions", () => {}), [el("div", { class: "menu" }, [
      ...(k ? [MI("copy", "Copy key", "Copy this key", copyKey, "Cmd+C")] : []),
      MI("paste", "Paste at playhead", "Paste the copied key at the playhead", pasteKey, "Cmd+V"),
      ...(k ? [MI("dup", "Duplicate", "Copy this key to the playhead (or +1 s)", duplicateKey, "Shift+D"),
        MI("move", "Move in editor", "Editor view: the gizmo moves this key's camera", toEditor("translate"), "G"),
        MI("rotate", "Rotate in editor", "Editor view: the gizmo turns this key's camera", toEditor("rotate"), "R")] : []),
      MI("frame", "Key editor view", "Add a key at the playhead from what the editor view sees", keyEditorView, "V")])]);
    aside.append(el("div", { class: "khead" }, [el("h3", {}, k ? `Key ${sel + 1} of ${state.keys.length}` : `${state.keys.length} keys`), actions]));
    if (!k) { aside.append(el("div", { class: "hint" }, "Press I to key the camera at the playhead, or pick a preset.")); return; }
    const num = (to, label, prop, step, min, max, fmt = (v) => v, title) => {
      const i = el("input", { type: "number", step, min, max, value: (+k[prop]).toFixed(prop === "lens" || prop === "roll" ? 1 : 3) });
      i.onchange = () => { k[prop] = fmt(+i.value); state.keys.sort((a, b) => a.t - b.t); sel = state.keys.indexOf(k); rebuild(); };
      to.append(el("label", { title }, [label, i]));
    };
    const vec = (to, label, prop, disabled) => {
      const ins = k[prop].map((v, ax) => { const i = el("input", { type: "number", step: (dist0 * 0.01).toFixed(4), value: (+v).toFixed(2), title: "xyz"[ax], style: "padding:4px 3px;font-size:12px;min-width:0" });
        if (disabled) i.disabled = true;
        i.onchange = () => { k[prop][ax] = +i.value; live = false; rebuild(); }; return i; });
      to.append(el("label", {}, [label, el("div", { style: "display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px" }, ins)]));
    };
    num(aside, "Time (s)", "t", 1 / +inFps.value, 0, +inDur.value, (v) => {
      const t = Math.min(+inDur.value, Math.max(0, snap(v)));
      return state.keys.some((q) => q !== k && Math.abs(q.t - t) < 0.5 / +inFps.value) ? k.t : t; });
    num(aside, "Lens (mm)", "lens", 1, 8, 300);
    const aimSel = el("select", { title: "Where this key looks: a fixed point, or the subject's head or body as they move" },
      [["fixed", "Fixed point"], ["head", "Follow head"], ["body", "Follow body"]].map(([v, l]) => el("option", { value: v, disabled: v !== "fixed" && !(TRACK && TRACK[v]) ? "" : null }, l)));
    aimSel.value = k.aim || "fixed"; aimSel.onchange = () => { k.aim = aimSel.value; if (k.aim !== "fixed") gizmoOn = "camera"; rebuild(); };
    aside.append(el("label", {}, ["Aim", aimSel]));
    if ((state.scene_mode || "keys") === "auto") {
      const fz = el("input", { type: "checkbox", style: "width:auto;justify-self:start" }); fz.checked = !!k.freeze;
      fz.onchange = () => { k.freeze = fz.checked; rebuild(); };
      aside.append(el("label", { title: "Freeze the capture until the next key. The camera keeps moving." }, ["Freeze time", fz]));
    } else num(aside, "Scene (0-1)", "scene", 0.01, 0, 1, (v) => Math.min(1, Math.max(0, v)), "Where in the capture this key is (keyed scene time)");
    const e = el("select", { title: "How the camera moves to the next key. ease in/out = slows into and out of each key, linear = constant speed, hold = stay here, then cut" },
      ["smooth", "linear", "hold"].map((x) => el("option", { value: x, selected: x === k.ease ? "" : null }, EASE_LABEL[x])));
    e.value = k.ease; e.onchange = () => { k.ease = e.value; rebuild(); };
    aside.append(el("label", {}, ["Ease", e]));
    const adv = el("details", {}, [el("summary", {}, "Advanced")]); adv.open = advancedOpen; adv.ontoggle = () => { advancedOpen = adv.open; };
    vec(adv, "Camera", "eye"); vec(adv, "Target", "target", TRACK && TRACK[k.aim]); num(adv, "Roll (°)", "roll", 1, -180, 180);
    adv.append(el("div", { class: "hint" }, `Capture time here: ${(sample(S(), k.t, dist0).scene * meta.seconds).toFixed(2)} of ${meta.seconds.toFixed(2)} s`));
    aside.append(adv);
    aside.append(el("div", { class: "kfoot" }, [
      IB("update", "Replace this key's camera with the current view", () => { const nk = keyFromCam(k.t); Object.assign(k, { eye: nk.eye, target: nk.target, lens: nk.lens, roll: nk.roll }); live = false; rebuild(); },
        "Update from camera", { show: true, cls: "quiet" }),
      el("span", { class: "grow" }), IB("trash", "Delete key (X in editor view)", deleteKey, "Delete")]));
  }

  // ---------- timeline ----------
  // lanes: ruler (seek) | camera keys (select, Shift+click to add, drag to retime; one drag moves a whole selection) | capture time
  const TL = { pad: 58, g: null };
  const lanes = (h) => ({ r1: 20, c0: 22, c1: 74, k: 44, s0: 80, s1: h - 6 });
  function tToX(t, w) { return TL.pad + (t / Math.max(+inDur.value, 1e-3)) * (w - 2 * TL.pad); }
  function xToT(x, w) { return Math.min(+inDur.value, Math.max(0, ((x - TL.pad) / (w - 2 * TL.pad)) * +inDur.value)); }
  function drawTimeline() {
    const r = tlc.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (!r.width || !r.height) return;
    if (tlc.width !== Math.round(r.width * dpr) || tlc.height !== Math.round(r.height * dpr)) { tlc.width = Math.round(r.width * dpr); tlc.height = Math.round(r.height * dpr); }
    const g = tlc.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = r.width, h = r.height, D = +inDur.value, F = fpsV(), L = lanes(h), keys = state.keys, eps = 0.5 / F;
    g.fillStyle = "#0b0b0b"; g.fillRect(0, 0, w, h);
    g.fillStyle = "#101010"; g.fillRect(0, L.c0, w, L.c1 - L.c0); g.fillRect(0, L.s0, w, L.s1 - L.s0);
    // ruler: seconds, frame ticks when they fit, OUT boundary and the last rendered frame
    g.font = "500 11px \"Space Grotesk\", system-ui"; g.textBaseline = "alphabetic";
    const pxF = (w - 2 * TL.pad) / Math.max(D * F, 1), every = pxF >= 6 ? 1 : pxF >= 2 ? Math.ceil(F / 4) : F;
    g.fillStyle = "#333"; for (let f = 0; f <= D * F + 1e-6; f += every) g.fillRect(tToX(f / F, w), L.r1 - 5, 1, 5);
    g.fillStyle = "#666"; for (let t = 0; t <= D + 1e-6; t += 1) { const x = tToX(t, w); g.fillRect(x, L.r1 - 10, 1, 10); g.fillText(`${t}s`, x + 3, 12); }
    const xo = tToX(D, w); g.fillStyle = "#3a3a3a"; g.fillRect(xo, 0, 1, h); g.fillStyle = "#777"; g.fillText("OUT", xo - 26, 12);
    g.fillStyle = "#2b2b2b"; g.fillRect(tToX(lastT(), w), L.c0, 1, h - L.c0);
    g.fillStyle = "#6a6a6a"; g.fillText("camera", 6, L.k + 4); g.fillText("capture", 6, (L.s0 + L.s1) / 2 + 4);
    const ks = keys.map((k, i) => [k, i]).sort((a, b) => a[0].t - b[0].t), shotEase = easeSel.value === "smooth";
    // capture lane: freeze bands (the capture stops, the camera keeps moving), then the sampled capture-time curve
    ks.forEach(([k], n) => {
      if (!k.freeze || n === ks.length - 1 || (state.scene_mode || "keys") !== "auto") return;
      const x0 = tToX(k.t, w), x1 = tToX(ks[n + 1][0].t, w);
      g.fillStyle = "rgba(90,160,255,.16)"; g.fillRect(x0, L.s0, x1 - x0, L.s1 - L.s0);
      g.strokeStyle = "rgba(90,160,255,.35)"; g.lineWidth = 1; g.beginPath();
      for (let x = x0 - (L.s1 - L.s0); x < x1; x += 7) { g.moveTo(Math.max(x, x0), L.s1 - Math.max(0, x0 - x)); g.lineTo(Math.min(x + (L.s1 - L.s0), x1), L.s0 + Math.max(0, x + (L.s1 - L.s0) - x1)); }
      g.stroke(); g.fillStyle = "#7fb2ff"; g.fillText("FREEZE", x0 + 4, L.s1 - 4);
    });
    if (keys.length) {
      g.strokeStyle = "#c8ff00"; g.lineWidth = 1.5; g.beginPath(); let prev = null;
      for (let x = TL.pad; x <= w - TL.pad; x += 3) {
        const sc = sample(S(), xToT(x, w), dist0).scene, y = L.s1 - 3 - sc * (L.s1 - L.s0 - 16);
        if (prev === null || Math.abs(sc - prev) > 0.5) g.moveTo(x, y); else g.lineTo(x, y); prev = sc;   // capture loops jump, not slide
      }
      g.stroke();
    }
    // camera lane: segment ease badges, keys, real arrival under the whole-shot ease
    g.textAlign = "center";
    ks.forEach(([k], n) => {
      if (n === ks.length - 1) return;
      const x0 = tToX(k.t, w), x1 = tToX(ks[n + 1][0].t, w), e = k.ease || "smooth";
      g.strokeStyle = e === "hold" ? "#ff9d3d" : "#3a3a3a"; g.lineWidth = 2; g.setLineDash(e === "hold" ? [4, 3] : []);
      g.beginPath(); g.moveTo(x0 + 8, L.k); g.lineTo(x1 - 8, L.k); g.stroke(); g.setLineDash([]);
      if (x1 - x0 > 56) { g.fillStyle = e === "hold" ? "#ff9d3d" : "#888"; g.fillText(shotEase && e !== "hold" ? "whole shot" : EASE_LABEL[e] || e, (x0 + x1) / 2, L.k + 22); }
    });
    g.textAlign = "left";
    ks.forEach(([k, i]) => {
      const out = k.t > D + 1e-6, x = tToX(Math.min(k.t, D), w), on = i === sel || (multi.size > 1 && multi.has(i));
      const blocked = TL.g && TL.g.blocked && TL.g.keys && TL.g.keys.includes(k);
      g.fillStyle = out || blocked ? "#ff3d6e" : on ? "#c8ff00" : "#cfcfcf";
      g.beginPath(); g.moveTo(x, L.k - 8); g.lineTo(x + 8, L.k); g.lineTo(x, L.k + 8); g.lineTo(x - 8, L.k); g.fill();
      if (out) { g.fillStyle = "#ff3d6e"; g.fillText("!", x + 10, L.k + 4); }
      const a = arrival(k);
      if (Math.abs(a - k.t) > eps) {   // where the camera really gets there (whole-shot ease)
        const xa = tToX(a, w); g.strokeStyle = on ? "#c8ff00" : "#8a8a8a"; g.lineWidth = 1; g.setLineDash([2, 2]);
        g.beginPath(); g.moveTo(x, L.k + 9); g.lineTo(xa, L.c1 - 4); g.stroke(); g.setLineDash([]);
        g.beginPath(); g.moveTo(xa, L.c1 - 9); g.lineTo(xa + 4, L.c1 - 5); g.lineTo(xa, L.c1 - 1); g.lineTo(xa - 4, L.c1 - 5); g.closePath(); g.stroke();
      }
    });
    // playhead with a handle on the ruler
    const xp = tToX(playhead, w); g.fillStyle = "#ff3d6e"; g.fillRect(xp - 1, 0, 2, h);
    g.beginPath(); g.moveTo(xp - 6, 0); g.lineTo(xp + 6, 0); g.lineTo(xp, 8); g.fill();
    const sc = keys.length ? sample(S(), playhead, dist0).scene : sceneT;
    g.fillStyle = "#c8ff00"; g.fillText(`${(sc * meta.seconds).toFixed(2)} / ${meta.seconds.toFixed(2)} s`, Math.max(TL.pad + 4, Math.min(xp + 6, w - 92)), L.s0 + 11);
  }
  const tlPos = (ev) => { const r = tlc.getBoundingClientRect(); return { x: ev.clientX - r.left, y: ev.clientY - r.top, w: r.width, h: r.height }; };
  function endGesture(commit) {
    const G = TL.g; TL.g = null; if (!G || G.scrub) return;
    if (!commit && G.moved) { G.keys.forEach((k, n) => (k.t = G.t0[n])); state.keys.sort((a, b) => a.t - b.t); }
    sel = state.keys.indexOf(G.sel); if (multi.size > 1) { multi.clear(); G.keys.forEach((k) => multi.add(state.keys.indexOf(k))); }
    history.mute = G.mute; if (commit && G.moved) record();
    rebuild();
  }
  tlc.onpointerdown = (ev) => {
    if (ev.button !== 0) return;
    tlc.focus(); playing = false;
    const { x, y, w, h } = tlPos(ev), L = lanes(h);
    let hit = -1;
    if (y >= L.c0 && y <= L.c1) {
      let best = 9;
      state.keys.forEach((k, i) => { const d = Math.abs(tToX(Math.min(k.t, +inDur.value), w) - x); if (d < best) { best = d; hit = i; } });
    }
    if (hit >= 0 && ev.shiftKey) {   // the same selection as Shift+click in the viewport
      if (!multi.size && sel >= 0) multi.add(sel);
      if (multi.has(hit) && multi.size > 1) { multi.delete(hit); if (sel === hit) sel = [...multi][0]; } else { multi.add(hit); sel = hit; }
      live = false; rebuild(); return;
    }
    if (hit >= 0) {
      if (!(multi.size > 1 && multi.has(hit))) multi.clear();
      sel = hit; const k = state.keys[hit], group = multi.size > 1 ? [...multi].map((i) => state.keys[i]) : [k];
      TL.g = { keys: group, t0: group.map((q) => q.t), x0: x, moved: false, sel: k, mute: history.mute, blocked: false };
      history.mute = true; playhead = arrival(k); live = false; rebuild();
    } else { TL.g = { scrub: true }; seek(xToT(x, w)); }
    tlc.setPointerCapture(ev.pointerId);
  };
  tlc.onpointermove = (ev) => {
    const G = TL.g; if (!G) return;
    const { x, w } = tlPos(ev);
    if (G.scrub) { seek(xToT(x, w)); return; }
    if (!G.moved && Math.abs(x - G.x0) < 3) return;
    G.moved = true;
    const D = +inDur.value, F = fpsV();
    let d = ((x - G.x0) / (w - 2 * TL.pad)) * D; if (!ev.altKey) d = Math.round(d * F) / F;
    d = Math.max(-Math.min(...G.t0), Math.min(D - Math.max(...G.t0), d));   // the group keeps its spacing at the shot's ends
    const nt = G.t0.map((t) => t + d), others = state.keys.filter((k) => !G.keys.includes(k));
    G.blocked = nt.some((t) => others.some((o) => Math.abs(o.t - t) < 0.5 / F));   // never land on another key's frame
    if (!G.blocked) { G.keys.forEach((k, n) => (k.t = nt[n])); state.keys.sort((a, b) => a.t - b.t); sel = state.keys.indexOf(G.sel); playhead = arrival(G.sel); rebuildPath(); }
    drawTimeline();
  };
  tlc.onpointerup = () => endGesture(true);
  tlc.onpointercancel = () => endGesture(false);
  tlc.onlostpointercapture = () => { if (TL.g) endGesture(true); };

  // ---------- input ----------
  const keysDown = new Set();
  const onKey = (ev) => {
    if (minimised || ev.target instanceof HTMLInputElement || ev.target instanceof HTMLSelectElement) return;
    ev.stopPropagation();
    const k = ev.key.toLowerCase(), mod = ev.metaKey || ev.ctrlKey;
    if (k === "control") { ctrlHeld = ev.type === "keydown"; applySnap(); }
    if (ev.type === "keydown" && mod && k === "z") { ev.shiftKey ? redo() : undo(); ev.preventDefault(); return; }
    if (ev.type === "keydown" && mod && k === "y") { redo(); ev.preventDefault(); return; }
    if (ev.type === "keydown" && mod && k === "c") { copyKey(); ev.preventDefault(); return; }
    if (ev.type === "keydown" && mod && k === "v") { pasteKey(); ev.preventDefault(); return; }
    if (ev.type === "keydown" && ev.shiftKey && k === "d") { duplicateKey(); ev.preventDefault(); return; }
    if (ev.type === "keydown" && ((view === "editor" && k === "x") || k === "delete" || k === "backspace")) { deleteKey(); ev.preventDefault(); return; }
    if (ev.type === "keydown" && view === "editor" && k === "v") { keyEditorView(); return; }
    if (ev.type === "keydown") {
      const np = { Numpad1: ["front", "back"], Numpad3: ["right", "left"], Numpad7: ["top", "bottom"] }[ev.code];
      if (np) { goView(np[mod ? 1 : 0]); ev.preventDefault(); return; }
      if (k === "." || ev.code === "NumpadDecimal") { editPan.set(0, 0, 0); editTween = null; ev.preventDefault(); return; }
      const tk = { home: () => seek(0), end: () => seek(lastT()), arrowleft: () => (ev.shiftKey ? jumpKey(-1) : stepFrame(-1)), arrowright: () => (ev.shiftKey ? jumpKey(1) : stepFrame(1)),
                   j: () => shuttle(-1), k: () => (playing = false), l: () => shuttle(1), i: () => addKey() }[k];
      if (tk && !(ev.repeat && (k === "i" || k === "j" || k === "l"))) { tk(); ev.preventDefault(); return; }
      if (tk) { ev.preventDefault(); return; }
      if (k === "escape" && TL.g) { endGesture(false); ev.preventDefault(); return; }
      if (k === "escape" && pop) { setPop(null); ev.preventDefault(); return; }
      if (k === " ") { togglePlay(); ev.preventDefault(); }
      else if (k === "c") setView(view === "camera" ? "editor" : "camera");
      else if (view === "editor" && k === "g") { if (gizmoOn !== "target") setGizmo("translate"); }
      else if (view === "editor" && k === "r") { gizmoOn = "camera"; setGizmo("rotate"); attachGizmo(); }
      else if (k === "escape") close(false);
      else keysDown.add(k);
    } else keysDown.delete(k);
  };
  window.addEventListener("keydown", onKey, true); window.addEventListener("keyup", onKey, true);
  let dragging = null;
  const ray = new THREE.Raycaster();
  pip.onclick = () => setView("camera");
  renderer.domElement.addEventListener("dblclick", (ev) => {      // double-click a key camera: look through it
    if (view !== "editor") return;
    const r = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1), edit);
    const hit = ray.intersectObjects(pickers)[0];
    if (hit) { sel = hit.object.userData.key; playhead = state.keys[sel].t; live = false; setView("camera"); rebuild(); }
  });
  renderer.domElement.addEventListener("pointerdown", (ev) => {
    if (view === "editor" && (tc.axis !== null || tc.dragging)) return;          // the gizmo owns this drag
    if (view === "editor") {
      const r = renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1), edit);
      if (targetObj && targetObj.userData.target && ray.intersectObject(targetObj)[0]) { gizmoOn = "target"; attachGizmo(); return; }
      const hit = ray.intersectObjects(pickers)[0];
      if (hit) {
        const i = hit.object.userData.key;
        if (ev.shiftKey) {   // add / remove this key from the selection
          if (!multi.size && sel >= 0) multi.add(sel);
          if (multi.has(i) && multi.size > 1) { multi.delete(i); if (sel === i) sel = [...multi][0]; } else { multi.add(i); sel = i; }
        } else { multi.clear(); sel = i; }
        gizmoOn = "camera"; setGizmo(tc.mode === "rotate" ? "rotate" : "translate"); playing = false; playhead = arrival(state.keys[sel]); live = false; rebuild(); return;
      }
    }
    dragging = { x: ev.clientX, y: ev.clientY, b: ev.button, pan: ev.shiftKey || ev.button === 1 }; renderer.domElement.setPointerCapture(ev.pointerId);
  });
  renderer.domElement.onpointerup = () => (dragging = null);
  renderer.domElement.onpointermove = (ev) => {
    if (!dragging || tc.dragging) return;
    const dx = ev.clientX - dragging.x, dy = ev.clientY - dragging.y; dragging.x = ev.clientX; dragging.y = ev.clientY;
    if (view === "camera") { cam.yaw -= dx * 0.004; cam.pitch = THREE.MathUtils.clamp(cam.pitch - dy * 0.004, -1.5, 1.5); live = true; }
    else if (dragging.pan) {   // move the view across the scene (Blender's Shift + middle drag)
      const m = edit.matrixWorld.elements, k = editOrbit.r * 0.0016; editTween = null;
      editPan.add(new THREE.Vector3(m[0], m[1], m[2]).multiplyScalar(-dx * k)).add(new THREE.Vector3(m[4], m[5], m[6]).multiplyScalar(dy * k));
    } else { editTween = null; editOrbit.az -= dx * 0.006; editOrbit.el = THREE.MathUtils.clamp(editOrbit.el + dy * 0.006, -POLE, POLE); }
  };
  // ---------- view gizmo (Blender-style) ----------
  const NAV = [["front", front, "#ff5a5a", "FRONT"], ["back", front, "#ff5a5a", "BACK", -1], ["left", side, "#7bd14a", "L"], ["right", side, "#7bd14a", "R", -1],
               ["top", up, "#5aa0ff", "TOP"], ["bottom", up, "#5aa0ff", "BTM", -1]];
  let navDots = [], navDrag = null;
  function drawNav() {
    const g = navg.getContext("2d"), m = edit.matrixWorld.elements, cx = 110, L = 74;
    const R = new THREE.Vector3(m[0], m[1], m[2]), U = new THREE.Vector3(m[4], m[5], m[6]), B = new THREE.Vector3(m[8], m[9], m[10]);
    navDots = NAV.map(([name, v, col, lab, sgn = 1]) => ({ name, col, lab, neg: sgn < 0, x: cx + v.dot(R) * L * sgn, y: cx - v.dot(U) * L * sgn, z: v.dot(B) * sgn }))
      .sort((a, b) => a.z - b.z);
    g.clearRect(0, 0, 220, 220); g.lineWidth = 4; g.font = "700 15px \"Unbounded\", system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
    for (const d of navDots) {
      if (!d.neg) { g.strokeStyle = d.col; g.beginPath(); g.moveTo(cx, cx); g.lineTo(d.x, d.y); g.stroke(); }
      const r = d.lab.length > 1 ? 24 : 15; g.beginPath(); g.arc(d.x, d.y, r, 0, 2 * Math.PI);
      g.fillStyle = d.neg ? "rgba(20,20,20,.85)" : d.col; g.fill(); g.strokeStyle = d.col; g.lineWidth = 3; g.stroke(); g.lineWidth = 4;
      g.fillStyle = d.neg ? d.col : "#111"; g.fillText(d.lab, d.x, d.y + 1); d.r = r;
    }
  }
  navg.onpointerdown = (ev) => { navDrag = { x: ev.clientX, y: ev.clientY, moved: false }; navg.setPointerCapture(ev.pointerId); ev.stopPropagation(); };
  navg.onpointermove = (ev) => {
    if (!navDrag) return; const dx = ev.clientX - navDrag.x, dy = ev.clientY - navDrag.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) navDrag.moved = true;
    navDrag.x = ev.clientX; navDrag.y = ev.clientY; editTween = null;
    editOrbit.az -= dx * 0.012; editOrbit.el = THREE.MathUtils.clamp(editOrbit.el + dy * 0.012, -POLE, POLE);
  };
  navg.onpointerup = (ev) => {
    const d = navDrag; navDrag = null; if (!d || d.moved) return;
    const b = navg.getBoundingClientRect(), x = ((ev.clientX - b.left) / b.width) * 220, y = ((ev.clientY - b.top) / b.height) * 220;
    const hit = [...navDots].reverse().find((q) => Math.hypot(q.x - x, q.y - y) <= q.r + 4);   // front-most first
    if (hit) goView(hit.name);
  };
  renderer.domElement.onwheel = (ev) => {
    ev.preventDefault();
    if (view === "camera") { cam.lens = THREE.MathUtils.clamp(cam.lens * (ev.deltaY > 0 ? 0.95 : 1.05), 8, 300); live = true; }
    else editOrbit.r = THREE.MathUtils.clamp(editOrbit.r * (ev.deltaY > 0 ? 1.08 : 0.92), dist0 * 0.3, dist0 * 20);
  };

  // ---------- loop ----------
  let last = performance.now(), alive = true;
  function resize() {
    const r = vp.getBoundingClientRect(); renderer.setSize(r.width, r.height, false);
    edit.aspect = r.width / r.height; edit.updateProjectionMatrix(); drawTimeline();
  }
  const ro = new ResizeObserver(resize); ro.observe(vp); ro.observe(tl);
  function frame(now) {
    if (!alive) return;
    if (minimised) { requestAnimationFrame(frame); return; }   // no GPU work while the window is down
    if (bPlay.dataset.playing !== String(playing)) {   // transport icon follows every way of starting/stopping playback
      bPlay.dataset.playing = String(playing); bPlay.replaceChildren(icon(playing ? "pause" : "play"));
      bPlay.title = playing ? "Pause (Space)" : "Play (Space)"; bPlay.setAttribute("aria-label", playing ? "Pause" : "Play");
    }
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (playing) {
      const span = nFrames() / fpsV(); playhead += dt * rate * previewMul;
      if (playhead > lastT() + 1e-9) { if (loopOn) playhead -= span; else { playhead = lastT(); playing = false; } }
      if (playhead < 0) { if (loopOn) playhead += span; else { playhead = 0; playing = false; } }
      playhead = Math.min(Math.max(0, playhead), lastT()); live = false; drawTimeline();
    }
    { const f = frameOf(playhead), F = fpsV(), txt = `${String(Math.floor(f / F)).padStart(2, "0")}:${String(f % F).padStart(2, "0")}`;
      if (tcOut.textContent !== txt) tcOut.textContent = txt;
      if (document.activeElement !== inFrame && inFrame.value !== String(f)) { inFrame.value = String(f); inFrame.max = String(nFrames() - 1); }
      const rt = playing && (rate !== 1 || previewMul !== 1) ? `${rate < 0 ? "◀" : "▶"}${Math.abs(rate * previewMul)}x` : "";
      if (rateOut.textContent !== rt) rateOut.textContent = rt; }
    // fly
    if (view === "camera" && keysDown.size) {
      const sp = dist0 * (keysDown.has("shift") ? 1.6 : 0.5) * dt, d = camDir(), r = d.clone().cross(up).normalize();
      if (keysDown.has("w")) cam.eye.addScaledVector(d, sp); if (keysDown.has("s")) cam.eye.addScaledVector(d, -sp);
      if (keysDown.has("d")) cam.eye.addScaledVector(r, sp); if (keysDown.has("a")) cam.eye.addScaledVector(r, -sp);
      if (keysDown.has("e")) cam.eye.addScaledVector(up, sp); if (keysDown.has("q")) cam.eye.addScaledVector(up, -sp);
      if (keysDown.has("r")) cam.roll -= 40 * dt; if (keysDown.has("t")) cam.roll += 40 * dt;
      if ([..."wasdqert"].some((c) => keysDown.has(c))) live = true;
    }
    // evaluate the path unless the user is flying a new shot
    if (!live && state.keys.length) {
      const q = sample(S(), playhead, dist0);
      cam.eye.set(...q.eye); aimAt(new THREE.Vector3(...q.target)); cam.lens = q.lens; cam.roll = q.roll; sceneT = q.scene;
    }
    mat.uniforms.uScene.value = sceneT;
    if (meta.p2s) {   // she moves through the clip's setting (the source camera's real path): place her there at this moment
      const m = meta.p2s[Math.min(meta.p2s.length - 1, Math.max(0, Math.round(sceneT * (meta.p2s.length - 1))))];
      points.matrixAutoUpdate = false; points.matrix.set(...m); points.matrixWorldNeedsUpdate = true;
    }
    shot.position.copy(cam.eye); shot.fov = lensToFov(cam.lens); shot.updateProjectionMatrix();
    shot.lookAt(cam.eye.clone().add(camDir())); shot.rotateZ((-cam.roll * Math.PI) / 180); shot.updateMatrixWorld();
    shotProxy.position.copy(shot.position); shotProxy.quaternion.copy(shot.quaternion); shotProxy.fov = shot.fov;
    shotProxy.updateProjectionMatrix(); shotProxy.updateMatrixWorld(); helper.update();
    const r = vp.getBoundingClientRect();
    renderer.setScissorTest(false); renderer.setViewport(0, 0, r.width, r.height); renderer.setClearColor(0x050505); renderer.clear();
    if (view === "camera") {   // letterboxed 9:16 shot
      pip.style.display = "none"; navg.style.display = "none";
      const w = Math.min(r.width, r.height * ASPECT), h = w / ASPECT, x = (r.width - w) / 2, y = (r.height - h) / 2;
      renderer.setViewport(x, y, w, h); renderer.setScissor(x, y, w, h); renderer.setScissorTest(true); renderer.setClearColor(0x000000); renderer.clear();
      mat.uniforms.uPx.value = h / (2 * Math.tan((shot.fov * Math.PI) / 360));
      if (settingPts) settingPts.visible = settingSel.value === "both";
      renderer.render(scene, shot);
    } else {
      if (editTween) {
        const x = Math.min(1, (now - editTween.t0) / 280), e = x * x * (3 - 2 * x);
        editOrbit.az = editTween.az0 + (editTween.az1 - editTween.az0) * e; editOrbit.el = editTween.el0 + (editTween.el1 - editTween.el0) * e;
        if (x >= 1) editTween = null;
      }
      const pivot = centre.clone().add(editPan);
      edit.position.copy(pivot).addScaledVector(front, Math.cos(editOrbit.el) * Math.cos(editOrbit.az) * editOrbit.r)
        .addScaledVector(side, Math.cos(editOrbit.el) * Math.sin(editOrbit.az) * editOrbit.r).addScaledVector(up, Math.sin(editOrbit.el) * editOrbit.r);
      edit.lookAt(pivot); edit.updateMatrixWorld(); drawNav(); mat.uniforms.uPx.value = r.height / (2 * Math.tan((edit.fov * Math.PI) / 360));
      if (settingPts) settingPts.visible = settingSel.value !== "off";
      renderer.render(scene, edit);
      if (settingPts) settingPts.visible = settingSel.value === "both";   // the SHOT inset matches the camera view
      const ph = Math.round(Math.min(r.height * 0.34, (r.width * 0.4) / ASPECT)), pw = Math.round(ph * ASPECT), m = 12;
      const vis = [helper.visible, keyGroup.visible, pathLine.visible, tcHelper.visible, rigDots.visible];
      helper.visible = keyGroup.visible = pathLine.visible = tcHelper.visible = rigDots.visible = false;
      renderer.setViewport(r.width - pw - m, m, pw, ph); renderer.setScissor(r.width - pw - m, m, pw, ph); renderer.setScissorTest(true);
      renderer.setClearColor(0x000000); renderer.clear();
      mat.uniforms.uPx.value = ph / (2 * Math.tan((shot.fov * Math.PI) / 360));
      renderer.render(scene, shot); renderer.setScissorTest(false);
      [helper.visible, keyGroup.visible, pathLine.visible, tcHelper.visible, rigDots.visible] = vis;
      navg.style.display = "block"; pip.style.display = "block"; pip.style.width = pw + "px"; pip.style.height = ph + "px";
      pip.firstChild.textContent = sel >= 0 && Math.abs(state.keys[sel]?.t - playhead) < 1e-6 ? `KEY ${sel + 1}` : "SHOT";
    }
    // only what isn't shown elsewhere: the flying camera isn't keyed yet (time, scene, lens and roll live in the timeline and the key panel)
    const hudTxt = live ? `<span class="live">● camera moved: I to key it at ${snap(playhead).toFixed(2)} s</span>` : "";
    if (hud.innerHTML !== hudTxt) hud.innerHTML = hudTxt;
    requestAnimationFrame(frame);
  }
  [inDur, inFps, inSpeed].forEach((i) => (i.oninput = () => { if (+inDur.value > 0) playhead = Math.min(playhead, +inDur.value); rebuild(); }));
  prevSel.value = "1x"; prevSel.onchange = () => { previewMul = parseFloat(prevSel.value); prevSel.blur(); };
  inFrame.onchange = () => { seek(Math.min(nFrames() - 1, Math.max(0, Math.round(+inFrame.value))) / fpsV()); inFrame.blur(); };
  bLoop.classList.add("on");
  rebuild(); requestAnimationFrame(frame);

  // ---------- import / export ----------
  // a path file keeps the raw keys (exact on the same capture) plus a portable copy: positions in the subject's frame, turned so the
  // real input camera sits at 0 degrees and scaled by the capture distance, so a move keyed on one clip lands the same way on another
  function relFrame() {
    let f = front.clone(), s = side.clone();
    if (meta.source) {
      const o = new THREE.Vector3(...meta.source.eye).sub(centre), a = Math.atan2(o.dot(side), o.dot(front));
      f = front.clone().multiplyScalar(Math.cos(a)).addScaledVector(side, Math.sin(a));
      s = front.clone().multiplyScalar(-Math.sin(a)).addScaledVector(side, Math.cos(a));
    }
    return { f, s };
  }
  const toRel = (p) => { const { f, s } = relFrame(), v = new THREE.Vector3(...p).sub(centre); return [v.dot(f), v.dot(s), v.dot(up)].map((x) => x / dist0); };
  const fromRel = (r) => { const { f, s } = relFrame(); return centre.clone().addScaledVector(f, r[0] * dist0).addScaledVector(s, r[1] * dist0).addScaledVector(up, r[2] * dist0).toArray(); };
  function exportPath() {
    // keys on the real input camera are tagged: on another clip they take that clip's real camera (position, aim and lens)
    const onSrc = (k) => meta.source && new THREE.Vector3(...k.eye).distanceTo(new THREE.Vector3(...meta.source.eye)) < dist0 * 1e-3;
    const st = S(), cr = camRel(meta, state.follow);
    const out = { dropcam_path: 1, state: st, rel: st.keys.map((k) => ({ eye: toRel(k.eye), target: toRel(k.target), source: onSrc(k) || undefined })),
                  cam: cr ? st.keys.map((k) => cr.to(k)) : undefined };
    const a = el("a", { href: URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" })), download: "camera_path.json" });
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  function importPath(text) {
    multi.clear();
    let d; try { d = JSON.parse(text); } catch { return alertMsg("Not a JSON file"); }
    const st = d.state || d;   // a bare director_state (copied off a node) also loads
    if (!Array.isArray(st.keys)) return alertMsg("No camera keys in this file");
    const keys = JSON.parse(JSON.stringify(st.keys));
    const cr = Array.isArray(d.cam) && d.cam.length === keys.length && camRel(meta, st.follow);
    if (st.space_dir !== trainDir && cr) keys.forEach((k, i) => cr.from(d.cam[i], k));   // camera-relative: from this clip's real camera
    else if (st.space_dir !== trainDir) {
      if (!Array.isArray(d.rel) || d.rel.length !== keys.length) return alertMsg("This path was made on another capture and has no portable copy: export it again from the Director");
      keys.forEach((k, i) => {
        if (d.rel[i].source && meta.source) Object.assign(k, { eye: [...meta.source.eye], target: [...meta.source.target], lens: meta.source.lens });
        else { k.eye = fromRel(d.rel[i].eye); k.target = fromRel(d.rel[i].target); }
      });
    }
    state.keys = keys; state.scene_mode = st.scene_mode || "auto";
    if (st.duration) inDur.value = st.duration; if (st.fps) inFps.value = st.fps;
    inSpeed.value = st.speed ?? 1;
    if (st.scene_end) endSel.value = st.scene_end; if (st.path) pathSel.value = st.path;
    easeSel.value = st.shot_ease === "smooth" ? "smooth" : "keys"; if (st.follow !== undefined) { state.follow = st.follow; followSel.value = st.follow === false ? "off" : "on"; }
    // aspect stays this clip's: the shot shape follows the input, not the clip the path was keyed on
    sel = 0; live = false; rebuild();
  }
  function alertMsg(t) { const m = el("div", { class: "msg", style: "pointer-events:none" }, t); vp.append(m); setTimeout(() => m.remove(), 3000); }

  // ---------- minimise ----------
  let minimised = false;
  const tab = el("div", { style: "position:fixed;right:16px;bottom:16px;z-index:100000;display:flex;gap:6px;align-items:center;padding:6px 8px;background:#080808;border:1px solid #c8ff00;border-radius:8px;font:13px system-ui,sans-serif;color:#e6e6e6" });
  function minimise(on) {
    minimised = on;
    if (on) {   // save to the node so the graph can be queued with this path while the window is down
      wState.value = JSON.stringify(S()); node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true);
      playing = false; keysDown.clear(); setPop(null); root.style.display = "none";
      tab.replaceChildren(el("b", { style: "color:#c8ff00;letter-spacing:.06em" }, "VIRTUAL CAMERA"),
        el("button", { style: "background:#c8ff00;color:#000;border:0;border-radius:6px;padding:4px 8px;font-weight:700;cursor:pointer", onclick: () => minimise(false) }, "Open"));
      document.body.append(tab); node.__s3dRestore = () => minimise(false);
    } else { tab.remove(); delete node.__s3dRestore; root.style.display = ""; last = performance.now(); resize(); }
  }

  function close(save) {
    // remove the window first: a teardown error must never leave it stuck open. If the capture failed to open, the editor below
    // (state helpers, render loop, corner tab) was never built, so every step after this is guarded
    root.remove(); delete node.__s3dRestore;
    const safe = (f) => { try { f(); } catch (e) { console.warn("Virtual Camera teardown:", e); } };
    if (save) safe(() => { wState.value = JSON.stringify(S()); node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); });
    safe(() => { alive = false; }); safe(() => tab.remove()); safe(() => document.removeEventListener("pointerdown", onOutside, true));
    safe(() => ro.disconnect());
    safe(() => window.removeEventListener("keydown", onKey, true)); safe(() => window.removeEventListener("keyup", onKey, true));
    safe(() => tc.detach()); safe(() => tc.dispose?.()); safe(() => renderer.dispose()); safe(() => geo.dispose());
  }
}

app.registerExtension({
  name: "sam3dbx.director",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "Sam3dbxCaptureSubject") {
      // a proper video input: the clip plays in the node, looping only the trimmed part, with what the trim becomes
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = created?.apply(this, arguments);
        const W = (n) => this.widgets.find((w) => w.name === n), vw = W("video"), sw = W("start_time"), ew = W("end_time");
        const vid = el("video", { muted: "", playsinline: "", style: "width:100%;max-height:360px;object-fit:contain;background:#000;border-radius:6px;display:block" });
        vid.muted = true;
        const label = el("div", { style: "font:12px system-ui,sans-serif;color:#9a9a9a;padding:4px 2px 0;white-space:pre-wrap" });
        const box = el("div", { style: "width:100%" }, [vid, label]);
        let boxH = 220;   // follows the clip's aspect once it loads, so the node frame wraps the player and the readout
        this.addDOMWidget("trim_preview", "preview", box, { serialize: false, hideOnZoom: false, getMinHeight: () => boxH, getMaxHeight: () => boxH });
        const range = () => {
          const s0 = Math.max(0, +sw.value || 0), d = isFinite(vid.duration) ? vid.duration : Infinity;
          const e0 = +ew.value > s0 ? +ew.value : s0 + 121 / 25;
          return [Math.min(s0, d), Math.min(e0, d)];
        };
        const show = () => {
          const [s0, e0] = range(), len = e0 - s0, d = isFinite(vid.duration) ? `${vid.duration.toFixed(2)} s clip` : "";
          label.textContent = len >= 0.5 ? `Selected ${s0.toFixed(2)} to ${e0.toFixed(2)} s (${len.toFixed(2)} s)${d ? " · " + d : ""}`
            : `The trim is ${len.toFixed(2)} s: pick a start and end at least 0.5 s apart${d ? " · " + d : ""}`;
          if (vid.readyState >= 1 && (vid.currentTime < s0 - 0.05 || vid.currentTime > e0)) vid.currentTime = s0;
        };
        const dropStock = () => {   // the frontend's own full-length player for upload inputs: the trim preview replaces it
          const i = this.widgets.findIndex((w) => w.name === "video-preview");
          if (i < 0) return;
          const w = this.widgets[i]; try { w.onRemove?.(); } catch {} w.element?.remove?.(); this.widgets.splice(i, 1);
          this.setSize([this.size[0], this.computeSize()[1]]); this.setDirtyCanvas?.(true, true);
        };
        const load = () => {
          setTimeout(dropStock, 0); setTimeout(dropStock, 300);
          const v = vw?.value; if (!v) return;
          const i = v.lastIndexOf("/"), sub = i >= 0 ? v.slice(0, i) : "", fn = i >= 0 ? v.slice(i + 1) : v;
          vid.src = api.apiURL(`/view?filename=${encodeURIComponent(fn)}&type=input&subfolder=${encodeURIComponent(sub)}`);
        };
        vid.onloadedmetadata = () => {
          dropStock(); vid.currentTime = range()[0]; show(); vid.play().catch(() => {});
          boxH = Math.round(Math.min(360, (this.size[0] - 20) * vid.videoHeight / Math.max(1, vid.videoWidth))) + 44;
          this.setSize([this.size[0], this.computeSize()[1]]); this.setDirtyCanvas?.(true, true);
        };
        const tick = () => {   // loop the trim: back to start at the end point (frame-accurate where the browser supports it)
          const [s0, e0] = range();
          if (vid.currentTime >= e0 || vid.ended) { vid.currentTime = s0; vid.play().catch(() => {}); }
          vid.requestVideoFrameCallback ? vid.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
        };
        vid.requestVideoFrameCallback ? vid.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
        for (const [w, f] of [[vw, load], [sw, show], [ew, show]]) {
          if (!w) continue;
          const cb = w.callback; w.callback = function () { const out = cb?.apply(this, arguments); f(); return out; };
        }
        const conf = this.onConfigure;
        this.onConfigure = function () {
          const out = conf?.apply(this, arguments);
          // a node saved by an older version: values that are no longer options go back to the defaults, old unlinked sockets go
          const defs = nodeData.input?.required || {};
          for (const w of this.widgets) {
            const d = defs[w.name]; if (!d) continue;
            if (d[0] === "STRING" && typeof w.value !== "string") { w.value = d[1]?.default ?? ""; continue; }   // e.g. name, added after the save
            if (!Array.isArray(d[0])) continue;
            const renamed = { "fast (mannequin)": "mannequin only (no training)", "fast + 4DGS splat": "4DGS splat (train)" };   // old saves keep their choice
            if (!d[0].includes(w.value)) w.value = d[0].includes(renamed[w.value]) ? renamed[w.value] : d[1]?.default ?? d[0][0];
          }
          const known = new Set([...Object.keys(defs), ...Object.keys(nodeData.input?.optional || {})]);
          for (let i = (this.inputs || []).length - 1; i >= 0; i--) if (!known.has(this.inputs[i].name) && this.inputs[i].link == null) this.removeInput(i);
          load(); return out;
        };
        const draw = this.onDrawForeground;   // the stock player can be added after load: drop it whenever it shows up
        this.onDrawForeground = function () { if (this.widgets?.some((w) => w.name === "video-preview")) dropStock(); return draw?.apply(this, arguments); };
        load(); show();
        return r;
      };
      // a finished capture becomes the capture of every Director wired to it, so "Open Virtual Camera" opens it straight away
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        executed?.apply(this, arguments);
        const dir = msg?.sam3dbx_capture?.[0]; if (!dir) return;
        this.__capture = dir;
        for (const id of this.outputs?.[0]?.links || []) {
          const l = app.graph.links[id], n = l && app.graph.getNodeById(l.target_id);
          const w = n?.type === "Sam3dbxCameraDirector" && n.widgets.find((x) => x.name === "capture");
          if (!w) continue;
          if (w.options?.values && !w.options.values.includes(dir)) w.options.values.unshift(dir);
          w.value = dir; n.setDirtyCanvas?.(true, true);
        }
      };
      return;
    }
    if (nodeData.name !== "Sam3dbxCameraDirector") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      this.addWidget("button", "🎬 Open Virtual Camera", null, () => openDirector(this));
      const pick = el("input", { type: "file", accept: ".json,application/json", style: "display:none" }); document.body.append(pick);
      pick.onchange = async () => {
        const f = pick.files[0]; pick.value = ""; if (!f) return;
        try { await importToNode(this, await f.text()); } catch (e) { alert("Camera path not imported: " + e.message); }
      };
      this.addWidget("button", "📂 Import camera path (.json)", null, () => pick.click());
      const onRemoved = this.onRemoved; this.onRemoved = function () { pick.remove(); return onRemoved?.apply(this, arguments); };
      // the camera path lives in director_state (saved with the workflow) but is edited in the Director, not as raw JSON
      const w = this.widgets.find((x) => x.name === "director_state");
      if (w) { w.hidden = true; w.computeSize = () => [0, -4]; const e = w.element || w.inputEl; if (e) e.style.display = "none"; }
      return r;
    };
  },
});
