// Runs the real 4D camera editor (web/director.js, copied in at publish) on Example 1's capture, with no ComfyUI and no GPU:
// the editor's capture routes are answered from files exported from that capture, and a stand-in Virtual Camera node holds the move.
// The editor opens by itself inside the page (#editor-host) and reopens if it is closed.
const DATA = new URL("./data/", import.meta.url);
const ROUTES = { resolve: "resolve.json", meta: "meta.json", points: "points.bin", setting: "setting.bin" };
const realFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url = typeof input === "string" ? input : input.url, m = url.match(/^\/sam3dbx\/director\/(\w+)/);
  if (!m) return realFetch(input, init);
  if (m[1] === "convert") return Promise.resolve(new Response(init.body, { headers: { "Content-Type": "application/json" } }));   // one capture: nothing to convert
  return realFetch(new URL(ROUTES[m[1]], DATA));
};

// The editor listens for keys on the whole window. In the page it only gets them after a click inside it, so Space and the
// arrow keys still scroll the page until then.
const host = document.querySelector("#editor-host");
let active = false;
document.addEventListener("pointerdown", (ev) => { active = host.contains(ev.target); host.classList.toggle("active", active); }, true);
const realAdd = window.addEventListener.bind(window), realRemove = window.removeEventListener.bind(window), gated = new Map();
window.addEventListener = (type, fn, opts) => {
  if (type !== "keydown" && type !== "keyup") return realAdd(type, fn, opts);
  const g = (ev) => (active ? fn(ev) : undefined); gated.set(fn, g); return realAdd(type, g, opts);
};
window.removeEventListener = (type, fn, opts) => realRemove(type, gated.get(fn) || fn, opts);

const { app } = await import("./scripts/app.js");
await import("./ext/web/director.js?v=2");
const move = await (await realFetch(new URL("example1_move.json", DATA))).text();

// a Virtual Camera node, as ComfyUI would create it, then the extension's own setup adds the Open button
function VirtualCamera() {
  this.type = "Sam3dbxCameraDirector"; this.inputs = []; this.outputs = [];
  this.widgets = [{ name: "capture", value: "example1" }, { name: "mode", value: "4DGS (splat)" }, { name: "director_state", value: move }];
}
VirtualCamera.prototype.addWidget = function (type, name, value, callback) { const w = { type, name, value, callback }; this.widgets.push(w); return w; };
for (const ext of app.extensions) await ext.beforeRegisterNodeDef?.(VirtualCamera, { name: "Sam3dbxCameraDirector", input: {} });
const node = new VirtualCamera(); node.onNodeCreated?.();
const open = node.widgets.find((w) => w.name.includes("Open Virtual Camera"));

// the editor appends its window to <body>: move it into the page, and open it again whenever it is closed
new MutationObserver((recs) => {
  for (const r of recs) for (const n of r.addedNodes) if (n.classList?.contains("s3d-dir")) { host.replaceChildren(n); window.dispatchEvent(new Event("resize")); }
}).observe(document.body, { childList: true });
new MutationObserver(() => { if (!host.querySelector(".s3d-dir")) open.callback(); }).observe(host, { childList: true });
open.callback();

const reset = document.querySelector("#reset-move");
reset.disabled = false;
reset.addEventListener("click", () => {   // put the rendered move back, then reopen on it (the editor's own Cancel closes it cleanly)
  node.widgets.find((w) => w.name === "director_state").value = move;
  const cancel = host.querySelector('[title="Cancel (Esc)"]');
  if (cancel) cancel.click(); else open.callback();
});
