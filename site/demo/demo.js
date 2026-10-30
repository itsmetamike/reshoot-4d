// Runs the real 4D camera editor (web/director.js, copied in at publish) on Example 1's capture, with no ComfyUI and no GPU:
// the editor's capture routes are answered from files exported from that capture, and a stand-in Virtual Camera node holds the move.
const DATA = new URL("./data/", import.meta.url);
const ROUTES = { resolve: "resolve.json", meta: "meta.json", points: "points.bin", setting: "setting.bin" };
const realFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url = typeof input === "string" ? input : input.url, m = url.match(/^\/sam3dbx\/director\/(\w+)/);
  if (!m) return realFetch(input, init);
  if (m[1] === "convert") return Promise.resolve(new Response(init.body, { headers: { "Content-Type": "application/json" } }));   // one capture: nothing to convert
  return realFetch(new URL(ROUTES[m[1]], DATA));
};

const { app } = await import("./scripts/app.js");
await import("./ext/web/director.js");
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

const btn = document.querySelector("#open-editor");
btn.disabled = false; btn.textContent = "Open the 4D camera editor";
btn.addEventListener("click", () => open.callback());
document.querySelector("#reset-move").addEventListener("click", () => {
  node.widgets.find((w) => w.name === "director_state").value = move; btn.focus();
});
