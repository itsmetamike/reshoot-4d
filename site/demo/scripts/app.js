// Stand-in for ComfyUI's app module so the 4D camera editor (web/director.js, unchanged) runs on a static page.
// registerExtension keeps the extension; the demo page builds a Virtual Camera node from it.
export const app = {
  extensions: [],
  graph: { links: {}, getNodeById: () => null, setDirtyCanvas() {} },
  registerExtension(ext) { this.extensions.push(ext); },
};
