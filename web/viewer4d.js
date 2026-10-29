// Sam3dbx 4D Viewer node: the FreeTimeGS browser player (served by this pack at /sam3dbx/player) embedded in the node.
import { app } from "../../scripts/app.js";

const playerUrl = (tsog) =>
  `/sam3dbx/player/index.html?src=${encodeURIComponent(location.origin + "/sam3dbx/file?p=" + encodeURIComponent(tsog))}&points=all`;

app.registerExtension({
  name: "sam3dbx.view4d",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Sam3dbxView4D") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const frame = document.createElement("iframe");
      frame.style.cssText = "width:100%;height:100%;border:0;border-radius:6px;background:#000;pointer-events:auto";
      frame.setAttribute("allow", "fullscreen; xr-spatial-tracking");
      this.addDOMWidget("player", "sam3dbx_player", frame, { serialize: false, getMinHeight: () => 420 });
      const show = (tsog) => { if (tsog && frame.dataset.tsog !== tsog) { frame.dataset.tsog = tsog; frame.src = playerUrl(tsog); } };
      const cap = this.widgets.find((w) => w.name === "capture");
      const cb = cap.callback;
      cap.callback = (v, ...rest) => { show(`${v}/scene.tsog`); return cb?.(v, ...rest); };
      this.__show = show;
      setTimeout(() => cap.value && !String(cap.value).startsWith("(") && show(`${cap.value}/scene.tsog`), 50);
      this.setSize([400, 760]);
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      executed?.apply(this, arguments);
      if (msg?.sam3dbx_tsog?.[0]) this.__show?.(msg.sam3dbx_tsog[0]);
    };
  },
});
