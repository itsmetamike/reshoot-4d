"""Routes the 4D camera editor (web/director.js) and the 4D Viewer node use, served by ComfyUI itself: capture metadata,
points and setting for the editor, path conversion, the browser splat player and result files from whitelisted folders.
"""
import json
from pathlib import Path

from aiohttp import web
from server import PromptServer

import folder_paths

HERE = Path(__file__).parent
from .config import ROOT, FTGS, TRAIN_PY
OUT = ROOT / "out"
UPLOADS = ROOT / "uploads"
routes = PromptServer.instance.routes
PLAYER = FTGS / "player"   # FTGS browser player (static, no build step)


def _allowed(p: Path):
    p = p.resolve()
    roots = [OUT.resolve(), UPLOADS.resolve(), Path(folder_paths.get_input_directory()).resolve(),
             (Path(folder_paths.get_output_directory()) / "sam3dbx").resolve()]
    return any(p.is_relative_to(r) for r in roots)


@routes.get("/sam3dbx/player/{name:.*}")
async def player(request):
    p = (PLAYER / request.match_info["name"]).resolve()
    if not p.is_file() or not p.is_relative_to(PLAYER.resolve()):
        return web.Response(status=404)
    return web.FileResponse(p)




def _preview(train_dir: Path):
    """Build (or reuse) the Director's browser preview of a capture: preview_meta.json + preview_points.bin."""
    import subprocess
    meta, pts = train_dir / "preview_meta.json", train_dir / "preview_points.bin"
    ckpts = sorted(train_dir.glob("ckpts/ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        if meta.is_file() and pts.is_file():   # fast-mode capture: the mannequin preview is written by fast_rig.py
            return meta, pts
        raise FileNotFoundError("no checkpoint")
    if not (meta.is_file() and pts.is_file() and meta.stat().st_mtime >= ckpts[-1].stat().st_mtime):
        subprocess.run([TRAIN_PY, str(HERE / "dump_points.py"), f"--train_dir={train_dir}"], check=True, capture_output=True,
                       env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    return meta, pts


@routes.get("/sam3dbx/director/resolve")
async def director_resolve(request):
    """Capture folder for the Director's fast / 4DGS toggle (the fast capture is built on demand)."""
    import asyncio
    from . import capture_resolve as cr
    td = Path(request.query.get("train_dir", ""))
    if not td.is_dir() or not _allowed(td):
        return web.json_response({"error": f"capture not found: {td}"}, status=404)
    try:
        d, k = await asyncio.get_running_loop().run_in_executor(None, cr.resolve, td, request.query.get("mode", cr.SPLAT))
    except Exception as e:  # noqa: BLE001 - surface the reason in the Director window
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response({"dir": str(d), "kind": k})


@routes.post("/sam3dbx/director/convert")
async def director_convert(request):
    from . import capture_resolve as cr
    b = await request.json()
    if not all(Path(d).is_dir() and _allowed(Path(d)) for d in (b["from"], b["to"])):
        return web.Response(status=404)
    return web.json_response(cr.convert_state(b["state"], b["from"], b["to"]))


@routes.get("/sam3dbx/director/meta")
async def director_meta(request):
    import asyncio
    td = Path(request.query.get("train_dir", ""))
    if not td.is_dir() or not _allowed(td):
        return web.Response(status=404)
    try:
        meta, _ = await asyncio.get_running_loop().run_in_executor(None, _preview, td)
    except Exception as e:  # noqa: BLE001 - surface the reason to the Director window
        return web.json_response({"error": str(e)}, status=500)
    from . import capture_resolve as cr
    from . import shot as S
    m = S.scene_meta(td, json.loads(meta.read_text()))   # in the setting's frame when the capture has one
    clip = td.parent / "clip.json"   # Capture Subject's trim: the 121 frames span this many real seconds
    if clip.is_file():
        m["real_seconds"] = json.loads(clip.read_text())["seconds"]
    if not cr.source_camera(td):
        return web.json_response(m)
    # the real input camera at the first frame (source_cam.py), in the Director's training space: presets start the shot on it
    import numpy as np
    cap = S.Capture(td, zoom_views=False); c2w = cap.pose(cap.source_cam, 0); K = cap.cams[cap.source_cam][1]
    eye = c2w[:3, 3]; d = float(np.linalg.norm(np.array(m["centre"]) - eye))
    if "follow" in m.get("track", {}):   # keys live relative to her (Virtual Camera follow): the real camera at frame 0 as such a key
        from . import director as D
        fl = {"f": m["track"]["follow"]}; eye = eye - (np.array(D._track_at(fl, "f", 0.0)) - np.array(D._track_at(fl, "f", 0.5)))
    m["source"] = {"eye": eye.tolist(), "target": (eye + c2w[:3, 2] * d).tolist(), "lens": 36.0 * K[1, 1] / cap.cams[cap.source_cam][5],
                   "aspect": cap.cams[cap.source_cam][4] / cap.cams[cap.source_cam][5]}
    return web.json_response(m)


@routes.get("/sam3dbx/director/points")
async def director_points(request):
    td = Path(request.query.get("train_dir", ""))
    p = td / "preview_points.bin"
    if not p.is_file() or not _allowed(p):
        return web.Response(status=404)
    return web.FileResponse(p, headers={"Content-Type": "application/octet-stream"})


@routes.get("/sam3dbx/director/setting")
async def director_setting(request):
    """The clip's real setting as static points (preview_extras.py): float32 xyz then uint8 rgb, count in the meta's setting_n."""
    td = Path(request.query.get("train_dir", ""))
    p = td / "setting_points.bin"
    if not p.is_file() or not _allowed(p):
        return web.Response(status=404)
    return web.FileResponse(p, headers={"Content-Type": "application/octet-stream"})


@routes.get("/sam3dbx/file")
async def file(request):
    p = Path(request.query.get("p", ""))
    if not p.is_file() or not _allowed(p):
        return web.Response(status=404)
    return web.FileResponse(p)