"""Capture modes. One clip can have two captures sharing the same generated views (the `source` in dataset/manifest.json):
  fast   = the SMPL-X mannequin, no training (built in seconds from the views, so it can always be made on demand)
  4DGS   = a trained FreeTimeGS splat
The 4D Director's mode toggle picks which one to preview and render. Camera keys are stored in the space of the capture they were
designed in (state["space_dir"]); switching mode converts them with the two captures' world_to_training matrices.
Pure numpy + stdlib, so the web routes and nodes can import it without torch."""
import json, subprocess
from pathlib import Path

import numpy as np

from .config import ROOT, GEN_PY, TRAIN_PY
HERE = Path(__file__).resolve().parent
FAST, SPLAT = "fast (mannequin)", "4DGS (splat)"
MODES = [FAST, SPLAT]


def kind(d):
    d = Path(d)
    if any(d.glob("ckpts/ckpt_*.pt")):
        return "splat"
    if (d / "preview_meta.json").is_file() and (d / "dataset" / "manifest.json").is_file():
        return "fast"
    return None


def source(d):
    return json.loads((Path(d) / "dataset" / "manifest.json").read_text()).get("source")


def _family(d):
    src, out = source(d), ROOT / "out"
    found = []
    for pat in ("*/dataset/manifest.json", "*/*/dataset/manifest.json", "*/*/*/dataset/manifest.json"):
        for m in out.glob(pat):
            try:
                if source(m.parent.parent) == src:
                    found.append(m.parent.parent)
            except Exception:  # noqa: BLE001 - unreadable manifest: not part of the family
                pass
    return found


def source_camera(d):
    """The real input camera solve (source_cam.py) for this capture: its own, else a sibling's. It is stored in the capture's world,
    which the fast capture and a splat trained from the same views share, so a splat uses the fast capture's solve as is."""
    d = Path(d); own = d / "source_camera.json"
    if own.is_file():
        return own
    try:
        return next((s / "source_camera.json" for s in _family(d) if (s / "source_camera.json").is_file()), None)
    except Exception:  # noqa: BLE001 - no manifest: no family, no real camera
        return None


def ensure_preview(d):
    """A trained splat needs preview_meta.json (rig, head/body tracks) and preview_points.bin for the Director and Shot Prep; build them
    on first use (CPU only)."""
    import os
    d = Path(d); meta, pts = d / "preview_meta.json", d / "preview_points.bin"
    ck = sorted(d.glob("ckpts/ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    if ck and not (meta.is_file() and pts.is_file() and meta.stat().st_mtime >= ck[-1].stat().st_mtime):
        r = subprocess.run([TRAIN_PY, str(HERE / "dump_points.py"), f"--train_dir={d}"], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        if r.returncode != 0:
            raise RuntimeError("could not build the splat preview: " + r.stderr[-800:])
    return d


def resolve(td, mode):
    """(capture dir for `mode`, kind). Builds the fast capture on demand; a missing splat is an error with a clear message."""
    td = Path(td); want = "fast" if str(mode).startswith("fast") else "splat"
    if kind(td) == want:
        return (ensure_preview(td) if want == "splat" else td), want
    have = [d for d in _family(td) if kind(d) == want]
    if have:
        d = max(have, key=lambda d: d.stat().st_mtime)
        return (ensure_preview(d) if want == "splat" else d), want
    if want == "fast":
        src = Path(source(td))
        if not (src / "cameras.json").is_file() or not (src / "preprocessing" / "viewer_geometry.npz").is_file():
            raise ValueError(f"Fast mode needs an upstream (GVHMR) capture; {src} has no cameras.json / body mesh. Use 4DGS mode for this one.")
        dest = src.parent / f"fast_{src.name}"
        r = subprocess.run([GEN_PY, str(HERE / "scripts" / "fast_rig.py"), str(src), str(dest)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("could not build the fast capture: " + r.stderr[-800:])
        return dest, "fast"
    if kind(td) == "fast":   # a quick capture (no views, no splat): the mannequin preview is all there is, and all a Retake needs
        print("[sam3dbx] no splat for this capture: using its mannequin preview")
        return td, "fast"
    raise ValueError("This capture has no trained splat yet. Run Capture with 4DGS splat (train), or set the Virtual Camera mode to fast (mannequin).")


def _space(d):
    f = Path(d) / "normalization.json"
    return np.array(json.loads(f.read_text())["world_to_training"]) if f.is_file() else np.eye(4)


def convert_state(state, src_dir, dst_dir):
    """Re-express the Director keys (eye, target) from one capture's space in another's."""
    M = _space(dst_dir) @ np.linalg.inv(_space(src_dir))
    for k in state.get("keys", []):
        for f in ("eye", "target"):
            k[f] = (M @ np.array([*k[f], 1.0]))[:3].tolist()
    state["space_dir"] = str(dst_dir)
    return state
