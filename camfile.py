"""Camera files (.fbx, .abc, .usd/.usda/.usdc/.usdz) for the Virtual Camera: Blender reads the file headless
(tools/camfile_blender.py), lists its cameras, and bakes the picked one per frame. The move is kept relative to its own first frame
and returned in capture units, so the Director can start it on this clip's real camera: no round trip, no shared scene needed.

  rel    per frame, OpenCV camera-to-camera-0 (x right, y down, z forward), translation in capture units (metres / her 1.7 m scale)
  zoom   focal length relative to the first frame (2 = twice the focal length)
  roll   is in rel; the Director turns it into its roll channel
Focus distance (metres and capture units), f-stop and lens shift ride along for the camera features that will use them.
"""
import hashlib, json, math, subprocess, tempfile
from pathlib import Path

import numpy as np

from .config import BLENDER

EXTS = (".fbx", ".abc", ".usd", ".usda", ".usdc", ".usdz")
SCRIPT = Path(__file__).parent / "tools" / "camfile_blender.py"
EXPORT_SCRIPT = Path(__file__).parent / "tools" / "camexport_blender.py"
EXPORT_EXT = {"fbx": ".fbx", "usd": ".usdc", "abc": ".abc"}
CV = np.diag([1.0, -1.0, -1.0])   # Blender camera axes (x right, y up, looks down -z) to OpenCV (x right, y down, z forward)


def store(data: bytes, name: str, folder: Path) -> Path:
    """Keep an upload by content hash, so picking a camera re-reads the same file."""
    ext = Path(name).suffix.lower()
    if ext not in EXTS:
        raise ValueError(f"not a camera file: {name} (fbx, abc, usd, usda, usdc, usdz)")
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{hashlib.sha1(data).hexdigest()[:16]}{ext}"
    if not p.is_file():
        p.write_bytes(data)
    return p


def _blender(path: Path, camera=None):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.json"
        cmd = [BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(SCRIPT), "--", str(path), str(out)]
        try:
            r = subprocess.run(cmd + ([camera] if camera else []), capture_output=True, text=True, timeout=300)
        except FileNotFoundError:
            raise RuntimeError("Camera files (.fbx/.abc/.usd) are read with Blender, which isn't installed or isn't on PATH. Install Blender "
                               "(free) and set \"blender\" in config.local.json or SAM3DBX_BLENDER to its path. JSON camera paths work without it.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender took more than 5 minutes to read the camera file")
        if r.returncode or not out.is_file():
            tail = [l for l in (r.stderr or r.stdout).splitlines() if l.strip()][-3:]
            raise RuntimeError("Blender could not read the camera file: " + " | ".join(tail))
        return json.loads(out.read_text())


def list_cameras(path: Path):
    """What the file holds: cameras (name, frame range, lens) and the other object types."""
    return _blender(path)


def bake(path: Path, camera: str, metres_per_unit: float):
    """The picked camera as a move relative to its first frame, in capture units."""
    b = _blender(path, camera)
    F = b["frames"]
    if not F:
        raise ValueError(f"{camera} has no frames")
    C = []
    for fr in F:
        m = np.eye(4); m[:3, :3] = np.array(fr["rot"]) @ CV; m[:3, 3] = fr["pos"]; C.append(m)
    C = np.stack(C); rel = np.einsum("ij,njk->nik", np.linalg.inv(C[0]), C)
    rel[:, :3, 3] /= metres_per_unit
    tan0 = F[0]["tan_half"]
    travel = float(np.linalg.norm(rel[:, :3, 3] * metres_per_unit, axis=1).max())
    return {"file": b["file"], "camera": b["camera"], "fps": b["fps"], "duration": F[-1]["t"],
            "t": [fr["t"] for fr in F], "rel": rel.round(6).tolist(), "zoom": [round(tan0 / fr["tan_half"], 5) for fr in F],
            "lens_mm": [round(fr["lens"], 3) for fr in F], "shift": [fr["shift"] for fr in F],
            "focus_m": [round(fr["dof"]["focus"], 4) for fr in F], "focus": [round(fr["dof"]["focus"] / metres_per_unit, 5) for fr in F],
            "fstop": [fr["dof"]["fstop"] for fr in F], "dof": bool(F[0]["dof"]["use"]), "type": F[0]["type"],
            "travel_m": round(travel, 3), "metres_per_unit": metres_per_unit}


def _camfx_at(keys, t, name, subject_m):
    """A key setting at camera time t (keys that set it; focus between keys in diopters), or None."""
    ks = sorted((k for k in keys if name in k), key=lambda k: k["t"])
    if not ks:
        return None
    val = (lambda v: subject_m if v == "subject" else float(v)) if name == "focus" else float
    if t <= ks[0]["t"] or len(ks) == 1:
        return val(ks[0][name])
    if t >= ks[-1]["t"]:
        return val(ks[-1][name])
    j = max(i for i in range(len(ks) - 1) if ks[i]["t"] <= t)
    u = (t - ks[j]["t"]) / max(ks[j + 1]["t"] - ks[j]["t"], 1e-9); a, b = val(ks[j][name]), val(ks[j + 1][name])
    return 1.0 / ((1 - u) / a + u / b) if name == "focus" else a + (b - a) * u


def export_camera(state, train_dir, fmt):
    """The Virtual Camera move as an animated camera file (Blender writes it): metres, z-up, the subject at the origin, one key per frame,
    with lens, focus and f-stop; the real source camera and a Subject empty ride along for reference."""
    from . import director as D, rig as RIG, shot as S
    if fmt not in EXPORT_EXT:
        raise ValueError(f"export format: fbx, usd or abc, not {fmt!r}")
    td = Path(train_dir)
    meta = S.scene_meta(td, json.loads((td / "preview_meta.json").read_text()))
    up, front, side = (np.array(meta[k], float) for k in ("up", "front", "side"))
    M = np.stack([side, front, up])                       # capture -> Blender (x = side, y = front, z = up)
    if np.linalg.det(M) < 0:
        M[0] = -M[0]
    centre = np.array(meta["centre"], float)
    m = S.metres_per_unit(S.Capture(td, zoom_views=False, scene=True))
    rows = D.rows(state, meta["dist0"], [up.tolist(), front.tolist(), side.tolist()], meta.get("track"))
    fps = float(state.get("fps") or 24)
    keys = state.get("keys") or []
    to_b = lambda c2w: {"pos": (M @ (c2w[:3, 3] - centre) * m).tolist(), "rot": (M @ c2w[:3, :3] @ CV).tolist()}
    frames = []
    for i, r in enumerate(rows):
        eye, tgt = np.array(r[1:4]), np.array(r[4:7])
        c2w = RIG.look_at(eye, tgt, up, r[7]); fov = math.radians(r[8])
        fr = dict(to_b(c2w), lens=18.0 / math.tan(fov / 2))
        subj = float(np.linalg.norm(tgt - eye)) * m
        fs, fo = _camfx_at(keys, i / fps, "fstop", subj), _camfx_at(keys, i / fps, "focus", subj)
        if fs or fo:
            fr.update(fstop=fs or 2.8, focus=fo or subj)
        frames.append(fr)
    a = str(state.get("aspect") or "9:16")
    try:
        aw, ah = (float(x) for x in a.split(":")); aspect = aw / ah
    except ValueError:
        aspect = 9 / 16
    J = {"fps": fps, "aspect": aspect, "frames": frames}
    try:
        cap = S.Capture(td, zoom_views=False)
        if cap.source_cam is not None:
            src = cap.pose(cap.source_cam, 0); K = cap.cams[cap.source_cam][1]
            J["source"] = dict(to_b(src), lens=36.0 * K[1][1] / cap.cams[cap.source_cam][5])
    except Exception:  # noqa: BLE001 - no source camera: export the move alone
        pass
    with tempfile.TemporaryDirectory() as tmp:
        jp, out = Path(tmp) / "frames.json", Path(tmp) / f"camera{EXPORT_EXT[fmt]}"
        jp.write_text(json.dumps(J))
        try:
            r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(EXPORT_SCRIPT), "--", str(jp), str(out)],
                               capture_output=True, text=True, timeout=300)
        except FileNotFoundError:
            raise RuntimeError("Camera file export uses Blender, which isn't installed or isn't on PATH: set \"blender\" in config.local.json")
        if r.returncode or not out.is_file():
            tail = [l for l in (r.stderr or r.stdout).splitlines() if l.strip()][-3:]
            raise RuntimeError("Blender could not write the camera file: " + " | ".join(tail))
        return out.read_bytes(), out.name
