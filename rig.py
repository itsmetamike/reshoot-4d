"""Shared rig fitting for a FreeTimeGS training folder (used by render_path.py and dump_points.py; trainer venv).

Cameras live in COLMAP (world units, OpenCV axes); splats live in the trainer's normalised space, so camera centres go
through normalization.json first. The orbit frame: centre = point nearest every camera axis, up = mean camera -y,
front = horizontal direction from centre to camera 0, plus the mean training focal length.
"""
import json, math, sys
from pathlib import Path

import numpy as np

try:
    from .config import FTGS
except ImportError:   # run as a script (render_path.py, mesh_depth.py ...)
    from config import FTGS
sys.path[:0] = [str(FTGS / "src"), str(FTGS)]


def fit_rig(train_dir):
    from datasets.read_write_model import read_cameras_binary, read_images_binary   # trainer venv only (COLMAP reader)
    train = Path(train_dir)
    sparse = next(p for p in (train / "dataset" / "sparse").rglob("cameras.bin")).parent
    cams, ims = read_cameras_binary(sparse / "cameras.bin"), read_images_binary(sparse / "images.bin")
    c2ws, fys, size = [], [], None
    for im in ims.values():
        w2c = np.eye(4); w2c[:3, :3] = im.qvec2rotmat(); w2c[:3, 3] = im.tvec
        c2ws.append(np.linalg.inv(w2c))
        c = cams[im.camera_id]; p = c.params
        fys.append(p[0] if c.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL") else p[1]); size = (c.width, c.height)
    c2ws = np.stack(c2ws)
    norm = train / "normalization.json"
    if norm.is_file():
        W2T = np.array(json.loads(norm.read_text())["world_to_training"])
        c2ws[:, :3, 3] = (W2T[:3, :3] @ c2ws[:, :3, 3].T).T + W2T[:3, 3]
    return rig_from_cams(c2ws, float(np.mean(fys)), size)


def rig_from_cams(c2ws, fy, size):
    """Orbit rig of a ring of OpenCV cameras (c2w, first = front): centre = least-squares meeting point of the view axes."""
    pos, fwd = c2ws[:, :3, 3], c2ws[:, :3, 2]
    up = -c2ws[:, :3, 1].mean(0); up /= np.linalg.norm(up)
    A = sum(np.eye(3) - np.outer(d, d) for d in fwd); b = sum((np.eye(3) - np.outer(d, d)) @ o for d, o in zip(fwd, pos))
    centre = np.linalg.solve(A, b)
    rel = pos - centre; h = rel @ up
    radius = float(np.linalg.norm(rel - np.outer(h, up), axis=1).mean())
    front = rel[0] - (rel[0] @ up) * up; front /= np.linalg.norm(front)
    return {"centre": centre, "up": up, "front": front, "side": np.cross(up, front), "radius": radius,
            "base_pitch": math.degrees(math.atan2(float(h.mean()), radius)), "dist0": math.hypot(radius, float(h.mean())),
            "fov_y": math.degrees(2 * math.atan(size[1] / 2 / fy)), "fy": fy, "size": size, "cam_pos": pos}


def look_at(eye, target, up, roll_deg=0.0):
    """OpenCV camera-to-world (x right, y down, z forward)."""
    z = target - eye; z = z / np.linalg.norm(z)
    x = np.cross(z, up); x = x / np.linalg.norm(x); y = np.cross(z, x)
    if roll_deg:
        r = math.radians(roll_deg); x, y = x * math.cos(r) + y * math.sin(r), -x * math.sin(r) + y * math.cos(r)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m
