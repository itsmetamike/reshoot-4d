"""Register a zoomed camera set (a second 4DAnyone run on a head-and-shoulders crop of the same clip) to a capture, so Shot Prep
can pick close-up identity anchors from it. Both runs reconstruct the same SMPL-X body (same topology, same frames), so a
similarity transform (scale, rotation, translation) fitted over the body vertices maps the zoomed set's world into the main one.
Usage: add_zoom_views.py main_capture_dir zoom_capture_dir [name=zoom]
Writes <main source>/extra_views.json (shared by the clip's fast and 4DGS captures)."""
import json, sys
from pathlib import Path

import numpy as np


def src_of(cap):
    return Path(json.loads((Path(cap) / "dataset" / "manifest.json").read_text())["source"])


main_src, zoom_src = src_of(sys.argv[1]), src_of(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else "zoom"
A = np.load(main_src / "preprocessing" / "viewer_geometry.npz")["vertices"].astype(np.float64)
B = np.load(zoom_src / "preprocessing" / "viewer_geometry.npz")["vertices"].astype(np.float64)
n = min(len(A), len(B))
# fit on the upper body only: the zoom run saw head, shoulders and torso but had to guess the legs (outside its crop)
h = A[0] @ np.array([0.0, 1.0, 0.0]); upper = np.flatnonzero(h >= np.quantile(h, 0.6))
X = B[:n:4][:, upper].reshape(-1, 3); Y = A[:n:4][:, upper].reshape(-1, 3)   # zoom-set points -> main-set points


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = float(np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean())
    return s, R, my - s * R @ mx


keep = np.ones(len(X), bool)
for _ in range(3):   # trimmed refit: drop the worst 20 % (hands and hair-adjacent vertices that differ between runs)
    s, R, t = umeyama(X[keep], Y[keep])
    err = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1); keep = err <= np.quantile(err, 0.8)
M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = t
# per-frame fits: the two runs agree on the body at every frame but can track its heading slightly differently over time
# (a spin), so anchors use the transform of their own frame
per, per_err = [], []
for f in range(n):
    Xf, Yf = B[f][upper], A[f][upper]; sf, Rf, tf = umeyama(Xf, Yf)
    Mf = np.eye(4); Mf[:3, :3] = sf * Rf; Mf[:3, 3] = tf; per.append(Mf.tolist())
    per_err.append(float(np.sqrt((np.linalg.norm((sf * (Rf @ Xf.T)).T + tf - Yf, axis=1) ** 2).mean())))
rms = float(np.sqrt((err[keep] ** 2).mean()))
f = main_src / "extra_views.json"
sets = [e for e in (json.loads(f.read_text()) if f.is_file() else []) if e.get("name") != name]
sets.append({"name": name, "source": str(zoom_src), "to_main_world": M.tolist(), "scale": s, "rms_m": rms, "valid_top": 0.65,
             "to_main_world_per_frame": per, "per_frame_rms_m_max": max(per_err)})
f.write_text(json.dumps(sets, indent=1))
print(json.dumps({"wrote": str(f), "per_frame_rms_cm_max": round(100 * max(per_err), 2), "global_scale": round(s, 4), "global_rms_m": round(rms, 4), "rot_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2)}))
