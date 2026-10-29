"""Virtual Camera preview extras for a capture, run after the setting solve (scene_layer.ensure):
- colour: a quick capture's mannequin takes the clip's real colours. Each surface point is projected into the real frame at its time
  slice with the solved source camera; points the source camera can't see there (facing away or behind another part of the body) take their
  colour from the slices where it could, and points it never saw stay light grey.
- setting: the clip's real setting (scene_layer.build: DA3 depth of the real frames, person masked out) as a point layer the
  Virtual Camera draws around the person, written as setting_points.bin (float32 xyz then uint8 rgb) with its count in the meta.
usage (ComfyUI venv): python preview_extras.py <capture dir> [setting points=200000]"""
import json, sys
from pathlib import Path

import numpy as np

from . import scene_layer as L
from . import shot as S



def colour_mannequin(td):
    td = Path(td); meta = json.loads((td / "preview_meta.json").read_text())
    if not meta.get("fast_mode") or meta.get("coloured"):   # splats have their own colours; done once per capture
        return bool(meta.get("coloured"))
    cap = S.Capture(td, zoom_views=False, scene=False); sc = cap.source_cam
    if sc is None:
        return False
    P_, K_ = meta.get("mannequin", [5000, 31])   # surface points and time slices fast_rig.py drew
    geo = np.load(cap.source / "preprocessing" / "viewer_geometry.npz"); V = geo["vertices"].astype(np.float64); F = geo["faces"]
    # the same surface samples fast_rig.py drew (same seed, same areas)
    tri = V[0][F]; area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(0); fi = rng.choice(len(F), P_, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(P_)), rng.random(P_); bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    kf = np.linspace(0, len(V) - 1, K_).round().astype(int)
    Kc = np.array(cap.cams[sc][1], float); video = cap.cams[sc][2]
    col = np.zeros((K_, P_, 3)); seen = np.zeros((K_, P_), bool)
    for k, f in enumerate(kf):
        t = V[f][F[fi]]; X = (t * bary[..., None]).sum(1); n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        c2w = cap.pose(sc, min(f, cap.frames - 1)); w2c = np.linalg.inv(c2w); Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
        img = S.video_frame(video, int(f)); h, w = img.shape[:2]
        z = Xc[:, 2]; u = Kc[0, 0] * Xc[:, 0] / np.maximum(z, 1e-6) + Kc[0, 2]; v = Kc[1, 1] * Xc[:, 1] / np.maximum(z, 1e-6) + Kc[1, 2]
        ok = (z > 1e-3) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & (np.einsum("ij,ij->i", n, c2w[:3, 3] - X) > 0)
        # occlusion: a coarse depth buffer of the points themselves (an arm in front of the chest hides the chest behind it)
        cell = max(2, h // 160); gu, gv = (u // cell).astype(int), (v // cell).astype(int); key = gv * (w // cell + 2) + gu
        zmin = np.full(key.max() + 1 if len(key) else 1, np.inf); np.minimum.at(zmin, key[ok], z[ok])
        ok &= z <= zmin[key] + 0.03 * np.median(z[ok]) if ok.any() else ok
        col[k, ok] = img[v[ok].round().astype(int), u[ok].round().astype(int)]; seen[k] = ok
    # unseen at a slice: the point's median colour over the slices that saw it; never seen: light grey
    med = np.full((P_, 3), 200.0)
    for p in np.where(seen.any(0))[0]:
        med[p] = np.median(col[seen[:, p], p], 0)
    col[~seen] = np.broadcast_to(med, col.shape)[~seen]
    path = td / "preview_points.bin"; buf = path.read_bytes(); n = meta["n"]
    if n != K_ * P_:
        return False
    f32 = buf[: n * meta.get("stride", 9) * 4]
    path.write_bytes(f32 + col.reshape(-1, 3).clip(0, 255).astype(np.uint8).tobytes())
    meta["coloured"] = True; (td / "preview_meta.json").write_text(json.dumps(meta))
    return True


def setting_points(td, max_points=200000):
    td = Path(td); meta = json.loads((td / "preview_meta.json").read_text())
    if meta.get("setting_n") and (td / "setting_points.bin").is_file():
        return meta["setting_n"]
    cap = S.Capture(td, zoom_views=False, scene=True)
    if cap.source_cam is None or cap.p2s is None:
        return 0
    bg = L.build(cap, td / "scene")
    if bg is None:
        return 0
    z = np.load(bg); M, C = z["means"], z["rgb"]
    if len(M) > max_points:
        keep = np.random.default_rng(0).choice(len(M), max_points, replace=False); M, C = M[keep], C[keep]
    (td / "setting_points.bin").write_bytes(M.astype("<f4").tobytes() + C.astype(np.uint8).tobytes())
    meta = json.loads((td / "preview_meta.json").read_text()); meta["setting_n"] = int(len(M))
    (td / "preview_meta.json").write_text(json.dumps(meta))
    return len(M)


def run(td, max_points=200000):
    return {"coloured": colour_mannequin(td), "setting_points": setting_points(td, max_points)}


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200000)))
