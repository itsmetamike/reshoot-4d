"""FAST MODE: a training-free capture the 4D Director, Shot Prep and Photoreal Shoot can use, straight from an adapted
upstream result. World space = training space (identity normalisation). The rig comes from the generated views' cameras;
the Director's preview is the moving SMPL-X body (the mannequin), written in the same point format as a splat preview:
surface points sampled per time slice, each moving with its velocity and fading in and out around its slice time.
Usage: fast_rig.py adapt_dir out_dir"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rig import rig_from_cams  # noqa: E402

adapt, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
(out / "dataset").mkdir(parents=True, exist_ok=True)
(out / "normalization.json").write_text(json.dumps({"world_to_training": np.eye(4).tolist(), "training_to_world": np.eye(4).tolist()}))
meta_src = json.loads((adapt / "metadata.json").read_text())
num, den = (str(meta_src["output"]["fps"]).split("/") + ["1"])[:2]; fps = float(num) / float(den)
frames = int(meta_src["output"]["frames_per_video"])
(out / "dataset" / "manifest.json").write_text(json.dumps({"source": str(adapt), "fps": meta_src["output"]["fps"], "frames": frames, "fast_mode": True}))

# rig from the generated cameras, front camera first (the Director's "front")
cj = json.loads((adapt / "cameras.json").read_text()); fid = cj.get("front_camera_ids", [0])[0]
cams = sorted(cj["cameras"], key=lambda c: c["camera_id"] != fid)
rig = rig_from_cams(np.stack([np.array(c["camera_to_world"], float) for c in cams]), float(np.mean([c["K"][1][1] for c in cams])),
                    (cams[0]["image_width"], cams[0]["image_height"]))

geo = np.load(adapt / "preprocessing" / "viewer_geometry.npz"); V = geo["vertices"].astype(np.float64); F = geo["faces"]
up = rig["up"]
h = V @ up; top = h >= np.quantile(h, 0.96, axis=1, keepdims=True)
head = np.stack([V[f][top[f]].mean(0) for f in range(len(V))]); body = V.mean(1)

# mannequin preview: P surface points x K time slices (FreeTimeGS-style: p = mean + v (t - time), opacity * exp(-0.5 ((t - time)/dur)^2))
P, K = 20000, 31   # dense enough to read as a person once coloured from the clip (preview_extras.py)
tri = V[0][F]; area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
rng = np.random.default_rng(0); fi = rng.choice(len(F), P, p=area / area.sum())
r1, r2 = np.sqrt(rng.random(P)), rng.random(P); bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
surf = lambda f: (V[f][F[fi]] * bary[..., None]).sum(1)             # P x 3 at capture frame f
kf = np.linspace(0, len(V) - 1, K).round().astype(int); kt = kf / (len(V) - 1)
dt = 1.0 / (K - 1)
rows, cols = [], []
shade = 0.55 + 0.4 * ((surf(0) @ up) - (surf(0) @ up).min()) / max(np.ptp(surf(0) @ up), 1e-6)   # light grey, brighter higher up
for k, f in enumerate(kf):
    a, b = kf[max(k - 1, 0)], kf[min(k + 1, K - 1)]
    vel = (surf(b) - surf(a)) / max(kt[min(k + 1, K - 1)] - kt[max(k - 1, 0)], 1e-6)
    rows.append(np.concatenate([surf(f), vel, np.full((P, 1), kt[k]), np.full((P, 1), dt * 0.45), np.ones((P, 1))], 1))
    cols.append(np.repeat(shade[:, None], 3, 1))
f32 = np.concatenate(rows).astype("<f4"); u8 = (np.concatenate(cols) * 255).astype(np.uint8)
(out / "preview_points.bin").write_bytes(f32.tobytes() + u8.tobytes())
meta = {"n": int(len(f32)), "stride": 9, "frames": frames, "fps": fps, "seconds": frames / fps, "use_velocity": True,
        **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rig.items() if k != "cam_pos"}, "cam_pos": rig["cam_pos"].tolist(),
        "ckpt": "fast mode (SMPL-X mannequin, no splat)", "fast_mode": True, "mannequin": [P, K],
        "track": {"head": head.round(5).tolist(), "body": body.round(5).tolist()}, "height": float(np.ptp(h[0]))}
(out / "preview_meta.json").write_text(json.dumps(meta))
print(json.dumps({"out": str(out), "frames": frames, "points": meta["n"], "dist0": round(rig["dist0"], 3), "height": round(meta["height"], 3)}))
