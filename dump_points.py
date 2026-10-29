"""Export a light preview of a FreeTimeGS capture for the 4D Director viewport (trainer venv, CPU only).

Keeps the N most opaque Gaussians with their canonical position, velocity, time centre, duration and DC colour; the browser
evaluates position = mean + v (t - time) and opacity * exp(-0.5 ((t - time) / dur)^2) in a shader, as FreeTimeGS does.
Writes preview_points.bin (float32 xyz|v|time|dur|opa, uint8 rgb) and preview_meta.json next to the checkpoint.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rig import FTGS, fit_rig  # noqa: E402
from viewer_4d import Config4D, Splats4D  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--train_dir", required=True); ap.add_argument("--n", type=int, default=150000)
a = ap.parse_args()
train = Path(a.train_dir)
ckpt = sorted(train.glob("ckpts/ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
sp = Splats4D(str(ckpt), Config4D(ckpt=str(ckpt), device="cpu", precompute_visibility=False, use_spatial_filter=True))
keep = torch.nonzero(sp.base_mask).squeeze(1)
score = sp.base_opacity[keep] * torch.exp(sp.scales[keep]).mean(1).clamp(max=0.05)   # favour visible, non-tiny splats
idx = keep[torch.argsort(score, descending=True)[: a.n]]
C0 = 0.28209479177387814
rgb = (0.5 + C0 * sp.sh0[idx, 0]).clamp(0, 1)
f32 = torch.cat([sp.means[idx], sp.velocities[idx], sp.times[idx], torch.exp(sp.durations[idx]), sp.base_opacity[idx, None]], 1)
f32 = f32.float().numpy().astype("<f4"); u8 = (rgb.numpy() * 255).astype(np.uint8)
(train / "preview_points.bin").write_bytes(f32.tobytes() + u8.tobytes())
rig = fit_rig(train)
man = json.loads((train / "dataset" / "manifest.json").read_text())
num, den = (str(man["fps"]).split("/") + ["1"])[:2]
meta = {"n": int(len(idx)), "stride": 9, "frames": man["frames"], "fps": float(num) / float(den),
        "seconds": man["frames"] / (float(num) / float(den)), "use_velocity": bool(sp.use_velocity),
        **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rig.items() if k != "cam_pos"},
        "cam_pos": rig["cam_pos"].tolist(), "ckpt": ckpt.name}
# subject tracks for the Director's "follow head / body" aim: per-frame points from the body mesh the capture was seeded from
# (preprocessing/viewer_geometry.npz in the source result, world units), moved into the splats' normalised space
try:
    src = Path(man["source"]); geo = np.load(src / "preprocessing" / "viewer_geometry.npz")["vertices"].astype(np.float64)
    norm = train / "normalization.json"
    if norm.is_file():
        W2T = np.array(json.loads(norm.read_text())["world_to_training"])
        geo = geo @ W2T[:3, :3].T + W2T[:3, 3]
    up = np.array(rig["up"]); h = geo @ up
    top = h >= np.quantile(h, 0.96, axis=1, keepdims=True)
    head = np.stack([geo[f][top[f]].mean(0) for f in range(len(geo))])
    body = geo.mean(1)
    meta["track"] = {"head": head.round(5).tolist(), "body": body.round(5).tolist()}
except Exception as e:  # noqa: BLE001 - tracking is optional; the Director falls back to fixed aim
    meta["track_error"] = str(e)
(train / "preview_meta.json").write_text(json.dumps(meta))
print(json.dumps({"n": meta["n"], "bytes": f32.nbytes + u8.nbytes}))
