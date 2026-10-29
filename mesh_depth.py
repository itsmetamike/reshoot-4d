"""Depth video of the capture's SMPL-X body mesh ("mannequin") along a camera path: the no-splat control for A/B tests and fast
previews. The mesh surface is sampled into tiny opaque Gaussians and rasterised with gsplat, so the depth matches render_path.py.
Usage (trainer venv): mesh_depth.py --train_dir T --path frames.json --out depth.mp4 [--width 720 --height 1280]
"""
import argparse, json, math, subprocess, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rig import fit_rig, look_at  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train_dir", required=True); ap.add_argument("--path", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--width", type=int, default=720); ap.add_argument("--height", type=int, default=1280)
ap.add_argument("--points", type=int, default=400_000)
ap.add_argument("--per_frame", action="store_true")   # normalise each frame on its own (keeps facial relief through push-ins)
a = ap.parse_args()
train = Path(a.train_dir)
src = Path(json.loads((train / "dataset" / "manifest.json").read_text())["source"])
geo = np.load(src / "preprocessing" / "viewer_geometry.npz")
V = geo["vertices"].astype(np.float64); F = geo["faces"]
W2T = np.array(json.loads((train / "normalization.json").read_text())["world_to_training"])
V = V @ W2T[:3, :3].T + W2T[:3, 3]
pm = train / "preview_meta.json"
up = np.array(json.loads(pm.read_text())["up"]) if pm.is_file() else fit_rig(train)["up"]

# fixed area-weighted barycentric samples (from frame 0) so the surface sampling is stable over time
tri = V[0][F]; area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
rng = np.random.default_rng(0); fi = rng.choice(len(F), a.points, p=area / area.sum())
r1, r2 = np.sqrt(rng.random(a.points)), rng.random(a.points); bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
scale = math.sqrt(area.sum() / a.points) * 1.6   # overlap enough to stay hole-free at close-up range
dev = "cuda"
Vt = torch.tensor(V, dtype=torch.float32, device=dev); Fi = torch.tensor(F[fi], device=dev); B = torch.tensor(bary, dtype=torch.float32, device=dev)
quats = torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(a.points, 4).contiguous()
scales = torch.full((a.points, 3), scale, device=dev); opac = torch.ones(a.points, device=dev); cols = torch.full((a.points, 3), 0.5, device=dev)

frames = json.loads(Path(a.path).read_text())
fps = frames.get("fps", 24); frames = frames["frames"]
depths = []
with torch.no_grad():
    for f in frames:
        s = min(max(f[0], 0.0), 1.0) * (len(V) - 1); i0 = int(math.floor(s)); i1 = min(i0 + 1, len(V) - 1); w = s - i0
        Vs = Vt[i0] * (1 - w) + Vt[i1] * w
        means = (Vs[Fi] * B[..., None]).sum(1)
        vm = torch.linalg.inv(torch.tensor(look_at(np.array(f[1:4]), np.array(f[4:7]), up, f[7]), dtype=torch.float32, device=dev))[None]
        fy = a.height / 2 / math.tan(math.radians(f[8]) / 2)
        K = torch.tensor([[fy, 0, a.width / 2], [0, fy, a.height / 2], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
        out, alpha, _ = rasterization(means, quats, scales, opac, cols, vm, K, a.width, a.height, packed=True, near_plane=0.01,
                                      far_plane=1e10, render_mode="RGB+ED")
        d = out[0, ..., 3]; depths.append(torch.where(alpha[0, ..., 0] > 0.5, d, torch.zeros_like(d)).cpu())
D = torch.stack(depths)


def norm(D):
    valid = D[D > 0]; valid = valid[torch.randperm(valid.numel())[:4_000_000]]
    if not valid.numel():
        return torch.zeros_like(D)
    lo, hi = torch.quantile(valid, 0.01).item(), torch.quantile(valid, 0.99).item()
    return torch.where(D > 0, 1.0 - ((D - lo) / max(hi - lo, 1e-6)).clamp(0, 1) * 0.85, torch.zeros_like(D))   # same mapping as render_path


vis = torch.stack([norm(d) for d in D]) if a.per_frame else norm(D)
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{a.width}x{a.height}", "-r", str(fps), "-i", "-",
                      "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
p.stdin.write((vis.numpy() * 255).astype(np.uint8).tobytes()); p.stdin.close(); p.wait()
print(json.dumps({"frames": len(frames), "out": a.out, "points": a.points, "scale": scale}))
