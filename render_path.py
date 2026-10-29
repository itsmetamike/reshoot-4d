"""Render a FreeTimeGS 4D checkpoint along a camera path (trainer venv: gsplat + Splats4D from viewer_4d).

Two path formats:
  rows   - orbit rows (t_src 0..1, yaw_deg, pitch_deg, dist_scale, roll_deg) from the Beat Camera; the orbit is fitted to the
           training rig, so yaw 0 / pitch 0 / dist 1 reproduces the front training view.
  frames - free cameras from the 4D Director: (t_src 0..1, eye xyz, target xyz, roll_deg, fov_y_deg) in scene coordinates.
Writes straight RGBA PNG frames (alpha = splat coverage).
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rig import fit_rig, look_at  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402
from viewer_4d import Config4D, Splats4D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train_dir", required=True)        # folder with ckpts/ and dataset/sparse
ap.add_argument("--path", required=True)             # camera path json (rows or frames)
ap.add_argument("--out", required=True)              # output folder for frames
ap.add_argument("--width", type=int, default=704)
ap.add_argument("--height", type=int, default=1280)
ap.add_argument("--fov_scale", type=float, default=1.0)
ap.add_argument("--look_drop", type=float, default=0.0)  # rows only: lower the look-at point by this x training camera distance
ap.add_argument("--needle_ratio", type=float, default=0.0)  # >0: drop Gaussians whose longest/shortest axis exceeds this
ap.add_argument("--depth_mp4", default="")  # also write a depth video (near = white, normalised over the clip) for depth control
ap.add_argument("--depth_per_frame", action="store_true")  # normalise each depth frame on its own (keeps relief through push-ins)
ap.add_argument("--env", default="")    # scene dir with world_env.json: the 360 world (env_map) behind everything, no fused set
ap.add_argument("--near", default="")   # json {capture frame: npz}: the real near content per frame (car, barrier), drawn with her
ap.add_argument("--scene", default="")  # background layer (scene_layer.py npz): static Gaussians rendered with the person; alpha stays hers
a = ap.parse_args()

train = Path(a.train_dir)
ckpt = sorted(train.glob("ckpts/ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
cfg = Config4D(ckpt=str(ckpt), precompute_visibility=False, use_spatial_filter=False)
sp = Splats4D(str(ckpt), cfg)
T = cfg.total_frames
if a.needle_ratio > 0:   # needles = long thin splats the trainer leaves where feet meet the (masked-out) floor
    sc = torch.exp(sp.scales); keep = sc.max(1).values / sc.min(1).values.clamp(min=1e-8) <= a.needle_ratio
    sp.base_mask = sp.base_mask & keep
    print(f"needle filter: dropped {(~keep).sum().item():,} of {len(keep):,}")

rig = fit_rig(train)
P2S = np.load(train / "scene" / "p2s.npy") if (train / "scene" / "p2s.npy").is_file() else None   # her frame -> the setting's, per frame


def quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2; x = math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2; z = math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x, y, z = math.copysign(x, R[2, 1] - R[1, 2]), math.copysign(y, R[0, 2] - R[2, 0]), math.copysign(z, R[1, 0] - R[0, 1])
    return torch.tensor([w, x, y, z], dtype=torch.float32, device="cuda")


def qmul(a, b):
    w1, x1, y1, z1 = a.unbind(-1); w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
centre, up, front, side, dist0 = rig["centre"], rig["up"], rig["front"], rig["side"], rig["dist0"]


def K_for(fov_y):
    fy = a.height / 2 / math.tan(math.radians(fov_y) / 2)
    return torch.tensor([[fy, 0, a.width / 2], [0, fy, a.height / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")[None]


K_rig = K_for(rig["fov_y"] / a.fov_scale) if a.fov_scale != 1 else K_for(rig["fov_y"])
path = json.loads(Path(a.path).read_text())
if "frames" in path:
    cams = [(f[0], np.array(f[1:4]), np.array(f[4:7]), f[7], f[8]) for f in path["frames"]]
else:
    cams = []
    for t, yaw, pitch, dist, roll in path["rows"]:
        yr, pr = math.radians(yaw), math.radians(rig["base_pitch"] + pitch)
        target = centre - up * a.look_drop * dist0
        eye = target + dist0 * dist * (math.cos(pr) * (math.cos(yr) * front + math.sin(yr) * side) + math.sin(pr) * up)
        cams.append((t, eye, target, roll, None))

bg = None
if a.scene:   # one Gaussian per plate pixel: isotropic, opaque, SH DC colour only (plus the environment beside it, when built)
    env = Path(a.scene).with_name("environment.npz")   # the mirrored environment beside it, when built
    zs = [np.load(a.scene)] + ([np.load(env)] if env.is_file() else [])
    z = {k: np.concatenate([q[k] for q in zs]) for k in ("means", "rgb", "scale")}; n = len(z["means"]); dev = "cuda"
    sh = torch.zeros(n, 16, 3, device=dev); sh[:, 0] = (torch.tensor(z["rgb"], device=dev).float() / 255 - 0.5) / 0.28209479
    bg = (torch.tensor(z["means"], device=dev), torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(n, 4).contiguous(),
          torch.tensor(z["scale"], device=dev)[:, None].expand(n, 3).contiguous(), torch.ones(n, device=dev), sh)

ENV = None; NEAR = {}
if a.env:
    from env_map import EnvMap
    ENV = EnvMap(a.env)
if a.near:
    NEAR = {int(k): v for k, v in json.loads(Path(a.near).read_text()).items()}


def near_g(f):
    """The near real content at capture frame f, as Gaussians (cached on the GPU between frames that share it)."""
    if not NEAR:
        return None
    g = min(NEAR, key=lambda k: abs(k - f))
    if near_g.key != g:
        z = np.load(NEAR[g]); n = len(z["means"]); dev = "cuda"
        sh = torch.zeros(n, 16, 3, device=dev); sh[:, 0] = (torch.tensor(z["rgb"], device=dev).float() / 255 - 0.5) / 0.28209479
        near_g.val = (torch.tensor(z["means"], device=dev), torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(n, 4).contiguous(),
                      torch.tensor(z["scale"], device=dev)[:, None].expand(n, 3).contiguous() * 1.6, torch.ones(n, device=dev), sh)
        near_g.key = g
    return near_g.val
near_g.key = None


out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
depths = []
with torch.no_grad():
    for i, (t, eye, target, roll, fov) in enumerate(cams):
        t = float(min(max(t, 0.0), 1.0))
        means, quats, scales, opac, colors, _ = sp.get_gaussians_at_time(t, int(round(t * (T - 1))), apply_visibility_mask=False)
        m = sp.base_mask
        if P2S is not None:   # place her in the setting at this moment (the source camera's real path, scene_layer.person_to_scene)
            M = P2S[min(int(round(t * (len(P2S) - 1))), len(P2S) - 1)]; R = torch.tensor(M[:3, :3], dtype=torch.float32, device="cuda")
            means = means @ R.T + torch.tensor(M[:3, 3], dtype=torch.float32, device="cuda")
            quats = qmul(quat(M[:3, :3]).expand_as(quats), quats / quats.norm(dim=-1, keepdim=True))
        vm = torch.linalg.inv(torch.tensor(look_at(eye, target, up, roll), dtype=torch.float32, device="cuda"))[None]
        KK = K_rig if fov is None else K_for(fov)
        G = [means[m], quats[m], scales[m], opac[m], colors[m]]
        nb = near_g(int(round(t * (T - 1)))) if NEAR else None
        if bg is not None or nb is not None:   # her coverage alone first (the alpha matte), then her with the setting (colour and depth)
            _, alpha, _ = rasterization(*G, vm, KK, a.width, a.height, sh_degree=3, packed=True, near_plane=0.01, far_plane=1e10)
            for extra in ((bg,) if bg is not None else ()) + ((nb,) if nb is not None else ()):
                G = [torch.cat([g, b.to(g.dtype)]) for g, b in zip(G, extra)]
        rgb, a2, _ = rasterization(*G, vm, KK, a.width, a.height, sh_degree=3, packed=True, near_plane=0.01, far_plane=1e10,
                                   render_mode="RGB+ED" if a.depth_mp4 else "RGB")
        if bg is None and nb is None:
            alpha = a2
        if a.depth_mp4:   # expected depth; keep only where the splat covers the pixel
            d = rgb[0, ..., 3]; depths.append(torch.where(a2[0, ..., 0] > 0.5, d, torch.zeros_like(d)).cpu()); rgb = rgb[..., :3]
        rgb = rgb[..., :3]
        if ENV is not None:   # the 360 world behind her and the near content (gsplat colours are premultiplied by coverage)
            c2w = look_at(eye, target, up, roll); fyy = float(KK[0, 1, 1])
            env_rgb, env_d = ENV.render(c2w[:3, 3], c2w[:3, :3], fyy, a.width, a.height, with_depth=True)
            rgb = rgb + env_rgb[None] * (1 - a2)
            if a.depth_mp4:   # the floor's depth where nothing nearer covers it
                depths[-1] = torch.where(depths[-1] > 0, depths[-1], env_d.cpu())
        rgba = torch.cat([rgb[0].clamp(0, 1), alpha[0].clamp(0, 1)], -1)
        Image.fromarray((rgba.cpu().numpy() * 255).astype(np.uint8), "RGBA").save(out / f"{i:05d}.png")
        if ENV is not None:   # coverage by her and the near footage together (Shot Prep weights its anchors by layer)
            Image.fromarray((a2[0, ..., 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out / f"{i:05d}_fg.png")
if a.depth_mp4 and depths:
    import subprocess
    def norm(D):
        valid = D[D > 0]
        valid = valid[torch.randperm(valid.numel())[:4_000_000]]   # torch.quantile caps its input size; a sample is plenty
        lo, hi = (torch.quantile(valid, 0.01).item(), torch.quantile(valid, 0.99).item()) if valid.numel() else (0.0, 1.0)
        return torch.where(D > 0, 1.0 - ((D - lo) / max(hi - lo, 1e-6)).clamp(0, 1) * 0.85, torch.zeros_like(D))   # near = white, empty = black

    def norm_inv(D):   # with the setting the depth runs from her to the skyline: disparity keeps her relief and the near set
        v = torch.where(D > 0, 1.0 / D.clamp(min=1e-6), torch.zeros_like(D)); s_ = v[v > 0]
        s_ = s_[torch.randperm(s_.numel())[:4_000_000]]
        lo, hi = (torch.quantile(s_, 0.01).item(), torch.quantile(s_, 0.995).item()) if s_.numel() else (0.0, 1.0)
        return torch.where(D > 0, 0.08 + ((v - lo) / max(hi - lo, 1e-9)).clamp(0, 1) * 0.92, torch.zeros_like(D))
    if bg is not None or NEAR:
        norm = norm_inv

    D = torch.stack(depths)
    vis = torch.stack([norm(d) for d in D]) if a.depth_per_frame else norm(D)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{a.width}x{a.height}", "-r",
                          str(path.get("fps", 24)), "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", a.depth_mp4], stdin=subprocess.PIPE)
    p.stdin.write((vis.numpy() * 255).astype(np.uint8).tobytes()); p.stdin.close(); p.wait()
print(json.dumps({"frames": len(cams), "out": str(out), "centre": centre.tolist(), "radius": rig["radius"], "base_pitch": rig["base_pitch"]}))
