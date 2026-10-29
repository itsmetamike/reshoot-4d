"""The clip's setting as static Gaussians, so a shot keeps the real place around the person (the splat holds only the person).

Every other frame of the clip goes through Depth Anything 3 multiview (ComfyUI core) in one pass: depth for each frame plus the source camera's
camera for each, all consistent with each other, so a camera that walks round her still gives one static setting. That reconstruction is
placed in the capture by the reference view (the body mesh's depth there sets the scale, the solved real camera the pose), the person is
masked out of every view and the rest fused: one Gaussian per pixel in the capture's training space. The real frame itself (person
included) goes the same way for Shot Prep: rendered from the shot camera it is a real-pixel anchor wherever the shot sees what the source camera
saw. Runs inside ComfyUI.
"""
import json, math
from pathlib import Path

import cv2
import numpy as np
import torch

from . import shot as S


_M = None


def _moge():
    global _M
    if _M is None:
        import comfy.utils, folder_paths
        from comfy.ldm.moge.model import MoGeModel
        names = [n for n in folder_paths.get_filename_list("geometry_estimation") if "moge" in n.lower()]
        if not names:
            raise FileNotFoundError("The background layer needs MoGe-2 in models/geometry_estimation "
                                    "(moge_2_vitl_normal_fp16.safetensors, see docs/SETUP.md)")
        _M = MoGeModel(comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("geometry_estimation", names[0]), safe_load=True))
    return _M


def _sky(img, K):
    """Where MoGe-2 sees no surface (the sky): DA3 Base has no sky head and puts the sky a few metres off."""
    fov_x = math.degrees(2 * math.atan(img.shape[1] / 2 / K[0, 0]))
    t = torch.from_numpy(np.ascontiguousarray(img)).float().div(255).permute(2, 0, 1)[None]
    with torch.no_grad():
        d = _moge().infer(t, resolution_level=9, fov_x=fov_x, force_projection=True, apply_mask=True)["depth"][0]
    return ~torch.isfinite(d).cpu().numpy()


def _person(cap, f, w, h, grow=0.02):
    """Where the person is in the real frame f: the body mesh's regions, grown a little more for loose hair and clothes."""
    m = (S._regions(cap, cap.source_cam, f, w, h)[0] < 0.5).astype(np.uint8)
    r = max(3, int(grow * h)) | 1
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))) > 0


def _mesh_scale(cap, f, d, Kd):
    """Training units per depth unit at frame f: the body mesh's nearest surface per cell (projected with the solved real camera) against
    the depth map there (sampled with the depth model's own K, so a differing focal estimate cannot shift the samples)."""
    sc = cap.source_cam; K = cap.cams[sc][1]; c2w = cap.pose(sc, f)
    if not hasattr(cap, "_V"):
        cap._V = cap.geometry()
    q = (cap._V[min(f, len(cap._V) - 1)] - c2w[:3, 3]) @ c2w[:3, :3]; q = q[q[:, 2] > 1e-3]
    uv = (q / q[:, 2:3]) @ K.T; h, w = d.shape; cell = 6
    x, y = (uv[:, 0] // cell).astype(int), (uv[:, 1] // cell).astype(int); ok = (x >= 0) & (x < w // cell) & (y >= 0) & (y < h // cell)
    near = np.full((h // cell, w // cell), np.inf); np.minimum.at(near, (y[ok], x[ok]), q[ok, 2])
    ys, xs = np.nonzero(np.isfinite(near))
    ray = np.stack([(xs * cell + cell / 2 - K[0, 2]) / K[0, 0], (ys * cell + cell / 2 - K[1, 2]) / K[1, 1], np.ones(len(xs))], 1)
    pu = (ray @ np.asarray(Kd, float).T); pu = pu[:, :2] / pu[:, 2:3]   # the same ray in the depth model's pixels
    ok2 = (pu[:, 0] >= 0) & (pu[:, 0] < w) & (pu[:, 1] >= 0) & (pu[:, 1] < h); xs, ys, pu = xs[ok2], ys[ok2], pu[ok2]
    md = d[pu[:, 1].astype(int), pu[:, 0].astype(int)]; good = np.isfinite(md) & (md > 0)
    if good.sum() < 20:
        raise ValueError("background layer: the depth map and the body mesh do not overlap in the real frame")
    return float(np.median(near[ys[good], xs[good]] / md[good]))


DA3_STRIDE, FUSE_STEP = 2, 8   # frames into DA3 (anchors take the nearest), and between the views fused into the background
_D = None
_SOLVED = {}


def _da3():
    global _D
    if _D is None:
        import comfy.sd, folder_paths
        names = [n for n in folder_paths.get_filename_list("geometry_estimation") if "depth_anything_3" in n and ("base" in n or "small" in n)]
        if not names:
            raise FileNotFoundError("The background layer needs Depth Anything 3 Base (or Small) in models/geometry_estimation "
                                    "(depth_anything_3_base.safetensors, see docs/SETUP.md)")
        _D = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("geometry_estimation", sorted(names)[0]),
                                           model_options={"dtype": torch.float32})   # fp16 trips its camera decoder; it is small
    return _D


def _solve(cap, out):
    """DA3 multiview over the clip, placed in the capture's training space, cached as out/da3.npz: frame ids, depth (training units),
    camera-to-world per frame (training space), K per frame (full size), confidence."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); p = out / "da3.npz"
    if str(p) in _SOLVED:   # loaded once per process: it is a few hundred MB and every anchor asks for it
        return _SOLVED[str(p)]
    if p.is_file():
        _SOLVED.clear(); _SOLVED[str(p)] = dict(np.load(p)); return _SOLVED[str(p)]
    import comfy.model_management as mm
    from comfy.ldm.depth_anything_3 import preprocess as pre
    cap = S.Capture(cap.dir, zoom_views=False, scene=False)   # aligned in the body frame: p2s is measured against it
    sc = cap.source_cam; video = cap.cams[sc][2]; F = cap.frames; f0 = F // 2
    ids = sorted(set(range(0, F, DA3_STRIDE)) | {f0}); imgs = np.stack([S.video_frame(video, f) for f in ids]); n, H, W = imgs.shape[:3]
    model = _da3(); mm.load_model_gpu(model); net = model.model.diffusion_model
    x = pre.preprocess_image(torch.from_numpy(imgs).float().div(255).to(mm.get_torch_device()), process_res=504, method="upper_bound_resize")
    h2, w2 = x.shape[-2:]
    with torch.no_grad():
        o = net(x.to(dtype=net.dtype or torch.float32).unsqueeze(0), use_ray_pose=False, ref_view_strategy="middle")
    up = lambda t: torch.nn.functional.interpolate(t.float().reshape(n, 1, h2, w2), size=(H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    depth, conf = up(o["depth"]), up(o["depth_conf"])
    E = o["extrinsics"].float().cpu().numpy()[0]; Ks = o["intrinsics"].float().cpu().numpy()[0].copy()
    Ks[:, 0] *= W / w2; Ks[:, 1] *= H / h2
    f_ = np.sqrt(Ks[:, 0, 0] * Ks[:, 1, 1]); Ks[:, 0, 0] = Ks[:, 1, 1] = f_   # square pixels (its two focal estimates can differ)
    w2c = np.tile(np.eye(4), (n, 1, 1)); w2c[:, :3, :4] = E; c2w = np.linalg.inv(w2c)   # DA3 world
    # DA3 world -> training space: the reference view's DA3 camera onto its solved real camera, scaled by the body mesh's depth there
    r = ids.index(f0); s = _mesh_scale(cap, f0, depth[r], Ks[r])
    A = cap.pose(sc, f0) @ np.diag([s, s, s, 1.0]) @ w2c[r]
    res = {"ids": np.array(ids), "depth": (depth * s).astype(np.float32), "c2w": np.einsum("ij,njk->nik", A, c2w) @ np.diag([1 / s, 1 / s, 1 / s, 1.0]),
           "K": Ks, "conf": conf.astype(np.float16), "s": s}
    np.savez(p, **res)
    return res


def _cloud(img, z, K, c2w, keep):
    """Pixels (where keep) -> Gaussians in training space, one per pixel with a one-pixel footprint; near depth edges are dropped (they
    would hang between the near and the far surface)."""
    h, w = z.shape; lz = np.log(np.maximum(z, 1e-6)); k = np.ones((3, 3), np.uint8)
    keep = keep & ~(((cv2.dilate(lz, k) - cv2.erode(lz, k)) > 0.06) & (z < np.percentile(z, 80)))
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)[keep]
    return {"means": (X @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32), "rgb": img[keep].astype(np.uint8),
            "scale": (z[keep] / K[0, 0] * 0.75).astype(np.float32)}, keep


def _with_sky(z, img, K):
    """Depth with the sky pushed far behind everything else (several times the farthest surface), so it barely moves with the camera."""
    sky = _sky(img, K); far = float(np.percentile(z[~sky], 99)) * 4 if (~sky).any() else float(z.max())
    return np.where(sky, far, z).astype(np.float32)


def _mirror(i, lo, hi):
    """Index i folded into [lo, hi] by mirror repetition (a triangle wave), so tiled texture meets itself without seams."""
    n = max(hi - lo, 1); k = np.mod(i - lo, 2 * n)
    return lo + np.where(k <= n, k, 2 * n - k)


def _environment(M, C, centre, up, ring_w=1440, ring_h=360, cell=0.03, ground_r=6.0):
    """The setting beyond what the source camera saw, continued from what it did, so a camera turned anywhere sees the clip's world rather
    than a void for the video model to invent. Far field: the seen distant points (skyline, sky) mapped by direction onto a ring
    and mirror-tiled round the full 360 degrees. Ground: the floor plane under her, its seen texture mirror-tiled outwards. Only
    directions and floor cells nothing was seen in are added. Returns Gaussians like _cloud."""
    up = up / np.linalg.norm(up); e1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    rel = M - centre; h = rel @ up; x, z = rel @ e1, rel @ e2; dist = np.linalg.norm(rel, axis=1)
    out = {"means": [], "rgb": [], "scale": []}
    # far field ring: azimuth x elevation (-25..75 degrees: the city drops away below the roofline), from the distant points only (skyline and sky, not the car or the railing)
    sky = dist > 0.6 * float(np.percentile(dist, 99.5)); scene_r = float(np.percentile(dist[~sky], 95)) if (~sky).any() else 1.0
    far = dist > 0.4 * scene_r   # the skyline and the sky; the car and the railing sit nearer
    if far.sum() > 1000:
        R = float(np.median(dist[far & ~sky])) if (far & ~sky).any() else float(np.median(dist[far])); az = np.arctan2(z[far], x[far]); el = np.arcsin(np.clip(h[far] / dist[far], -1, 1))
        ci = ((az + np.pi) / (2 * np.pi) * ring_w).astype(int) % ring_w; ri = np.clip(((el + np.radians(25)) / np.radians(100) * ring_h).astype(int), 0, ring_h - 1)
        acc = np.zeros((ring_h, ring_w, 3)); cnt = np.zeros((ring_h, ring_w)); np.add.at(acc, (ri, ci), C[far]); np.add.at(cnt, (ri, ci), 1)
        cols = np.nonzero(cnt.sum(0) > 0)[0]
        if len(cols) > 8:
            # the seen arc (the longest run of covered columns, round the wrap): everything else mirrors it
            cov = cnt.sum(0) > 0; ext = np.concatenate([cov, cov]); best, start, run, s0 = 0, 0, 0, 0
            for i, v in enumerate(ext):
                run, s0 = (run + 1, s0) if v else (0, i + 1)
                if run > best and run <= ring_w:
                    best, start = run, s0
            lo, hi = start, start + best - 1
            img = np.where(cnt[..., None] > 0, acc / np.maximum(cnt[..., None], 1), np.nan)
            rowm = np.nanmedian(img, 1)   # each height's typical colour across what was seen: the sky's gradient, the skyline's tone
            for r in range(ring_h):
                if np.isnan(rowm[r, 0]):
                    near = [q for q in range(ring_h) if not np.isnan(rowm[q, 0])]
                    rowm[r] = rowm[min(near, key=lambda q: abs(q - r))] if near else 0
            for c in range(ring_w):   # gaps in a seen column take that height's typical colour (no vertical smears)
                miss = np.isnan(img[:, c, 0])
                if (~miss).any():
                    img[miss, c] = rowm[miss]
            seen = set(np.arange(lo, hi + 1) % ring_w)
            for c in range(ring_w):
                if c in seen:
                    continue
                src = int(_mirror(np.array([c if c >= lo else c + ring_w]), lo, hi)[0]) % ring_w
                if np.isnan(img[0, src, 0]):
                    continue
                a = (c + 0.5) / ring_w * 2 * np.pi - np.pi; e = (np.arange(ring_h) + 0.5) / ring_h * np.radians(100) - np.radians(25)
                d = np.cos(e)[:, None] * (np.cos(a) * e1 + np.sin(a) * e2) + np.sin(e)[:, None] * up
                out["means"].append(centre + R * d); out["rgb"].append(img[:, src]); out["scale"].append(np.full(ring_h, R * 2 * np.pi / ring_w * 0.9))
    # ground: the floor under her as a plane, seen cells mirror-tiled out to ground_r
    floor = (np.sqrt(x ** 2 + z ** 2) < ground_r) & (h < np.percentile(h[dist < 2.0], 10) + 0.15) if (dist < 2.0).any() else np.zeros(len(M), bool)
    if floor.sum() > 1000:
        P = M[floor]; ctr = P.mean(0); _, _, Vt = np.linalg.svd(P - ctr, full_matrices=False); nrm = Vt[2] * np.sign(Vt[2] @ up)
        g1 = np.cross(nrm, e2); g1 /= np.linalg.norm(g1); g2 = np.cross(nrm, g1)
        n = int(2 * ground_r / cell); u = ((P - ctr) @ g1 / cell + n / 2).astype(int); v = ((P - ctr) @ g2 / cell + n / 2).astype(int)
        ok = (u >= 0) & (u < n) & (v >= 0) & (v < n)
        acc = np.zeros((n, n, 3)); cnt = np.zeros((n, n)); np.add.at(acc, (v[ok], u[ok]), C[floor][ok]); np.add.at(cnt, (v[ok], u[ok]), 1)
        vs, us = np.nonzero(cnt > 0)
        if len(us) > 100:
            u0, u1, v0, v1 = np.percentile(us, 5), np.percentile(us, 95), np.percentile(vs, 5), np.percentile(vs, 95)
            tex = acc / np.maximum(cnt[..., None], 1)
            gv, gu = np.nonzero(cnt == 0)
            su = np.clip(_mirror(gu, int(u0), int(u1)), 0, n - 1); sv = np.clip(_mirror(gv, int(v0), int(v1)), 0, n - 1)
            keep = cnt[sv, su] > 0; gu, gv, su, sv = gu[keep], gv[keep], su[keep], sv[keep]
            out["means"].append(ctr + ((gu + 0.5 - n / 2) * cell)[:, None] * g1 + ((gv + 0.5 - n / 2) * cell)[:, None] * g2)
            out["rgb"].append(tex[sv, su]); out["scale"].append(np.full(len(gu), cell * 0.7))
    if not out["means"]:
        return {"means": np.zeros((0, 3)), "rgb": np.zeros((0, 3)), "scale": np.zeros(0)}
    return {k: np.concatenate(v) for k, v in out.items()}


def build(cap, out_dir):
    """Background layer of a capture, cached in out_dir/background.npz: every FUSE_STEP-th DA3 view with the person masked out, fused
    (one point per spot, the view nearest the middle of the clip wins). Returns its path, or None without a solved real camera."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); bg = out / "background.npz"
    if bg.is_file() or cap.source_cam is None:
        return bg if bg.is_file() else None
    d = _solve(cap, out); ids = list(d["ids"]); f0 = cap.frames // 2; video = cap.cams[cap.source_cam][2]
    parts = []
    for f in sorted([f for f in ids if f % FUSE_STEP == 0] + [f0], key=lambda f: abs(f - f0)):
        i = ids.index(f); img = S.video_frame(video, f); h, w = img.shape[:2]
        c, _ = _cloud(img, _with_sky(d["depth"][i], img, d["K"][i]), d["K"][i], d["c2w"][i], ~_person(cap, f, w, h))
        parts.append(c)
    # last and weakest: the reference view with her painted out (colour and depth), for ground no view saw behind her
    i = ids.index(f0); img = S.video_frame(video, f0); h, w = img.shape[:2]; m = _person(cap, f0, w, h).astype(np.uint8)
    z = _with_sky(d["depth"][i], img, d["K"][i])
    zf = z.copy(); x = np.arange(w)
    for y in range(h):   # depth across the gap, row by row from either side (floors and walls behind a person are near-linear)
        g = m[y] > 0
        if g.any() and (~g).sum() >= 2:
            zf[y, g] = np.interp(x[g], x[~g], z[y, ~g])
    c, _ = _cloud(cv2.inpaint(img, m, 9, cv2.INPAINT_TELEA), zf, d["K"][i], d["c2w"][i], m > 0)
    parts.append(c)
    M = np.concatenate([c["means"] for c in parts]); C = np.concatenate([c["rgb"] for c in parts]); Fp = np.concatenate([c["scale"] for c in parts])
    # one point per spot: binned in the reference camera (pixels and log depth), the earliest (nearest to the middle) view kept
    w2c = np.linalg.inv(d["c2w"][ids.index(f0)]); K = d["K"][ids.index(f0)]; Xr = M @ w2c[:3, :3].T + w2c[:3, 3]; zr = np.maximum(Xr[:, 2], 1e-6)
    key = np.stack([np.round(Xr[:, 0] / zr * K[0, 0] / 1.5), np.round(Xr[:, 1] / zr * K[1, 1] / 1.5), np.round(np.log(zr) / 0.03)], 1).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True); first.sort()
    M, C, Fp = M[first], C[first], Fp[first]
    E = _environment(M, C, np.array(cap.body).mean(0), cap.up)   # what the source camera never saw, continued from what it did
    # the environment apart: depth and the splat render take both, anchors only what the source camera really saw (a guessed backdrop pinned
    # as an anchor gets copied, blur and all)
    np.savez(out / "environment.npz", means=E["means"].astype(np.float32), rgb=E["rgb"].astype(np.uint8), scale=E["scale"].astype(np.float32))
    np.savez(bg, means=M.astype(np.float32), rgb=C.astype(np.uint8), scale=Fp.astype(np.float32), scale_m=float(d["s"]))
    (out / "background.json").write_text(json.dumps({"frame": f0, "views": len(parts), "gaussians": int(len(first)), "environment": int(len(E["means"])),
                                                     "units_per_da3": float(d["s"])}))
    return bg


def frame_cloud(cap, f, out_dir):
    """The real frame nearest f that DA3 saw (person included) as Gaussians in the same placement as the background, cached as
    out_dir/frame_NNN.npz, plus a person flag per Gaussian."""
    out = Path(out_dir); d = _solve(cap, out); ids = list(d["ids"]); f = min(ids, key=lambda g: abs(g - f)); p = out / f"frame_{f:03d}.npz"
    if p.is_file():
        return p
    i = ids.index(f); img = S.video_frame(cap.cams[cap.source_cam][2], f); h, w = img.shape[:2]
    # placed by the same (smoothed) real camera the splat uses, so seen from that camera her real pixels land on her splat
    c2w = cap.pose(cap.source_cam, f) if getattr(cap, "p2s", None) is not None else d["c2w"][i]
    c, keep = _cloud(img, _with_sky(d["depth"][i], img, d["K"][i]), d["K"][i], c2w, np.ones((h, w), bool))
    np.savez(p, **c, person=_person(cap, f, w, h, grow=0.0)[keep])
    return p


def person_to_scene(cap, out_dir):
    """Per capture frame, the rigid transform taking the person's own frame (her body track, where the source camera's motion leaked into
    her path) into the setting's (DA3's camera track): the real camera seen both ways, C_setting(f) @ inv(C_body(f)), the TRAM
    composition. Smoothed (her body solve jitters), cached as out_dir/p2s.npy."""
    out = Path(out_dir); p = out / "p2s.npy"
    if p.is_file():
        return p
    d = _solve(cap, out); ids = list(d["ids"]); cap = S.Capture(cap.dir, zoom_views=False, scene=False); sc = cap.source_cam; F = cap.frames
    raw = {f: d["c2w"][i] @ np.linalg.inv(cap.pose(sc, f)) for i, f in enumerate(ids)}
    M = []
    for f in range(F):   # frames DA3 skipped: between their neighbours
        a = max(g for g in ids if g <= f); b = min(g for g in ids if g >= f); w = 0 if a == b else (f - a) / (b - a)
        M.append(raw[a] * (1 - w) + raw[b] * w)
    M = np.array(M); h = 10; out_m = M.copy()
    for f in range(F):
        w = M[max(0, f - h):f + h + 1].mean(0); U, _, Vt = np.linalg.svd(w[:3, :3])
        out_m[f, :3, :3] = U @ Vt; out_m[f, :3, 3] = w[:3, 3]; out_m[f, 3] = [0, 0, 0, 1]
    np.save(p, out_m)
    return p


def ensure(train_dir):
    """The setting solve for a capture (DA3 cameras and depth, person-to-setting transform), so every later step works in the
    setting's frame. Returns False (and the capture stays in its body frame) when there is no real camera or no DA3 model."""
    try:
        cap = S.Capture(train_dir, zoom_views=False, scene=False)
        if cap.source_cam is None:
            return False
        person_to_scene(cap, Path(train_dir) / "scene")
        return True
    except Exception as e:  # noqa: BLE001 - the capture still works in its own frame
        print(f"[sam3dbx] setting solve: {e}")
        return False



def _equirect(M, C, o, e1, e2, up, W, H):
    """Points seen from o into an equirect image (azimuth 0 = e1 at the centre column, up = top): colour, nearest distance, coverage."""
    rel = M - o; dist = np.linalg.norm(rel, axis=1); az = np.arctan2(rel @ e2, rel @ e1); el = np.arcsin(np.clip(rel @ up / np.maximum(dist, 1e-9), -1, 1))
    u = ((az / (2 * np.pi) + 0.5) * W).astype(int) % W; v = np.clip(((0.5 - el / np.pi) * H).astype(int), 0, H - 1)
    order = np.argsort(-dist); img = np.zeros((H, W, 3), np.uint8); dep = np.zeros((H, W), np.float32); seen = np.zeros((H, W), np.uint8)
    img[v[order], u[order]] = C[order]; dep[v[order], u[order]] = dist[order]; seen[v[order], u[order]] = 255
    k = np.ones((5, 5), np.uint8)
    return img, dep, cv2.morphologyEx(seen, cv2.MORPH_CLOSE, k) > 0


def register_panorama(cap, out_dir, pano):
    """A generated 360 panorama of the empty set (uint8 HxWx3, 2:1) registered to the clip: turned (and if need be mirrored) so it
    matches, in colour, what the source camera saw from where it stood, and saved with its frame for env_map.EnvMap (skybox lookup with
    ground projection). Returns the world_env.json path."""
    from PIL import Image
    out = Path(out_dir); bg = np.load(out / "background.npz"); M, C = bg["means"], bg["rgb"]
    H, W = pano.shape[:2]; up = cap.up / np.linalg.norm(cap.up); c = cap.pose(cap.source_cam, cap.frames // 2)
    o = c[:3, 3]; e1 = c[:3, 2] - up * (c[:3, 2] @ up); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    seen_img, _, seen = _equirect(M, C, o, e1, e2, up, W, H)
    sw, sh_ = 512, 256; lab = lambda a: cv2.cvtColor(cv2.GaussianBlur(cv2.resize(a, (sw, sh_), interpolation=cv2.INTER_AREA), (0, 0), 2), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = lab(seen_img); m = cv2.resize(seen.astype(np.uint8), (sw, sh_), interpolation=cv2.INTER_NEAREST) > 0; pl = lab(pano); best = (np.inf, 0, False)
    for flip in (False, True):   # the panorama may run either way round
        q = pl[:, ::-1] if flip else pl
        for k in range(sw):
            err = float(np.abs(ref[m] - np.roll(q, k, 1)[m]).mean())
            if err < best[0]:
                best = (err, k, flip)
    _, k, flip = best
    pano = np.roll(np.ascontiguousarray(pano[:, ::-1]) if flip else pano, k * W // sw, 1)
    Image.fromarray(pano).save(out / "panorama_aligned.png")
    if not hasattr(cap, "_V"):
        cap._V = cap.geometry()
    floor = float(np.percentile(cap._V.reshape(-1, 3) @ up, 0.5))   # her soles over the clip: the floor she dances on
    env = {"panorama": str(out / "panorama_aligned.png"), "origin": o.tolist(), "e1": e1.tolist(), "e2": e2.tolist(), "up": up.tolist(),
           "floor": floor, "floor_radius": 6.0, "yaw_px": int(k * W // sw), "mirrored": bool(flip)}
    (out / "world_env.json").write_text(json.dumps(env))
    return out / "world_env.json"


def near_cloud(cap, f, out_dir, radius=4.0):
    """The real frame nearest f lifted (DA3), keeping only what is near her and not her: the car, the barrier, the floor round her.
    It moves with the footage frame by frame, so a vehicle that drives keeps driving. Cached as out_dir/near_NNN.npz."""
    out = Path(out_dir); z = np.load(frame_cloud(cap, f, out)); g = int(Path(str(frame_cloud(cap, f, out))).stem.split("_")[1])
    p = out / f"near_{g:03d}.npz"
    if not p.is_file():
        c = np.array(cap.body)[min(g, len(cap.body) - 1)]
        keep = (np.linalg.norm(z["means"] - c, axis=1) < radius) & ~z["person"]
        # and nothing close to her body: the 2D person mask misses loose limbs and hair at its edge, which would float in the shot
        from scipy.spatial import cKDTree
        if not hasattr(cap, "_V"):
            cap._V = cap.geometry()
        V = cap._V[min(g, len(cap._V) - 1)]; idx = np.nonzero(keep)[0]
        dd, _ = cKDTree(V).query(z["means"][idx], distance_upper_bound=0.15); keep[idx[np.isfinite(dd)]] = False
        # the floor itself comes from the 360 world's ground projection (one smooth surface): only what stands on it stays here, or the
        # two floors meet in a step the video model reads as something near the lens
        up = cap.up / np.linalg.norm(cap.up); floor = float(np.percentile(cap._V.reshape(-1, 3) @ up, 0.5))
        keep &= (z["means"] @ up) > floor + 0.12
        np.savez(p, means=z["means"][keep], rgb=z["rgb"][keep], scale=z["scale"][keep])
    return p
