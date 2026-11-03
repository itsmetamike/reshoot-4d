"""Camera FX worker: the lens and optical effects on a finished clip, run on the small GPU by the Camera FX node (camera_fx.py) with
ComfyUI's own Python, so ComfyUI's built-in MoGe-2 gives metric depth without loading onto the big card.

  python camfx_run.py <comfyui dir> <job dir>     job dir holds frames.npy (N,H,W,3 uint8) and params.json; writes out.npy, depth.npy, info.json

Order, as in a camera: depth of field on the full frame (thin lens, metric depth), then the lens (zoom = crop about the centre plus shift
and roll), then vignette and grain on the final frame.
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

comfy_dir, job = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(comfy_dir)); sys.argv = sys.argv[:1]
P = json.loads((job / "params.json").read_text())
frames = np.load(job / "frames.npy", mmap_mode="r"); N, H, W, _ = frames.shape
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dh = 896 if H >= W else round(896 * H / W); dw = round(dh * W / H)   # depth and matte at about half size: fits the 8 GB card
import folder_paths


def curve(a, b, ease="smooth"):
    u = np.linspace(0.0, 1.0, N)
    if ease == "smooth":
        u = u * u * (3 - 2 * u)
    return a + (b - a) * u


# keyframes: [{"t": 0..1 (or "f": frame), "zoom": 1.5, "fstop": 0.7, "focus": "subject" | metres, "roll": 10, "ease": "smooth"|"linear"|"hold"}, ...]
# Each setting interpolates between the keys that set it; a setting no key sets keeps its plain value (or its start/end ramp).
KEYS = sorted(([dict(k, t=k["f"] / max(N - 1, 1)) if "f" in k and "t" not in k else k for k in P.get("keys") or []]), key=lambda k: float(k["t"]))
U = np.clip(np.array(P["u"], float), 0, 1) if len(P.get("u") or []) == N else np.linspace(0.0, 1.0, N)   # camera time per frame (whole-shot easing)


def _ease(u, kind):
    return u * u * (3 - 2 * u) if kind == "smooth" else (np.floor(u) if kind == "hold" else u)


def keyed(name, default, conv=float):
    """Per-frame values of one setting from the keys that set it: `default` (a scalar or per-frame array) when none do."""
    ks = [k for k in KEYS if name in k]
    if not ks:
        return np.broadcast_to(np.asarray(default, float), (N,)).copy() if conv is float else [default] * N
    ts = [float(k["t"]) for k in ks]; vs = [conv(k[name]) if conv is float else k[name] for k in ks]
    out = []
    for u in U:
        if u <= ts[0] or len(ks) == 1:
            out.append((vs[0], vs[0], 0.0))
        elif u >= ts[-1]:
            out.append((vs[-1], vs[-1], 0.0))
        else:
            j = max(i for i in range(len(ts) - 1) if ts[i] <= u)
            e = _ease((u - ts[j]) / max(ts[j + 1] - ts[j], 1e-9), ks[j].get("ease", "smooth"))
            out.append((vs[j], vs[j + 1], float(e)))
    return np.array([a + (b - a) * e for a, b, e in out]) if conv is float else out


def curve(a, b, ease="smooth"):
    return a + (b - a) * _ease(U, ease)


zoom = keyed("zoom", np.array(P["zoom"], float) if isinstance(P.get("zoom"), list) else curve(P.get("zoom_start", 1.0), P.get("zoom_end", 1.0)))
roll = keyed("roll", curve(P.get("roll_start", 0.0), P.get("roll_end", 0.0)))
shift_x = keyed("shift_x", curve(P.get("shift_x_start", 0.0), P.get("shift_x_end", 0.0)))
shift_y = keyed("shift_y", curve(P.get("shift_y_start", 0.0), P.get("shift_y_end", 0.0)))
fstop = keyed("fstop", float(P.get("fstop", 2.8)))
vignette = keyed("vignette", float(P.get("vignette", 0.0)))
grain = keyed("grain", float(P.get("grain", 0.0)))
distortion = keyed("distortion", float(P.get("distortion", 0.0)))   # -0.3 pincushion .. +0.3 barrel
chroma = keyed("chromatic_aberration", float(P.get("chromatic_aberration", 0.0)))
bloom = keyed("bloom", float(P.get("bloom", 0.0)))
halation = keyed("halation", float(P.get("halation", 0.0)))
ANAM = max(1.0, min(2.0, float(P.get("anamorphic", 1.0))))   # oval bokeh: vertical blur this much taller than wide
focus_keys = keyed("focus", "subject" if not P.get("focus_m", 0) > 0 else float(P["focus_m"]), conv=None)   # (from, to, ease) per frame
needs_subject = any(v == "subject" for a, b, _ in (x if isinstance(x, tuple) else (x, x, 0) for x in focus_keys) for v in (a, b))
lens0 = float(P.get("lens_mm", 50.0)); SENSOR_H = 36.0   # the Director's lens convention: mm on a 36 mm tall frame


def print_step(i, n):
    print(f"STEP {i} {n}", flush=True)


# ---------- depth (MoGe-2, metric) ----------
depth = None
if P.get("dof") or P.get("passes"):
    import comfy.utils
    import folder_paths
    from comfy.ldm.moge.model import MoGeModel
    sd = comfy.utils.load_torch_file(P.get("moge_path") or folder_paths.get_full_path_or_raise("geometry_estimation", "moge_2_vitl_normal_fp16.safetensors"), safe_load=True)
    moge = MoGeModel(sd)
    fov_x = math.degrees(2 * math.atan(SENSOR_H * W / H / 2 / lens0))   # the clip's own lens, so the depth is in the right scale
    dl = []
    for i in range(0, N, 2):
        x = torch.from_numpy(np.ascontiguousarray(frames[i:i + 2])).float().div(255).permute(0, 3, 1, 2)
        x = F.interpolate(x, (dh, dw), mode="area")
        with torch.no_grad():
            g = moge.infer(x, resolution_level=P.get("depth_level", 6), fov_x=fov_x, force_projection=True, apply_mask=False, refine_steps=0)
        d = g["depth"].float()[:, None].nan_to_num(nan=1e3, posinf=1e3)
        dl.append(F.interpolate(d, (H, W), mode="bilinear", align_corners=False)[:, 0].cpu()); print_step(i, N * 2)
    depth = torch.cat(dl).nan_to_num(nan=1e3, posinf=1e3).clamp(0.05, 1e3)   # (N, H, W) metres; sky/invalid = far
    del moge; torch.cuda.empty_cache()

# subject matte (soft alpha): matte.npy from the node's matte input when wired (any matting model), else BiRefNet's cutout (ComfyUI's
# background removal). It keeps the person's edges out of the background blur and gives the auto focus its subject.
alpha = None   # (N, dh, dw) uint8, upsampled per frame
aim_subject = P.get("zoom_aim", "subject") == "subject" and float(zoom.max()) > 1.001
use_matte = (depth is not None and (P.get("edge_matte", True) or needs_subject)) or aim_subject or bool(P.get("passes"))
if use_matte:
    if (job / "matte.npy").is_file():
        mt = torch.from_numpy(np.load(job / "matte.npy")).float()[:, None]
        alpha = (F.interpolate(mt, (dh, dw), mode="bilinear", align_corners=False)[:, 0].clamp(0, 1) * 255).byte()
    else:
        from comfy import bg_removal_model as BG
        seg = BG.load(P.get("subject_path") or folder_paths.get_full_path_or_raise("background_removal", "birefnet.safetensors"))
        al = []
        for i in range(0, N, 4):
            x = torch.from_numpy(np.ascontiguousarray(frames[i:i + 4])).float().div(255)
            x = F.interpolate(x.permute(0, 3, 1, 2), (dh, dw), mode="area").permute(0, 2, 3, 1)
            with torch.no_grad():
                m = seg.encode_image(x).float()[:, None]
            al.append((F.interpolate(m, (dh, dw), mode="bilinear", align_corners=False)[:, 0].clamp(0, 1).cpu() * 255).byte())
        alpha = torch.cat(al); del seg; torch.cuda.empty_cache()

full_alpha = lambda i: F.interpolate(alpha[i][None, None].float().div(255), (H, W), mode="bilinear", align_corners=False)[0, 0]

# zoom aim: the subject's upper body (from the matte), steadied over time, so a zoom keeps the person in frame; the crop never leaves the frame
aim_x, aim_y = np.zeros(N), np.zeros(N)   # normalised -1..1
if aim_subject and alpha is not None:
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, dh), torch.linspace(-1, 1, dw), indexing="ij")
    ax, ay, ok = np.zeros(N), np.zeros(N), np.zeros(N, bool)
    for i in range(N):
        a = alpha[i].float() / 255
        if a.sum() > 0.005 * dh * dw:
            rows = (a > 0.5).any(1).nonzero()
            top, bot = (float(ys[rows.min(), 0]), float(ys[rows.max(), 0])) if len(rows) else (-1.0, 1.0)
            ax[i] = float((a * xs).sum() / a.sum()); ay[i] = top + 0.3 * (bot - top); ok[i] = True
    if ok.any():
        idx = np.arange(N); ax = np.interp(idx, idx[ok], ax[ok]); ay = np.interp(idx, idx[ok], ay[ok])
        k = 15; ker = np.ones(k) / k
        aim_x = np.convolve(np.pad(ax, k // 2, mode="edge"), ker, "valid"); aim_y = np.convolve(np.pad(ay, k // 2, mode="edge"), ker, "valid")


# focus per frame: keys give metres or "subject" (the subject matte's median depth, steadied; the middle of the frame when none is found);
# between keys the focus moves in diopters (1/distance), as a focus pull does
focus = None
if depth is not None:
    subj = None
    if needs_subject:
        f = np.zeros(N)
        for i in range(N):
            m = full_alpha(i) > 0.5 if alpha is not None else torch.zeros(H, W, dtype=torch.bool); d = depth[i]
            f[i] = float(d[m].median()) if m.sum() > 0.005 * H * W else float(d[int(H * 0.25):int(H * 0.75), int(W * 0.3):int(W * 0.7)].quantile(0.2))
        k = 9; pad = np.pad(f, (k // 2, k // 2), mode="edge")
        subj = np.array([np.median(pad[i:i + k]) for i in range(N)])
    val = lambda v, i: float(subj[i]) if v == "subject" else float(v)
    focus = np.array([1.0 / ((1 - e) / val(a, i) + e / val(b, i)) for i, (a, b, e) in
                      enumerate(x if isinstance(x, tuple) else (x, x, 0.0) for x in focus_keys)])


def _gauss(L, device):
    k = int(math.ceil(L * 2)) * 2 + 1
    ax = torch.arange(k, device=device) - k // 2
    g = torch.exp(-(ax.float() ** 2) / (2 * (L / 1.6) ** 2)); return g / g.sum(), k


def _blur(t, L, Ly=None):
    """Separable gaussian: L across, Ly up and down (taller = anamorphic oval bokeh)."""
    c = t.shape[1]; g, k = _gauss(L, t.device); gy, ky = _gauss(Ly or L, t.device)
    t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(t, (0, 0, ky // 2, ky // 2), mode="replicate"), gy.view(1, 1, ky, 1).repeat(c, 1, 1, 1), groups=c)


def _layered(img, r, w, levels):
    """Each pixel takes the blur level matching its radius; the blur gathers colour weighted by w."""
    stack = [(0.0, img)] + [(L, _blur(img * w, L, L * ANAM) / _blur(w, L, L * ANAM).clamp_min(1e-4)) for L in levels[1:]]
    out = img.clone()
    for (l0, a), (l1, b) in zip(stack[:-1], stack[1:]):
        m = (r >= l0) & (r < l1); u = ((r - l0) / (l1 - l0)).clamp(0, 1)
        out = torch.where(m[None, None], a * (1 - u) + b * u, out)
    return torch.where((r >= levels[-1])[None, None], stack[-1][1], out)


def dof(img, z, zf, f_mm, fstop, z_scale, a=None):
    """Thin-lens blur: circle of confusion per pixel from metric depth. Blur gathers mostly from at-or-behind the focus plane. With a subject
    matte the background is blurred from background pixels only (filling behind the person from around them) and the person, blurred by
    its own depth, goes back on top through the matte, so hair and shoulders never smear into the background."""
    coc_mm = (f_mm * f_mm) / (fstop * max(zf * 1000 - f_mm, 1.0)) * (z - zf).abs() / z   # on the sensor
    cap = float(P.get("max_blur_px", 40))
    r = (coc_mm / SENSOR_H * H * z_scale / 2).clamp(0, cap)   # radius in input pixels (the crop later magnifies by zoom)
    levels = [0.0, 1.5, 3, 5, 8, 12, 18, 26, 40, 56, 80]
    levels = [L for L in levels if L < cap] + [cap]
    w = torch.where(z[None, None] >= zf * 0.97, 1.0, 0.15)
    person = _layered(img, r, w, levels)
    if a is None:
        return person, r
    A = a[None, None]; bgw = (1 - A).clamp_min(0) ** 2
    # the background's blur radius under the person: filled in from the background around them
    big = max(levels[-1], 26.0)
    r_fill = (_blur(r[None, None] * bgw, big) / _blur(bgw, big).clamp_min(1e-4))[0, 0]
    r_bg = torch.where(a > 0.02, torch.maximum(r_fill, r * (1 - a)), r)
    back = _layered(img, r_bg, w * bgw + 1e-6, levels)
    return person * A + back * (1 - A), torch.where(a > 0.5, r, r_bg)


def lens(img, z, ro, sx, sy, cx=0.0, cy=0.0, k1=0.0, ca=0.0):
    """Zoom (a focal change, aimed at cx, cy and kept inside the frame), lens shift (fraction of the frame), roll, radial distortion (k1:
    + barrel, - pincushion) and chromatic aberration (red and blue at slightly different radial scales), as one resample."""
    a = math.radians(ro); c, s = math.cos(a), math.sin(a); asp = W / H
    # the rotated window must stay inside the real frame (no invented edges): zoom in just enough to cover the roll, as a stabiliser does,
    # then keep the window's centre (aim plus lens shift) where its rotated half-extents still fit
    ex, ey = abs(c) + abs(s) / asp, abs(s) * asp + abs(c)   # half-extents at zoom 1, in normalised units
    z = max(z, ex, ey)
    lx, ly = max(0.0, 1 - ex / z), max(0.0, 1 - ey / z)
    tx, ty = max(-lx, min(lx, sx * 2 + cx)), max(-ly, min(ly, -sy * 2 + cy))
    th = torch.tensor([[c / z, -s / z / asp, tx], [s / z * asp, c / z, ty]], dtype=torch.float32, device=img.device)[None]
    if k1 == 0 and ca == 0:
        grid = F.affine_grid(th, list(img.shape), align_corners=False)
        return F.grid_sample(img, grid, mode="bicubic", padding_mode="reflection", align_corners=False).clamp(0, 1)
    base = F.affine_grid(torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]], device=img.device), list(img.shape), align_corners=False)
    q = base * torch.tensor([asp, 1.0], device=img.device); r2 = (q ** 2).sum(-1, keepdim=True) / (asp * asp + 1)   # 0 centre .. 1 corner
    def warp(scale):
        g = base * (1 + k1 * r2) * scale / (max(1 + k1, 1.0) * (1 + abs(ca) * 0.008))   # never reaches outside the frame
        x, y = g[..., 0], g[..., 1]
        return torch.stack([th[0, 0, 0] * x + th[0, 0, 1] * y + th[0, 0, 2], th[0, 1, 0] * x + th[0, 1, 1] * y + th[0, 1, 2]], -1)
    s_ca = ca * 0.008
    chans = [F.grid_sample(img[:, c:c + 1], warp(1 + d), mode="bicubic", padding_mode="reflection", align_corners=False)
             for c, d in ((0, s_ca), (1, 0.0), (2, -s_ca))]
    return torch.cat(chans, 1).clamp(0, 1)


def glow(img, b, h):
    """Bloom (highlights spread) and halation (a warm red glow around bright edges, as on film)."""
    if b <= 0 and h <= 0:
        return img
    small = F.interpolate(img, scale_factor=0.25, mode="area")
    lum = small.mean(1, keepdim=True); hi = (lum - 0.72).clamp_min(0) / 0.28
    out = img
    if b > 0:
        out = out + b * 0.8 * F.interpolate(_blur(small * hi, 10), size=img.shape[-2:], mode="bilinear", align_corners=False)
    if h > 0:
        tint = torch.tensor([1.0, 0.35, 0.12], device=img.device).view(1, 3, 1, 1)
        out = out + h * 0.6 * F.interpolate(_blur(hi, 4) * tint, size=img.shape[-2:], mode="bilinear", align_corners=False)
    return out.clamp(0, 1)


yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
rad2 = (xx * (W / H)) ** 2 + yy ** 2; rad2 = rad2 / rad2.max()
gen = torch.Generator(device=dev).manual_seed(int(P.get("seed", 0)))
out = np.lib.format.open_memmap(job / "out.npy", mode="w+", dtype=np.uint8, shape=(N, H, W, 3))
dprev = np.zeros((N, H // 4, W // 4), np.float32)
if P.get("passes") and depth is not None:
    dq = F.interpolate(depth[:, None], (H // 2, W // 2), mode="area")[:, 0]; disp = 1.0 / dq
    lo, hi = float(disp.quantile(0.01)), float(disp.quantile(0.99))
    np.save(job / "depth_pass.npy", ((disp - lo) / max(hi - lo, 1e-6)).clamp(0, 1).numpy().astype(np.float16))
if P.get("passes") and alpha is not None:
    np.save(job / "matte_pass.npy", alpha.numpy())
for i in range(N):
    img = torch.from_numpy(np.ascontiguousarray(frames[i])).to(dev).float().div(255).permute(2, 0, 1)[None]
    if depth is not None and P.get("dof"):
        z = depth[i].to(dev)
        img, r = dof(img, z, float(focus[i]), lens0 * zoom[i], float(fstop[i]), 1.0 / zoom[i],
                     full_alpha(i).to(dev) if alpha is not None and P.get("edge_matte", True) else None)
        dprev[i] = F.interpolate(r[None, None], (H // 4, W // 4), mode="area")[0, 0].cpu().numpy()
    img = lens(img, float(zoom[i]), float(roll[i]), float(shift_x[i]), float(shift_y[i]), float(aim_x[i]), float(aim_y[i]),
               float(distortion[i]), float(chroma[i]))
    img = glow(img, float(bloom[i]), float(halation[i]))
    img = img * (1 - float(vignette[i]) * rad2 ** 1.5).clamp(0, 1)
    if grain[i] > 0:
        n = torch.randn((1, 1, H, W), generator=gen, device=dev) * float(grain[i]) * 0.06
        img = (img + n * (1.2 - img.mean(1, keepdim=True))).clamp(0, 1)   # grain shows most in the mids and shadows
    out[i] = (img[0].permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
    print_step(N + i if depth is not None else i, N * 2 if depth is not None else N)
out.flush(); np.save(job / "blur.npy", dprev)
(job / "info.json").write_text(json.dumps({"frames": N, "zoom": [float(zoom[0]), float(zoom[-1])],
                                           "focus_m": None if focus is None else [round(float(focus.min()), 2), round(float(focus.max()), 2)],
                                           "fstop": [float(fstop.min()), float(fstop.max())], "keys": len(KEYS), "lens_mm": lens0}))
print("DONE", flush=True)
