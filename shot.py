"""Shot planning for the photoreal pass: capture loading, identity anchors cropped from the capture's own generated views,
chunk plans, and camera presets. Shared by the ComfyUI nodes (nodes_super.py) and the scripts. numpy + PIL + ffmpeg only.

A capture folder is either a trained 4D splat (Train 4DGS output) or a fast-mode capture (fast_rig: no training). Both have
dataset/manifest.json -> the source result (cameras.json + videos/NN.mp4 + preprocessing/viewer_geometry.npz),
normalization.json (world -> training space) and preview_meta.json (up vector, head/body tracks).
"""
import json, math, subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .capture_resolve import source_camera

W, H, FPS = 768, 1280, 24   # LTX shot size (multiples of 128 for IC-LoRA control) and frame rate


class Capture:
    def __init__(self, cap_dir, zoom_views=True, scene=True):
        """scene: when the capture has a setting solve (scene_layer: scene/p2s.npy), everything is placed in the setting's frame: the
        person's mesh, tracks and generated cameras move by the per-frame person-to-setting transform, so she and the clip's world
        (and the source camera that walked alongside her) agree. False = the capture's own body frame, as the trainer saw it."""
        self.dir = Path(cap_dir)
        self.meta = json.loads((self.dir / "preview_meta.json").read_text())
        self.up = np.array(self.meta["up"], float); self.up /= np.linalg.norm(self.up)
        self.head = np.array(self.meta["track"]["head"]); self.body = np.array(self.meta["track"]["body"])
        self.source = Path(json.loads((self.dir / "dataset" / "manifest.json").read_text())["source"])
        if not (self.source / "cameras.json").is_file():
            raise ValueError(f"{self.source} has no cameras.json: identity anchors need an upstream (GVHMR) capture")
        self.W2T = np.array(json.loads((self.dir / "normalization.json").read_text())["world_to_training"])
        self.fast = not any(self.dir.glob("ckpts/ckpt_*.pt"))
        rig = json.loads((self.source / "cameras.json").read_text())
        self.front_id = rig.get("front_camera_ids", [0])[0]
        self.cams = []   # (c2w in training space, K, video path, camera_id, width, height, set)
        self._add_cams(self.source, np.eye(4), "wide")
        extra = self.source / "extra_views.json"   # zoomed camera sets registered to this capture (scripts/add_zoom_views.py)
        if zoom_views and extra.is_file():
            for e in json.loads(extra.read_text()):
                per = e.get("to_main_world_per_frame")
                self._add_cams(Path(e["source"]), [np.array(m) for m in per] if per else np.array(e["to_main_world"]), e.get("name", "zoom"),
                               e.get("valid_top", 1.0))
        self.frames = len(self.head)
        self.source_cam = None   # the real input camera (source_cam.py): anchors come from the footage itself where the shot sees what it saw
        sc = source_camera(self.dir)
        if sc:
            s = json.loads(sc.read_text()); self.source_cam = len(self.cams)
            self.cams.append(([self._place(np.array(p), np.eye(4)) for p in _smooth_poses(s["c2w"])], np.array(s["K"], float), Path(s["video"]),
                              "", s["width"], s["height"], "source", 1.0))
        p2s = self.dir / "scene" / "p2s.npy"
        self.p2s = np.load(p2s) if scene and p2s.is_file() else None
        if self.p2s is not None:
            M = self.p2s; F = self.frames
            at = lambda f: M[min(max(int(round(f)), 0), len(M) - 1)]
            self.head = np.array([at(f)[:3, :3] @ x + at(f)[:3, 3] for f, x in enumerate(self.head)])
            self.body = np.array([at(f)[:3, :3] @ x + at(f)[:3, 3] for f, x in enumerate(self.body)])
            self.cams = [([at(f) @ (c[0][min(f, len(c[0]) - 1)] if isinstance(c[0], list) else c[0]) for f in range(F)],) + tuple(c[1:])
                         for c in self.cams]

    def _place(self, c2w0, Mi):
        A = self.W2T @ Mi; s = np.cbrt(np.linalg.det(A[:3, :3])); c2w = c2w0.copy()
        c2w[:3, 3] = A[:3, :3] @ c2w0[:3, 3] + A[:3, 3]; c2w[:3, :3] = A[:3, :3] @ c2w0[:3, :3] / s
        return c2w

    def _add_cams(self, src, M, name, valid_top=1.0):
        """Cameras of a generated view set; M maps that set's world into this capture's world (a similarity, or one per capture
        frame for a set registered frame by frame), then W2T to training space. Per-frame sets store a list of c2w, one per frame."""
        for c in json.loads((src / "cameras.json").read_text())["cameras"]:
            c2w0 = np.array(c["camera_to_world"], float)
            pose = [self._place(c2w0, Mi) for Mi in M] if isinstance(M, list) else self._place(c2w0, M)
            self.cams.append((pose, np.array(c["K"], float), src / c["video"], c["camera_id"], c["image_width"], c["image_height"], name, valid_top))

    def pose(self, i, frame):
        """c2w of camera i at capture frame `frame` (fixed for the main ring, per-frame for frame-registered sets)."""
        p = self.cams[i][0]
        return p[min(max(int(round(frame)), 0), len(p) - 1)] if isinstance(p, list) else p

    def geometry(self):
        """SMPL-X body mesh per capture frame in training space."""
        V = np.load(self.source / "preprocessing" / "viewer_geometry.npz")["vertices"].astype(np.float64)
        V = V @ self.W2T[:3, :3].T + self.W2T[:3, 3]
        if self.p2s is not None:
            M = self.p2s[np.minimum(np.arange(len(V)), len(self.p2s) - 1)]
            V = np.einsum("fij,fvj->fvi", M[:, :3, :3], V) + M[:, None, :3, 3]
        return V


def scene_meta(train_dir, m):
    """A capture's Virtual Camera meta in the setting's frame when it has one: head/body tracks and the subject centre moved with her,
    plus her per-frame transform (row-major 4x4s) for the viewport. Unchanged without a setting solve."""
    p = Path(train_dir) / "scene" / "p2s.npy"
    if not p.is_file() or "track" not in m:
        return m
    M = np.load(p); at = lambda f: M[min(max(int(round(f)), 0), len(M) - 1)]
    tr = {k: [(at(f)[:3, :3] @ np.array(x) + at(f)[:3, 3]).tolist() for f, x in enumerate(v)] for k, v in m["track"].items()}
    tr["follow"] = [(at(f)[:3, :3] @ np.array(m["centre"]) + at(f)[:3, 3]).tolist() for f in range(len(M))]   # where her centre travels
    mid = at(len(M) // 2)
    return {**m, "track": tr, "centre": (mid[:3, :3] @ np.array(m["centre"]) + mid[:3, 3]).tolist(), "p2s": [x.reshape(-1).tolist() for x in M]}


def _smooth_poses(c2w, k=9):
    """The real camera per frame, steadied: the solve from her body jitters (pose noise reads as camera shake), so the track is median
    filtered then averaged over k frames, position and rotation alike (rotation re-orthonormalised). A handheld source camera's real drift stays."""
    P = np.array(c2w, float)
    if len(P) < 3:
        return P
    h = k // 2; n = len(P); out = P.copy()
    med = np.stack([np.median(P[max(0, i - h):i + h + 1], 0) for i in range(n)])
    for i in range(n):
        w = med[max(0, i - h):i + h + 1].mean(0); U, _, Vt = np.linalg.svd(w[:3, :3])
        out[i, :3, :3] = U @ Vt; out[i, :3, 3] = w[:3, 3]
    return out


def look_at(eye, tgt, up, roll=0.0):
    """OpenCV camera rotation (x right, y down, z forward), as rig.look_at."""
    z = tgt - eye; z = z / np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    if roll:
        r = math.radians(roll); x, y = x * math.cos(r) + y * math.sin(r), -x * math.sin(r) + y * math.cos(r)
    return np.stack([x, y, z], 1)


def video_frame(path, idx):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    from io import BytesIO
    return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))


def _regions(cap, i, f, w, h):
    """Soft masks (h, w) of background / legs / torso+arms / head+hair for capture camera i at frame f, from the body mesh projected
    into that camera. Regions by height along the body and distance to the head track; the head is grown to take in the hair."""
    if not hasattr(cap, "_V"):
        cap._V = cap.geometry()
    V = cap._V[min(f, len(cap._V) - 1)]; K = cap.cams[i][1]; c2w = cap.pose(i, f)
    q = (V - c2w[:3, 3]) @ c2w[:3, :3]; ok = q[:, 2] > 1e-3; uv = (q[ok] / q[ok, 2:3]) @ K.T
    hgt = V @ cap.up; lo, span = hgt.min(), np.ptp(hgt)
    head = np.array([np.interp(f, np.arange(cap.frames), cap.head[:, k]) for k in range(3)])
    lab = np.where(np.linalg.norm(V - head, axis=1) < 0.09 * span, 3, np.where(hgt < lo + 0.47 * span, 1, 2))[ok]
    px = np.ptp(uv[:, 1]) / span if len(uv) else 0.0          # pixels per unit height in this view
    sc = 8; gw, gh = w // sc + 1, h // sc + 1; out = {}
    for k, grow in ((1, 0.03), (2, 0.03), (3, 0.06)):
        g = np.zeros((gh, gw), np.uint8); m = lab == k; x, y = (uv[m, 0] / sc).astype(int), (uv[m, 1] / sc).astype(int)
        keep = (x >= 0) & (x < gw) & (y >= 0) & (y < gh); g[y[keep], x[keep]] = 255
        r = max(1, int(grow * span * px / sc)) | 1
        im = Image.fromarray(g).filter(ImageFilter.MaxFilter(min(r * 2 + 1, 31))).filter(ImageFilter.GaussianBlur(1.5)).resize((w, h), Image.BILINEAR)
        out[k] = np.asarray(im.filter(ImageFilter.GaussianBlur(max(4, h // 150))), np.float32) / 255.0
    # a soft partition: the four weights sum to 1 at every pixel (no unmatched seams); head wins over torso wins over legs
    out[2] = out[2] * (1 - out[3]); out[1] = out[1] * (1 - out[3]) * (1 - out[2])
    out[0] = np.clip(1 - out[1] - out[2] - out[3], 0, 1)
    return out


def match_source(cap, i, img, f):
    """Generated view -> the real person's colours: per region (background, legs, torso, head+hair), shift the generated frame's mean and
    spread to the real input frame's at the same capture frame. The generator drifts hair, skin and clothing colour; the real footage
    knows them for every frame of the clip, so this needs nothing per clip."""
    sc = cap.source_cam; src = video_frame(cap.cams[sc][2], f).astype(np.float32); res = np.zeros(img.shape, np.float32)
    rs, rg = _regions(cap, sc, f, src.shape[1], src.shape[0]), _regions(cap, i, f, img.shape[1], img.shape[0])
    for k in (0, 1, 2, 3):
        ws, wg = (rs[k].reshape(-1) > 0.8).astype(np.float32), (rg[k].reshape(-1) > 0.8).astype(np.float32)   # stats from region cores only
        if ws.sum() < 400 or wg.sum() < 400:
            res += img.astype(np.float32) * rg[k][..., None]; continue
        S, G = src.reshape(-1, 3), img.astype(np.float32).reshape(-1, 3)
        ms, mg = (S * ws[:, None]).sum(0) / ws.sum(), (G * wg[:, None]).sum(0) / wg.sum()
        ss = np.sqrt((((S - ms) ** 2) * ws[:, None]).sum(0) / ws.sum()) + 1e-3; sg = np.sqrt((((G - mg) ** 2) * wg[:, None]).sum(0) / wg.sum()) + 1e-3
        lo = 0.25 if k == 0 else 0.6   # the generated room can be busier than the real one (extra lamps, props): let its contrast flatten
        res += ((img.astype(np.float32) - mg) * np.clip(ss / sg, lo, 1.6) + ms) * rg[k][..., None]
    return np.clip(res, 0, 255).astype(np.uint8)


def anchor(cap, row, out_png, size=None):
    """Identity anchor for one camera-path row [scene, eye, target, roll, fov]: the training camera whose view direction is
    nearest the shot camera, its generated frame at the same scene time, cropped to the shot's framing (render frame corners
    projected at the look-at distance) and edge-padded where the shot sees past the generated frame. Returns the camera id."""
    W, H = size or (globals()["W"], globals()["H"])
    s, eye, tgt, roll, fov = row[0], np.array(row[1:4]), np.array(row[4:7]), row[7], row[8]
    R = look_at(eye, tgt, cap.up, roll); d = np.linalg.norm(tgt - eye)
    fy = 0.5 / math.tan(math.radians(fov) / 2)
    fi = min(max(s, 0.0), 1.0) * (cap.frames - 1)
    head = np.array([np.interp(fi, np.arange(cap.frames), cap.head[:, k]) for k in range(3)])
    corners = [eye + d * (R @ np.array([sx * (W / H) / (2 * fy), sy / (2 * fy), 1.0])) for sx in (-1, 1) for sy in (-1, 1)]

    def box(i):   # the shot's frame projected into generated camera i: centre, crop size (pixels), fraction inside that camera's image
        _, K, _, _, cw, ch, _, vt = cap.cams[i]; w2c = np.linalg.inv(cap.pose(i, fi)); uv = []
        ch = ch * vt   # a zoomed set only saw the top of its frame (the rest was guessed): anchors must come from that part
        for p in corners:
            q = w2c[:3, :3] @ p + w2c[:3, 3]; uv.append((K @ (q / q[2]))[:2])
        uv = np.array(uv); (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
        cx, cy, bh = (x0 + x1) / 2, (y0 + y1) / 2, max(y1 - y0, (x1 - x0) * H / W); bw = bh * W / H
        ix = max(0.0, min(cx + bw / 2, cw) - max(cx - bw / 2, 0)); iy = max(0.0, min(cy + bh / 2, ch) - max(cy - bh / 2, 0))
        hq = w2c[:3, :3] @ head + w2c[:3, 3]; hu, hv = (K @ (hq / hq[2]))[:2]   # the face must sit well inside the real image
        face_ok = hq[2] > 0 and 0.1 * cw <= hu <= 0.9 * cw and 0.08 * ch <= hv <= 0.85 * ch
        inside = ix * iy / max(bw * bh, 1e-6)
        return cx, cy, bh, bw, inside if face_ok else 0.0, inside

    # nearest viewing direction; among cameras about as well aligned (wide ring + zoomed sets), take the one that puts the most
    # of its own pixels on the shot's frame while still covering it, so close-ups get native detail instead of an upscaled crop
    dots = [float(cap.pose(i, fi)[:3, 2] @ R[:, 2]) for i in range(len(cap.cams))]; top = max(dots)
    cand = [(i, box(i)) for i in range(len(cap.cams)) if dots[i] >= top - 0.03]
    good = [(i, b) for i, b in cand if b[4] >= 0.7]   # face inside + most of the frame (edge padding covers a bit of wall)
    if not good:   # nothing covers the shot (e.g. a wide lens): take the camera that covers most of it, from any direction nearby
        cand = [(i, box(i)) for i in range(len(cap.cams)) if dots[i] >= top - 0.15]
    best, (cx, cy, bh, bw, _, _) = max(good, key=lambda ib: ib[1][2]) if good else max(cand, key=lambda ib: ib[1][5])
    sc = cap.source_cam   # within ~14 degrees of the real camera and covering the shot: the real frame beats any generated view
    # no face test for the real camera: where the shot sees what it saw, its frame is the truth even with the head near an edge
    if sc is not None and dots[sc] >= 0.97 and box(sc)[5] >= 0.7:
        best, (cx, cy, bh, bw, _, _) = sc, box(sc)
    c2w, (K, video, cid) = cap.pose(best, fi), cap.cams[best][1:4]
    a = video_frame(video, int(round(min(max(s, 0.0), 1.0) * (cap.frames - 1))))
    if cap.source_cam is not None and best != cap.source_cam:
        a = match_source(cap, best, a, int(round(fi)))
    pad = int(max(bw, bh))
    # outside the generated frame: a heavy blur of the edge-extended image (soft wall/floor), not raw smeared edge pixels, which
    # the video models copy as streaks; feathered into the real image so there is no hard border
    hh = a.shape[0]; top_c, bot_c = a[: max(1, hh // 20)].reshape(-1, 3).mean(0), a[-max(1, hh // 20):].reshape(-1, 3).mean(0)
    t = np.clip((np.arange(hh + 2 * pad) - pad) / max(hh - 1, 1), 0, 1)[:, None, None]       # wall tone above, floor tone below
    ext = Image.fromarray((top_c * (1 - t) + bot_c * t).repeat(a.shape[1] + 2 * pad, 1).astype(np.uint8))
    ext.paste(Image.fromarray(a), (pad, pad)); soft = ext.copy()
    soft.paste(Image.fromarray((top_c * (1 - t) + bot_c * t).repeat(a.shape[1] + 2 * pad, 1).astype(np.uint8)), (0, 0))
    m = Image.new("L", ext.size, 0); m.paste(255, (pad, pad, pad + a.shape[1], pad + a.shape[0]))
    m = m.filter(ImageFilter.GaussianBlur(max(4, a.shape[0] // 80)))
    img = Image.composite(ext, soft, m)
    crop = (int(cx - bw / 2) + pad, int(cy - bh / 2) + pad, int(cx + bw / 2) + pad, int(cy + bh / 2) + pad)
    img.crop(crop).resize((W, H), Image.LANCZOS).save(out_png)
    # where the anchor holds real pixels (the camera's own frame; for a zoomed set only its valid top): Photoreal Shoot passes this as
    # the guide's attention mask, so the padding has no pull on the render (padding was copied as box edges) and a portrait view can
    # anchor its strip of a landscape shot
    fe = max(4, a.shape[0] // 60); vt = cap.cams[best][7]
    mk = Image.new("L", ext.size, 0); mk.paste(255, (pad + fe, pad + fe, pad + a.shape[1] - fe, pad + int(a.shape[0] * vt) - fe))
    mk.filter(ImageFilter.GaussianBlur(fe / 2)).crop(crop).resize((W, H), Image.BILINEAR).save(Path(out_png).with_name(Path(out_png).stem + "_mask.png"))
    cov = (box(best)[5])
    return f"{cap.cams[best][6]}{cid}", round(float(cov), 3)   # camera (e.g. wide12 / zoom12), fraction of the shot frame it covers


def plan(n_frames, chunk_seconds, anchor_every, start=0.8, mid=0.5, end=0.5, every=0.6, fps=FPS):
    """Chunks of <= chunk_seconds; per chunk, anchor times (s, chunk-relative) with strengths. Chunks after the first start
    from the previous chunk's last frame (linked in the graph) plus a capture anchor one latent frame in."""
    dur = n_frames / fps; out = []; t0 = 0.0
    while t0 < dur - 1e-6:
        secs = min(chunk_seconds, dur - t0); f0 = int(round(t0 * fps)); n = min(int(round(secs * fps)), n_frames - f0)
        refs = [("start", 0, start if not out else None), (round(secs / 2, 3), min(n_frames - 1, f0 + n // 2), mid),
                ("end", f0 + n - 1, end)]
        if anchor_every > 0:
            t = anchor_every
            while t < secs - 0.25:
                if abs(t - secs / 2) > 0.25:
                    refs.append((round(t, 3), min(n_frames - 1, f0 + int(t * fps)), every))
                t += anchor_every
        if out:
            refs.append((round(8 / fps, 3), f0, 0.5))
        out.append({"t0": t0, "secs": secs, "f0": f0, "n": n, "refs": refs})
        t0 += secs
    return out
