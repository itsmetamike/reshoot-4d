"""Reshoot 4D: one video of a person -> synced multi-view videos (sam3dbx4danyone) -> FreeTimeGS 4D splat, inside ComfyUI,
then a virtual camera through it and an optional LTX 2.3 realism pass.

Wraps the capture CLIs under config.ROOT. Generation runs in repo/.venv (torch 2.8 cu128), training in
repo/third_party/FreeTimeGsVanilla/.venv (torch 2.10 cu128 + gsplat/fused-ssim/torch_scatter built against it). Both run as
subprocesses so they never share ComfyUI's torch; ComfyUI's models are unloaded first so generation has the GPU to itself.
"""
import json, math, os, subprocess, time
from pathlib import Path

import cv2
import numpy as np
import torch

import comfy.model_management as mm

from .beat_path import beat_path
from . import director
from . import app_routes  # noqa: F401  (registers the editor and player routes)

from .config import ROOT, REPO, GEN_PY, TRAIN_PY, CUDA128, CUDA_ARCH, GPUS   # paths + GPUs: see config.py
OUT = ROOT / "out" / "comfy"


def _small_gpu():
    """GPU key for small jobs (depth/splat renders): the second card when config names one, else the main card. On a shared card,
    ComfyUI's models are unloaded first so the job fits."""
    if GPUS["small"] == GPUS["big"]:
        _free_comfy()
    return "small"


def _env(gpu, train=False):
    env = {**os.environ, "UV_CACHE_DIR": str(ROOT / "uvcache"), "HF_HOME": str(ROOT / "hf"), "TMPDIR": str(ROOT / "tmp"),
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPUS[gpu]}
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if train:   # gsplat JIT needs the nvcc that matches the trainer's torch (12.8), with C++20 for torch 2.10 headers
        t = CUDA128 / "targets" / "x86_64-linux"
        env.update(CUDA_HOME=str(CUDA128), PATH=env["PATH"] + f":{CUDA128}/bin", TORCH_CUDA_ARCH_LIST=CUDA_ARCH,
                   CPATH=str(t / "include"), LIBRARY_PATH=f"{t}/lib:{CUDA128}/lib", NVCC_APPEND_FLAGS="-std=c++20")
    return env


CAT, CAT_ADV, CAT_LEGACY = "Reshoot 4D", "Reshoot 4D/Advanced", "Reshoot 4D/Legacy"
_STEPS = {"inference.py": "Generating the views", "upstream_adapter.py": "Preparing the views", "fast_rig.py": "Building the mannequin",
          "source_cam.py": "Solving the real camera", "train_4dgs.py": "Training the splat", "dump_points.py": "Building the splat preview",
          "render_path.py": "Rendering the splat", "mesh_depth.py": "Rendering depth"}


def _failed(cmd, code, log):
    """A short error for the node, the full output stays in the log."""
    step = _STEPS.get(Path(str(cmd[1])).name, Path(str(cmd[1])).name)
    tail = [l for l in Path(log).read_text(errors="replace").splitlines() if l.strip()][-4:]
    return RuntimeError(f"{step} failed (exit {code}). Full log: {log}\n" + "\n".join(tail))


def _run(cmd, log, env):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n"); f.flush()
        r = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env=env)
    if r.returncode != 0:
        raise _failed(cmd, r.returncode, log)


def _read_frames(path, height=None, stride=1):
    cap, frames, i = cv2.VideoCapture(str(path)), [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            if height and fr.shape[0] != height:
                fr = cv2.resize(fr, (round(fr.shape[1] * height / fr.shape[0] / 2) * 2, height), interpolation=cv2.INTER_AREA)
            frames.append(fr)
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return torch.from_numpy(np.stack(frames)).float() / 255.0


def _grid(result_dir, height):
    views = sorted((Path(result_dir) / "videos" / "dense").glob("*.mp4"))
    clips = [_read_frames(v, height) for v in views]
    n = min(len(c) for c in clips)
    return torch.cat([c[:n] for c in clips], dim=2)


def _orbit_plate(img, rows, width, height, blur):
    """Cylindrical background: one full plate width per 360 deg of yaw, zoom follows the camera distance."""
    pl = img.permute(2, 0, 1)[None]
    if blur > 0:
        k = int(blur * 3) * 2 + 1
        x = torch.arange(k) - k // 2; w = torch.exp(-x.float() ** 2 / (2 * blur ** 2)); w = (w / w.sum())
        pl = torch.nn.functional.conv2d(torch.nn.functional.pad(pl, (k // 2, k // 2, 0, 0), mode="circular"), w.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
        pl = torch.nn.functional.conv2d(torch.nn.functional.pad(pl, (0, 0, k // 2, k // 2), mode="replicate"), w.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
    pl = pl[0]                                            # 3 x h x w, wraps horizontally
    r = torch.tensor([[row[1], row[2], row[3]] for row in rows], dtype=torch.float32)   # yaw, pitch, dist
    d0 = float(r[:, 2].median())
    z = (d0 / r[:, 2]).clamp(1.0, 1.15)                    # closer camera -> slightly larger background
    aspect = pl.shape[2] / pl.shape[1]
    # normalised sampling grid per frame: x wraps by yaw (one plate width per 360 deg), y follows pitch; zoom about centre
    gx = torch.linspace(-1, 1, width)[None, None, :] / z[:, None, None] * (width / height) / aspect / 1.15
    gx = (gx - 2 * r[:, 0, None, None] / 360.0 + 1) % 2 - 1
    gy = torch.linspace(-1, 1, height)[None, :, None] / z[:, None, None] / 1.15 + (r[:, 1].clamp(-10, 10) / 10 * 0.12)[:, None, None]
    out = []
    for i in range(0, len(rows), 32):
        grid = torch.stack([gx[i:i + 32].expand(-1, height, width), gy[i:i + 32].expand(-1, height, width)], -1)
        out.append(torch.nn.functional.grid_sample(pl[None].expand(len(grid), -1, -1, -1), grid, mode="bilinear",
                                                   padding_mode="border", align_corners=True).permute(0, 2, 3, 1))
    return torch.cat(out)


def _stage_match(rgb, a, bg, grade, rim, shadow):
    """2D stage match for a studio-lit capture over a club plate: exposure/tint toward the plate, a rim in the plate's
    brightest hue along the silhouette, and a soft contact shadow under the lowest alpha. rgb is premultiplied."""
    n, h, w, _ = rgb.shape
    lum = lambda x: (x * torch.tensor([0.2126, 0.7152, 0.0722])).sum(-1)
    bl = lum(bg).mean((1, 2))                                              # per-frame plate brightness
    mean_c = bg.mean((1, 2)); mean_c = mean_c / mean_c.mean(-1, keepdim=True).clamp(min=1e-4)
    # exposure: pull the subject's mean luminance toward a level implied by the plate, never brighten and never below 0.45x,
    # so a white dress darkens in a dark club while black clothes stay as they are
    cov = a[..., 0].sum((1, 2)).clamp(min=1.0)
    subj = (lum(rgb).sum((1, 2)) / cov).clamp(min=1e-3)                    # unpremultiplied mean subject luminance
    target = (0.12 + 1.6 * bl).clamp(0.15, 0.6)
    expo = (target / subj).clamp(0.45, 1.0)
    g = (1 - grade) + grade * expo[:, None] * (0.7 + 0.3 * mean_c)        # n x 3
    out = rgb * g[:, None, None, :]
    if rim > 0:
        flat = bg.reshape(n, -1, 3); top = lum(flat).topk(max(1, flat.shape[1] // 200), dim=1).indices
        rim_c = torch.stack([flat[i, top[i]].mean(0) for i in range(n)])   # brightest ~0.5% of the plate (lasers, strobes)
        rim_c = rim_c / rim_c.max(-1, keepdim=True).values.clamp(min=1e-4)
        am = a.permute(0, 3, 1, 2)
        edge = (am - (-torch.nn.functional.max_pool2d(-am, 9, 1, 4))).clamp(0, 1).permute(0, 2, 3, 1)
        out = out + rim * 0.5 * edge * rim_c[:, None, None, :] * a
    if shadow > 0:
        ys = torch.arange(h).float()[None, :, None]; xs = torch.arange(w).float()[None, None, :]
        m = a[..., 0] > 0.5
        rows_any = m.any(2)                                                # n x h
        bottom = torch.where(rows_any.any(1), (rows_any.float() * torch.arange(h)).argmax(1).float(), torch.full((n,), float(h)))
        cx = torch.where(m.any((1, 2)), (m.float() * xs).sum((1, 2)) / m.float().sum((1, 2)).clamp(min=1), torch.full((n,), w / 2))
        width = (m.any(1).float().sum(1) * 0.55).clamp(min=w * 0.12)
        sh = torch.exp(-((xs - cx[:, None, None]) / width[:, None, None]) ** 2 - ((ys - bottom[:, None, None]) / (h * 0.018)) ** 2)
        bg = bg * (1 - shadow * 0.85 * sh[..., None])
    return out, bg


def _free_comfy():
    mm.unload_all_models(); mm.soft_empty_cache()


class Sam3dbxGenerateViews:
    """Generate synced views around one person from a video. Advanced capture step: Capture does this and more in one node."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"name": ("STRING", {"default": "person", "advanced": True, "tooltip": "Folder name (a timestamp is added)"}),
                             "views": ([6, 24], {"default": 6, "tooltip": "Number of generated views in one ring around the person"}),
                             "target_fps": ("STRING", {"default": "auto", "advanced": True, "tooltip": "auto keeps the input clock. A number spreads the 121 frames over a longer stretch"}),
                             "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1, "tooltip": "Where to start in the video (s)"}),
                             "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
                             "preview_height": ("INT", {"default": 384, "min": 128, "max": 1280, "step": 32, "advanced": True, "tooltip": "Height of each preview view"})},
                "optional": {"frames": ("IMAGE", {"tooltip": "Source frames"}),
                             "frames_fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 0.5, "tooltip": "Frame rate of the connected frames"}),
                             "video_path": ("STRING", {"default": "", "advanced": True, "tooltip": "A video file on the server, used instead of frames"})}}
    RETURN_TYPES = ("STRING", "IMAGE", "STRING"); RETURN_NAMES = ("result_dir", "views_grid", "info")
    OUTPUT_TOOLTIPS = ("Generated views folder, for Train 4DGS", "A few of the generated views", "What ran")
    FUNCTION = "run"; CATEGORY = CAT_ADV; OUTPUT_NODE = True

    def run(self, name, views, target_fps, start_time, seed, preview_height, frames=None, frames_fps=16.0, video_path=""):
        run = OUT / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"; run.mkdir(parents=True)
        log = run / "generate.log"
        if video_path.strip():
            src = Path(video_path.strip())
        elif frames is not None:
            src = run / "source.mp4"
            arr = (frames.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            h, w = arr.shape[1:3]
            p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(frames_fps),
                                  "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(src)], stdin=subprocess.PIPE)
            p.stdin.write(arr.tobytes()); p.stdin.close()
            if p.wait() != 0:
                raise RuntimeError("ffmpeg could not write the source video")
        else:
            raise ValueError("Connect frames or set video_path.")
        _free_comfy()
        t = time.time()
        _run([GEN_PY, "inference.py", f"--video_path={src}", f"--views_per_layer={views}", f"--target_fps={target_fps}",
              f"--start_time={start_time}", f"--seed={seed}", "--gpu_ids=[0]", f"--data_dir={run}"], log, _env("big"))
        result = next((run / "fdanyone").iterdir())
        info = f"{views} views -> {result}"
        return {"ui": {"sam3dbx_result_dir": [str(result)]}, "result": (str(result), _grid(result, preview_height), info)}


class Sam3dbxTrain4DGS:
    """Train a 4DGS splat from generated views. Advanced capture step: Capture does this in one node."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result_dir": ("STRING", {"forceInput": True, "tooltip": "Generated views folder"}),
                             "steps": ("INT", {"default": 15000, "min": 2000, "max": 60000, "step": 1000, "tooltip": "Training steps: more steps, a sharper splat"}),
                             "samples_per_keyframe": ("INT", {"default": 32768, "min": 4096, "max": 262144, "step": 4096, "advanced": True,
                                                              "tooltip": "Splat density: 32768 = 819k Gaussians"}),
                             "tag": ("STRING", {"default": "", "advanced": True, "tooltip": "Output folder suffix. Empty = steps and density"}),
                             "preview_height": ("INT", {"default": 640, "min": 128, "max": 1280, "step": 32, "advanced": True, "tooltip": "Height of the preview"})}}
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING"); RETURN_NAMES = ("train_dir", "tsog_path", "flythrough", "info")
    OUTPUT_TOOLTIPS = ("The trained capture, for the Virtual Camera", "The splat file", "The splat seen from its training cameras", "What ran")
    FUNCTION = "run"; CATEGORY = CAT_ADV; OUTPUT_NODE = True

    def run(self, result_dir, steps, samples_per_keyframe, tag, preview_height):
        gpu = "big"   # the main card
        out = Path(result_dir).parent.parent / f"train_{tag or f'{samples_per_keyframe}_{steps}'}"
        if gpu == "big":
            _free_comfy()
        _run([GEN_PY, "train_4dgs.py", f"--result_dir={result_dir}", f"--output_dir={out}", "--gpu_id=0",
              f"--steps={steps}", f"--samples_per_keyframe={samples_per_keyframe}"], out.with_suffix(".log"), _env(gpu, train=True))
        traj = sorted((out / "videos").glob("traj_4d_step*.mp4"))[-1]
        info = f"{steps} steps x {samples_per_keyframe}/keyframe -> {out / 'scene.tsog'}"
        return {"ui": {"sam3dbx_train_dir": [str(out)], "sam3dbx_tsog": [str(out / "scene.tsog")]},
                "result": (str(out), str(out / "scene.tsog"), _read_frames(traj, preview_height), info)}


def _captures():
    """4D captures for the Director dropdown, newest first: trained splats and fast-mode (mannequin) captures."""
    if not OUT.parent.exists():
        return ["(no captures yet)"]
    items = [d.parent for d in OUT.parent.rglob("ckpts") if (d.parent / "dataset" / "manifest.json").is_file()]
    items += [p.parent for p in OUT.parent.rglob("preview_points.bin") if (p.parent / "dataset" / "manifest.json").is_file()
              and not (p.parent / "ckpts").exists()]   # fast-mode captures (mannequin, no splat)
    # every capture folder stays listed so a saved workflow always validates; the Director's mode toggle picks the fast or 4DGS sibling
    return [str(d) for d in sorted(set(items), key=lambda d: d.stat().st_mtime, reverse=True)] or ["(no captures yet)"]


def _splats():
    """Trained captures only (the Splat Viewer plays scene.tsog)."""
    return [c for c in _captures() if (Path(c) / "scene.tsog").is_file()] or ["(no trained splats yet)"]


class Sam3dbxCameraDirector:
    """Direct a new camera move through your 4D capture. Open the Virtual Camera, add keys or pick a preset, set the lens,
    freeze time for bullet time. The move is saved with the workflow."""
    @classmethod
    def INPUT_TYPES(cls):
        from .capture_resolve import SPLAT, MODES
        return {"required": {"capture": (_captures(), {"tooltip": "A saved capture. A connected Capture node takes priority"}),
                             "mode": (MODES, {"default": SPLAT, "tooltip": "4DGS = the trained splat. fast = the mannequin. Camera keys carry over when you switch"}),
                             "director_state": ("STRING", {"default": "", "multiline": True, "tooltip": "The camera move, saved with the workflow. Edit it in the Virtual Camera"})},
                "optional": {"train_dir": ("STRING", {"forceInput": True, "tooltip": "Capture from the Capture node"})}}
    @classmethod
    def VALIDATE_INPUTS(cls, capture, **kw):
        return True   # a shared workflow names a capture from another machine; a wired Capture node replaces it, run() reports a missing one
    RETURN_TYPES = ("STRING", "STRING", "STRING"); RETURN_NAMES = ("train_dir", "camera_path", "info")
    OUTPUT_TOOLTIPS = ("The capture this move was made on", "The camera move, for Render Shot or Photoreal Prep", "Length, frame rate and key count")
    FUNCTION = "run"; CATEGORY = CAT

    def run(self, capture, mode, director_state, train_dir=None):
        from . import capture_resolve as cr
        td, _kind = cr.resolve(Path(train_dir or capture), mode)
        if not director_state.strip():
            raise ValueError("No camera move yet. Open the Virtual Camera on this node, add a key or pick a preset, then queue again.")
        state = json.loads(director_state)
        sp = state.get("space_dir")
        if sp and Path(sp).is_dir() and Path(sp).resolve() != td.resolve():
            try:
                same = cr.source(sp) == cr.source(td)
            except Exception:  # noqa: BLE001 - no manifest on one side: let the conversion below decide
                same = True
            if not same:   # keys from another clip would point at empty space here
                raise ValueError("This camera move was made on a different clip. Open the Virtual Camera on this capture and use Import "
                                 "to bring the move over (it re-fits it to this person).")
        if sp and Path(sp).is_dir() and Path(sp).resolve() != td.resolve():   # keys were designed in the other capture's space
            cr.convert_state(state, sp, td)
        meta = td / "preview_meta.json"
        m = json.loads(meta.read_text()) if meta.is_file() else {}
        if m:
            from . import shot as S
            m = S.scene_meta(td, m)   # keys aim at her where she is in the setting
        basis = (m["up"], m["front"], m["side"]) if m else None
        frames = director.rows(state, m.get("dist0", 1.0), basis, m.get("track"))
        path = td / f"camera_director_{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps({"fps": float(state.get("fps", 24)), "frames": frames, "state": state}))
        return (str(td), str(path), f"{'4DGS splat' if _kind == 'splat' else 'mannequin'}: {state['duration']} s at {state.get('fps', 24)} fps, {len(state['keys'])} keys")


class Sam3dbxView4D:
    """Look around a trained 4DGS capture in the node. Drag to orbit, scrub time, play the motion."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"capture": (_splats(), {"tooltip": "A trained 4DGS capture"})},
                "optional": {"train_dir": ("STRING", {"forceInput": True, "tooltip": "Capture from the Capture node"})}}
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("tsog_path",)
    OUTPUT_TOOLTIPS = ("The splat file this viewer plays",)
    FUNCTION = "run"; CATEGORY = CAT; OUTPUT_NODE = True

    def run(self, capture, train_dir=None):
        tsog = Path(train_dir or capture) / "scene.tsog"
        if not tsog.is_file():
            raise FileNotFoundError(f"No trained splat in {tsog.parent}. Run Capture with 4DGS splat (train).")
        return {"ui": {"sam3dbx_tsog": [str(tsog)]}, "result": (str(tsog),)}


class Sam3dbxLoadResult:
    """Reopen generated views to preview or retrain them. Advanced capture step."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result_dir": ("STRING", {"default": "", "tooltip": "A generated views folder on the server"}),
                             "preview_height": ("INT", {"default": 384, "min": 128, "max": 1280, "step": 32, "advanced": True, "tooltip": "Height of each preview view"})}}
    RETURN_TYPES = ("STRING", "IMAGE"); RETURN_NAMES = ("result_dir", "views_grid")
    OUTPUT_TOOLTIPS = ("Generated views folder, for Train 4DGS", "A few of the views")
    FUNCTION = "run"; CATEGORY = CAT_ADV

    def run(self, result_dir, preview_height):
        if not any(Path(result_dir.strip() or "/nonexistent").rglob("*.mp4")):
            raise FileNotFoundError(f"No generated views in '{result_dir}'. Pick a folder made by Generate Views or Capture.")
        return (result_dir, _grid(result_dir, preview_height))


class Sam3dbxBeatCamera:
    """Legacy music-timed camera path. Use the Virtual Camera for new workflows."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"train_dir": ("STRING", {"forceInput": True}),
                             "bpm": ("FLOAT", {"default": 138.0, "min": 40.0, "max": 220.0, "step": 0.01}),
                             "first_beat_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.001,
                                                        "tooltip": "First downbeat, in seconds from the start of the audio excerpt"}),
                             "duration_s": ("FLOAT", {"default": 13.91, "min": 1.0, "max": 120.0, "step": 0.01}),
                             "fps": ("FLOAT", {"default": 24.0, "min": 8.0, "max": 60.0, "step": 1.0}),
                             "orbit_deg_per_bar": ("FLOAT", {"default": 45.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                             "bullet_every_bars": ("INT", {"default": 4, "min": 0, "max": 32, "tooltip": "0 = only at the drop"}),
                             "bullet_sweep_deg": ("FLOAT", {"default": 180.0, "min": 0.0, "max": 720.0, "step": 5.0}),
                             "drop_s": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 120.0, "step": 0.01, "tooltip": "-1 = none"}),
                             "push": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01}),
                             "roll_deg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                             "speed": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 8.0, "step": 0.05, "tooltip": "Capture playback speed (x the capture clock). A slowed source, e.g. 1 s interpolated to 121 frames, plays real-time at ~4.84"}),
                             "distance": ("FLOAT", {"default": 1.25, "min": 0.5, "max": 3.0, "step": 0.05, "tooltip": "1 = training camera distance"}),
                             "pitch_deg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 45.0, "step": 1.0})}}
    RETURN_TYPES = ("STRING", "STRING"); RETURN_NAMES = ("camera_path", "info")
    FUNCTION = "run"; CATEGORY = CAT_LEGACY; DEPRECATED = True

    def run(self, train_dir, bpm, first_beat_s, duration_s, fps, orbit_deg_per_bar, bullet_every_bars, bullet_sweep_deg,
            drop_s, push, roll_deg, speed, distance, pitch_deg):
        man = json.loads((Path(train_dir) / "dataset" / "manifest.json").read_text())
        num, den = (str(man["fps"]).split("/") + ["1"])[:2]
        src_seconds = man["frames"] / (float(num) / float(den))
        rows, markers = beat_path(bpm, first_beat_s, duration_s, src_seconds, fps=fps, orbit_deg=orbit_deg_per_bar, push=push,
                                  roll_deg=roll_deg, bullet_every=bullet_every_bars, bullet_deg=bullet_sweep_deg, drop_s=drop_s,
                                  speed=speed, pitch_deg=pitch_deg, dist=distance)
        path = Path(train_dir) / f"camera_{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps({"fps": fps, "bpm": bpm, "first_beat_s": first_beat_s, "rows": rows, "markers": markers}))
        return (str(path), f"{len(rows)} frames, {markers['bars']} bars, bullet time on bars {[b + 1 for b in markers['bullet_bars']]}")


class Sam3dbxRender4DPath:
    """Render the 4DGS splat along the Virtual Camera move: the finished splat shot, its alpha matte and an mp4."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"train_dir": ("STRING", {"forceInput": True, "tooltip": "Capture from the Virtual Camera"}),
                             "camera_path": ("STRING", {"forceInput": True, "tooltip": "Camera move from the Virtual Camera"}),
                             "width": ("INT", {"default": 0, "min": 0, "max": 3840, "step": 16, "advanced": True, "tooltip": "0 = auto: the shot aspect, long side 1280"}),
                             "height": ("INT", {"default": 0, "min": 0, "max": 3840, "step": 16, "advanced": True, "tooltip": "0 = auto: the shot aspect, long side 1280"}),
                             "background": ("STRING", {"default": "#0b0b0b", "advanced": True, "tooltip": "Background colour (hex) when no plate is connected"}),
                             "audio_path": ("STRING", {"default": "", "advanced": True, "tooltip": "Optional audio file to add to the mp4"}),
                             "audio_start_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.001, "advanced": True, "tooltip": "Where the audio starts (s)"}),
                             "name": ("STRING", {"default": "shot", "advanced": True, "tooltip": "File name for the render"}),
                             "look_drop": ("FLOAT", {"default": 0.12, "min": -1.0, "max": 1.0, "step": 0.01, "advanced": True,
                                                     "tooltip": "Legacy Beat Camera paths only: lower the look-at point"})},
                "optional": {"plate": ("IMAGE", {"tooltip": "Optional background image or frames behind the splat"}),
                             "plate_mode": (["orbit-pan", "static"], {"default": "orbit-pan",
                                            "tooltip": "orbit-pan: a wide still wraps round like a cylinder and turns with the camera yaw (and zooms on push-ins)"}),
                             "plate_blur": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5, "tooltip": "Gaussian blur sigma (px) for depth separation"}),
                             "stage_grade": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Pull the subject's exposure and tint toward the plate"}),
                             "stage_rim": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Rim light in the plate's brightest colour along the silhouette"}),
                             "stage_shadow": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Soft contact shadow under the feet"}),
                             "edge_choke": ("INT", {"default": 1, "min": 0, "max": 6, "tooltip": "Erode the splat's coverage by this many px before compositing (kills matte halos)"}),
                             "preview_div": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "frames/alpha outputs are downscaled by this (the mp4 is full size)"}),
                             "setting": ("BOOLEAN", {"default": False, "tooltip": "Experimental: render the clip's setting around the person (rebuilt in 3D with "
                                                     "Depth Anything 3). Off = the person alone over the background colour or plate"})}}
    RETURN_TYPES = ("IMAGE", "MASK", "STRING"); RETURN_NAMES = ("frames", "alpha", "video_path")
    OUTPUT_TOOLTIPS = ("The shot (preview size, see preview_div; the mp4 is full size)", "Where the splat covers the frame (a matte)", "The full-size mp4")
    FUNCTION = "run"; CATEGORY = CAT_ADV; OUTPUT_NODE = True

    def run(self, train_dir, camera_path, width, height, background, audio_path, audio_start_s, name, look_drop=0.12, plate=None,
            plate_mode="orbit-pan", plate_blur=2.0, stage_grade=0.6, stage_rim=0.3, stage_shadow=0.6, edge_choke=1, preview_div=2, setting=False):
        if not any(Path(train_dir).glob("ckpts/ckpt_*.pt")):
            raise ValueError("This capture has no trained splat. Run Capture with 4DGS splat (train), or set the Virtual Camera to 4DGS.")
        out = Path(train_dir) / f"render_{name}_{time.strftime('%Y%m%d-%H%M%S')}"
        if not (width and height):   # follow the shot the Virtual Camera framed (its aspect follows the input clip), long side 1280
            a, b = (float(v) for v in str((json.loads(Path(camera_path).read_text()).get("state") or {}).get("aspect") or "9:16").split(":"))
            r = a / b
            if width:
                height = width / r
            elif height:
                width = height * r
            else:
                width, height = (1280, 1280 / r) if r >= 1 else (1280 * r, 1280)
            width, height = int(round(width / 16) * 16), int(round(height / 16) * 16)
        scene = None
        if setting and plate is None and "frames" in json.loads(Path(camera_path).read_text()):
            from . import shot as S, scene_layer as L
            try:
                cap = S.Capture(train_dir); scene = L.build(cap, cap.dir / "scene")
            except Exception as e:  # noqa: BLE001 - no real camera or no MoGe: the person alone, as before
                print(f"[sam3dbx] background layer: {e}")
        gpu = _small_gpu()
        _run([TRAIN_PY, Path(__file__).parent / "render_path.py", f"--train_dir={train_dir}", f"--path={camera_path}",
              f"--out={out / 'rgba'}", f"--width={width}", f"--height={height}", f"--look_drop={look_drop}", *([f"--scene={scene}"] if scene else [])],
             out.with_suffix(".log"), _env(gpu))
        from PIL import Image
        files = sorted((out / "rgba").glob("*.png"))
        cam = json.loads(Path(camera_path).read_text())
        fps = cam["fps"]
        if "frames" in cam:   # 4D Director path: derive orbit rows (yaw/pitch/distance about the rig centre) for the plate
            m = json.loads((Path(train_dir) / "preview_meta.json").read_text())
            c, up, fr, sd = (np.array(m[k]) for k in ("centre", "up", "front", "side"))
            cam["rows"] = []
            for f in cam["frames"]:
                rel = np.array(f[1:4]) - c; r = float(np.linalg.norm(rel)) or 1e-9
                cam["rows"].append([f[0], math.degrees(math.atan2(rel @ sd, rel @ fr)), math.degrees(math.asin(np.clip(rel @ up / r, -1, 1))) - m["base_pitch"],
                                    r / m["dist0"], f[7]])
        video = out.with_suffix(".mp4")
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if audio_path.strip():
            cmd += ["-ss", str(audio_start_s), "-i", audio_path.strip(), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "256k", "-shortest"]
        cmd += ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(video)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        # stream in chunks: a full clip as float32 (rgba + plate + composite) overflows ComfyUI's 24 GB memory cgroup
        pw, ph = max(2, width // preview_div), max(2, height // preview_div)
        previews, alphas = [], []
        h = background.lstrip("#"); colour = torch.tensor([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
        still = None
        if plate is not None and plate_mode != "orbit-pan":
            still = plate if plate.shape[1:3] == (height, width) else \
                torch.nn.functional.interpolate(plate.permute(0, 3, 1, 2), size=(height, width), mode="bilinear").permute(0, 2, 3, 1)
        for i in range(0, len(files), 32):
            rgba = torch.from_numpy(np.stack([np.asarray(Image.open(f)) for f in files[i:i + 32]])).float() / 255.0
            rgb, a = rgba[..., :3], rgba[..., 3:]
            if plate is not None and plate_mode == "orbit-pan":
                bg = _orbit_plate(plate[0], cam["rows"][i:i + len(rgb)], width, height, plate_blur)
            elif still is not None:
                bg = still[(torch.arange(i, i + len(rgb))) % len(still)]
            else:
                bg = colour.expand(len(rgb), height, width, 3)
            if plate is not None:
                if edge_choke > 0:             # erode coverage a little: removes the pale matte halo on dark subjects
                    k = int(edge_choke) * 2 + 1
                    ae = -torch.nn.functional.max_pool2d(-a.permute(0, 3, 1, 2), k, 1, k // 2).permute(0, 2, 3, 1)
                    rgb = rgb * (ae / a.clamp(min=1e-4)).clamp(0, 1); a = ae
                rgb, bg = _stage_match(rgb, a, bg, stage_grade, stage_rim, stage_shadow)
            comp = rgb.clamp(0, 1) if scene else (rgb + bg * (1 - a)).clamp(0, 1)   # gsplat colours are premultiplied by coverage; the setting fills behind her
            p.stdin.write((comp.numpy() * 255).astype(np.uint8).tobytes())
            previews.append(torch.nn.functional.interpolate(comp.permute(0, 3, 1, 2), size=(ph, pw), mode="area").permute(0, 2, 3, 1))
            alphas.append(torch.nn.functional.interpolate(a.permute(0, 3, 1, 2), size=(ph, pw), mode="area")[:, 0])
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError("ffmpeg failed writing the render")
        comp, a = torch.cat(previews), torch.cat(alphas)[..., None]
        return {"ui": {"sam3dbx_video": [str(video)]}, "result": (comp, a[..., 0], str(video))}


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {"Sam3dbxView4D": Sam3dbxView4D, "Sam3dbxCameraDirector": Sam3dbxCameraDirector, "Sam3dbxBeatCamera": Sam3dbxBeatCamera, "Sam3dbxRender4DPath": Sam3dbxRender4DPath, "Sam3dbxGenerateViews": Sam3dbxGenerateViews, "Sam3dbxTrain4DGS": Sam3dbxTrain4DGS,
                       "Sam3dbxLoadResult": Sam3dbxLoadResult}
NODE_DISPLAY_NAME_MAPPINGS = {"Sam3dbxView4D": "Reshoot 4D: Splat Viewer", "Sam3dbxCameraDirector": "Reshoot 4D: Virtual Camera",
                              "Sam3dbxBeatCamera": "Reshoot 4D: Beat Camera (legacy)", "Sam3dbxRender4DPath": "Reshoot 4D: Render Shot (splat preview)",
                              "Sam3dbxGenerateViews": "Reshoot 4D: Generate Views", "Sam3dbxTrain4DGS": "Reshoot 4D: Train 4DGS",
                              "Sam3dbxLoadResult": "Reshoot 4D: Load Views"}

from . import nodes_super  # noqa: E402  (super recipe nodes; imports the helpers above)
NODE_CLASS_MAPPINGS.update(nodes_super.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(nodes_super.NODE_DISPLAY_NAME_MAPPINGS)
from . import retake   # noqa: E402  (ReDirector retake + its photoreal pass)
NODE_CLASS_MAPPINGS.update(retake.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(retake.NODE_DISPLAY_NAME_MAPPINGS)
