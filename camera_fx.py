"""Reshoot 4D: Camera FX. Lens and optical effects on a finished clip (a Photoreal Pass, or any video): zoom, lens shift and roll as an
exact crop of the frame, thin-lens depth of field from ComfyUI's MoGe-2 metric depth, vignette and grain. The re-shoot only has to get
the camera's position right; these change in seconds without generating again.

Wired to the Virtual Camera (camera_path) and the Retake, the keys drive everything: their Camera settings (aperture, focus, vignette,
grain), and their lens, applied here as the crop the re-shoot left out. Nothing is applied that nothing asks for.

The work runs in tools/camfx_run.py on the small GPU (config gpu_small) with ComfyUI's own Python.
"""
import json, math, os, queue, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

import numpy as np
import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from . import director as D
from .config import GPUS

CAT = "Reshoot 4D"
WORKER = Path(__file__).parent / "tools" / "camfx_run.py"
COMFY_DIR = Path(folder_paths.base_path)
MOGE, SUBJECT = "moge_2_vitl_normal_fp16.safetensors", "birefnet.safetensors"
TIMEOUT_S = 3600


def _num(v, name, lo, hi):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"Camera FX: {name} must be a number, got {v!r}")
    if not math.isfinite(x) or not lo <= x <= hi:
        raise ValueError(f"Camera FX: {name} must be between {lo} and {hi}, got {v!r}")
    return x


def _check_keys(keys):
    """Keys from the path or the keyframes box: finite times and values in range, focus a positive distance or "subject"."""
    if not isinstance(keys, list) or not all(isinstance(k, dict) and ("t" in k or "f" in k) for k in keys):
        raise ValueError('Camera FX keyframes: a JSON list of keys, each with "t" (0..1) or "f" (frame)')
    for k in keys:
        if "t" in k:
            k["t"] = _num(k["t"], "key time t", 0, 1)
        if "f" in k:
            k["f"] = _num(k["f"], "key frame f", 0, 1e6)
        for n, lo, hi in (("fstop", 0.5, 64), ("vignette", 0, 1), ("grain", 0, 1), ("zoom", 0.25, 8), ("roll", -180, 180),
                          ("shift_x", -1, 1), ("shift_y", -1, 1), ("distortion", -0.3, 0.3), ("chromatic_aberration", 0, 1),
                          ("bloom", 0, 1), ("halation", 0, 1)):
            if n in k:
                k[n] = _num(k[n], n, lo, hi)
        if "focus" in k and k["focus"] != "subject":
            k["focus"] = _num(k["focus"], 'focus (metres or "subject")', 0.01, 1e5)
        if k.get("ease", "smooth") not in ("smooth", "linear", "hold"):
            raise ValueError(f"Camera FX: ease must be smooth, linear or hold, got {k['ease']!r}")
    return keys


def _shm_or_disk(nbytes):
    """Shared memory when it has room (fast), else a disk temp folder: small /dev/shm in containers would kill the worker."""
    shm = Path("/dev/shm")
    if shm.is_dir() and shutil.disk_usage(shm).free > nbytes * 1.5:
        return str(shm)
    return None


class Sam3dbxCameraFX:
    @classmethod
    def INPUT_TYPES(cls):
        F = lambda d, lo, hi, st, tip, adv=False: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": st, "tooltip": tip, "advanced": adv})
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The finished clip (Photoreal Pass output or any video)"}),
            "lens_mm": F(0.0, 0, 300, 0.1, "The clip's focal length, measured on a 36 mm tall frame as in the Virtual Camera editor: sets the depth scale and blur strength. "
                         "0 = from the Virtual Camera's first key (or 50 mm without one)"),
            "zoom_start": F(1.0, 1.0, 4.0, 0.01, "Without a Retake: focal length at the start, relative to the clip (2 = twice, a crop)"),
            "zoom_end": F(1.0, 1.0, 4.0, 0.01, "Without a Retake: focal length at the end"),
            "roll_start": F(0.0, -45, 45, 0.5, "Extra roll at the start (degrees)"),
            "roll_end": F(0.0, -45, 45, 0.5, "Extra roll at the end (degrees)"),
            "depth_of_field": (["auto", "on", "off"], {"default": "auto", "tooltip": "auto: on when the keys set an aperture or focus. on: always, with the aperture below unless keyed"}),
            "fstop": F(2.8, 0.7, 22, 0.1, "Aperture when depth of field is on and no key sets one: lower = shallower focus"),
            "focus_m": F(0.0, 0.0, 100, 0.05, "Focus distance in metres when no key sets one. 0 = the subject"),
            "vignette": F(0.0, 0.0, 1.0, 0.01, "Darkening towards the corners (keys override)"),
            "grain": F(0.0, 0.0, 1.0, 0.01, "Film grain (keys override)"),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": False})},
            "optional": {
                "zoom_aim": (["subject", "centre"], {"default": "subject", "tooltip": "subject: zooms aim at the person (kept in frame, steadied). centre: a true lens, zooming to the middle of the frame. The Virtual Camera's Shot setting overrides"}),
                "shift_x": F(0.0, -0.5, 0.5, 0.005, "Lens shift across (fraction of the frame), no perspective change", True),
                "shift_y": F(0.0, -0.5, 0.5, 0.005, "Lens shift up/down (fraction of the frame)", True),
                "max_blur_px": F(40.0, 4, 80, 1, "Cap on the blur radius (pixels)", True),
                "distortion": F(0.0, -0.3, 0.3, 0.01, "Lens distortion: + barrel (wide-angle bulge), - pincushion. The Virtual Camera's Shot setting overrides", True),
                "chromatic_aberration": F(0.0, 0.0, 1.0, 0.01, "Colour fringing towards the edges", True),
                "bloom": F(0.0, 0.0, 1.0, 0.01, "Highlights spread into a soft glow", True),
                "halation": F(0.0, 0.0, 1.0, 0.01, "Warm red glow around bright edges, as on film", True),
                "anamorphic": F(1.0, 1.0, 2.0, 0.05, "Oval bokeh: out-of-focus blur this much taller than wide (1 = round)", True),
                "passes": ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Also output the depth pass (near = white) and the subject matte, for compositing"}),
                "edge_matte": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Keep the person's edges (hair, shoulders) out of the background blur using a subject matte"}),
                "keyframes": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional, JSON, overrides the path's Camera settings. t = 0..1 across the clip (or f = frame). "
                    'Each setting moves between the keys that set it; focus is metres or "subject". ease: smooth / linear / hold. Example: '
                    '[{"t":0,"focus":"subject","fstop":4},{"t":0.5,"focus":30,"fstop":0.9},{"t":1,"focus":"subject","fstop":4}]'}),
                "matte": ("MASK", {"tooltip": "Optional subject matte from any matting node (e.g. a video matting model). Empty = BiRefNet's cutout"}),
                "retake": ("STRING", {"forceInput": True, "tooltip": "The Retake job file this clip came from: the lens the re-shoot was rendered at, so the keys' lens change is applied here as a crop"}),
                "camera_path": ("STRING", {"forceInput": True, "tooltip": "The Virtual Camera move: its keys' lens and Camera settings (aperture, focus, vignette, grain) and zoom aim drive the effects"})},
            "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE", "MASK"); RETURN_NAMES = ("images", "blur_map", "info", "depth", "matte")
    OUTPUT_TOOLTIPS = ("The clip with the camera effects", "Blur radius per pixel (white = most blur), to check the focus", "What was applied",
                       "Depth pass, near = white (with passes on)", "Subject matte (with passes on)")
    FUNCTION = "run"; CATEGORY = CAT

    def run(self, images, lens_mm, zoom_start, zoom_end, roll_start, roll_end, depth_of_field, fstop, focus_m, vignette, grain, seed,
            zoom_aim="subject", shift_x=0.0, shift_y=0.0, max_blur_px=40.0, distortion=0.0, chromatic_aberration=0.0, bloom=0.0, halation=0.0,
            anamorphic=1.0, passes=False, edge_matte=True, keyframes="", matte=None, retake=None, camera_path=None, unique_id=None):
        N = images.shape[0]
        P = {"zoom_start": zoom_start, "zoom_end": zoom_end, "roll_start": roll_start, "roll_end": roll_end,
             "shift_x_start": shift_x, "shift_x_end": shift_x, "shift_y_start": shift_y, "shift_y_end": shift_y,
             "fstop": fstop, "focus_m": focus_m, "vignette": vignette, "grain": grain, "seed": seed,
             "max_blur_px": max_blur_px, "edge_matte": bool(edge_matte), "zoom_aim": zoom_aim, "distortion": distortion,
             "chromatic_aberration": chromatic_aberration, "bloom": bloom, "halation": halation, "anamorphic": anamorphic, "passes": bool(passes)}
        lens = lens_mm if lens_mm > 0 else None
        if camera_path:
            cam = json.loads(Path(camera_path).read_text()); st = cam.get("state") or cam
            keys = sorted(st.get("keys") or [], key=lambda k: k["t"])
            if keys:
                dur = max(float(st.get("duration") or keys[-1]["t"] or 1.0), 1e-6)
                # each output frame at the camera time the Retake rendered it at (whole-shot easing included), so the keys' Camera
                # settings land on the same frames as the move
                ct = [D.shot_time(st, keys, dur * i / max(N - 1, 1)) for i in range(N)]
                P["u"] = [min(1.0, max(0.0, c / dur)) for c in ct]
                P["keys"] = [dict({"t": min(1.0, max(0.0, float(k["t"]) / dur)), "ease": k.get("ease", "smooth")},
                                  **{n: k[n] for n in ("fstop", "focus", "vignette", "grain") if n in k}) for k in keys]
                P["keys"] = [k for k in P["keys"] if len(k) > 2] or None
                lens = lens or float(keys[0].get("lens") or 50.0)
                if retake:   # the keys' lens now, over what the re-shoot was rendered at: edits to lens keys need no new render
                    z = np.load(json.loads(Path(retake).read_text())["npz"])
                    fov = np.radians([D.sample(st, dur * i / max(N - 1, 1))[8] for i in range(N)])
                    want = np.tan(fov[0] / 2) / np.tan(fov / 2)
                    got = z["retake_zoom"] if "retake_zoom" in z.files else (z["zoom"] if "zoom" in z.files else np.ones(1))
                    got = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(got)), got)
                    zc = want / got
                    if zc.min() < 0.98:
                        raise ValueError(f"Camera FX: the lens keys now go {1 / zc.min():.2f}x wider than the Retake was rendered at. "
                                         "A wider lens needs new picture: queue the Retake again.")
                    if np.ptp(zc) + abs(float(zc[0]) - 1) > 1e-3:
                        P["zoom"] = np.maximum(zc, 1.0).tolist()
            if st.get("zoom_aim"):
                P["zoom_aim"] = st["zoom_aim"]
            for n, v in (st.get("lens_fx") or {}).items():   # the shot's lens character from the Virtual Camera
                if n in ("distortion", "chromatic_aberration", "bloom", "halation", "anamorphic"):
                    P[n] = _num(v, n, -1, 2)
        elif retake:   # a Retake without its path: the crop it saved
            z = np.load(json.loads(Path(retake).read_text())["npz"])
            zc = z["post_zoom"] if "post_zoom" in z.files else None
            if zc is not None and np.ptp(zc) + abs(float(zc[0]) - 1) > 1e-3:
                P["zoom"] = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(zc)), zc).tolist()
        if keyframes and keyframes.strip():   # typed keys override the path's
            try:
                P["keys"] = json.loads(keyframes)
            except json.JSONDecodeError as e:
                raise ValueError(f"Camera FX keyframes are not valid JSON: {e}")
            P.pop("u", None)
        if P.get("keys"):
            _check_keys(P["keys"])
        P["lens_mm"] = lens or 50.0
        keyed = lambda n: any(n in k for k in P.get("keys") or [])
        P["dof"] = depth_of_field == "on" or (depth_of_field == "auto" and (keyed("fstop") or keyed("focus")))
        lens_fx = "zoom" in P or zoom_start != 1 or zoom_end != 1 or roll_start or roll_end or shift_x or shift_y or keyed("zoom") or keyed("roll")
        look = any(P[n] for n in ("distortion", "chromatic_aberration", "bloom", "halation")) or any(
            keyed(n) for n in ("vignette", "grain", "distortion", "chromatic_aberration", "bloom", "halation"))
        H, W = images.shape[1:3]
        empty = (torch.zeros(1, H, W, 3), torch.zeros(1, H, W))
        if not (P["dof"] or lens_fx or vignette > 0 or grain > 0 or look or P["passes"]):
            info = {"frames": N, "applied": "nothing: no Camera settings on the keys and none set here"}
            return (images, torch.zeros_like(images), json.dumps(info)) + empty
        if GPUS["small"] == GPUS["big"]:   # one card: make room for the depth and matte models
            mm.unload_all_models(); mm.soft_empty_cache()
        # model files as this server resolves them (extra model paths); the worker loads each only when it needs it, and says which is missing
        P["moge_path"] = folder_paths.get_full_path("geometry_estimation", MOGE)
        P["subject_path"] = folder_paths.get_full_path("background_removal", SUBJECT)
        job = Path(tempfile.mkdtemp(prefix="camfx_", dir=_shm_or_disk(2 * N * H * W * 3 + N * H * W * 2)))
        p = None
        try:
            np.save(job / "frames.npy", (images[..., :3].clamp(0, 1) * 255 + 0.5).byte().numpy())
            (job / "params.json").write_text(json.dumps(P))
            if matte is not None:
                mt = matte.float().clamp(0, 1)
                if mt.shape[0] != N:   # a single matte (or a shorter clip of them) is stretched over the clip
                    mt = mt[torch.linspace(0, mt.shape[0] - 1, N).round().long()]
                np.save(job / "matte.npy", mt.numpy().astype(np.float16))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS["small"], PYTHONUNBUFFERED="1")
            pbar = comfy.utils.ProgressBar(100)
            p = subprocess.Popen([sys.executable, str(WORKER), str(COMFY_DIR), str(job)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, env=env, cwd=str(COMFY_DIR))
            lines = queue.Queue()
            threading.Thread(target=lambda: [lines.put(l) for l in p.stdout] + [lines.put(None)], daemon=True).start()
            log, t0, done = [], time.time(), False
            while not done:
                mm.throw_exception_if_processing_interrupted()   # Cancel in ComfyUI stops the worker (finally below)
                if time.time() - t0 > TIMEOUT_S:
                    raise RuntimeError(f"Camera FX: no result after {TIMEOUT_S // 60} min, stopped")
                try:
                    line = lines.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    done = True; continue
                log.append(line.rstrip())
                if line.startswith("STEP "):
                    i, n = map(int, line.split()[1:3]); pbar.update_absolute(int(100 * (i + 1) / max(n, 1)), 100)
            if p.wait() != 0 or not (job / "info.json").is_file():
                raise RuntimeError("Camera FX failed:\n" + "\n".join(log[-12:]))
            out = torch.from_numpy(np.load(job / "out.npy")).float().div(255)
            blur = torch.from_numpy(np.load(job / "blur.npy"))
            blur = torch.nn.functional.interpolate(blur[:, None], (H, W), mode="bilinear", align_corners=False)[:, 0]
            blur = (blur / max(float(blur.max()), 1e-6))[..., None].repeat(1, 1, 1, 3)
            up = lambda a: torch.nn.functional.interpolate(torch.from_numpy(a.astype(np.float32))[:, None], (H, W), mode="bilinear", align_corners=False)[:, 0]
            dp = (up(np.load(job / "depth_pass.npy"))[..., None].repeat(1, 1, 1, 3),) if (job / "depth_pass.npy").is_file() else (empty[0],)
            mp = (up(np.load(job / "matte_pass.npy") / 255.0).clamp(0, 1),) if (job / "matte_pass.npy").is_file() else (empty[1],)
            return (out, blur, (job / "info.json").read_text()) + dp + mp
        finally:
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
            shutil.rmtree(job, ignore_errors=True)


def _source_audio(retake, seconds):
    """The original video's audio over the captured trim (clip.json), for a retake that plays in real time. None when there is none."""
    try:
        z = np.load(json.loads(Path(retake).read_text())["npz"])
        cap = Path(str(z["video"])).parent
        clip = json.loads((cap / "clip.json").read_text())
        src, start = clip["video"], float(clip.get("start", 0.0))
    except Exception:  # noqa: BLE001 - no clip record: nothing to add
        return None
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels", "-of", "csv=p=0", src],
                           capture_output=True, text=True)
    if probe.returncode or not probe.stdout.strip():
        return None
    ch = max(1, min(2, int(probe.stdout.strip().split(",")[0] or 2))); sr = 48000
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", src, "-vn", "-ac", str(ch), "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True).stdout
    if not raw:
        return None
    wav = torch.from_numpy(np.frombuffer(raw, np.float32).copy()).view(-1, ch).t()[None]
    return {"waveform": wav, "sample_rate": sr}


class Sam3dbxFinish:
    """Optional finishing for delivery: a standard frame rate with RIFE in-between frames (the ComfyUI-Frame-Interpolation pack's RIFE VFI,
    MIT), motion blur from a shutter angle (in-between frames averaged over the shutter), a lanczos resize, the source audio, and the camera
    path saved beside the output. The clip keeps its real-time length."""
    FITS = {"off": None, "fit 1080p": 1920, "fit 1440p": 2560, "fit 2160p": 3840}
    RATES = ["source", "24", "25", "30", "48", "50", "60"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The clip (Camera FX or Photoreal Pass output)"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01, "forceInput": True, "tooltip": "The clip's frame rate (Photoreal Pass fps)"}),
            "frame_rate": (cls.RATES, {"default": "source", "tooltip": "Output frame rate. Faster than the clip = RIFE in-between frames; same length either way"}),
            "shutter_angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0, "tooltip": "Motion blur: 0 = off, 180 = film look, 360 = smear. The Virtual Camera's Shot setting overrides"}),
            "upscale": (list(cls.FITS), {"default": "off", "tooltip": "Lanczos resize so the long side fits 1920 / 2560 / 3840 px (keeps the shape)"})},
            "optional": {
                "retake": ("STRING", {"forceInput": True, "tooltip": "The Retake job file: its source video's audio comes out on the audio output"}),
                "camera_path": ("STRING", {"forceInput": True, "tooltip": "The Virtual Camera move: its shutter angle, and a copy saved beside the output"}),
                "filename_prefix": ("STRING", {"default": "reshoot4d/final", "tooltip": "Where the camera path copy is saved (match the video save node)"}),
                "rife_model": ("STRING", {"default": "rife49.pth", "advanced": True, "tooltip": "RIFE checkpoint name in ComfyUI-Frame-Interpolation"})}}

    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO"); RETURN_NAMES = ("images", "fps", "audio")
    OUTPUT_TOOLTIPS = ("The finished clip", "Its frame rate (connect to the video save node)", "The source video's audio over the clip (empty if none)")
    FUNCTION = "run"; CATEGORY = CAT

    @staticmethod
    def _dense(frames, m, model):
        """RIFE in-betweens ((N - 1) * m + 1 frames in all) as uint8 chunks (start index, frames): only one short chunk is in memory at a time."""
        n = frames.shape[0]
        if m == 1 or n == 1:   # nothing to interpolate (a single frame is held)
            for i in range(0, n, 16):
                mm.throw_exception_if_processing_interrupted()
                yield i, frames[i:i + 16].clamp(0, 1).mul(255).round().to(torch.uint8)
            return
        import nodes
        rife = nodes.NODE_CLASS_MAPPINGS.get("RIFE VFI")
        if rife is None:
            raise RuntimeError("Finish: a faster frame rate or motion blur needs the ComfyUI-Frame-Interpolation pack (RIFE VFI). "
                               "Install it from the Manager, or set frame_rate to source and shutter_angle to 0.")
        C = max(1, 48 // m)   # about 48 in-betweens per chunk
        for i in range(0, n - 1, C):
            mm.throw_exception_if_processing_interrupted()
            d = rife().vfi(model, frames[i:min(n, i + C + 1)], clear_cache_after_n_frames=10, multiplier=m, fast_mode=True, ensemble=True)[0]
            d = d if i + C + 1 >= n else d[:-1]
            yield i * m, d.clamp_(0, 1).mul_(255).round_().to(torch.uint8)
            del d

    def run(self, images, fps, frame_rate, shutter_angle, upscale, retake=None, camera_path=None, filename_prefix="reshoot4d/final",
            rife_model="rife49.pth"):
        n, src = images.shape[0], float(fps)
        st = None
        if camera_path:
            cam = json.loads(Path(camera_path).read_text()); st = cam.get("state") or cam
            if st.get("shutter_angle") is not None:
                shutter_angle = float(st["shutter_angle"])
        dur = n / src
        target = src if frame_rate == "source" else float(frame_rate)
        angle = max(0.0, min(360.0, float(shutter_angle)))
        # in-between density: enough for the target rate, and about 6 samples inside each open shutter
        need = max(target, 6.0 * target * 360.0 / angle) if angle > 0 else target
        m = 1 if need <= src * 1.001 else min(16, int(math.ceil(need / src)))
        if m == 1 and abs(target - src) < 1e-3 and angle == 0:
            out = images
        else:
            # each output frame is a weighted sum of in-between frames: what the shutter sees (centred on the frame time), or the two
            # around it. The in-betweens stream through in chunks, so memory stays at one chunk plus the output
            rate = src * m; nd = (n - 1) * m + 1 if m > 1 and n > 1 else n
            k_out = max(1, int(round(dur * target)))
            W = []
            for k in range(k_out):
                t = k / target
                if angle > 0:
                    w = angle / 360.0 / target
                    b = min(nd - 1, int(math.ceil((t + w / 2) * rate))); a = min(b, max(0, int(math.floor((t - w / 2) * rate))))
                    W.append([(j, 1.0 / (b - a + 1)) for j in range(a, b + 1)])
                else:
                    x = min(t * rate, nd - 1); a = int(math.floor(x)); u = x - a; b = min(a + 1, nd - 1)
                    W.append([(a, 1 - u)] if b == a else [(a, 1 - u), (b, u)])
            out = torch.empty((k_out,) + tuple(images.shape[1:]), dtype=torch.float32)
            acc, k0 = {}, 0
            for s0, c in self._dense(images, m, rife_model):
                e = s0 + c.shape[0]
                for k in range(k0, k_out):
                    if W[k][0][0] >= e:
                        break
                    if W[k][-1][0] < s0:   # finished in an earlier chunk
                        continue
                    for j, wt in W[k]:
                        if s0 <= j < e:
                            f = c[j - s0].float().mul_(wt)
                            acc[k] = f if k not in acc else acc[k].add_(f)
                    if W[k][-1][0] < e:
                        out[k] = acc.pop(k).div_(255)
                while k0 < k_out and W[k0][-1][0] < e:
                    k0 += 1
                del c
            if k0 < k_out:
                raise RuntimeError(f"Finish: {k_out - k0} output frames were not filled")
        long = self.FITS[upscale]
        if long:
            h, w = out.shape[1:3]; sc = long / max(h, w)
            if abs(sc - 1) > 1e-3:
                nh, nw = round(h * sc / 2) * 2, round(w * sc / 2) * 2
                big = torch.empty((out.shape[0], nh, nw, out.shape[3]), dtype=torch.float32)
                for i in range(0, out.shape[0], 8):
                    mm.throw_exception_if_processing_interrupted()
                    big[i:i + 8] = comfy.utils.common_upscale(out[i:i + 8].movedim(-1, 1), nw, nh, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
                out = big
        if camera_path:   # the move beside the output, so a result can be reproduced
            root = Path(folder_paths.get_output_directory()).resolve()
            o = (root / f"{filename_prefix}_{time.strftime('%Y%m%d-%H%M%S')}_camera.json").resolve()
            if root not in o.parents:
                raise ValueError(f"Finish: filename_prefix must stay inside ComfyUI's output folder, not {filename_prefix!r}")
            o.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(camera_path, o)
        audio = _source_audio(retake, dur) if retake else None
        return (out, float(target), audio)


NODE_CLASS_MAPPINGS = {"Sam3dbxCameraFX": Sam3dbxCameraFX, "Sam3dbxFinish": Sam3dbxFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"Sam3dbxCameraFX": "Reshoot 4D: Camera FX", "Sam3dbxFinish": "Reshoot 4D: Finish (frame rate, motion blur, upscale, audio)"}
