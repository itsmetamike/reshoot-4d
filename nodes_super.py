"""Reshoot 4D nodes: Capture (video -> 4D capture, 4DGS splat by default) -> Virtual Camera (your camera move) -> Render Shot,
and the optional realism pass: Photoreal Prep (depth + identity anchors) -> Photoreal Pass (LTX 2.3).

Photoreal Pass is a node expansion: at run time it becomes the full LTX 2.3 graph for every segment of the shot (built by
ltx_graph.py), all inside the same prompt. Identical loader nodes across segments are merged so the models load once; each
segment after the first starts from the previous one's last frame; the segments are trimmed and joined into one IMAGE batch.
"""
import hashlib, json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import comfy.model_management as mm
import folder_paths
import nodes

from . import ltx_graph as sg
from . import shot as S
from . import capture_resolve as CR
from . import scene_layer as L
_pkg = sys.modules[__package__]   # the package __init__ imports this module after defining its helpers
GEN_PY, TRAIN_PY, ROOT, _env, _free_comfy, _read_frames, _small_gpu = (_pkg.GEN_PY, _pkg.TRAIN_PY, _pkg.ROOT, _pkg._env, _pkg._free_comfy, _pkg._read_frames, _pkg._small_gpu)
CAT, CAT_ADV, CAT_LEGACY, _failed = _pkg.CAT, _pkg.CAT_ADV, _pkg.CAT_LEGACY, _pkg._failed

from .config import UPSTREAM, UP_PY  # noqa: E402
NODES_DIR = Path(__file__).resolve().parent
FAST_OUT = ROOT / "out" / "comfy"


def _shot_size(aspect, budget=1024 * 1792, m=128):
    """'W:H' -> 'WxH' for the photoreal pass: the closest ratio in multiples of 128 (IC-LoRA control) within the 1024x1792 pixel budget,
    so any shot aspect the Director picks (9:16, 4:5, 1:1, 16:9 ...) renders at the same cost."""
    a, b = (float(v) for v in str(aspect).split(":")); r = a / b
    best = min(((w, h) for w in range(m, 4097, m) for h in range(m, 4097, m) if w * h <= budget),
               key=lambda wh: (abs(math.log(wh[0] / wh[1] / r)) > 0.04, -wh[0] * wh[1], abs(math.log(wh[0] / wh[1] / r))))
    return f"{best[0]}x{best[1]}"


def _bars(video, start, secs=4.84):
    """ffmpeg crop filter (with trailing comma) removing black bars that stay put over the capture window, or "" if there are none.
    cropdetect keeps the largest picture area it sees across the window, so a dark frame can't crop real content."""
    r = subprocess.run(["ffmpeg", "-v", "info", "-ss", str(start), "-t", str(secs), "-i", str(video), "-vf", "cropdetect=limit=24:round=2:reset=0",
                        "-f", "null", "-"], capture_output=True, text=True)
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
                           capture_output=True, text=True).stdout.strip().split(",")
    if not found or len(probe) < 2:
        return ""
    w, h, x, y = map(int, found[-1]); W, H = int(probe[0]), int(probe[1])
    return f"crop={w}:{h}:{x}:{y}," if w * h < 0.96 * W * H and w > 0.3 * W and h > 0.3 * H else ""   # only real bars, never a near-black clip


def _clip_key(video, start, end, views):
    """Identity of a capture request: the video's bytes (size + head + tail, so a re-upload under another name still matches), the trim and the views."""
    p = Path(video); h = hashlib.sha1(f"{p.stat().st_size}:{start:.3f}:{end:.3f}:{views}".encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20)); f.seek(max(0, p.stat().st_size - (1 << 20))); h.update(f.read(1 << 20))
    return h.hexdigest()


def _sh(cmd, log, env, cwd):
    # own process group, polled for ComfyUI's Cancel: the stages spawn their own workers, so Cancel stops the whole group
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n"); f.flush()
        p = subprocess.Popen(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=str(cwd), env=env, start_new_session=True)
        while p.poll() is None:
            if mm.processing_interrupted():
                try: os.killpg(p.pid, 15)
                except ProcessLookupError: pass
                try: p.wait(10)
                except subprocess.TimeoutExpired: pass
                try: os.killpg(p.pid, 9)   # whatever in the group outlived the grace period
                except ProcessLookupError: pass
                p.wait(); f.write("\n(cancelled)\n"); mm.throw_exception_if_processing_interrupted()
            time.sleep(1)
    if p.returncode != 0:
        raise _failed(cmd, p.returncode, log)


class Sam3dbxCaptureSubject:
    """Turn a video of one person into a 4D capture. Pick the clip, trim it in the preview, queue. Then open the capture in the
    Virtual Camera. Queueing again with the same clip and trim reuses the capture."""
    MODES = ["4DGS splat (train)", "mannequin only (no training)", "quick (for Retake: no views, no splat)"]

    @classmethod
    def INPUT_TYPES(cls):
        inp = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types([f for f in os.listdir(inp) if os.path.isfile(os.path.join(inp, f))], ["video"])
        return {"required": {"video": (sorted(files), {"video_upload": True, "tooltip": "Any clip of one person: any size, aspect or frame rate"}),
                             "name": ("STRING", {"default": "", "tooltip": "Capture name (its folder). Empty = the video's file name"}),
                             "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01, "tooltip": "Trim start (s). The preview loops the trimmed part"}),
                             "end_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01, "tooltip": "Trim end (s). 0 = 4.84 s from the start. Any length works: about 2 to 8 s keeps motion smooth"}),
                             "mode": (cls.MODES, {"default": cls.MODES[0], "tooltip": "4DGS splat: generated views + a trained splat (~1 h). mannequin: generated views, no training. "
                                      "quick: tracking, real camera and setting only (~4 min), everything Retake needs, with a mannequin preview in the Virtual Camera"}),
                             "views": ([24, 48], {"default": 24, "tooltip": "Generated views around the person. 48 adds higher and lower rings: cleaner from high angles"}),
                             "train_steps": ("INT", {"default": 15000, "min": 2000, "max": 60000, "step": 1000, "tooltip": "4DGS only: more steps, a sharper splat"})},
                "optional": {"subject": (["largest", "left", "right"], {"default": "largest", "tooltip": "Who to capture when the clip has more than one person: the biggest, or the left-most / right-most. The others stay in the setting"})}}
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "IMAGE", "IMAGE"); RETURN_NAMES = ("capture_dir", "views_grid", "info", "trimmed_clip", "splat_preview")
    OUTPUT_TOOLTIPS = ("The capture, for the Virtual Camera", "Six of the generated views over time", "Mode, trim and views",
                       "The 121 frames the capture is built from (after trim, bar crop and resample)",
                       "The trained 4D splat seen from its training cameras (one black frame in mannequin mode)")
    FUNCTION = "run"; CATEGORY = CAT; OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, video, **kw):
        return os.path.getmtime(folder_paths.get_annotated_filepath(video))

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kw):
        return True if folder_paths.exists_annotated_filepath(video) else f"Invalid video file: {video}"

    def run(self, video, start_time, end_time, mode, views, train_steps, name="", subject="largest"):
        preview_height, splat_mode, quick = 320, mode == self.MODES[0], mode == self.MODES[2]
        src_in = folder_paths.get_annotated_filepath(video)
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip() or Path(src_in).stem)[:40] or "clip"
        ts = time.strftime("%Y%m%d-%H%M%S"); run = FAST_OUT / f"cap_{name}_{ts}"; run.mkdir(parents=True); log = run / "capture.log"
        src = run / "source_121.mp4"
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate:format=duration", "-of", "json", src_in],
                               capture_output=True, text=True)
        pj = json.loads(probe.stdout or "{}"); total = float(pj.get("format", {}).get("duration", 0) or 0)
        num, den = (pj.get("streams") or [{}])[0].get("r_frame_rate", "25/1").split("/"); src_fps = float(num) / float(den or 1)
        end = end_time if end_time > start_time else start_time + 121 / 25
        if total:
            end = min(end, total)
        secs = end - start_time
        if secs < 0.5:
            raise ValueError(f"The trimmed clip is {secs:.2f} s (video is {total:.2f} s long): pick a start and end at least 0.5 s apart.")
        # the same clip, trim and views were captured before (e.g. queued again after a restart cleared ComfyUI's cache): reuse it
        key = _clip_key(src_in, start_time, end, views) + ("" if subject == "largest" else f"-{subject}") + ("-quick" if quick else "")
        old = next((c.parent for c in sorted(FAST_OUT.glob("cap_*/clip.json"), reverse=True)
                    if json.loads(c.read_text()).get("key") == key and (c.parent / "capture" / "preview_meta.json").is_file()
                    and (c.parent / "capture" / "source_camera.json").is_file()), None)
        t = time.time(); cap = run / "capture"
        if old:
            shutil.rmtree(run); run, log, cap = old, old / "capture.log", old / "capture"
        else:
            # 121 frames spread over the trimmed clip, then re-clocked to 25 fps for 4DAnyone. Too few source frames: motion-interpolate
            # instead of repeating frames (repeats read as stutter in the views and the splat)
            rate = 121 / secs
            pick = f"fps={rate:.6f}" if secs * src_fps >= 121 else f"minterpolate=fps={rate:.6f}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
            crop = _bars(src_in, start_time, secs)   # letterbox / pillarbox bars are not part of the scene: the views and the shot would copy them
            _sh(["ffmpeg", "-v", "error", "-y", "-ss", str(start_time), "-t", f"{secs + 2 / rate:.4f}", "-i", src_in, "-frames:v", "121", "-vf",
                            f"{crop}scale=-2:1280:flags=lanczos,{pick},tpad=stop_mode=clone:stop=3,setpts=N/25/TB", "-r", "25", "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(src)], log, os.environ.copy(), run)
            n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(src)],
                                   capture_output=True, text=True).stdout.strip() or 0)
            if n < 121:
                raise ValueError(f"Only {n} of the 121 frames came out of {src_in} between {start_time:.2f} and {end:.2f} s.")
            # the real length of the clip: the Director plays the 121 frames back over this, so shots run in real time
            (run / "clip.json").write_text(json.dumps({"video": src_in, "start": start_time, "end": end, "seconds": secs, "source_fps": src_fps, "key": key}))
            _free_comfy()
            env = _env("3090"); env["PATH"] = f"{ROOT / 'uvenv' / 'bin'}:{env['PATH']}"
            if subject != "largest":
                env["SAM3DBX_SUBJECT"] = subject   # patches/upstream_subject_pick.patch
            if quick:
                env["SAM3DBX_SKIP_GENERATION"] = "1"   # patches/upstream_quick_capture.patch: skeleton renders stand in for the views
            rings = ["--views_per_layer=24"] if views == 24 else ["--views_per_layer=16", "--layer_pitches=[-10,15,35]"]
            _sh([UP_PY, "inference.py", f"--video_path={src}", f"--output_dir={run / 'upstream'}", *rings, "--gpu_ids=[0]"], log, env, UPSTREAM)
            res = next((p.parent for p in (run / "upstream").rglob("cameras.json")), None)
            if res is None:
                raise RuntimeError(f"no cameras.json under {run / 'upstream'}; log {log}")
            _sh([UP_PY, NODES_DIR / "upstream_adapter.py", f"--upstream_result={res}", f"--dest={run / 'adapt'}", f"--upstream_root={UPSTREAM}"], log, env, UPSTREAM)
            cap = run / "capture"
            _sh([GEN_PY, NODES_DIR / "scripts" / "fast_rig.py", run / "adapt", cap], log, env, UPSTREAM)
            _sh([UP_PY, NODES_DIR / "source_cam.py", cap], log, env, NODES_DIR)   # the real camera, for shots that start on it + real-frame anchors
        sp = run / f"splat_{train_steps // 1000}k"
        if splat_mode and not any(sp.glob("ckpts/ckpt_*.pt")):
            _free_comfy()
            _sh([GEN_PY, "train_4dgs.py", f"--result_dir={run / 'adapt'}", f"--output_dir={sp}", "--gpu_id=0", f"--steps={train_steps}"],
                log, _env("3090", train=True), ROOT / "repo")
            _sh([TRAIN_PY, NODES_DIR / "dump_points.py", f"--train_dir={sp}"], log, {**_env("3090", train=True), "CUDA_VISIBLE_DEVICES": ""}, NODES_DIR)
        L.ensure(sp if splat_mode else cap)   # the clip's setting and the source camera's real path (DA3): shots work in the setting's frame
        try:   # Virtual Camera preview: the real setting around her, and real colours on a quick capture's mannequin
            from . import preview_extras
            print("[sam3dbx] preview extras", preview_extras.run(sp if splat_mode else cap))
        except Exception as e:  # noqa: BLE001 - the plain preview still works
            print(f"[sam3dbx] preview extras: {e}")
        vids = sorted((run / "adapt" / "videos").glob("*.mp4"))[:: max(1, views // 6)][:6]
        clips = [_read_frames(v, preview_height, stride=4) for v in vids]; k = min(len(c) for c in clips)
        info = f"{mode}: {secs:.2f} s clip ({start_time:.2f}-{end:.2f} s), {views} views{f' + {train_steps} training steps' if splat_mode else ''} -> {cap}"
        trimmed = _read_frames(run / "source_121.mp4", preview_height)
        traj = sorted((sp / "videos").glob("traj_4d_step*.mp4"))
        splat = _read_frames(traj[-1], preview_height) if splat_mode and traj else torch.zeros(1, 64, 64, 3)
        return {"ui": {"sam3dbx_capture": [str(cap)]}, "result": (str(cap), torch.cat([c[:k] for c in clips], 2), info, trimmed, splat)}


class Sam3dbxShotPrep:
    """Optional realism pass, step 1: depth and identity anchors along the camera move, for Photoreal Pass (LTX)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"capture_dir": ("STRING", {"forceInput": True, "tooltip": "Capture from the Virtual Camera"}),
                             "camera_path": ("STRING", {"forceInput": True, "tooltip": "Camera move from the Virtual Camera"}),
                             "depth_from": (["auto", "splat", "mannequin"], {"default": "auto", "advanced": True, "tooltip": "auto uses the splat when the capture is trained"}),
                             "chunk_seconds": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 5.0, "step": 0.5, "advanced": True, "tooltip": "Longest LTX segment (s). Lower it if the GPU runs out of memory"}),
                             "anchor_every": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.5, "advanced": True, "tooltip": "Extra identity anchors every N s. 0 = start, middle and end only"}),
                             "anchor_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "Strength of the extra anchors"}),
                             "resolution": (["auto", "768x1280", "1024x1792", "1280x768", "1792x1024"], {"default": "auto", "tooltip": "Shot size. auto follows the shot aspect at the same pixel cost"}),
                             "zoom_views": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Use close-up views when the capture has them"})},
                "optional": {"generated_anchors": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "Scale for anchors cut from generated views. Real-frame anchors always stay strong. 0 = real frames only"}),
                             "free_orbit_set": ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Experimental: rebuild the clip's setting in 3D (Depth Anything 3) so "
                                                "moves far from the real camera keep a background. Off = moves near the real camera, carried by the footage itself"})}}
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "STRING"); RETURN_NAMES = ("shot", "anchors", "depth", "info")
    OUTPUT_TOOLTIPS = ("The prepared shot, for Photoreal Pass", "Identity anchors (preview)", "Depth along the move (preview)", "Frames, segments and anchors")
    FUNCTION = "run"; CATEGORY = CAT_LEGACY; OUTPUT_NODE = True; DEPRECATED = True

    def run(self, capture_dir, camera_path, depth_from, chunk_seconds, anchor_every, anchor_strength, resolution="auto", zoom_views=True,
            generated_anchors=1.0, free_orbit_set=False):
        note = ""
        if not CR.source_camera(capture_dir):   # captures from before source_cam.py: solve the real camera now
            try:
                _sh([UP_PY, NODES_DIR / "source_cam.py", capture_dir], Path(capture_dir) / "source_cam.log", _env(_small_gpu()), NODES_DIR)
            except Exception as e:  # noqa: BLE001 - no real camera only means generated-view anchors, as before
                print(f"[sam3dbx] source camera: {e}"); note = ", real camera not solved (generated-view anchors only)"
        cp = json.loads(Path(camera_path).read_text()); rows = cp["frames"]; fps = float(cp.get("fps", S.FPS))
        if resolution == "auto":
            resolution = _shot_size((cp.get("state") or {}).get("aspect") or "9:16")
        cap = S.Capture(capture_dir, zoom_views); SW, SH = (int(v) for v in resolution.split("x"))
        tag = f"{cap.dir.name}_{time.strftime('%Y%m%d-%H%M%S')}"; inp = Path(folder_paths.get_input_directory())
        work = cap.dir / f"shot_{tag}"; work.mkdir(parents=True); log = work / "prep.log"
        mode = ("mannequin" if cap.fast else "splat") if depth_from == "auto" else depth_from
        gpu = _small_gpu()
        if mode == "splat" and cap.fast:
            raise ValueError("This capture has no trained splat: use depth_from = mannequin (or auto)")
        depth = work / "depth.mp4"; bg = None
        if free_orbit_set and mode == "splat" and cap.source_cam is not None:   # the clip's setting as Gaussians around her: depth, colour and anchors keep the place
            try:
                bg = L.build(cap, cap.dir / "scene")
            except Exception as e:  # noqa: BLE001 - without it the shot is the person alone, as before
                print(f"[sam3dbx] background layer: {e}"); note += ", no background layer"
        world = bg is not None and (cap.dir / "scene" / "world_env.json").is_file()   # the 360 World node ran for this capture
        layers = []
        if world:   # the real near content (car, barrier) per frame, drawn with her over the 360 world
            fs = sorted({int(round(min(max(r[0], 0.0), 1.0) * (cap.frames - 1))) for r in rows})
            (work / "near.json").write_text(json.dumps({f: str(L.near_cloud(cap, f, cap.dir / "scene")) for f in fs}))
            layers = [f"--env={cap.dir / 'scene'}", f"--near={work / 'near.json'}"]
        elif bg:
            layers = [f"--scene={bg}"]
        if mode == "splat":
            _sh([TRAIN_PY, NODES_DIR / "render_path.py", f"--train_dir={cap.dir}", f"--path={camera_path}", f"--out={work / 'rgba'}",
                 f"--width={SW}", f"--height={SH}", f"--depth_mp4={depth}", "--depth_per_frame", *layers],
                log, _env(gpu, train=True), NODES_DIR)
        else:
            _sh([TRAIN_PY, NODES_DIR / "mesh_depth.py", f"--train_dir={cap.dir}", f"--path={camera_path}", f"--out={depth}",
                 f"--width={SW}", f"--height={SH}", "--per_frame"], log, _env(gpu, train=True), NODES_DIR)
        chunks, pngs = [], []
        for k, c in enumerate(S.plan(len(rows), chunk_seconds, anchor_every, every=anchor_strength, fps=fps)):
            ctrl = f"sx_{tag}_ctrl_{k}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(depth), "-vf", f"select=between(n\\,{c['f0']}\\,{c['f0'] + c['n'] - 1}),setpts=N/{fps}/TB",
                            "-r", str(fps), "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(inp / ctrl)], check=True)
            src = None
            if mode == "splat":   # the splat's own colour render of this chunk (over grey, or with its setting), for Photoreal Shoot's source_strength
                src = f"sx_{tag}_src_{k}.mp4"
                comp = ["-filter_complex", "[1]format=rgb24[o]", "-map", "[o]"] if bg else ["-filter_complex", "[0][1]overlay=shortest=1"]
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x808080:s={SW}x{SH}:r={fps}", "-framerate", str(fps),
                                "-start_number", str(c["f0"]), "-i", str(work / "rgba" / "%05d.png"), *comp,
                                "-frames:v", str(c["n"]), "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(inp / src)], check=True)
            refs = []
            for j, (at, fi, st) in enumerate(c["refs"]):
                if st is None:      # chunk start after the first: linked to the previous chunk's last frame in the graph
                    refs.append({"prev": True, "at": "start", "strength": 0.8}); continue
                name = f"sx_{tag}_c{k}_r{j}.png"; cid, cov = S.anchor(cap, rows[fi], inp / name, (SW, SH))
                refs.append({"image": name, "mask": name[:-4] + "_mask.png", "at": at, "strength": st, "cam": cid, "frame": fi, "coverage": cov})
            chunks.append({"secs": round(c["n"] / fps, 4), "n": c["n"], "ctrl": ctrl, "refs": refs, **({"src": src} if src else {})})
        if bg:   # the real frame at each anchor's moment, lifted to 3D and seen from the shot camera
            jobs = [{"cloud": str(L.frame_cloud(cap, int(round(min(max(rows[r["frame"]][0], 0.0), 1.0) * (cap.frames - 1))), cap.dir / "scene")),
                     "background": str(bg), "row": rows[r["frame"]], "width": SW, "height": SH, "out": str(work / f"real_{r['image']}")}
                    for ch in chunks for r in ch["refs"] if r.get("image")]
            if jobs:
                (work / "real_jobs.json").write_text(json.dumps(jobs))
                _sh([TRAIN_PY, NODES_DIR / "render_cloud.py", f"--jobs={work / 'real_jobs.json'}", f"--up={json.dumps(cap.up.tolist())}"],
                    log, _env(gpu, train=True), NODES_DIR)
        for ch in chunks:
            keep = []
            for r in ch["refs"]:
                if not r.get("image"):
                    keep.append(r); continue
                st = r["strength"]
                if world:   # the shot's own composite (360 world, near footage, her) is the anchor; her real pixels where the source camera saw her
                    st = _world_anchor(cap, rows[r["frame"]], r, inp, work, st)
                elif bg:   # every anchor from the shot camera (a raw source frame would pin LTX to the source camera's framing and undo the move)
                    st = _real_anchor(cap, rows[r["frame"]], r, inp, work, st, generated_anchors)
                elif r["coverage"] < 0.25:   # the anchor's mask keeps padding out of the render, but a sliver this small says too little; depth holds it
                    continue
                else:
                    # real frames hold the first and last frame hard; in between they guide softly (hard ones every second make LTX lurch)
                    st = (max(st, 0.9) if r["at"] in ("start", "end") else st) if r["cam"] == "source" else round(st * generated_anchors, 3)
                if st <= 0:
                    continue
                r["strength"] = st; pngs.append(inp / r["image"]); keep.append(r)
            ch["refs"] = keep
        spec = {"tag": tag, "frames": len(rows), "fps": fps, "width": SW, "height": SH, "depth_from": mode, "depth_video": str(depth), "chunks": chunks}
        if cap.source_cam is not None:   # the tracked person alone (her mesh's box in the real first frame): the auto prompt describes only her
            sc = cap.source_cam; f0 = int(round(rows[0][0] * (cap.frames - 1))); K = cap.cams[sc][1]; c2w = cap.pose(sc, f0)
            q = (cap.geometry()[f0] - c2w[:3, 3]) @ c2w[:3, :3]; uv = (q / q[:, 2:3]) @ K.T
            im = Image.fromarray(S.video_frame(cap.cams[sc][2], f0)); (x0, y0), (x1, y1) = uv[:, :2].min(0), uv[:, :2].max(0)
            mx, my = (x1 - x0) * 0.15, (y1 - y0) * 0.08
            im.crop((max(0, int(x0 - mx)), max(0, int(y0 - my)), min(im.width, int(x1 + mx)), min(im.height, int(y1 + my)))).save(inp / f"sx_{tag}_subject.png")
            im.save(inp / f"sx_{tag}_scene.png")   # the real frame whole: the setting is described from the video, never a generated view
            spec["subject_image"], spec["scene_image"] = f"sx_{tag}_subject.png", f"sx_{tag}_scene.png"
        shot_path = work / "shot.json"; shot_path.write_text(json.dumps(spec, indent=1))
        anchors = (torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGB")) for p in pngs])).float() / 255.0 if pngs
                   else torch.zeros(1, 64, 64, 3))
        sets = sorted({r["cam"].rstrip("0123456789") for c in chunks for r in c["refs"] if "cam" in r})
        info = f"{len(rows)} frames at {fps:g} fps, {len(chunks)} segments, {len(pngs)} anchors ({', '.join(sets)}), depth from {mode}{note}"
        return {"ui": {"sam3dbx_shot": [str(shot_path)]}, "result": (str(shot_path), anchors, _read_frames(depth, 320), info)}


REAL_DEG = 30.0
WORLD_PULL = 0.35
WORLD_ANCHOR = 1.0   # anchor weight on the 360 world (env_map) where it is the background   # anchor weight where only the generated 360 world (not footage) is behind the shot   # within this of the source camera's view the real person, lifted to 3D, still looks right; past it she flattens


def _real_anchor(cap, row, r, inp, work, st, generated_anchors):
    """Merge the real frame seen from the shot camera (render_cloud.py) into a generated-view anchor, in place. Near the source camera's view the
    real frame wins everywhere it lands (her face and the setting are the footage); further round she comes from the generated view and
    the real frame keeps only the setting around her. Returns the anchor strength (0 = drop it)."""
    from PIL import ImageFilter
    A = np.asarray(Image.open(inp / r["image"]).convert("RGB"), np.float32); M = np.asarray(Image.open(inp / r["mask"]).convert("L"), np.float32) / 255
    rp = work / f"real_{r['image']}"; Wr = np.asarray(Image.open(rp).convert("RGB"), np.float32)
    Wa = np.asarray(Image.open(rp.with_name(rp.stem + "_alpha.png")), np.float32) / 255
    h, w = Wa.shape; Wa = np.clip((Wa - 0.5) * 4, 0, 1)   # only where the real pixels are dense, not the scattered fringe
    Wc = Wr / np.maximum(np.asarray(Image.open(rp.with_name(rp.stem + "_alpha.png")), np.float32)[..., None] / 255, 1e-3)   # un-premultiply
    eye, tgt = np.array(row[1:4]), np.array(row[4:7]); fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
    sc = cap.source_cam; fi = int(round(min(max(row[0], 0.0), 1.0) * (cap.frames - 1)))
    near = float(fwd @ cap.pose(sc, fi)[:3, 2]) >= math.cos(math.radians(REAL_DEG))
    sil = Image.open(work / "rgba" / f"{r['frame']:05d}.png").getchannel("A")
    Q = np.asarray(Image.open(rp.with_name(rp.stem + "_person.png")).point(lambda v: 255 if v > 20 else 0).filter(ImageFilter.MaxFilter(21))
                   .filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
    if near:   # her real pixels only where they land on her splat (the real frame and the splat can place her a little apart): no second her
        Pin = np.asarray(sil.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(4)), np.float32) / 255
        P = Q * (1 - Pin); Pt = np.zeros((h, w), np.float32)
    else:   # her region in the shot (the splat's own coverage at this frame), grown so the real frame's flattened edge stays out
        Pt = np.asarray(sil.filter(ImageFilter.MinFilter(5)), np.float32) / 255   # tight: only her, never the generated view's backdrop
        if r["cam"] == "source":   # the nearest camera was the source camera itself: its frame has her at the source camera's framing, not this one
            Pt = np.zeros((h, w), np.float32)
        P = np.asarray(sil.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
        # and the real frame's own person, wherever her flattened copy lands, plus its scattered fringe: no second her in the anchor
        P = np.maximum(P, Q)
    use = Wa * (1 - P)
    # outside the real pixels and her: a soft spread of the real setting (normalised blur), nothing sharp for LTX to copy
    import cv2
    k = (0, 0); sig = max(h, w) / 25
    soft = cv2.GaussianBlur(Wc * use[..., None], k, sig) / np.maximum(cv2.GaussianBlur(use, k, sig), 1e-3)[..., None]
    out = Wc * use[..., None] + (A * Pt[..., None] + soft * (1 - Pt[..., None])) * (1 - use[..., None])
    rq = rp.with_name(rp.stem + "_real.png")
    if rq.is_file():   # the generated world guides softly (layout, colour); only footage pins hard, so LTX draws the world's detail itself
        Rl = np.clip((np.asarray(Image.open(rq), np.float32) / 255 - 0.5) * 4, 0, 1)
        use = use * (Rl + WORLD_PULL * (1 - Rl))
    mask = np.maximum(use, M * Pt) if not near else use   # the seam round her and anything no real pixel reached stay free for LTX
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(inp / r["image"])
    Image.fromarray((mask * 255).astype(np.uint8)).save(inp / r["mask"])
    real_cov = float(use.mean())
    r["cam"] = ("real" if near else "real+") + r["cam"]; r["coverage"] = round(max(r["coverage"], real_cov), 3)
    if near:
        return max(st, 0.9)
    return round(st * generated_anchors, 3) if r["coverage"] >= 0.25 else 0.0


def _world_anchor(cap, row, r, inp, work, st):
    """Anchor from the shot's own composite render (360 world behind, real near content, her splat): footage pinned hard, the world
    firmly (WORLD_ANCHOR), her splat left free (it is soft; LTX draws her from depth and the prompt), except near the source camera's view,
    where her real pixels from the lifted frame go in at full weight where they land on her. Returns the anchor strength."""
    from PIL import ImageFilter
    f = r["frame"]; img = Image.open(work / "rgba" / f"{f:05d}.png").convert("RGBA")
    out = np.asarray(img.convert("RGB"), np.float32); her = np.asarray(img.getchannel("A"), np.float32) / 255
    fg = np.asarray(Image.open(work / "rgba" / f"{f:05d}_fg.png"), np.float32) / 255
    nearc = np.clip(fg - her, 0, 1); env = 1 - fg
    mask = nearc + WORLD_ANCHOR * env
    eye, tgt = np.array(row[1:4]), np.array(row[4:7]); fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
    fi = int(round(min(max(row[0], 0.0), 1.0) * (cap.frames - 1)))
    near = float(fwd @ cap.pose(cap.source_cam, fi)[:3, 2]) >= math.cos(math.radians(REAL_DEG))
    rp = work / f"real_{r['image']}"
    on_phone = r.get("cam") == "source" and float(fwd @ cap.pose(cap.source_cam, fi)[:3, 2]) >= 0.995
    if on_phone:   # the shot camera is the source camera's: the real frame as shot is the anchor (as the footage-locked push-in does)
        return max(st, 0.9)
    if near and rp.is_file():   # near the source camera's view: the whole real frame (her included) seen from the shot camera; the world fills its gaps
        Wr = np.asarray(Image.open(rp).convert("RGB"), np.float32); Wa = np.asarray(Image.open(rp.with_name(rp.stem + "_alpha.png")), np.float32) / 255
        use = np.clip((Wa - 0.5) * 4, 0, 1)
        Wc = Wr / np.maximum(Wa[..., None], 1e-3)
        out = Wc * use[..., None] + out * (1 - use[..., None]); mask = np.maximum(mask, use)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(inp / r["image"])
    Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).save(inp / r["mask"])
    r["cam"] = ("world+real" if near else "world"); r["coverage"] = 1.0
    return max(st, 0.9) if near else max(st, 0.8)


def _merge_identical(graph, out_id):
    """Common-subexpression merge: nodes with the same class and the same (canonicalised) inputs become one, so the LTX models,
    text encoder and VAEs are loaded once for all chunks. Returns (graph, canonical out_id)."""
    canon, seen = {}, {}

    def key(nid):
        if nid in canon:
            return canon[nid]
        n = graph[nid]; inputs = {}
        for k, v in n["inputs"].items():
            inputs[k] = [key(v[0]), v[1]] if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in graph else v
        sig = json.dumps([n["class_type"], inputs], sort_keys=True, default=str)
        canon[nid] = seen.setdefault(sig, nid); graph[nid] = {"class_type": n["class_type"], "inputs": inputs}
        return canon[nid]

    for nid in list(graph):
        key(nid)
    return {nid: graph[nid] for nid in set(canon.values())}, canon[out_id]


class Sam3dbxPhotorealShoot:
    """Optional realism pass, step 2: LTX 2.3 renders the shot as footage, held to the capture's pose and camera by depth control
    and to the person by the identity anchors. The prompt is written from the video when subject and scene are empty."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"shot": ("STRING", {"forceInput": True, "tooltip": "Prepared shot from Photoreal Prep"}),
                             "seed": ("INT", {"default": 11, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
                             "control_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How closely LTX follows the capture's depth (pose and camera)"}),
                             "control_lora_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.05, "advanced": True, "tooltip": "Strength of the depth-control LoRA"}),
                             "source_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "0 starts from noise. Higher starts from the splat render: more of the splat, less new detail"}),
                             "detailer": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Detailer LoRA in the final stage"}),
                             "subject": ("STRING", {"multiline": True, "default": "", "advanced": True, "tooltip": "Optional: who (hair, clothes, look). Empty = described from the video"}),
                             "scene": ("STRING", {"multiline": True, "default": "", "advanced": True, "tooltip": "Optional: where (backdrop, floor, light). Empty = described from the video"})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("frames",)
    OUTPUT_TOOLTIPS = ("The photoreal shot, at the prepared size and frame rate",)
    FUNCTION = "run"; CATEGORY = CAT_LEGACY; DEPRECATED = True

    def run(self, shot, seed, control_strength, control_lora_strength, detailer, subject, scene, unique_id=None, source_strength=0.0, **legacy):
        if legacy.get("engine", "LTX 2.3") != "LTX 2.3":
            raise ValueError("This workflow uses a retired engine. Photoreal Pass runs on LTX 2.3 only.")
        spec_shot = json.loads(Path(shot).read_text()); P = f"{unique_id}.sx."
        tail = " Sharp natural detail in the face, eyes, hair strands, skin and fabric. The camera framing follows the guide exactly. No text."
        prompt = f"Photoreal footage of {subject.strip()}, {scene.strip()}." + tail
        graph, cat, link = {}, None, None
        # the first anchor (the real input frame when the shot starts on the real camera); else the tracked person's crop
        first = next((r["image"] for c in spec_shot["chunks"] for r in c["refs"] if r.get("image")), None) or spec_shot.get("subject_image")
        if not (subject.strip() and scene.strip()) and not first:
            raise ValueError("Nothing to describe the shot from: no anchors and no source frame. Fill in subject and scene (advanced).")
        if not (subject.strip() and scene.strip()):
            # no description given: LTX's own Gemma describes the tracked person (her crop, when Shot Prep has one: other people in the
            # frame are not in the capture, and describing them hands her their look) and the setting without people, from the first
            # anchor (the real input frame when the shot starts on the real camera), so the prompt comes from the video itself
            # the describer: Qwen3-VL 8B when installed (reads hair, eyewear and shoe colour reliably), else LTX's own Gemma (its fp4 vision
            # guessed dark straight hair and black shoes for a light-haired woman in white sneakers)
            vlm = next((n for n in folder_paths.get_filename_list("text_encoders") if n.startswith("qwen3vl_8b")), None)
            graph[f"{P}cap.clip"] = ({"class_type": "CLIPLoader", "inputs": {"clip_name": vlm, "type": "ideogram4", "device": "default"}} if vlm else
                                     {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": sg.MODELS["gemma"], "clip_name2": sg.MODELS["proj"], "type": "ltxv", "device": "default"}})
            samp = {"sampling_mode": "on", "sampling_mode.temperature": 0.2, "sampling_mode.top_k": 64, "sampling_mode.top_p": 0.95,
                    "sampling_mode.min_p": 0.05, "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": seed}

            def caption(key, image, ask):
                graph[f"{P}cap.{key}.img"] = {"class_type": "LoadImage", "inputs": {"image": image, "upload": "image"}}
                graph[f"{P}cap.{key}"] = {"class_type": "TextGenerate", "inputs": {"clip": [f"{P}cap.clip", 0], "image": [f"{P}cap.{key}.img", 0], "max_length": 160,
                                                                                    "thinking": False, "use_default_template": True, "prompt": ask, **samp}}
                graph[f"{P}cap.{key}.flat"] = {"class_type": "StringReplace", "inputs": {"string": [f"{P}cap.{key}", 0], "find": "\n", "replace": " "}}
                return [f"{P}cap.{key}.flat", 0]
            who = subject.strip() or caption("who", spec_shot.get("subject_image") or first,
                          "Describe the one main person in this image for a video generation model: apparent age, sex, build, hair colour and style, face, "
                          "every clothing item with its exact colour, footwear and its colour (bare feet only if no shoes are visible). Only that person; ignore anyone else. Plain colour words, no "
                          "invented details. Reply with the description only, one paragraph under 60 words.")
            where = scene.strip() or caption("where", spec_shot.get("scene_image") or first,
                            "Describe only the setting of this image for a video generation model: the place, backdrop, floor, every notable object (vehicles, "
                            "furniture, buildings) with its exact colour, the sky, lighting and colours. Do not "
                            "describe any people. Plain words, no invented details. Reply with the description only, one paragraph under 50 words.")
            graph[f"{P}cap.p0"] = {"class_type": "StringConcatenate", "inputs": {"string_a": "Photoreal footage of one person. ", "string_b": who, "delimiter": ""}}
            graph[f"{P}cap.p1"] = {"class_type": "StringConcatenate", "inputs": {"string_a": [f"{P}cap.p0", 0], "string_b": where, "delimiter": " Setting: "}}
            graph[f"{P}cap.p2"] = {"class_type": "StringConcatenate", "inputs": {"string_a": [f"{P}cap.p1", 0], "string_b": tail, "delimiter": ""}}
            graph[f"{P}cap.show"] = {"class_type": "PreviewAny", "inputs": {"source": [f"{P}cap.p2", 0]}}
            link = [f"{P}cap.p2", 0]
        for k, c in enumerate(spec_shot["chunks"]):
            refs = [{"link": [f"{P}k{k - 1}.last", 0], "at": r["at"], "strength": r["strength"], "name": "previous chunk"} if r.get("prev")
                    else {"image": r["image"], "mask": r.get("mask"), "at": r["at"], "strength": r["strength"]} for r in c["refs"]]
            spec = {"prompt": prompt, "video": {"seconds": c["secs"], "fps": spec_shot["fps"], "width": spec_shot["width"], "height": spec_shot["height"], "seed": seed},
                    "control": {"video": c["ctrl"], "type": "depth", "strength": control_strength, "lora_strength": control_lora_strength},
                    "refs": refs, "prefix": f"sam3dbx/shoot_{spec_shot['tag']}_{k}", "return_image": True}
            if link:
                spec["_prompt_link"] = link
            if source_strength > 0 and c.get("src"):
                spec["source"] = {"video": c["src"], "strength": source_strength}
            if not detailer:
                spec["detail"] = {"lora": None}
            api, dec = sg.build(spec)
            ren = {nid: f"{P}k{k}.{nid}" for nid in api}
            for nid, n in api.items():
                inputs = {key: ([ren[v[0]], v[1]] if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in ren else v)
                          for key, v in n["inputs"].items()}
                graph[ren[nid]] = {"class_type": n["class_type"], "inputs": inputs}
            trim, last = f"{P}k{k}.trim", f"{P}k{k}.last"
            graph[trim] = {"class_type": "ImageFromBatch", "inputs": {"image": [ren[dec], 0], "batch_index": 0, "length": c["n"]}}
            graph[last] = {"class_type": "ImageFromBatch", "inputs": {"image": [trim, 0], "batch_index": c["n"] - 1, "length": 1}}
            if cat is None:
                cat = trim
            else:
                graph[f"{P}cat{k}"] = {"class_type": "ImageBatch", "inputs": {"image1": [cat, 0], "image2": [trim, 0]}}; cat = f"{P}cat{k}"
        graph, out = _merge_identical(graph, cat)
        return {"result": ([out, 0],), "expand": graph}


WORLD_MODELS = {"unet": "qwen_image_2.1_int8_convrot.safetensors", "te": "qwen3vl_8b_int8_convrot.safetensors",
                "vae": "qwen_image_2.1_vae_bf16.safetensors", "lora": "pano360_qwen21_edit_v1.safetensors"}


def _world_dir(capture_dir):
    td, _ = CR.resolve(Path(capture_dir), "4DGS (splat)"); return td


class Sam3dbxWorldFrames:
    """360 World, step 1: the clip's setting solved (DA3) and three real frames (start, middle, end) for the panorama."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"capture_dir": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "IMAGE"); RETURN_NAMES = ("capture_dir", "start", "middle", "end")
    FUNCTION = "run"; CATEGORY = CAT_LEGACY; DEPRECATED = True

    def run(self, capture_dir):
        td = _world_dir(capture_dir)
        if not L.ensure(td):
            raise ValueError("360 World needs the setting solve (a real camera and Depth Anything 3 Base in models/geometry_estimation)")
        cap = S.Capture(td, zoom_views=False); L.build(cap, td / "scene"); v = cap.cams[cap.source_cam][2]
        im = lambda f: torch.from_numpy(S.video_frame(v, f)).float()[None] / 255.0
        return (str(td), im(0), im(cap.frames // 2), im(cap.frames - 1))


class Sam3dbxWorldLift:
    """360 World, last step: the generated panorama of the empty set registered to what the source camera really saw (turned, and mirrored if
    need be). Every shot of this capture then shows it behind her and the real near footage (env_map: skybox + ground projection)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"capture_dir": ("STRING", {"forceInput": True}), "panorama": ("IMAGE",)}}
    RETURN_TYPES = ("STRING", "STRING"); RETURN_NAMES = ("capture_dir", "info")
    FUNCTION = "run"; CATEGORY = CAT_LEGACY; OUTPUT_NODE = True; DEPRECATED = True

    def run(self, capture_dir, panorama):
        td = Path(capture_dir); pano = np.ascontiguousarray((panorama[0, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))   # RGB (an upscaler may add alpha)
        Image.fromarray(pano).save(td / "scene" / "panorama.png")
        L.register_panorama(S.Capture(td, zoom_views=False), td / "scene", pano)
        w = json.loads((td / "scene" / "world_env.json").read_text())
        _free_comfy()   # Qwen-Image and the upscaler are done with: give the memory back before the shot
        return (str(td), f"360 world: {pano.shape[1]}x{pano.shape[0]} panorama, turned {w['yaw_px']} px{', mirrored' if w['mirrored'] else ''}")


class Sam3dbxWorld:
    """Builds a 360 world round the clip once, so a camera turned anywhere sees the clip's own place instead of a void. Three real
    frames with people and vehicles removed (Qwen-Image 2.1 edit), described by Qwen3-VL, become one 360 panorama of the empty set
    (Qwen-Image 2.1 + pano360 LoRA, SeedVR2 2x), registered to the real camera. Shots then show it behind her, with the real car and
    whatever else is near her taken from the footage frame by frame. Bypass it to keep shots near the real camera."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"capture_dir": ("STRING", {"forceInput": True, "tooltip": "Capture (from Capture)"}),
                             "seed": ("INT", {"default": 20261026, "min": 0, "max": 2**31 - 1, "control_after_generate": True})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("STRING", "IMAGE", "STRING"); RETURN_NAMES = ("capture_dir", "panorama", "info")
    OUTPUT_TOOLTIPS = ("The capture, for the Virtual Camera", "The 360 panorama (preview)", "World summary")
    FUNCTION = "run"; CATEGORY = CAT_LEGACY; DEPRECATED = True

    def run(self, capture_dir, seed, unique_id=None):
        have = {k: v in folder_paths.get_filename_list({"unet": "diffusion_models", "te": "text_encoders", "vae": "vae", "lora": "loras"}[k])
                for k, v in WORLD_MODELS.items()}
        if not all(have.values()):
            raise ValueError("360 World needs: " + ", ".join(WORLD_MODELS[k] for k, ok in have.items() if not ok) + " (see docs/SETUP.md)")
        P = f"{unique_id}.w."; g = {}
        g[P + "frames"] = {"class_type": "Sam3dbxWorldFrames", "inputs": {"capture_dir": capture_dir}}
        g[P + "unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": WORLD_MODELS["unet"], "weight_dtype": "default"}}
        g[P + "te"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": WORLD_MODELS["te"], "type": "qwen_image", "device": "default"}}
        g[P + "vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": WORLD_MODELS["vae"]}}
        clean = []
        for k in range(3):   # the person removed from each real frame, so she is not painted into the world
            g[f"{P}e{k}"] = {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": [P + "te", 0], "vae": [P + "vae", 0], "images.image_1": [P + "frames", k + 1],
                             "prompt": "Remove every person and every vehicle completely and restore what was behind them: the floor, the barriers and the "
                                       "view, as the place looks empty. Keep everything else exactly the same.",
                             "negative_prompt": "person, people, car, vehicle, shadow, blur, smudge", "resolution": 1024}}
            g[f"{P}k{k}"] = {"class_type": "KSampler", "inputs": {"model": [P + "unet", 0], "positive": [f"{P}e{k}", 0], "negative": [f"{P}e{k}", 1],
                             "latent_image": [f"{P}e{k}", 2], "seed": seed + k, "steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
            g[f"{P}d{k}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"{P}k{k}", 0], "vae": [P + "vae", 0]}}; clean.append([f"{P}d{k}", 0])
        g[P + "vl"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": WORLD_MODELS["te"], "type": "ideogram4", "device": "default"}}
        g[P + "cap"] = {"class_type": "TextGenerate", "inputs": {"clip": [P + "vl", 0], "image": [P + "frames", 2], "max_length": 180, "thinking": False,
                        "use_default_template": True, "sampling_mode": "off",
                        "prompt": "Describe only the fixed setting of this image for an image generation model: the place, the floor, barriers and "
                                  "buildings with their exact colours, the skyline, the sky and the light. Do not describe people or vehicles. "
                                  "Plain words, no invented details. Reply with the description only, one paragraph under 70 words."}}
        g[P + "p"] = {"class_type": "StringConcatenate", "inputs": {"string_b": [P + "cap", 0], "delimiter": "",
                      "string_a": "Turn these photos into one seamless 360-degree equirectangular panorama (2:1) of this exact place, seen from where "
                                  "the photographer stands. Keep the same floor, barriers, buildings and sky, with the left and right edges meeting. "
                                  "No people and no vehicles. Scene: "}}
        g[P + "lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [P + "unet", 0], "lora_name": WORLD_MODELS["lora"], "strength_model": 1.0}}
        g[P + "enc"] = {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": [P + "te", 0], "vae": [P + "vae", 0], "images.image_1": clean[1],
                        "images.image_2": clean[0], "images.image_3": clean[2], "prompt": [P + "p", 0], "resolution": 1024,
                        "negative_prompt": "people, person, car, vehicle, blurry patch, text, watermark, fisheye circle, black border"}}
        g[P + "lat"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 2048, "height": 1024, "batch_size": 1}}
        g[P + "ks"] = {"class_type": "KSampler", "inputs": {"model": [P + "lora", 0], "positive": [P + "enc", 0], "negative": [P + "enc", 1],
                       "latent_image": [P + "lat", 0], "seed": seed, "steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
        g[P + "pano"] = {"class_type": "VAEDecode", "inputs": {"samples": [P + "ks", 0], "vae": [P + "vae", 0]}}
        pano = [P + "pano", 0]
        if "SeedVR2VideoUpscaler" in nodes.NODE_CLASS_MAPPINGS:   # 2x sharper (SeedVR2): a panorama pixel spans several shot pixels otherwise
            g[P + "sdit"] = {"class_type": "SeedVR2LoadDiTModel", "inputs": {"model": "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "device": "cuda:0"}}
            g[P + "svae"] = {"class_type": "SeedVR2LoadVAEModel", "inputs": {"model": "ema_vae_fp16.safetensors", "device": "cuda:0", "encode_tiled": True, "decode_tiled": True}}
            g[P + "up"] = {"class_type": "SeedVR2VideoUpscaler", "inputs": {"image": pano, "dit": [P + "sdit", 0], "vae": [P + "svae", 0], "seed": seed,
                           "resolution": 2048, "max_resolution": 4096, "batch_size": 1, "uniform_batch_size": False, "color_correction": "lab"}}
            pano = [P + "up", 0]
        g[P + "lift"] = {"class_type": "Sam3dbxWorldLift", "inputs": {"capture_dir": [P + "frames", 0], "panorama": pano}}
        return {"result": ([P + "lift", 0], pano, [P + "lift", 1]), "expand": g}


NODE_CLASS_MAPPINGS = {"Sam3dbxCaptureSubject": Sam3dbxCaptureSubject, "Sam3dbxShotPrep": Sam3dbxShotPrep, "Sam3dbxPhotorealShoot": Sam3dbxPhotorealShoot,
                       "Sam3dbxWorld": Sam3dbxWorld, "Sam3dbxWorldFrames": Sam3dbxWorldFrames, "Sam3dbxWorldLift": Sam3dbxWorldLift}
NODE_DISPLAY_NAME_MAPPINGS = {"Sam3dbxCaptureSubject": "Reshoot 4D: Capture", "Sam3dbxShotPrep": "Reshoot 4D: Splat Photoreal Prep (legacy)",
                              "Sam3dbxPhotorealShoot": "Reshoot 4D: Splat Photoreal Pass (legacy)", "Sam3dbxWorld": "Reshoot 4D: 360 World (legacy)",
                              "Sam3dbxWorldFrames": "Reshoot 4D: 360 World frames (legacy)", "Sam3dbxWorldLift": "Reshoot 4D: 360 World lift (legacy)"}
