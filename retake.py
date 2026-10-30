"""Reshoot 4D Retake: the whole source clip re-shot along a Virtual Camera move by ReDirector (Apache-2.0, Wan2.1-Fun 1.3B
Control-Camera), then Photoreal Pass (retake): LTX 2.3 v2v over the retake.

Retake      node expansion: describes the first frame with Qwen3-VL (unless a prompt is given), then Retake (run).
Retake run  builds the condition path (the real camera path from the capture's camera solve, metric from her height) and the target
            path (the Virtual Camera keys, re-based on the source camera's frame-0 pose), unloads ComfyUI's models and runs ReDirector in its
            own venv (redirector_worker.py: torch 2.8 + DiffSynth never share ComfyUI's torch), with live progress and Cancel.
Photoreal Pass (retake)  node expansion into the LTX 2.3 graph (ltx_graph.py): starts from the retake partway down the schedule,
            the retake's edges as union control and the real first frame as the start guide.
"""
import json, math, os, select, shutil, subprocess, time
from pathlib import Path

import cv2
import numpy as np
import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from . import crossview as CV
from . import director as D
from . import ltx_graph as sg
from . import rig as RIG
from . import shot as S
from .config import GPUS, REDIRECTOR, RD_PY

CAT = "Reshoot 4D"
SIZES = {"480p": (480, 832), "720p": (720, 1280)}   # (short, long) side; ReDirector was trained at 480p, 720p needs ~21 GB


def build_paths(td, state, n=81):
    """Condition path (the real camera, smoothed) and target path (the Virtual Camera move) in ReDirector's camera format: OpenCV
    camera-to-world, both relative to their own first frame, translation in metres (her height = 1.7 m). The move must start on the
    real camera: ReDirector re-shoots the clip from where it was filmed."""
    cap = S.Capture(td, zoom_views=False, scene=True)
    if cap.source_cam is None:
        raise ValueError("This capture has no real camera solve, so the clip can't be re-shot. Capture it again with the current Capture node.")
    sc = cap.source_cam; F = cap.frames
    K = np.array(cap.cams[sc][1], float); video = str(cap.cams[sc][2]); W, H = cap.cams[sc][4], cap.cams[sc][5]
    fs = np.round(np.linspace(0, F - 1, n)).astype(int)
    C = np.stack([cap.pose(sc, f) for f in fs])
    meta = S.scene_meta(td, json.loads((Path(td) / "preview_meta.json").read_text()))
    basis = [np.array(meta[k], float) for k in ("up", "front", "side")]; up = basis[0]
    V = cap.geometry(); Vm = V[fs[n // 2]]; ax = np.linalg.svd(Vm - Vm.mean(0), full_matrices=False)[2][0]
    m = 1.7 / float((Vm @ ax).max() - (Vm @ ax).min())   # metres per capture unit
    run = Path(td).parent if (Path(td).parent / "clip.json").is_file() else Path(td).parent.parent
    secs = float(json.loads((run / "clip.json").read_text())["seconds"]) if (run / "clip.json").is_file() else F / 25.0
    dur = float(state["duration"])   # the move is stretched over the retake (the whole source clip, in real time)
    rows = [D.sample(state, dur * i / (n - 1), meta["dist0"], [b.tolist() for b in basis], meta.get("track")) for i in range(n)]
    T = np.stack([RIG.look_at(np.array(r[1:4]), np.array(r[4:7]), up, r[7]) for r in rows])
    d_eye = float(np.linalg.norm(T[0][:3, 3] - C[0][:3, 3]) * m)
    d_rot = math.degrees(math.acos(np.clip((np.trace(T[0][:3, :3].T @ C[0][:3, :3]) - 1) / 2, -1, 1)))
    if d_eye > 0.15 or d_rot > 10:
        raise ValueError(f"The move's first key is {d_eye:.2f} m / {d_rot:.0f} deg away from the real camera. A retake starts where the clip "
                         "was filmed: put the first key on the real camera (any preset does), then queue again.")
    Q = T[0][:3, :3].T @ C[0][:3, :3]   # the source camera's own tilt/roll against the level key, carried along (frame 0 = the source camera exactly)
    T[:, :3, :3] = T[:, :3, :3] @ Q; T[:, :3, 3] += C[0][:3, 3] - T[0][:3, 3]
    rel = lambda X: np.einsum("ij,njk->nik", np.linalg.inv(X[0]), X)
    cond, targ = rel(C), rel(T); cond[:, :3, 3] *= m; targ[:, :3, 3] *= m
    return dict(seconds=secs, frames=fs, video=video, K=K, W=W, H=H, cond=cond, targ=targ, metres_per_unit=m,
                info=f"travel {np.linalg.norm(targ[-1, :3, 3]):.2f} m, first key {d_eye:.2f} m / {d_rot:.1f} deg off the source camera")


def _first_frame(video, out, w, h):
    """The real first frame, cropped to the retake's aspect: the caption's image and Photoreal Pass's start guide."""
    cap = cv2.VideoCapture(str(video)); ok, fr = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"can't read {video}")
    fh, fw = fr.shape[:2]; s = max(w / fw, h / fh); fr = cv2.resize(fr, (round(fw * s), round(fh * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = (fr.shape[0] - h) // 2, (fr.shape[1] - w) // 2
    cv2.imwrite(str(out), fr[y0:y0 + h, x0:x0 + w])


def _ltx_size(w, h):
    return (1024, 1792) if h >= w else (1792, 1024)


def _prep_ltx(mp4, name, fps):
    """The retake as LTX's source video and its CPU Canny at LTX size (full-res Canny on the GPU runs out of memory)."""
    inp = Path(folder_paths.get_input_directory())
    r = cv2.VideoCapture(str(mp4)); fr = []
    while True:
        ok, f = r.read()
        if not ok:
            break
        fr.append(f)
    r.release()
    h, w = fr[0].shape[:2]; LW, LH = _ltx_size(w, h)
    a = cv2.VideoWriter(str(inp / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    c = cv2.VideoWriter(str(inp / f"{name}_canny.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (LW, LH))
    for f in fr:
        a.write(f)
        s = max(LW / w, LH / h); g = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_LANCZOS4)
        y0, x0 = (g.shape[0] - LH) // 2, (g.shape[1] - LW) // 2; g = g[y0:y0 + LH, x0:x0 + LW]
        c.write(cv2.cvtColor(cv2.Canny(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), 51, 153), cv2.COLOR_GRAY2BGR))
    a.release(); c.release()
    return fr


CAPTION_ASK = ("Describe this video frame in one sentence for a video generation model: the person (hair colour and style, clothing, shoes, "
               "accessories) and what they are doing, then the setting and lighting. Be precise about colours. No preamble.")


class Sam3dbxRetake:
    """Re-shoot the whole clip along your Virtual Camera move (ReDirector). The person, the real setting and the motion come from the
    footage itself; the camera follows your keys. The move must start on the real camera. The prompt is written from the first frame
    when left empty."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"train_dir": ("STRING", {"forceInput": True, "tooltip": "Capture, from the Virtual Camera"}),
                             "camera_path": ("STRING", {"forceInput": True, "tooltip": "The camera move, from the Virtual Camera"}),
                             "resolution": (list(SIZES), {"default": "480p", "tooltip": "480p: ~25 min, 12 GB (what ReDirector was trained at). 720p: ~1 h 50, 21 GB"}),
                             "steps": ("INT", {"default": 50, "min": 10, "max": 100, "tooltip": "Fewer than ~40 steps judders in motion"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": False}),
                             "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional. Empty = Qwen3-VL describes the first frame"})},
                "optional": {"cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.5, "advanced": True})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT"); RETURN_NAMES = ("frames", "retake", "fps")
    OUTPUT_TOOLTIPS = ("The retake (preview of the mp4)", "The retake, for Photoreal Pass (retake)", "Frame rate: the clip plays in real time")
    FUNCTION = "run"; CATEGORY = CAT

    def run(self, train_dir, camera_path, resolution, steps, seed, prompt, cfg=5.0, unique_id=None):
        cam = json.loads(Path(camera_path).read_text()); state = cam.get("state") or cam
        short, long_ = SIZES[resolution]
        z = build_paths(train_dir, state)
        w, h = (short, long_) if z["H"] >= z["W"] else (long_, short)
        job_dir = Path(train_dir) / f"retake_{time.strftime('%Y%m%d-%H%M%S')}"; job_dir.mkdir(parents=True)
        np.savez(job_dir / "path.npz", **{k: v for k, v in z.items() if k != "info"})
        tag = f"rt_{Path(train_dir).parent.name}_{job_dir.name}"[:80]
        f0 = f"{tag}_f0.png"; _first_frame(z["video"], Path(folder_paths.get_input_directory()) / f0, w, h)
        job = {"npz": str(job_dir / "path.npz"), "out": str(job_dir / "retake.mp4"), "width": w, "height": h, "steps": steps, "seed": seed,
               "cfg": cfg, "fps": len(z["frames"]) / z["seconds"], "frames": len(z["frames"]), "tag": tag, "first_frame": f0, "info": z["info"]}
        (job_dir / "job.json").write_text(json.dumps(job, indent=1))
        P = f"{unique_id}.rt."; g = {}
        if prompt.strip():
            text = prompt.strip()
        else:   # the prompt from the video itself: Qwen3-VL on the real first frame
            vlm = next((n for n in folder_paths.get_filename_list("text_encoders") if n.startswith("qwen3vl_8b")), None)
            if not vlm:
                raise ValueError("No prompt and no Qwen3-VL 8B text encoder (qwen3vl_8b*) to write one. Type a prompt or install the model.")
            g[P + "vl"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": vlm, "type": "ideogram4", "device": "default"}}
            g[P + "im"] = {"class_type": "LoadImage", "inputs": {"image": f0, "upload": "image"}}
            g[P + "cap"] = {"class_type": "TextGenerate", "inputs": {"clip": [P + "vl", 0], "image": [P + "im", 0], "max_length": 160, "thinking": False,
                                                                       "use_default_template": True, "sampling_mode": "off", "prompt": CAPTION_ASK}}
            g[P + "flat"] = {"class_type": "StringReplace", "inputs": {"string": [P + "cap", 0], "find": "\n", "replace": " "}}
            g[P + "show"] = {"class_type": "PreviewAny", "inputs": {"source": [P + "flat", 0]}}
            text = [P + "flat", 0]
        g[P + "run"] = {"class_type": "Sam3dbxRetakeRun", "inputs": {"job": str(job_dir / "job.json"), "prompt": text}}
        return {"result": ([P + "run", 0], [P + "run", 1], job["fps"]), "expand": g}


class Sam3dbxRetakeRun:
    """Runs one prepared retake in ReDirector's own environment (added by Retake; not needed on its own)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": ("STRING", {"forceInput": True}), "prompt": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE", "STRING"); RETURN_NAMES = ("frames", "retake")
    FUNCTION = "run"; CATEGORY = CAT + "/Advanced"

    def run(self, job, prompt):
        jp = Path(job); j = json.loads(jp.read_text())
        j["prompt"] = " ".join(str(prompt).split()); jp.write_text(json.dumps(j, indent=1))
        if not Path(RD_PY).is_file():
            raise RuntimeError(f"ReDirector isn't installed at {REDIRECTOR} (no .venv). See docs/REDIRECTOR.md.")
        mm.unload_all_models(); mm.soft_empty_cache()   # ReDirector needs the card: 12 GB at 480p, 21 GB at 720p
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPUS["big"], "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "PYTHONUNBUFFERED": "1"}
        log = jp.with_name("retake.log"); pbar = comfy.utils.ProgressBar(int(j["steps"]))
        with open(log, "w") as lf:
            p = subprocess.Popen([RD_PY, str(Path(__file__).parent / "redirector_worker.py"), str(REDIRECTOR), str(jp)], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=env, cwd=str(REDIRECTOR), text=True, bufsize=1)
            try:
                while True:
                    if mm.processing_interrupted():
                        p.kill(); p.wait(); mm.throw_exception_if_processing_interrupted()
                    ready, _, _ = select.select([p.stdout], [], [], 1.0)
                    if not ready:
                        if p.poll() is not None:
                            break
                        continue
                    line = p.stdout.readline()
                    if not line:
                        break
                    lf.write(line); lf.flush()
                    if line.startswith("STEP "):
                        i, n = map(int, line.split()[1:3]); pbar.update_absolute(i, n)
            finally:
                if p.poll() is None:
                    p.kill()
            code = p.wait()
        if code != 0 or not Path(j["out"]).is_file():
            tail = [l for l in log.read_text(errors="replace").splitlines() if l.strip()][-4:]
            raise RuntimeError(f"ReDirector failed (exit {code}). Full log: {log}\n" + "\n".join(tail))
        fr = _prep_ltx(j["out"], j["tag"], j["fps"])
        frames = torch.from_numpy(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in fr])).float() / 255.0
        return (frames, str(jp))


class Sam3dbxRetakePhotoreal:
    """Photoreal Pass over a Retake: LTX 2.3 starts from the retake partway down its schedule. CrossView (default): the CrossView-Warp
    IC-LoRA steered by a depth warp of the real video along the move and by the real video itself. Union control: the retake's edges
    and the real first frame as guides."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"retake": ("STRING", {"forceInput": True, "tooltip": "From Retake"}),
                             "seed": ("INT", {"default": 11, "min": 0, "max": 2**31 - 1, "control_after_generate": False}),
                             "start_strength": ("FLOAT", {"default": 0.35, "min": 0.1, "max": 1.0, "step": 0.05,
                                                          "tooltip": "Where LTX starts on its schedule. 0.35 keeps the retake's pose and camera; 0.5 starts changing the pose"}),
                             "control_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "How closely LTX follows the retake's edges"}),
                             "control_lora_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05, "advanced": True}),
                             "first_frame_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True,
                                                                "tooltip": "The real first frame as the start guide (identity)"}),
                             "detailer": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Detailer LoRA in the final stage"})},
                "optional": {"pass_type": (["crossview", "union control"], {"default": "crossview",
                                           "tooltip": "crossview: the real video, warped along the move, guides LTX (sharper, holds large moves). "
                                                      "union control: the retake's edges guide LTX (the start, control and detailer settings above)"}),
                             "crossview_start": ("FLOAT", {"default": 0.7, "min": 0.3, "max": 1.0, "step": 0.05, "advanced": True,
                                                          "tooltip": "CrossView: where LTX starts on its schedule over the retake. 1.0 ignores the retake"})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("IMAGE", "FLOAT"); RETURN_NAMES = ("frames", "fps")
    OUTPUT_TOOLTIPS = ("The photoreal shot at 1024x1792 (or 1792x1024)", "Frame rate: the clip plays in real time")
    FUNCTION = "run"; CATEGORY = CAT

    def run(self, retake, seed, start_strength, control_strength, control_lora_strength, first_frame_strength, detailer, pass_type="crossview",
            crossview_start=0.7, unique_id=None):
        j = json.loads(Path(retake).read_text()); fps = float(j["fps"]); LW, LH = _ltx_size(j["width"], j["height"])
        if pass_type == "crossview":
            api, dec, info = CV.build(j, LW, LH, fps, seed, crossview_start, detailer)
            print(f"[Reshoot 4D] CrossView: pivot {info['pivot_z']:.2f} m, hfov {info['hfov']:.1f} deg, {len(info['keys'])} keys", flush=True)
            return self._expand(api, dec, fps, unique_id)
        spec = {"prompt": j["prompt"] + " Sharp natural detail.", "video": {"seconds": (j["frames"] - 1) / fps + 1e-3, "fps": fps, "width": LW, "height": LH, "seed": seed},
                "source": {"video": j["tag"] + ".mp4", "strength": start_strength},
                "control": {"video": j["tag"] + "_canny.mp4", "type": "precomputed", "strength": control_strength, "lora_strength": control_lora_strength},
                "refs": [{"image": j["first_frame"], "at": "start", "strength": first_frame_strength}],
                "prefix": f"reshoot4d/{j['tag']}", "return_image": True}
        if not detailer:
            spec["detail"] = {"lora": None}
        api, dec = sg.build(spec)
        return self._expand(api, dec, fps, unique_id)

    @staticmethod
    def _expand(api, dec, fps, unique_id):
        P = f"{unique_id}.rp."; g = {}
        for nid, n in api.items():
            g[P + nid] = {"class_type": n["class_type"], "inputs": {k: ([P + v[0], v[1]] if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in api else v)
                                                                      for k, v in n["inputs"].items()}}
        return {"result": ([P + dec, 0], fps), "expand": g}


NODE_CLASS_MAPPINGS = {"Sam3dbxRetake": Sam3dbxRetake, "Sam3dbxRetakeRun": Sam3dbxRetakeRun, "Sam3dbxRetakePhotoreal": Sam3dbxRetakePhotoreal}
NODE_DISPLAY_NAME_MAPPINGS = {"Sam3dbxRetake": "Reshoot 4D: Retake", "Sam3dbxRetakeRun": "Reshoot 4D: Retake (run)",
                              "Sam3dbxRetakePhotoreal": "Reshoot 4D: Photoreal Pass"}
