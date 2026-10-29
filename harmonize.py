"""DiffusionHarmonizer (NVIDIA, CVPR 2026) over a rendered clip: single-step Cosmos-Predict2-0.6B enhancer, each frame
conditioned on its 4 previously enhanced frames (temporal). Mirrors src/inference_pix2pix_turbo_harmonizer.py (same model
construction, normalisation and offsets) but takes/returns mp4, runs at a portrait size, and keeps references in memory.
Runs in <root>/harmonizer/.venv from the repo's src/ folder.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision import transforms

from config import HARMONIZER as H  # noqa: E402
sys.path.insert(0, str(H / "src"))
from pix2pix_turbo_harmonizer import Pix2Pix_Turbo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)                  # mp4
ap.add_argument("--output", required=True)                 # mp4
ap.add_argument("--width", type=int, default=576)          # model size; 576x1024 = portrait of the trained 1024x576
ap.add_argument("--height", type=int, default=1024)
ap.add_argument("--timestep", type=int, default=250)
ap.add_argument("--nontemporal", action="store_true")
ap.add_argument("--max_frames", type=int, default=100000)
a = ap.parse_args()

dev, dtype = torch.device("cuda"), torch.bfloat16
t0 = time.time()
model = Pix2Pix_Turbo(pretrained_path=str(H / "models" / "diffusion_harmonizer.pkl"), timestep=a.timestep, train_full_unet=True,
                      freeze_vae=False, vae_skip_connection=False, use_sched=True, device=dev, dtype=dtype)
model.set_eval(); model = model.to(device=dev, dtype=dtype)
load_s = time.time() - t0

probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height,r_frame_rate",
                                   "-of", "json", a.input], capture_output=True, text=True).stdout)["streams"][0]
W, Hh, fps = probe["width"], probe["height"], probe["r_frame_rate"]
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.input, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", fps, "-i", "-",
                        "-i", a.input, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "copy",
                        "-shortest", a.output], stdin=subprocess.PIPE)
norm = transforms.Normalize([0.5], [0.5]); size = (a.width, a.height)
offsets = [-1, -2, -3, -4]; history = []; n = 0; t1 = time.time()
with torch.no_grad():
    while n < a.max_frames:
        raw = dec.stdout.read(W * Hh * 3)
        if len(raw) < W * Hh * 3:
            break
        img = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(Hh, W, 3)).resize(size, Image.BILINEAR)
        c_t = norm(transforms.ToTensor()(img)).unsqueeze(0).to(dev, dtype)
        if not a.nontemporal and len(history) >= 4:
            stacked = rearrange(torch.stack([c_t] + [history[o] for o in offsets], dim=1), "b v c h w -> b c v h w")
        else:
            stacked = rearrange(c_t.unsqueeze(1), "b v c h w -> b c v h w")
        out = rearrange(model(stacked).float(), "b c v h w -> b v c h w")[0, 0]
        history.append(out.to(dtype).unsqueeze(0)); history = history[-4:]
        rgb = (torch.clamp(out * 0.5 + 0.5, 0, 1).cpu() * 255).byte().permute(1, 2, 0).numpy()
        enc.stdin.write(np.asarray(Image.fromarray(rgb).resize((W, Hh), Image.BICUBIC)).tobytes()); n += 1
enc.stdin.close(); enc.wait(); dec.terminate()
run_s = time.time() - t1
print(json.dumps({"frames": n, "load_s": round(load_s, 1), "run_s": round(run_s, 1), "s_per_frame": round(run_s / max(n, 1), 3),
                  "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "output": a.output}))
