"""Render real frames lifted to Gaussians (scene_layer.frame_cloud) from shot cameras, for Shot Prep's real-pixel anchors (trainer venv).

--jobs: json list of {"cloud": frame npz, "background": npz or null, "row": [scene, eye xyz, target xyz, roll, fov_y],
"width", "height", "out": png}. Writes out (RGB), out_alpha.png (where the frame or the plate lands) and out_person.png (where the
real person lands). The frame is drawn over the clean plate, so ground she uncovers as the camera moves still has the real setting.
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rig import look_at  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--up", required=True)   # the capture's up vector, json
a = ap.parse_args()
up = np.array(json.loads(a.up), float); dev = "cuda"


def gauss(npz, person_colour=False):
    z = np.load(npz); n = len(z["means"])   # world.npz has no person flag (only the frame clouds do)
    rgb = torch.tensor(z["rgb"], device=dev).float() / 255
    if person_colour:   # a white-on-black person flag instead of colour
        rgb = torch.tensor(z["person"], device=dev).float()[:, None].expand(n, 3)
    sh = torch.zeros(n, 1, 3, device=dev); sh[:, 0] = (rgb - 0.5) / 0.28209479
    return [torch.tensor(z["means"], device=dev), torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(n, 4).contiguous(),
            torch.tensor(z["scale"], device=dev)[:, None].expand(n, 3).contiguous() * 1.6, torch.ones(n, device=dev), sh]


with torch.no_grad():
    for j in json.loads(Path(a.jobs).read_text()):
        W, H = j["width"], j["height"]; r = j["row"]
        fy = H / 2 / math.tan(math.radians(r[8]) / 2)
        K = torch.tensor([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
        vm = torch.linalg.inv(torch.tensor(look_at(np.array(r[1:4]), np.array(r[4:7]), up, r[7]), dtype=torch.float32, device=dev))[None]
        R = lambda G: rasterization(*G, vm, K, W, H, sh_degree=0, packed=True, near_plane=0.01, far_plane=1e10)[:2]
        rgb, alpha = R(gauss(j["cloud"]))   # the real frame always wins; the plate only fills where the frame has nothing
        per, _ = R(gauss(j["cloud"], person_colour=True))
        real_alpha = alpha
        if j.get("background"):
            b = gauss(j["background"])
            prgb, palpha = R(b); real_alpha = alpha + palpha * (1 - alpha)   # coverage by what the source camera really saw (frame and fused set)
            rgb = rgb + prgb * (1 - alpha); alpha = alpha + palpha * (1 - alpha)
        out = Path(j["out"])
        Image.fromarray((rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out)
        Image.fromarray((alpha[0, ..., 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out.with_name(out.stem + "_alpha.png"))
        Image.fromarray((per[0, ..., 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out.with_name(out.stem + "_person.png"))
        Image.fromarray((real_alpha[0, ..., 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out.with_name(out.stem + "_real.png"))
print(json.dumps({"rendered": len(json.loads(Path(a.jobs).read_text()))}))
