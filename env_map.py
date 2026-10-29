"""The clip's 360 world as an environment map: a generated panorama of the empty set (no person, no vehicles), registered to the
real camera, looked up per shot pixel by ray direction (a skybox), with ground projection for rays that meet the floor (looked up
from where the source camera stood towards the point they hit, so the floor sits right under her from any camera). torch + numpy only,
so the renderer (trainer venv) and the nodes (ComfyUI venv) share it.

scene/world_env.json: {"panorama": png path (aligned: column centre = the source camera's forward direction), "origin", "e1", "e2", "up",
"floor", "floor_radius"}
"""
import json, math
from pathlib import Path

import numpy as np
import torch


class EnvMap:
    def __init__(self, scene_dir, device="cuda"):
        from PIL import Image
        m = json.loads((Path(scene_dir) / "world_env.json").read_text()); self.m = m; self.device = device
        pano = np.asarray(Image.open(m["panorama"]).convert("RGB"), np.float32) / 255
        self.pano = torch.tensor(pano, device=device).permute(2, 0, 1)[None]   # 1x3xHxW
        t = lambda k: torch.tensor(m[k], dtype=torch.float32, device=device)
        self.o, self.e1, self.e2, self.up = t("origin"), t("e1"), t("e2"), t("up")
        self.floor, self.rg = float(m["floor"]), float(m.get("floor_radius", 6.0))

    def render(self, eye, c2w_rot, fy, width, height, with_depth=False):
        """The world seen from a camera at eye (rotation c2w_rot, OpenCV axes, focal fy in pixels): HxWx3 float in [0, 1]; with_depth
        also returns the camera depth of the floor where rays meet it (0 elsewhere: the skybox is at infinity)."""
        dev = self.device; eye = torch.tensor(np.asarray(eye), dtype=torch.float32, device=dev)
        R = torch.tensor(np.asarray(c2w_rot), dtype=torch.float32, device=dev)
        v, u = torch.meshgrid(torch.arange(height, device=dev) + 0.5, torch.arange(width, device=dev) + 0.5, indexing="ij")
        d = torch.stack([(u - width / 2) / fy, (v - height / 2) / fy, torch.ones_like(u)], -1) @ R.T
        d = d / d.norm(dim=-1, keepdim=True)
        du = d @ self.up; t = (self.floor - eye @ self.up) / torch.where(du < -1e-4, du, torch.full_like(du, -1e-4))
        X = eye + t[..., None] * d; rel = X - self.o; hor = rel - (rel @ self.up)[..., None] * self.up
        on_floor = (du < -1e-4) & (hor.norm(dim=-1) < self.rg)
        q = torch.where(on_floor[..., None], rel, d); q = q / q.norm(dim=-1, keepdim=True)
        az = torch.atan2(q @ self.e2, q @ self.e1); el = torch.asin((q @ self.up).clamp(-1, 1))
        grid = torch.stack([az / math.pi, -el / (math.pi / 2)], -1)[None]   # x: -1..1 over 360 degrees, y: -1 (up) .. 1 (down)
        img = torch.nn.functional.grid_sample(self.pano, grid, mode="bicubic", padding_mode="border", align_corners=False)
        img = img[0].permute(1, 2, 0).clamp(0, 1)
        if with_depth:   # z-depth (along the optical axis) of the floor hits, so the depth guide turns with the camera too
            fwd = R[:, 2]; z = t * (d @ fwd)
            return img, torch.where(on_floor, z, torch.zeros_like(z))
        return img
