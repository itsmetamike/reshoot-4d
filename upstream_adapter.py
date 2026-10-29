"""Turn an upstream 4DAnyone result (GVHMR + SMPL-X pose) into the result layout the 4DGS trainer reads.

Runs in the upstream venv (needs its fdanyone package, GVHMR and the user's own SMPL-X model). Writes, next to the upstream
result, a folder with metadata.json (output block), cameras.json (OpenCV rig, video paths relative to the folder) and
preprocessing/viewer_geometry.npz (per-frame SMPL-X mesh in the cameras' canonical_human_world frame), with the generated
videos symlinked in. The trainer then seeds its Gaussians from the SMPL-X surface instead of the SAM 3D Body mesh.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--upstream_result", required=True)   # folder with cameras.json, metadata.json, gvhmr/, videos
ap.add_argument("--dest", required=True)
ap.add_argument("--upstream_root", required=True)
a = ap.parse_args()

U = Path(a.upstream_root); sys.path.insert(0, str(U))
from fdanyone.assets import MHR70_REGRESSOR  # noqa: E402
from fdanyone.motion.result import MotionResult  # noqa: E402
from fdanyone.skeleton.pipeline import _body_geometry  # noqa: E402

src, dst = Path(a.upstream_result).resolve(), Path(a.dest).resolve()
(dst / "preprocessing").mkdir(parents=True, exist_ok=True)
rig = json.loads((src / "cameras.json").read_text())
if rig.get("world_frame", {}).get("name") != "canonical_human_world" or rig.get("camera_model") != "OPENCV":
    raise SystemExit(f"unexpected rig frame/model: {rig.get('world_frame')} {rig.get('camera_model')}")
for cam in rig["cameras"]:   # hard links: the trainer rejects videos that resolve outside the result folder (symlinks do)
    v = (src / cam["video"]).resolve(); link = dst / cam["video"]
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        os.link(v, link)
(dst / "cameras.json").write_text(json.dumps(rig))
meta = json.loads((src / "metadata.json").read_text())
out = meta.get("output", meta)
(dst / "metadata.json").write_text(json.dumps({"output": out, "source": "upstream-4DAnyone-GVHMR", "upstream": str(src)}))
motion = MotionResult.load(src / "gvhmr")
geo = _body_geometry(motion, U / "models" / MHR70_REGRESSOR, U / "third_party" / "GVHMR", "cpu", include_mesh=True)
verts = np.asarray(geo.mesh_world, dtype=np.float32); faces = np.asarray(geo.mesh_faces)
frames = int(out["frames_per_video"])
if verts.shape[0] != frames:      # motion may cover the source clock; resample to the generated frame count
    idx = np.round(np.linspace(0, verts.shape[0] - 1, frames)).astype(int); verts = verts[idx]
np.savez_compressed(dst / "preprocessing" / "viewer_geometry.npz", vertices=verts, faces=faces.astype(np.int64))
print(json.dumps({"dest": str(dst), "cameras": len(rig["cameras"]), "frames": frames, "vertices": list(verts.shape)}))
