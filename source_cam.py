"""The real (input video) camera in a capture's world, so shots can start on it and Shot Prep can anchor to real footage.

PnP of GVHMR's observed 2D COCO-17 keypoints against the capture's own SMPL-X mesh (-> SMPL -> COCO-17 joints), with GVHMR's
intrinsics. A tripod clip gets one camera solved over all frames (per-frame solves jitter by tens of cm on pose noise); a moving
camera keeps per-frame solves, smoothed. Writes <capture>/source_camera.json: c2w per capture frame (OpenCV, capture world), K
at the source clip's size, the clip path. Run with the upstream venv (torch, cv2, safetensors):
    python source_cam.py <capture_dir>
"""
import json, sys
from pathlib import Path

import cv2, numpy as np, torch
from safetensors.numpy import load_file

from config import UPSTREAM  # noqa: E402
GVHMR_BM = UPSTREAM / "third_party" / "GVHMR" / "hmr4d" / "utils" / "body_model"


def solve(cap_dir):
    cap = Path(cap_dir)
    adapt = Path(json.loads((cap / "dataset" / "manifest.json").read_text())["source"])
    upstream = Path(json.loads((adapt / "metadata.json").read_text())["upstream"])
    clip = next(iter(sorted(upstream.parent.glob("source_*.mp4"))), None)   # Capture Subject's trimmed input (1:1 with capture frames)
    if clip is None:
        raise FileNotFoundError(f"no source_*.mp4 next to {upstream}")
    V = np.load(adapt / "preprocessing" / "viewer_geometry.npz")["vertices"].astype(np.float64)
    jr = torch.load(GVHMR_BM / "smpl_coco17_J_regressor.pt"); jr = jr.to_dense().numpy() if hasattr(jr, "to_dense") else np.asarray(jr)
    J = np.einsum("jv,fvk->fjk", jr @ torch.load(GVHMR_BM / "smplx2smpl_sparse.pt").to_dense().numpy(), V)   # (F, 17, 3)
    m = load_file(str(upstream / "gvhmr" / "motion.safetensors")); mj = json.loads((upstream / "gvhmr" / "motion.json").read_text())
    K = m["K_fullimg"].astype(np.float64); kp = m["observed_keypoints_2d"].astype(np.float64); F = len(V)
    ok = kp[:, :, 2] > 0.5

    def pnp(X, x, Kf, guess=None):
        if guess is None:
            _, rv, tv = cv2.solvePnP(X, x, Kf, None, flags=cv2.SOLVEPNP_EPNP)
        else:
            rv, tv = guess
        _, rv, tv = cv2.solvePnP(X, x, Kf, None, rv.copy(), tv.copy(), True, flags=cv2.SOLVEPNP_ITERATIVE)
        return rv, tv

    def err(rv, tv, X, x, Kf):
        return float(np.linalg.norm(cv2.projectPoints(X, rv, tv, Kf, None)[0][:, 0] - x, axis=1).mean())

    kp2 = [np.ascontiguousarray(kp[f, ok[f], :2]) for f in range(F)]
    per = [pnp(J[f, ok[f]], kp2[f], K[f]) for f in range(F)]
    e_per = np.mean([err(*per[f], J[f, ok[f]], kp2[f], K[f]) for f in range(F)])
    Kc = np.median(K, 0)   # GVHMR's K is per frame but constant for a fixed lens
    Xa, xa = J[ok], np.ascontiguousarray(kp[ok][:, :2])
    st = pnp(Xa, xa, Kc, per[0]); e_st = err(*st, Xa, xa, Kc)
    static = e_st < 1.5 * e_per + 3.0
    if static:
        Rm = cv2.Rodrigues(st[0])[0]; c2w = np.eye(4); c2w[:3, :3] = Rm.T; c2w[:3, 3] = (-Rm.T @ st[1]).ravel(); poses = [c2w] * F
    else:
        pos, fwd, dwn = [], [], []
        for rv, tv in per:
            Rm = cv2.Rodrigues(rv)[0]; pos.append(-Rm.T @ tv.ravel()); fwd.append(Rm[2]); dwn.append(Rm[1])
        sm = lambda a: np.stack([np.convolve(np.pad(a[:, k], 4, mode="edge"), np.ones(9) / 9, "valid") for k in range(3)], 1)
        pos, fwd, dwn = sm(np.array(pos)), sm(np.array(fwd)), sm(np.array(dwn)); poses = []
        for p, z, y in zip(pos, fwd, dwn):
            z = z / np.linalg.norm(z); x = np.cross(y, z); x /= np.linalg.norm(x); y = np.cross(z, x)
            c2w = np.eye(4); c2w[:3, :3] = np.stack([x, y, z], 1); c2w[:3, 3] = p; poses.append(c2w)
    out = {"video": str(clip), "width": mj["image_width"], "height": mj["image_height"], "K": Kc.tolist(), "static": bool(static),
           "reproj_px": round(float(e_st if static else e_per), 2), "c2w": [p.tolist() for p in poses]}
    (cap / "source_camera.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    o = solve(sys.argv[1])
    print(json.dumps({k: o[k] for k in ("video", "width", "height", "static", "reproj_px")}))
