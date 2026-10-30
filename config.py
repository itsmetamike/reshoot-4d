"""Where the pack finds its environments and which GPUs it uses. Plain module (no package imports) so the subprocess scripts
(render_path.py, rig.py, source_cam.py ...) can `import config` too.

Each value comes from config.local.json next to this file, then an environment variable, then the default:
  root          SAM3DBX_ROOT        ~/reshoot4d      holds repo/ (sam3dbx4danyone), upstream/ (4DAnyone), cuda128/, out/
  gpu_big       SAM3DBX_GPU_BIG     0                CUDA device (index or UUID) for view generation and LTX, needs 24 GB
  gpu_small     SAM3DBX_GPU_SMALL   gpu_big          device for depth renders and 4DGS training (~2-3 GB); a second card lets
                                                     training run while the big one generates
  cuda_arch     SAM3DBX_CUDA_ARCH   8.6              TORCH_CUDA_ARCH_LIST for gsplat's JIT (8.6 = RTX 30, 8.9 = RTX 40, 12.0 = RTX 50)
  redirector    SAM3DBX_REDIRECTOR  root/redirector  ReDirector checkout with its own .venv and models/ (Retake node)
"""
import json, os
from pathlib import Path

_local = Path(__file__).resolve().parent / "config.local.json"
_cfg = json.loads(_local.read_text()) if _local.is_file() else {}


def _get(key, default):
    for v in (_cfg.get(key), os.environ.get(f"SAM3DBX_{key.upper()}")):   # 0 is a real GPU index: only None / "" fall through
        if v is not None and v != "":
            return v
    return default


ROOT = Path(_get("root", str(Path.home() / "reshoot4d")))
REPO = ROOT / "repo"
UPSTREAM = ROOT / "upstream"
FTGS = REPO / "third_party" / "FreeTimeGsVanilla"
GEN_PY = str(REPO / ".venv" / "bin" / "python")
TRAIN_PY = str(FTGS / ".venv" / "bin" / "python")
UP_PY = str(UPSTREAM / ".venv" / "bin" / "python")
CUDA128 = ROOT / "cuda128"
HARMONIZER = ROOT / "harmonizer"
REDIRECTOR = Path(_get("redirector", str(ROOT / "redirector")))
RD_PY = str(REDIRECTOR / ".venv" / "bin" / "python")
CUDA_ARCH = str(_get("cuda_arch", "8.6"))
_big = str(_get("gpu_big", "0"))
GPUS = {"big": _big, "small": str(_get("gpu_small", _big))}   # big card: view generation, LTX, ReDirector; small card: depth and splat renders
