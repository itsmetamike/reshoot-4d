# Setup (Linux, one NVIDIA GPU with 24 GB, tested on Ubuntu 24.04)

The documented setup uses Linux and an NVIDIA GPU with 24 GB VRAM, tested on Ubuntu 24.04. The default data root is `/data/sam3dbx`; override it in `config.local.json` (step 4).

```
/data/sam3dbx/
  upstream/   4DAnyone (tracking + synced views)
  repo/       sam3dbx4danyone, with third_party/FreeTimeGsVanilla (the 4D splat trainer)
  cuda128/    CUDA 12.8 nvcc for the trainer's extensions
  out/        captures
```

## 0. 4DAnyone (Python 3.11)

```bash
mkdir -p /data/sam3dbx && cd /data/sam3dbx
git clone https://github.com/ant-research/4DAnyone.git upstream && cd upstream
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_smplx.py      # SMPL-X needs a free registration on its site first
.venv/bin/python scripts/download_model.py
```

- Capture runs `inference.py` from this folder, so keep the venv at `upstream/.venv`.
- Missing models also download on first use, but doing it up front keeps the first capture from stalling.

## 1. Generator: sam3dbx4danyone (Python 3.11, torch 2.8 cu128)

```bash
cd /data/sam3dbx
git clone --recurse-submodules https://github.com/OpsiClear-4DGS/sam3dbx4danyone.git repo
# keep caches off a small root disk
cat > env.sh <<'EOF'
export UV_CACHE_DIR=/data/sam3dbx/uvcache UV_PYTHON_INSTALL_DIR=/data/sam3dbx/pythons HF_HOME=/data/sam3dbx/hf TMPDIR=/data/sam3dbx/tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EOF
source env.sh && mkdir -p tmp
cd repo && UV_HTTP_TIMEOUT=120 uv sync --extra trt
.venv/bin/python scripts/download_model.py --turbo=True      # ~21 GB
.venv/bin/python scripts/download_example.py
```

- If `uv sync` hangs on "Downloading torch" with no traffic, kill it and re-run with `UV_CONCURRENT_DOWNLOADS=1`.
- The bundled SageAttention wheel includes an sm80 kernel, so it runs on RTX 30 cards as-is.
- Generation needs almost all of a 24 GB card. Stop anything else using the GPU; the nodes unload ComfyUI's own models first.

## 2. Trainer: FreeTimeGsVanilla (Python 3.12, torch 2.10 **cu128**)

The lockfile's torch is 2.10+cu128, but uv's build isolation compiles gsplat / fused-ssim / torch_scatter against a cu130 torch, which gives `undefined symbol` errors at import. Build them against the venv's own torch with a matching CUDA 12.8 nvcc and C++20:

```bash
micromamba create -y -p /data/sam3dbx/cuda128 -c nvidia/label/cuda-12.8.1 -c conda-forge cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-nvrtc-dev
C=/data/sam3dbx/cuda128; T=$C/targets/x86_64-linux
export CUDA_HOME=$C PATH=$PATH:$C/bin TORCH_CUDA_ARCH_LIST=8.6 CC=/usr/bin/gcc CXX=/usr/bin/g++ \
       CPATH=$T/include LIBRARY_PATH=$T/lib:$C/lib NVCC_APPEND_FLAGS=-std=c++20 FORCE_CUDA=1 MAX_JOBS=8
cd /data/sam3dbx/repo/third_party/FreeTimeGsVanilla
uv sync --locked                       # may fail on the CUDA extensions; the venv is still created
P=.venv/bin/python; uv pip install --python $P setuptools wheel ninja
uv pip install --python $P --no-build-isolation --reinstall --no-deps --no-cache \
  "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157" \
  "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@b60e917c95afc449c5be33a634f1f457e116ff5e" \
  torch_scatter==2.1.2
```

- Use the system gcc as the host compiler. Conda's gcc can't see the system Python headers.
- `--no-cache` matters, because uv will otherwise reuse the broken wheels it built earlier.
- `TORCH_CUDA_ARCH_LIST` is your card's architecture: 8.6 for RTX 30, 8.9 for RTX 40, 12.0 for RTX 50. Use the same value as `cuda_arch` in step 4.
- gsplat compiles its kernels the first time it's imported; the nodes export the same CUDA variables.
- TSOG packaging needs Node 22: `npm ci --ignore-scripts --prefix third_party/tsog && npm run build --prefix third_party/tsog`.

## 3. The node pack

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/itsmetamike/reshoot-4d.git reshoot-4d
# optional: trainer overrides hook
cd /data/sam3dbx/repo && git apply /path/to/ComfyUI/custom_nodes/reshoot-4d/patches/train_worker_overrides.patch
# optional: Capture's "subject" choice (left / right person in videos with more than one)
cd /data/sam3dbx/upstream && git apply /path/to/ComfyUI/custom_nodes/reshoot-4d/patches/upstream_subject_pick.patch
# Capture's "quick" mode (tracking + source camera + setting, no generated views): needed for quick captures
cd /data/sam3dbx/upstream && git apply /path/to/ComfyUI/custom_nodes/reshoot-4d/patches/upstream_quick_capture.patch
```

The background layer (the video's setting around the person) uses ComfyUI's built-in Depth Anything 3 and MoGe-2 support. Put both in `ComfyUI/models/geometry_estimation/`:

- `depth_anything_3_base.safetensors` from [Comfy-Org/Depth-Anything-3](https://huggingface.co/Comfy-Org/Depth-Anything-3/tree/main/geometry_estimation). It handles depth and the source camera's motion across the video.
- `moge_2_vitl_normal_fp16.safetensors`. It finds the sky.

Install both setting models before using the main Retake workflow.

## 4. Config

Nothing to edit in the code. If your folder or GPUs differ from the defaults, create `config.local.json` next to `config.py` (or set the matching `SAM3DBX_*` environment variables):

```json
{"root": "/data/sam3dbx", "cuda_arch": "8.6"}
```

- One GPU: that's all. Every step runs on GPU 0 and ComfyUI's models are unloaded before the heavy steps.
- Two GPUs: add `"gpu_big": "0", "gpu_small": "1"` (indexes or UUIDs from `nvidia-smi -L`). Capture and training use the big one; depth and splat renders use the small one.

## 5. Photoreal Pass (retake), LTX 2.3

Custom nodes: [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) and [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes).

Models, in the usual ComfyUI folders:

| File | Folder |
|---|---|
| `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors` | `models/diffusion_models` |
| `gemma_3_12B_it_fp4_mixed.safetensors` | `models/text_encoders` |
| `ltx-2.3_text_projection_bf16.safetensors` | `models/text_encoders` |
| `LTX23_video_vae_bf16.safetensors` | `models/vae` |
| `LTX23_audio_vae_bf16.safetensors` | `models/vae` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `models/latent_upscale_models` |
| `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | `models/loras` |
| `ltx-2-19b-ic-lora-detailer.safetensors` (only with `detailer` on) | `models/loras` |

For automatic Retake captions, install `qwen3vl_8b_int8_convrot.safetensors` or the supported bf16 variant in `ComfyUI/models/text_encoders/`. Enter a manual prompt to bypass captioning. Gemma remains required for LTX text encoding.

Restart ComfyUI. The nodes are under **Reshoot 4D**.
