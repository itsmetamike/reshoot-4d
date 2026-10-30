# Install Reshoot 4D

Reshoot 4D runs Capture -> Virtual Camera -> Retake -> Photoreal Pass inside ComfyUI. The Virtual Camera opens the 4D camera editor. Start with [the Reshoot 4D workflow](../workflows/Reshoot%204D.json), then follow [Workflow](WORKFLOW.md).

## Hardware and storage

Every size, SHA-256, commit and version below was measured on the reference machine on 2026-10-28. Where something could not be measured, it says unknown. The full recipe has not yet been replayed on a clean machine.

The reference system is Ubuntu 24.04 with one NVIDIA RTX 3090 (24 GB VRAM) and 64 GB RAM. Use one 24 GB NVIDIA GPU and about 64 GB RAM to match the reference. Peak RAM and a tested minimum RAM requirement are unknown. config.py defaults both GPU selections to the same device. The second GPU is optional for depth and splat renders; GPU memory is separate.

| Measured directory relative to ROOT | Rounded allocated size | Included nested sizes |
| --- | --- | --- |
| repo | 41G | repo/.venv 15G; trainer .venv 8.8G |
| upstream | 28G | upstream/.venv 7.4G |
| redirector | 29G | redirector/.venv 8.7G; models 20G |
| cuda128 | 3.8G | CUDA 12.8 build toolchain |
| hf | 4.4M | Separate from the other cache listings |

The non-overlapping totals sum to about 101.8 GiB. Do not add nested environments or redirector/models again. The nine ComfyUI files used for automatic Retake captioning and default Photoreal Pass (CrossView) total 48,374,189,704 bytes (45.05 GiB). Published DA3 and MoGe files add about 1.12 GiB; their installed sizes are unknown. These account for about 148.0 GiB before ComfyUI, package/build caches, temporary files, videos and outputs. Reserve at least 200 GiB free as a planning allowance, with more for repeated captures. Each capture writes about 4.6 GB (measured for a 4.4 s video). A measured minimum disk requirement is unknown. Full checkout measurements include assets outside the public graph.

## Paths and repository revisions

Use a Linux machine with an NVIDIA driver, Git, uv, a C/C++ compiler, FFmpeg and ffprobe. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) and Node 22 before the commands below. On Ubuntu, Git, compilers and FFmpeg can be installed with `sudo apt-get install git build-essential python3-dev ffmpeg curl`. The 4DGS build also uses micromamba and Node 22 according to [SETUP.md](SETUP.md). Versions of Git, uv, FFmpeg, compilers and micromamba are unknown. Check that `ffmpeg -version` and `ffprobe -version` work in the service environment; Capture invokes both by name.

The following is an example layout. Install uv, micromamba and Node 22 using their linked instructions before continuing. `ROOT` is an absolute data directory you choose. The default in `config.py` is `~/reshoot4d`.

```text
ROOT/
  upstream/.venv/                         Capture, Python 3.11
  repo/.venv/                             Adaptation/generator, Python 3.11
  repo/third_party/FreeTimeGsVanilla/.venv/ Trainer, Python 3.12
  redirector/.venv/                       Retake, Python 3.11
  comfy_nodes/                            This node pack
  cuda128/                                CUDA 12.8 build toolchain
  pythons/  hf/  uvcache/  tmp/  out/
ComfyUI/
  custom_nodes/reshoot-4d/                 Checkout or link to ROOT/comfy_nodes
  models/                                 Logical model paths below
```

`config.py` sets `REPO=ROOT/repo`, `UPSTREAM=ROOT/upstream`, `FTGS=REPO/third_party/FreeTimeGsVanilla` and a separate Python executable under each `.venv/bin/python`. It also defines `ROOT/harmonizer` for a separate helper; use by the baseline Retake graph is not established. The baseline graph does not call that helper.

| Repository | Runtime target | Measured commit |
| --- | --- | --- |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI.git) | ComfyUI | `e638023d54497dbe0579565e5de4bb7076899592` |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | ComfyUI/custom_nodes/ComfyUI-KJNodes | `d3cfe21625e5170126ce06fbfcfe1d88108688c3` |
| [ComfyUI-CrossViewWarp](https://github.com/cseti007/ComfyUI-CrossViewWarp) | ComfyUI/custom_nodes/ComfyUI-CrossViewWarp | `3266a83a84eaa6833a9ebc38ac7cdd8789e44f5b` |
| [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) | ComfyUI/custom_nodes/ComfyUI-LTXVideo | `3bf3ca62595f1764c47d01c35c8e5dfe47e1a88f` |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite | `4d907bee61e92c2e65af3bd6383a4e4d356126d1` |
| [repo](https://github.com/OpsiClear-4DGS/sam3dbx4danyone.git) | ROOT/repo | `1356955f5c1c3ecf1ab636af52048d4e312c1069` |
| [upstream](https://github.com/ant-research/4DAnyone.git) | ROOT/upstream | `6dcf626d74bd95c42792f86b1d55e3841075c10c` |
| [GVHMR](https://github.com/zju3dv/GVHMR.git) | ROOT/upstream/third_party/GVHMR | `6ec3ca39336c50492c0fae65fba2fb831fc7d866` |
| [FreeTimeGsVanilla](https://github.com/OpsiClear-4DGS/FreeTimeGsVanilla.git) | ROOT/repo/third_party/FreeTimeGsVanilla | `42e2f9b974ff4b6841e97a1e388aac7f73f52a01` |
| [tsog](https://github.com/xiaomi-research/tsog.git) | ROOT/repo/third_party/tsog | `a58102d0489a68c4f7e18e5e1ddaf613256417dd` |
| [redirector](https://github.com/byeongjun-park/ReDirector.git) | ROOT/redirector | `ede9779ec09fb22b2120381deb39d824f1f3e3c7` |

The installed Reshoot 4D commit is unknown in the inventory. Use the submission checkout. Other installed custom-node packs are outside the baseline. LTXVideo is recorded for reference; the generated graph uses core LTX nodes, KJNodes, VideoHelperSuite and ComfyUI-CrossViewWarp. ManualSigmas is a [core node](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_ltx2_i2v_distilled.json).

After each clone below, check out the corresponding commit in this table before installing dependencies. Initialize submodules after pinning the parent and check out the nested commits from the table. Use HTTPS submodule URLs. Local patch state is unknown.

## Create the environments on a new machine

Keep the separate environments required by config.py and the subprocess launchers.

| Runtime | Measured Python | Measured torch | torch CUDA runtime |
| --- | --- | --- | --- |
| ROOT/upstream/.venv | 3.11.17 | 2.8.0+cu128 | 12.8 |
| ROOT/repo/.venv | 3.11.17 | 2.8.0+cu128 | 12.8 |
| ROOT/redirector/.venv | 3.11.17 | 2.8.0+cu128 | 12.8 |
| ROOT/repo/third_party/FreeTimeGsVanilla/.venv | 3.12.3 | 2.10.0+cu128 | 12.8 |
| ComfyUI (its own venv) | 3.12.3 | 2.14.0+cu130 | 13.0 |

The reference env.sh exports PATH with ROOT/uvenv/bin, UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, HF_HOME, TMPDIR, CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES set to the RTX 3090 UUID. The portable equivalent below uses device 0. Select your GPU index or UUID with nvidia-smi -L. Save these exports in ROOT/env.sh and source it before launching ComfyUI.

Choose writable absolute paths, then keep these exports in the shell or service that launches ComfyUI:

```bash
export SAM3DBX_ROOT=/data/reshoot4d
export COMFY_DIR=/opt/ComfyUI
export UV_CACHE_DIR="$SAM3DBX_ROOT/uvcache"
export UV_PYTHON_INSTALL_DIR="$SAM3DBX_ROOT/pythons"
export HF_HOME="$SAM3DBX_ROOT/hf"
export TMPDIR="$SAM3DBX_ROOT/tmp"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$SAM3DBX_ROOT" "$TMPDIR"
export CUDA_VISIBLE_DEVICES=0
export PATH="$SAM3DBX_ROOT/uvenv/bin:$PATH"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$SAM3DBX_ROOT/out"
uv python install 3.11.17 3.12.3
```

Clone and pin ComfyUI before adding custom nodes:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
git -C "$COMFY_DIR" checkout e638023d54497dbe0579565e5de4bb7076899592
```

Follow its [Linux installation instructions](https://github.com/Comfy-Org/ComfyUI#manual-install-windows-linux). Use ComfyUI's own Python environment for its requirements and custom-node requirements. The reference ComfyUI runs Python 3.12.3 with torch 2.14.0 cu130. Do not merge the four subprocess environments into ComfyUI's environment.

```bash
cd "$SAM3DBX_ROOT"
git clone --recurse-submodules https://github.com/ant-research/4DAnyone.git upstream
git clone --recurse-submodules https://github.com/OpsiClear-4DGS/sam3dbx4danyone.git repo
git -C upstream checkout 6dcf626d74bd95c42792f86b1d55e3841075c10c
git -C upstream submodule update --init --recursive
git -C upstream/third_party/GVHMR checkout 6ec3ca39336c50492c0fae65fba2fb831fc7d866
git -C repo checkout 1356955f5c1c3ecf1ab636af52048d4e312c1069
git -C repo submodule update --init --recursive
git -C repo/third_party/FreeTimeGsVanilla checkout 42e2f9b974ff4b6841e97a1e388aac7f73f52a01
git -C repo/third_party/tsog checkout a58102d0489a68c4f7e18e5e1ddaf613256417dd
cd upstream
uv venv --python 3.11.17 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python scripts/download_smplx.py
.venv/bin/python scripts/download_model.py
cd ../repo
UV_HTTP_TIMEOUT=120 uv sync --python 3.11.17 --extra trt
```

SMPL-X requires separate registration and acceptance of its terms. Complete its interactive download before running Capture in ComfyUI. Current upstream's model downloader pins its assets internally. Older SETUP.md instructions pass `--turbo=True` to the adaptation downloader, while the current public downloader has a different signature. Check the selected checkout's `scripts/download_model.py` before using that flag. The baseline Capture node invokes `upstream/inference.py`; adaptation's `.venv` is still needed for the rig and trainer launcher.

Build the trainer as described in [SETUP.md section 2](SETUP.md#2-trainer-freetimegsvanilla-python-312-torch-210-cu128). Its documented torch is 2.10 cu128; Capture and adaptation use torch 2.8. Use the matching CUDA toolchain when compiling extensions:

```bash
micromamba create -y -p "$SAM3DBX_ROOT/cuda128" -c nvidia/label/cuda-12.8.1 -c conda-forge cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-nvrtc-dev
export CUDA_HOME="$SAM3DBX_ROOT/cuda128"
export PATH="$PATH:$CUDA_HOME/bin"
export TORCH_CUDA_ARCH_LIST=8.6
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export CPATH="$CUDA_HOME/targets/x86_64-linux/include"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib"
export NVCC_APPEND_FLAGS=-std=c++20 FORCE_CUDA=1 MAX_JOBS=8
cd "$SAM3DBX_ROOT/repo/third_party/FreeTimeGsVanilla"
uv sync --locked --python 3.12.3
# If sync failed specifically while building these extensions, repair them below.
uv pip install --python .venv/bin/python setuptools wheel ninja
uv pip install --python .venv/bin/python --no-build-isolation --reinstall --no-deps --no-cache \
  'fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157' \
  'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@b60e917c95afc449c5be33a634f1f457e116ff5e' \
  torch_scatter==2.1.2
# TSOG packaging is optional for the public video workflow.
# From ROOT/repo, build the measured TSOG checkout if exporting assets:
cd "$SAM3DBX_ROOT/repo"
npm ci --ignore-scripts --prefix third_party/tsog
npm run build --prefix third_party/tsog
```

Use `8.6` for the reference RTX 30-series cards. Confirm that the selected trainer checkout creates a Python 3.12 environment and installs torch 2.10 cu128 before compiling. An unrelated sync error needs its own fix. Kernels can compile on first use, so an import is not a harmless live-machine audit.

## Install the pack and graph providers

```bash
cd "$COMFY_DIR/custom_nodes"
git clone https://github.com/itsmetamike/reshoot-4d.git reshoot-4d
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git -C ComfyUI-KJNodes checkout d3cfe21625e5170126ce06fbfcfe1d88108688c3
git -C ComfyUI-VideoHelperSuite checkout 4d907bee61e92c2e65af3bd6383a4e4d356126d1
git clone https://github.com/cseti007/ComfyUI-CrossViewWarp.git
git -C ComfyUI-CrossViewWarp checkout 3266a83a84eaa6833a9ebc38ac7cdd8789e44f5b   # no extra requirements; numba is an optional speed-up
cd "$COMFY_DIR"
uv venv --python 3.12.3 .venv
uv pip install --python .venv/bin/python torch==2.14.0 torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -r custom_nodes/ComfyUI-KJNodes/requirements.txt -r custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt
uv pip install --python .venv/bin/python opencv-python scipy
```

Check out ComfyUI, KJNodes and VideoHelperSuite at the measured commits before installing their requirements. For a new ComfyUI checkout, clone https://github.com/comfyanonymous/ComfyUI.git to COMFY_DIR first. The pack has no root requirements.txt. In ComfyUI's environment, install ComfyUI's requirements, both providers' requirements.txt files, OpenCV and SciPy. NumPy, Pillow and torch are also direct imports.

ComfyUI itself runs on Python 3.12.3 and torch 2.14.0 cu130 on the reference machine. Install torch 2.8.0 and torchvision 0.23.0 from https://download.pytorch.org/whl/cu128 in the upstream and ReDirector environments after their requirements, matching the measured runtime. Verify repo's uv sync resolves torch 2.8.0+cu128. The trainer must use 2.10.0+cu128.

The graph requires core ManualSigmas, LTX guide/conditioning/A/V latent nodes, GetICLoRAParameters, TextGenerate, DA3 and MoGe support. KJNodes supplies VAELoaderKJ. VideoHelperSuite supplies VHS_LoadVideo and VHS_VideoCombine. ComfyUI-CrossViewWarp supplies CrossViewWarp; MoGeInference and LoadMoGeModel are core nodes. Inspect the local /object_info endpoint for missing classes. The generated graph was inspected without loading weights.

Create `config.local.json` beside this pack's `config.py`:

```json
{"root": "/data/reshoot4d", "cuda_arch": "8.6", "gpu_big": "0", "gpu_small": "0"}
```

Local JSON takes precedence over `SAM3DBX_ROOT`, `SAM3DBX_CUDA_ARCH`, `SAM3DBX_GPU_BIG`, `SAM3DBX_GPU_SMALL` and `SAM3DBX_REDIRECTOR`. A stale local root will override your shell. For two cards, set `gpu_small` to the second device's index or UUID. `redirector` can override `ROOT/redirector`.

Check patches against your chosen Capture checkout before applying them on a fresh install:

```bash
export PACK_DIR="$COMFY_DIR/custom_nodes/reshoot-4d"
git -C "$SAM3DBX_ROOT/upstream" apply --check "$PACK_DIR/patches/upstream_quick_capture.patch"
git -C "$SAM3DBX_ROOT/upstream" apply "$PACK_DIR/patches/upstream_quick_capture.patch"
# Needed for left/right subject selection:
git -C "$SAM3DBX_ROOT/upstream" apply --check "$PACK_DIR/patches/upstream_subject_pick.patch"
git -C "$SAM3DBX_ROOT/upstream" apply "$PACK_DIR/patches/upstream_subject_pick.patch"
```

An apply failure means the selected revision needs review. Which patches were already applied to the measured revisions is unknown. The optional trainer override patch is documented in SETUP.md.

Install ReDirector in its separate Python 3.11 environment:

```bash
cd "$PACK_DIR"
git clone https://github.com/byeongjun-park/ReDirector.git "$SAM3DBX_ROOT/redirector"
git -C "$SAM3DBX_ROOT/redirector" checkout ede9779ec09fb22b2120381deb39d824f1f3e3c7
bash scripts/install_redirector.sh "$SAM3DBX_ROOT/redirector"
```

The local installer installs upstream requirements plus natsort with uv. Current upstream requires torch 2.8.0 and torchvision 0.23.0 using the cu128 index. It downloads from moving `main` URLs. Direct-download model revisions and complete dependency locks are unknown. Verify downloads against the recorded hashes below.

## Model manifest

Read [Models and licences](MODELS_AND_LICENSES.md) before downloading. Exact bytes and hashes below come from the inventory unless labelled public. Missing values are unknown. A matching published hash establishes file identity, while original download history can remain unknown. Moving-main source revisions are unknown unless a pinned release is stated.

The saved `workflows/Reshoot 4D.json` selects **4DGS splat (train)**, 24 views, 15,000 training steps, a blank Retake prompt, 480p/50-step Retake and enabled Photoreal detailer. Required models include Capture body/generation assets, DA3 and MoGe for the setting preview, all ReDirector components, Qwen for automatic captioning, and the eight LTX components. Quick Capture skips generated views and splat training. A manual prompt bypasses Qwen; disabling detailer removes its LoRA. Virtual Camera opens the 4D camera editor from the prepared capture and loads no additional pretrained weights.

### ComfyUI models

Targets below use complete paths relative to the installation directory. ReDirector and Capture use separate ROOT paths because their subprocess loaders do not search ComfyUI/models. The inventory records these model folders; additional path mappings are unknown. Open a source file link and use its Download action, keeping the exact basename in the target folder.

| Exact filename | Target folder | Bytes (measured unless public) | SHA-256 | Source |
| --- | --- | --- | --- | --- |
| `depth_anything_3_base.safetensors` | `ComfyUI/models/geometry_estimation/depth_anything_3_base.safetensors` | 541,524,124 bytes | `418c0d2ea857e2d1215fa51baa46833f499a62eb2400ec63d337aa20d326414f` | [Comfy-Org/Depth-Anything-3](https://huggingface.co/Comfy-Org/Depth-Anything-3/blob/657708b0ec5f218cbf14bff0bb2cb8c58ea3fe33/geometry_estimation/depth_anything_3_base.safetensors) |
| `moge_2_vitl_normal_fp16.safetensors` | `ComfyUI/models/geometry_estimation/moge_2_vitl_normal_fp16.safetensors` | 661,859,924 bytes | `cb1a692d03235671e959e81360d7b4d9f44aefadb1f852d6ca6aa17799d5e31f` | [Comfy-Org/MoGe](https://huggingface.co/Comfy-Org/MoGe/blob/7f19f325949e1e7bb31c720333936f03cf932f57/geometry_estimation/moge_2_vitl_normal_fp16.safetensors) |
| `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors` | 21,505,993,424 bytes | `30fe2173fdb18881eb2482ece17ca044bedb8f28a38550ea7230da796fb8b614` | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy/blob/a5a175044e7607277812c00b98a27718a4c42710/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors) |
| `gemma_3_12B_it_fp4_mixed.safetensors` | `ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` | 9,447,702,218 bytes | `aaca463d11e6d8d2a4bdb0d6299214c15ef78a3f73e0ef8113d5a9d0219b3f6d` | [Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2/blob/fff639871e076110fc09c4cc43353c82a1b15ef1/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors) |
| `ltx-2.3_text_projection_bf16.safetensors` | `ComfyUI/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors` | 2,312,149,072 bytes | `911d59bb4cb7708179c9a0045ea0fe41212ecfb77aed3a02702b7c0a8274911f` | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy/blob/1c032c322c3db1992a66a7690d414b4fe8e686d0/text_encoders/ltx-2.3_text_projection_bf16.safetensors) |
| `LTX23_video_vae_bf16.safetensors` | `ComfyUI/models/vae/LTX23_video_vae_bf16.safetensors` | 1,452,258,578 bytes | `01ea62d09bc139f95c5dee7b5c062ad6a3e6cd8be910a1983ac02e7eb5b8ee3b` | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy/blob/a422e5661a18bde257c0ef1c3761d8b9ce1eb645/vae/LTX23_video_vae_bf16.safetensors) |
| `LTX23_audio_vae_bf16.safetensors` | `ComfyUI/models/vae/LTX23_audio_vae_bf16.safetensors` | 364,855,188 bytes | `5bc10fa4adecf99dda132d916e23048cbd56797702c5fa50eb5d2079048a38c3` | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy/blob/a422e5661a18bde257c0ef1c3761d8b9ce1eb645/vae/LTX23_audio_vae_bf16.safetensors) |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 995,743,560 bytes | `5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed` | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3/blob/cd784f3198e9ec3efec60b66a0fd78aafe413a86/ltx-2.3-spatial-upscaler-x2-1.1.safetensors) |
| `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | `ComfyUI/models/loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | 654,465,352 bytes | `a1b888a87f661d27f08b394ae559e8e1050be33900bcc36a5cdf659e48f88d18` | [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/blob/b4d1c4d8c9e544e9bbbd6811bb4363708b6093ff/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors) |
| `ltx-2-19b-ic-lora-detailer.safetensors` | `ComfyUI/models/loras/ltx-2-19b-ic-lora-detailer.safetensors` | 2,617,401,920 bytes | `05efdae9e472e06d168e122f5ebb890e7ef348cc047cf9876da6504c36d7d0e2` | [Lightricks/LTX-2-19b-IC-LoRA-Detailer](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/blob/70afe09e20511305b7c0a471bded8e8babe7beab/ltx-2-19b-ic-lora-detailer.safetensors) |
| `LTX2.3-22B_IC-LoRA-CrossView-Warp_v2_6000.safetensors` | `ComfyUI/models/loras/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2_6000.safetensors` | 327,287,384 bytes | `e9796a6e14a8f8fadf0eda6c0ccc6c300c02bd5457fe9a0d4b91ffec39badda3` | [Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2) |
| `qwen3vl_8b_int8_convrot.safetensors` | `ComfyUI/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 9,350,798,360 bytes | `8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f` | [Kutches/Qwen](https://huggingface.co/Kutches/Qwen/blob/main/qwen3vl_8b_int8_convrot.safetensors) |

Gemma is required for LTX text encoding. Qwen is needed only when Retake's prompt is empty; enter a manual prompt to bypass it. The Qwen link identifies a published same-name conversion; the published SHA-256 matches the inventory. The audio VAE is loaded for A/V latent initialization even when only video frames are returned. Detailer is enabled in the example workflow and can be disabled. The 19B detailer is applied to the 2.3 22B base by the local graph; compatibility is not independently measured here.

Public file-upload revisions verified from HF commit pages: [LTX transformer a5a175044e7607277812c00b98a27718a4c42710](https://huggingface.co/Kijai/LTX2.3_comfy/commit/a5a175044e7607277812c00b98a27718a4c42710); [Gemma fff639871e076110fc09c4cc43353c82a1b15ef1](https://huggingface.co/Comfy-Org/ltx-2/commit/fff639871e076110fc09c4cc43353c82a1b15ef1). These identify published bytes. Original installed HF cache refs for these files are unknown. The installed DA3 and MoGe files match the published hashes.

### ReDirector models

Base target prefix: `ROOT/redirector/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera/`. The adapter goes in `ROOT/redirector/models/checkpoints/`. `redirector_worker.py` loads all four base weight components plus the adapter. Keep base `config.json` and the complete `google/umt5-xxl/` tokenizer directory from the same snapshot. Its exact installed tokenizer file list, sizes and hashes are unknown.

| Exact filename | Target | Bytes (measured unless public) | SHA-256 | Source |
| --- | --- | --- | --- | --- |
| `diffusion_pytorch_model.safetensors` | base prefix | 3,232,727,784 bytes | `1a648f6fbb543dd6ac9f2f9af1ca4bc1bbe460482ee4d7984d50565699e97398` | [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/blob/main/diffusion_pytorch_model.safetensors) |
| `Wan2.1_VAE.pth` | base prefix | 507,609,880 bytes | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` | [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/blob/main/Wan2.1_VAE.pth) |
| `models_t5_umt5-xxl-enc-bf16.pth` | base prefix | 11,361,920,418 bytes | `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d` | [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/blob/main/models_t5_umt5-xxl-enc-bf16.pth) |
| `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` | base prefix | 4,772,359,047 bytes | `628c9998b613391f193eb67ff68da9667d75f492911e4eb3decf23460a158c38` | [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/blob/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth) |
| `step20000.safetensors` | adapter folder | 752,616,056 bytes | `a7e25b0ab0b2493f00b790bd0ff0d21f2fce232f9a48dea82aaec1dd557c7a54` | [byeongjun-park/ReDirector](https://huggingface.co/byeongjun-park/ReDirector/blob/main/step20000.safetensors) |

### Capture models and body assets

The current public [asset resolver](https://github.com/ant-research/4DAnyone/blob/main/fdanyone/assets.py) pins `AntResearch/4DAnyone` to `4c80e87b805a5f8461cf339cdbe2fb4249e585aa`. The following filenames and layout come from that upstream release. The installed files below were measured on the reference machine. Targets are relative to `ROOT/upstream/models/`. Measured GVHMR runtime destinations are ROOT/upstream/third_party/GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt, hmr2/epoch=10-step=25000.ckpt, vitpose/vitpose-h-multi-coco.pth, yolo/yolov8x.pt and body_models/smplx/SMPLX_NEUTRAL.npz under that checkpoints directory.

| File and relative target | Use | Source | Installed size / SHA-256 |
| --- | --- | --- | --- |
| `4danyone/model.safetensors` | Generated views | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone/model.safetensors) | 12,296,825,310 bytes / `aff60b0db2d333bd9e960a9cf3333cc8dd40fe76614a22f75a0da72be4e8289f` |
| `4danyone/smplx_to_goliath70.pt` | Body regression | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone/smplx_to_goliath70.pt) | 541,437 bytes / `725139e6cd14f986cd835468b2d62ca7abb9ffe29beb204d11f816e8e3fab40f` |
| `4danyone/Wan2.2_VAE.pth` | Generated views | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone/Wan2.2_VAE.pth) | 2,818,839,170 bytes / `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` |
| `4danyone/prompt_context.safetensors` | Frozen conditioning | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone/prompt_context.safetensors) | 4,194,840 bytes / `50a71c4447c2121837f09619946a8190e2bddd76ed1baffac0a5f1b3b430a669` |
| `4danyone/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors` | Turbo views | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors) | 332,348,584 bytes / `0ace5244e3d1256f884662c261b017249796cf5b95f05d5ed93cc02a478967b8` |
| `gvhmr/gvhmr_siga24_release.ckpt` | Body tracking | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/gvhmr/gvhmr_siga24_release.ckpt) | 163,508,011 bytes / `4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91` |
| `gvhmr/epoch=10-step=25000.ckpt` | HMR2 body initialization | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/gvhmr/epoch=10-step=25000.ckpt) | 2,709,494,041 bytes / `2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e` |
| `gvhmr/vitpose-h-multi-coco.pth` | Pose estimation | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/gvhmr/vitpose-h-multi-coco.pth) | 2,549,075,546 bytes / `50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc` |
| `gvhmr/yolov8x.pt` | Person detection | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/gvhmr/yolov8x.pt) | 136,867,539 bytes / `c4d5a3f000d771762f03fc8b57ebd0aae324aeaefdd6e68492a9c4470f2d1e8b` |
| `perceptual/imagenet-vgg-verydeep-19-conv.safetensors` | Optional upstream perceptual reconstruction | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/perceptual/imagenet-vgg-verydeep-19-conv.safetensors) | 61,221,640 bytes / `0beaf06caed0d0b752022c756bdf1752e941652ea30cf91e593438797f3293db` |
| `birefnet/model.safetensors` | Foreground masks | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet/tree/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4) | 444,473,596 bytes / `9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154` |
| `body_models/smplx/SMPLX_NEUTRAL.npz` | Body geometry | [Official registered download](https://smpl-x.is.tue.mpg.de/) | 108,752,058 bytes / `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992` |

The pinned BiRefNet release also requires `BiRefNet_config.py`, `birefnet.py` and `config.json` in the same folder. Upstream creates GVHMR compatibility links under `third_party/GVHMR/inputs/checkpoints/`. The camera solve also loads `smpl_coco17_J_regressor.pt` and `smplx2smpl_sparse.pt` from `ROOT/upstream/third_party/GVHMR/hmr4d/utils/body_model/`; measured tensor metadata is below; individual asset licence provenance is unknown.

The installed Turbo LoRA matches the hash upstream publishes. Quick mode skips generated-view synthesis, while the example workflow selects 4DGS mode. Generated `ckpts/ckpt_*.pt` files are per-capture training outputs, with job-specific sizes and hashes.

### Optional and legacy files

These are outside the saved public graph. Sources, revisions and licences for these exact conversions are unknown. Presence in the inventory does not make them required.

| Exact filename | Node / branch | Target | Measured bytes | Measured SHA-256 | Source |
| --- | --- | --- | --- | --- | --- |
| `Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors` | Splat Photoreal Pass (legacy), sheet identity helper | `ComfyUI/models/loras/Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors` | 1,308,756,480 | `4c7804265c5e8a284c0613fb6fd63d114f029429e9837e3ea289521d7ef93ffa` | unknown |
| `flux-2-klein-9b-fp8.safetensors` | ltx_graph still-refinement helper; no public node enables it | `ComfyUI/models/diffusion_models/flux-2-klein-9b-fp8.safetensors` | 9,433,061,528 | `865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee` | unknown |
| `flux2-vae.safetensors` | ltx_graph still-refinement helper; no public node enables it | `ComfyUI/models/vae/flux2-vae.safetensors` | 336,211,292 | `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3` | unknown |
| `qwen_3_8b_fp8mixed.safetensors` | ltx_graph still-refinement helper; no public node enables it | `ComfyUI/models/text_encoders/qwen_3_8b_fp8mixed.safetensors` | 8,664,848,742 | `abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6` | unknown |
| `lenovo_qwen21.safetensors` | ltx_graph still-generation helper; no public node enables it | `ComfyUI/models/loras/lenovo_qwen21.safetensors` | 79,743,896 | `7f1e95c3ec1e9c7cea197f7b6f80f601af8e6b479f6ead78ab16df542f447c0e` | unknown |
| `ltx-2.3-id-lora-celebvhq-3k.safetensors` | ltx_graph reference-voice helper; no public node enables it | `ComfyUI/models/loras/ltx-2.3-id-lora-celebvhq-3k.safetensors` | 1,157,884,304 | `12e6be9c52c83047cb71470c1c63b64ebc8b13dc0dfa34400f6576ba2cd88ecf` | unknown |
| `qwen_image_2.1_int8_convrot.safetensors` | 360 World (legacy) | `ComfyUI/models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | 7,256,783,064 | `cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d` | unknown |
| `qwen_image_2.1_vae_bf16.safetensors` | 360 World (legacy) | `ComfyUI/models/vae/qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` | unknown |
| `pano360_qwen21_edit_v1.safetensors` | 360 World (legacy) | `ComfyUI/models/loras/pano360_qwen21_edit_v1.safetensors` | 159,436,400 | `846a39df32e95307207727e16c8d2d88f873f10c590f72450da33a6a2dc5203c` | unknown |
| `ema_vae_fp16.safetensors` | 360 World (legacy), SeedVR2 upscale | `ComfyUI/models/SEEDVR2/ema_vae_fp16.safetensors` | 501,324,814 | `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1` | unknown |
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | 360 World (legacy), SeedVR2 upscale | `ComfyUI/models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors` | 3,391,544,696 | `3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff` | unknown |
| Best_FaceID_v1.0_LoRA.safetensors | ltx_graph identity helper | ComfyUI/models/loras/Best_FaceID_v1.0_LoRA.safetensors | unknown | unknown | unknown |
| ltx-2.3-22b-ic-lora-deblur-0.9.safetensors | scripts/face_pass.py | ComfyUI/models/loras/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors | unknown | unknown | unknown |

The perceptual VGG-19 row above is optional upstream reconstruction. H3 character sheets reference an external workflow whose model closure is unknown. SAM 3D Body, DINOv3 and MHR full checkpoints belong to alternate Generate Views preprocessing. Public Capture uses GVHMR and the separate MHR70 regression asset. Exact alternate checkpoint filenames, bytes, hashes and revisions are unknown.

### Required source-camera tensors

source_cam.py loads both tensors, including quick Capture. Their individual licences are unknown.

| Filename | Runtime target | Measured bytes | Measured SHA-256 | Source |
| --- | --- | --- | --- | --- |
| `smpl_coco17_J_regressor.pt` | `ROOT/upstream/third_party/GVHMR/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt` | 469,227 | `bacdaf756629493994cc869f4c27d179f5e4a5d06b8797ee3dcb94571522079f` | [GVHMR measured commit](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt) |
| `smplx2smpl_sparse.pt` | `ROOT/upstream/third_party/GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt` | 208,935 | `0fc821a9e79ec3e76d6a9796b96d5bef8cd67055e18497bbe370d8aed9e07e06` | [GVHMR measured commit](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/utils/body_model/smplx2smpl_sparse.pt) |

### Verify downloaded bytes

For each file, compare its size and SHA-256 with the corresponding row. For example, from the installation directory:

```bash
stat -c '%s' ComfyUI/models/vae/LTX23_video_vae_bf16.safetensors
sha256sum ComfyUI/models/vae/LTX23_video_vae_bf16.safetensors
```

A different hash is a different artifact, even when its filename matches. Rows marked unknown cannot be verified against a measured reference hash. Capture body/checkpoint files are installed by the upstream downloader with compatibility links under ROOT/upstream/third_party/GVHMR/inputs/checkpoints. The inventory measured those linked paths. Exact additional model files used internally by the pinned trainer are unknown beyond the inventory and visible launchers.

## First run on the new installation

1. Start ComfyUI with the configured environment after installation. Wait for provider imports to finish. From COMFY_DIR, source ROOT/env.sh and run .venv/bin/python main.py. Open the local ComfyUI interface.
2. Open `workflows/Reshoot 4D.json`. Upload your own video into Capture and replace the example filename. Set a short trim, initially about four seconds as in the example.
3. Keep 4DGS mode for the example's splat preview, or select quick mode after applying its patch. Queue Capture on the new machine. Open the 4D camera editor, pick a preset or key a modest move starting on the real camera, and save it to the node. The workflow ships with no camera keys, so every move is made on your own capture.
4. Enter a manual Retake prompt for the first run, or install Qwen for automatic captioning. Use 480p, 50 steps and CFG 5 as in the example.
5. Queue Retake and Photoreal Pass. Connect the returned FPS to the output video's frame rate. The public Photoreal Pass returns frames and FPS; the example uses VHS_VideoCombine to save them under `ComfyUI/output/reshoot4d/`.

If a class or model is missing, check its exact name against the manifest and `/object_info` before rendering again. See [Troubleshooting](TROUBLESHOOTING.md) for environment and extension errors.

## Timings on one RTX 3090

Measured on the reference machine (one RTX 3090) with the example settings: a 4.4 s video, 24 views, 15,000 training steps, 480p / 50-step Retake and the Photoreal detailer on. Capture times come from one full run's capture.log and ComfyUI's prompt history (2026-10-28, 15:09 to 16:13). The rest of the pipeline comes from Example 1's ComfyUI prompt total.

| Stage | Time |
| --- | --- |
| Capture: tracking preprocess | 22 s |
| Capture: body tracking and view generation, all 24 views | 46 min 15 s |
| Capture: adaptation, rig and source-camera solve | under 30 s |
| Capture: 4DGS training, 15,000 steps | 16 min 14 s |
| Capture: splat preview and setting export | about 45 s |
| **Capture total** | **63 min 31 s** |
| Virtual Camera | interactive |
| **Retake + Photoreal Pass (LTX 2.3), one camera move** | **28 min 21 s** |
| Photoreal Pass alone, CrossView + detailer, from a finished retake | 5 min 22 s and 5 min 44 s (two clips) |

Queueing the same video, trim and views again reuses the finished capture, so each extra camera move costs only the Retake + Photoreal Pass time. The 28 min 21 s total was measured with the earlier `union control` pass; the CrossView row was timed separately from ComfyUI's prompt history (2026-10-29).

## Remaining unknowns

Tokenizer/config metadata; original curl download revisions; ComfyUI Python/torch; complete dependency locks and CUDA package inventory; local patches; minimum RAM/disk; separate Retake/LTX stage times. A complete fresh installation and all model combinations have not been run during this review.
