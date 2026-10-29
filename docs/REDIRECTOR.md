# Retake (ReDirector)

**Reshoot 4D: Retake** reshoots the source video along the path authored in the 4D camera editor. It uses
[ReDirector](https://github.com/byeongjun-park/ReDirector) (Apache-2.0), which is built on
Wan2.1-Fun-V1.1 1.3B Control-Camera (Apache-2.0).
- **Condition path:** the real camera path from the capture's camera solve, scaled using an assumed body height.
- **Target path:** your keys, re-based on the source camera's first-frame pose. The move must start on the real camera.

**Photoreal Pass (retake)** uses LTX 2.3 to render every final frame:
- it uses the retake as video-to-video input with start strength 0.35 in the first stage;
- the retake's Canny edges (computed on the CPU) act as union control;
- the real first frame is the start guide.

## Install

```bash
scripts/install_redirector.sh                # clones to $SAM3DBX_ROOT/redirector, ~20 GB weights, own .venv
```

ReDirector runs in its own Python (torch 2.8 + DiffSynth), started by the node as a subprocess. It never shares ComfyUI's torch, and ComfyUI's
models are unloaded first. If you install it elsewhere, set `"redirector": "/path"` in `config.local.json`.

## Reported resource use (RTX 3090)

These reported measurements describe individual Retake runs; rendering time and memory use vary with hardware, settings and video length.

| resolution | time (50 steps) | peak VRAM |
|---|---|---|
| 480p (trained size) | ~25 min | 12 GB |
| 720p | ~1 h 50 | 21 GB |

Use 50 steps as a starting point. When the prompt is empty, Qwen3-VL 8B describes the first frame; enter a manual prompt to bypass captioning.
