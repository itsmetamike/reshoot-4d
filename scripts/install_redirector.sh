#!/bin/bash
# Install ReDirector (Apache-2.0) for the Retake node: its own venv (torch 2.8 + DiffSynth) and ~20 GB of weights.
# usage: scripts/install_redirector.sh [target dir, default $SAM3DBX_ROOT/redirector or ~/reshoot4d/redirector]
set -euo pipefail
T=${1:-${SAM3DBX_ROOT:-$HOME/reshoot4d}/redirector}
[ -d "$T/.git" ] || git clone https://github.com/byeongjun-park/ReDirector.git "$T"
cd "$T"
M=models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera; B=https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/resolve/main
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
mkdir -p "$M" models/checkpoints
for f in diffusion_pytorch_model.safetensors Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth config.json; do
  [ -s "$M/$f" ] || { curl -fL "${AUTH[@]}" -o "$M/$f.part" "$B/$f" && mv "$M/$f.part" "$M/$f"; }
done
for f in $(curl -s "${AUTH[@]}" "https://huggingface.co/api/models/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/tree/main/google/umt5-xxl" \
           | python3 -c "import json,sys;print(' '.join(x['path'] for x in json.load(sys.stdin) if x['type']=='file'))"); do
  mkdir -p "$M/$(dirname "$f")"; [ -s "$M/$f" ] || curl -fL "${AUTH[@]}" -o "$M/$f" "$B/$f"
done
[ -s models/checkpoints/step20000.safetensors ] || curl -fL -o models/checkpoints/step20000.safetensors \
  https://huggingface.co/byeongjun-park/ReDirector/resolve/main/step20000.safetensors
uv venv -q --python 3.11 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt natsort
echo "ReDirector ready in $T. If it isn't at \$SAM3DBX_ROOT/redirector, set \"redirector\": \"$T\" in config.local.json."
