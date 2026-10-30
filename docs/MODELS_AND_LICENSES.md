# Models, code and licences

Reshoot 4D is authored by itsmetamike and licensed under [AGPL-3.0](../LICENSE). Third-party code, model weights, body assets and schemas retain their own terms.

**The current Capture path includes research and non-commercial restrictions.** [GVHMR](https://github.com/zju3dv/GVHMR/blob/main/LICENSE) restricts use to education, research and non-profit purposes. [SMPL-X](https://smpl-x.is.tue.mpg.de/modellicense.html) uses non-commercial scientific research terms, with a separate commercial licensing route. The Turbo view-generation LoRA uses **CC BY-NC-SA 4.0**. These restrictions also matter when Retake and LTX have different terms.

[Install](INSTALL.md#model-manifest) gives measured bytes, SHA-256 values, runtime paths, revisions and which models are required or optional.

The saved workflow selects 4DGS Capture, blank-prompt automatic captioning and enabled detailer. All ComfyUI rows below except Union Control are required for those settings; Union Control is needed only for the `union control` pass. Manual prompts bypass Qwen; disabling detailer removes its LoRA. Quick Capture skips view generation and splat training. Optional branches are excluded from baseline requirements.

## ComfyUI model weights

This table matches Install's ComfyUI manifest. Source links identify the converted artifact, where available. The final graph loads the audio VAE even though the public node returns video frames and FPS.

| Exact filename | Role | Source artifact | Licence |
| --- | --- | --- | --- |
| `depth_anything_3_base.safetensors` | Setting depth | [Comfy-Org/Depth-Anything-3](https://huggingface.co/Comfy-Org/Depth-Anything-3/blob/657708b0ec5f218cbf14bff0bb2cb8c58ea3fe33/geometry_estimation/depth_anything_3_base.safetensors) | Apache-2.0 |
| `moge_2_vitl_normal_fp16.safetensors` | Setting sky mask; CrossView depth of the real video | [Comfy-Org/MoGe](https://huggingface.co/Comfy-Org/MoGe/blob/7f19f325949e1e7bb31c720333936f03cf932f57/geometry_estimation/moge_2_vitl_normal_fp16.safetensors) | MIT |
| `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors` | Final synthesis | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy/blob/a5a175044e7607277812c00b98a27718a4c42710/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors) | LTX-2 Community License |
| `gemma_3_12B_it_fp4_mixed.safetensors` | LTX text encoding | [Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2/blob/fff639871e076110fc09c4cc43353c82a1b15ef1/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors) | Gemma terms |
| `ltx-2.3_text_projection_bf16.safetensors` | LTX text projection | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy/blob/1c032c322c3db1992a66a7690d414b4fe8e686d0/text_encoders/ltx-2.3_text_projection_bf16.safetensors) | LTX-2 Community License |
| `LTX23_video_vae_bf16.safetensors` | LTX video VAE | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy/blob/a422e5661a18bde257c0ef1c3761d8b9ce1eb645/vae/LTX23_video_vae_bf16.safetensors) | LTX-2 Community License |
| `LTX23_audio_vae_bf16.safetensors` | LTX audio latent initialization | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy/blob/a422e5661a18bde257c0ef1c3761d8b9ce1eb645/vae/LTX23_audio_vae_bf16.safetensors) | LTX-2 Community License |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | LTX second stage | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3/blob/cd784f3198e9ec3efec60b66a0fd78aafe413a86/ltx-2.3-spatial-upscaler-x2-1.1.safetensors) | LTX-2 Community License |
| `LTX2.3-22B_IC-LoRA-CrossView-Warp_v2_6000.safetensors` | Photoreal Pass guides (CrossView, default) | [Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2) | Apache-2.0 (LTX-2 base terms still apply) |
| `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | Retake Canny control (`union control` pass only) | [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/blob/b4d1c4d8c9e544e9bbbd6811bb4363708b6093ff/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors) | LTX-2 Community License |
| `ltx-2-19b-ic-lora-detailer.safetensors` | Detailer, on by default | [Lightricks/LTX-2-19b-IC-LoRA-Detailer](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/blob/70afe09e20511305b7c0a471bded8e8babe7beab/ltx-2-19b-ic-lora-detailer.safetensors) | LTX-2 Community License |
| `qwen3vl_8b_int8_convrot.safetensors` | Automatic Retake caption only | [Kutches/Qwen](https://huggingface.co/Kutches/Qwen/blob/main/qwen3vl_8b_int8_convrot.safetensors) | Base Qwen3-VL Instruct: Apache-2.0; conversion terms unknown |

[LTX 2.3's official card](https://huggingface.co/Lightricks/LTX-2.3), the [conversion repository](https://huggingface.co/Kijai/LTX2.3_comfy), [Union Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) and [Detailer](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer) declare LTX community terms. Retain and read the agreement supplied with each release. The Detailer card identifies a 19B base; this pack applies it to the 2.3 22B base.

The [CrossView-Warp card](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2) declares Apache-2.0 for the LoRA weights and points to the LTX-2 base-model terms. Its [ATTRIBUTION.md](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2/blob/main/ATTRIBUTION.md) lists the third-party assets used to render its training scenes (CC-BY, CC0 and CMU motion-capture data, among others) and notes that the licence of the MoGe-2 checkpoint used for its training depth is unverified. Inference use does not ship those assets; read that file before any commercial use. The [ComfyUI-CrossViewWarp](https://github.com/cseti007/ComfyUI-CrossViewWarp) node is Apache-2.0.

The [Gemma base card](https://huggingface.co/google/gemma-3-12b-it) declares Gemma terms. The enclosing Comfy-Org/ltx-2 repository's LTX label does not replace Gemma's terms for the text encoder. [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) declares Apache-2.0; the exact base variant and additional permissions for the linked Qwen conversion are unknown. A manual Retake prompt bypasses Qwen. Gemma remains required for LTX.

[DA3's conversion card](https://huggingface.co/Comfy-Org/Depth-Anything-3) declares Apache-2.0. [MoGe's conversion card](https://huggingface.co/Comfy-Org/MoGe) declares MIT for its weights. Code and weight licences can differ.

## ReDirector model weights and tokenizer

The base directory and adapter destination are specified in [Install](INSTALL.md#redirector-models). The worker loads all components below.

| Exact file or directory | Source | Licence evidence |
| --- | --- | --- |
| `diffusion_pytorch_model.safetensors` | [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/blob/main/diffusion_pytorch_model.safetensors) | Apache-2.0 declared by the model card |
| `Wan2.1_VAE.pth` | [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/blob/main/Wan2.1_VAE.pth) | Apache-2.0 declared by the model card |
| `models_t5_umt5-xxl-enc-bf16.pth` | [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/blob/main/models_t5_umt5-xxl-enc-bf16.pth) | Apache-2.0 declared by the model card; individual encoder origins and notices unknown |
| `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` | [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/blob/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth) | Apache-2.0 declared by the model card; individual encoder origins and notices unknown |
| `step20000.safetensors` | [byeongjun-park/ReDirector](https://huggingface.co/byeongjun-park/ReDirector/blob/main/step20000.safetensors) | Apache-2.0 declared by the model card |
| `config.json`, `google/umt5-xxl/` | [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera/tree/main) | Bundle declares Apache-2.0; exact installed tokenizer files and individual notices unknown |

The [ReDirector card](https://huggingface.co/byeongjun-park/ReDirector) and [Wan Control-Camera card](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera) both declare Apache-2.0. UMT5 and OpenCLIP component origins require matching individual notices. The local installer uses direct moving-main downloads; the inventory supplies all five weight sizes and three encoder/VAE hashes. All five weight hashes are in Install. HF revisions and tokenizer hashes are unknown.

## Capture model weights and body assets

The current public [asset resolver](https://github.com/ant-research/4DAnyone/blob/main/fdanyone/assets.py) and [HF model card](https://huggingface.co/AntResearch/4DAnyone) identify these files. Paths are relative to `ROOT/upstream/models/`. The [upstream third-party notices](https://github.com/ant-research/4DAnyone/blob/main/docs/THIRD_PARTY_NOTICES.md) explain the split terms.

| File | Source | Licence / restriction |
| --- | --- | --- |
| `4danyone/model.safetensors` | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone/model.safetensors) | Apache-2.0, first-party checkpoint |
| `4danyone/smplx_to_goliath70.pt` | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone/smplx_to_goliath70.pt) | Apache-2.0 tensors; Sapiens2-derived schema retains separate terms |
| `4danyone/Wan2.2_VAE.pth` | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone/Wan2.2_VAE.pth) | Apache-2.0 Wan assets |
| `4danyone/prompt_context.safetensors` | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone/prompt_context.safetensors) | Apache-2.0 Wan-derived asset |
| `4danyone/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors` | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/4danyone/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors) | **NON-COMMERCIAL: CC BY-NC-SA 4.0** |
| `gvhmr/gvhmr_siga24_release.ckpt` | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/gvhmr/gvhmr_siga24_release.ckpt) | **RESEARCH / EDUCATION / NON-PROFIT ONLY**; additional weight terms unknown |
| `gvhmr/epoch=10-step=25000.ckpt` | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/gvhmr/epoch=10-step=25000.ckpt) | Individual weight licence unknown |
| `gvhmr/vitpose-h-multi-coco.pth` | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/gvhmr/vitpose-h-multi-coco.pth) | Individual weight licence unknown |
| `gvhmr/yolov8x.pt` | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/gvhmr/yolov8x.pt) | Individual weight licence unknown |
| `perceptual/imagenet-vgg-verydeep-19-conv.safetensors` | [AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/perceptual/imagenet-vgg-verydeep-19-conv.safetensors) | CC BY 4.0 per upstream notices |
| `birefnet/model.safetensors`, `BiRefNet_config.py`, `birefnet.py`, `config.json` | [ZhengPeng7/BiRefNet at e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4](https://huggingface.co/ZhengPeng7/BiRefNet/tree/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4) | MIT per upstream notices |
| `body_models/smplx/SMPLX_NEUTRAL.npz` | [Official SMPL-X download](https://smpl-x.is.tue.mpg.de/) | **NON-COMMERCIAL / RESEARCH TERMS**; registration required; redistribution restricted |
| `smpl_coco17_J_regressor.pt`, `smplx2smpl_sparse.pt` under GVHMR's `hmr4d/utils/body_model/` | [GVHMR body-model source directory](https://github.com/zju3dv/GVHMR/tree/main/hmr4d/utils/body_model) | Exact individual asset provenance and licences unknown; do not assume GVHMR's code licence covers them |

Turbo provenance is documented through [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) revision `86c2b0442e01eeee630b48fd7efc0cd37af03252` and [the original Turbo licence](https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo/blob/main/LICENSE.md). **Non-commercial terms apply to that LoRA.** It belongs to generated-view Capture; the installed file matches the published hash.

The MHR70 regression tensor has Apache-2.0 tensor terms and separate Sapiens2-derived keypoint-schema terms. Upstream records provenance from SAM 3D Body, MHR conversion tools and DNA-Rendering. Those provenance sources do not establish that their full models or dataset are loaded by this pack's GVHMR route. Preserve the schema notice; exact installed notices remain unknown.

Generated `ckpts/ckpt_*.pt` splats are per-capture outputs. They are not separately downloaded pretrained models. Record the source video's permissions and the applicable pipeline terms with each distributed capture.

## Optional and compatibility branches

These names occur in `nodes_super.py`, `ltx_graph.py`, the optional face script or the older fast-shot helper. They are outside the baseline Retake example. Measured optional-file bytes and hashes are in Install. Exact conversion sources, revisions and permissions remain unknown unless stated.

| Exact file or component | Role | Source and licence status |
| --- | --- | --- |
| `qwen_image_2.1_int8_convrot.safetensors`, `qwen_image_2.1_vae_bf16.safetensors` | 360 World (legacy) and ltx_graph stills helper | Exact conversion source and licence unknown |
| `pano360_qwen21_edit_v1.safetensors` | 360 World (legacy) panorama adapter | Exact source and licence unknown |
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors`, `ema_vae_fp16.safetensors` | 360 World (legacy), SeedVR2 upscale and older finishing helper | Exact conversion sources and licences unknown |
| `Best_FaceID_v1.0_LoRA.safetensors`, `Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors` | Splat Photoreal Pass (legacy), identity helper | Exact sources and licences unknown |
| `ltx-2.3-id-lora-celebvhq-3k.safetensors` | ltx_graph reference-voice helper | Exact source and licence unknown |
| `lenovo_qwen21.safetensors` | ltx_graph still-generation helper | Exact source and licence unknown |
| `flux-2-klein-9b-fp8.safetensors`, `qwen_3_8b_fp8mixed.safetensors`, `flux2-vae.safetensors` | ltx_graph still-refinement helper | Exact conversions and licences unknown |
| `ltx-2.3-22b-ic-lora-deblur-0.9.safetensors` | Optional face-pass script | Exact source and licence unknown |
| H3 character-sheet branch and referenced external workflow | Optional character sheets | Full asset closure and licences unknown |
| SAM 3D Body, DINOv3 and MHR checkpoints | Alternate adaptation route | Exact installed models and loading status unknown; SAM 3D Body has [custom terms](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE) |

## Code and runtime terms

| Component | Source / notice | Status |
| --- | --- | --- |
| Reshoot 4D | [Local licence](../LICENSE) | AGPL-3.0; author itsmetamike |
| 4DAnyone | [Third-party notices](https://github.com/ant-research/4DAnyone/blob/main/docs/THIRD_PARTY_NOTICES.md) | Apache-2.0 first-party code; third-party restrictions retained |
| sam3dbx4danyone | [Project](https://github.com/OpsiClear-4DGS/sam3dbx4danyone) | AGPL-3.0-only additions; inherited notices retained |
| FreeTimeGsVanilla | [Project](https://github.com/OpsiClear-4DGS/FreeTimeGsVanilla) | AGPL-3.0 stated by project; measured commit in Install; local patches unknown |
| ReDirector | [Licence](https://github.com/byeongjun-park/ReDirector/blob/main/LICENSE) | Apache-2.0 upstream; measured commit in Install; local patches unknown |
| DiffSynth | [Licence](https://github.com/modelscope/DiffSynth-Studio/blob/main/LICENSE) | Apache-2.0 upstream; exact DiffSynth revision unknown |
| GVHMR | [Licence](https://github.com/zju3dv/GVHMR/blob/main/LICENSE) | **Education, research and non-profit only**; retain notices |
| gsplat | [Licence](https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE) | Apache-2.0 upstream; installed native-kernel notices unknown |
| Three.js and TransformControls | `web/lib/THREE_LICENSE` | MIT notice present locally |
| ComfyUI and custom nodes | Sources in [Install](INSTALL.md#paths-and-repository-revisions) | Measured commits in Install; complete notice inventory unknown |
| TSOG codecs/player | [Project](https://github.com/OpsiClear-4DGS/tsog) | Apache-2.0 stated by project; measured ROOT/repo/third_party/tsog commit in Install; trainer-nested commit unknown |
| CUDA, PyTorch, attention libraries, native extensions, FFmpeg/codecs and other Python packages | Environment inventories required | Exact installed versions, build flags and notice closure unknown |

Preserve source-video permissions and model, code and schema notices. Remaining unknowns include individual tracking/body utility weight licences, optional conversion provenance, exact installed schema notices, complete dependency/build locks and tokenizer/configuration metadata. Measured commits and hashes do not resolve these separate licence questions.
