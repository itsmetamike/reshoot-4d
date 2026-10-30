# Troubleshooting

See [Install](INSTALL.md) for setup and [Retake](REDIRECTOR.md) for the Retake runtime.

| Symptom | Check and action |
| --- | --- |
| Nodes do not load | Inspect ComfyUI startup errors. Check required custom-node providers and environment versions in INSTALL.md. |
| Capture cannot find environments | Check root in config.local.json or SAM3DBX_ROOT. Defaults expect upstream/.venv, repo/.venv and the FreeTimeGsVanilla trainer environment. |
| CUDA extension undefined symbol | Follow SETUP.md's matching torch/CUDA 12.8 build instructions, no-build-isolation and no-cache guidance for the documented trainer. Do not mix ComfyUI and trainer wheels. |
| Missing depth/scene models | Check DA3 Base and MoGe-2 under models/geometry_estimation; Capture and Retake require source-camera and setting metadata. |
| Retake says ReDirector is not installed | Check that the configured redirector path contains .venv/bin/python and required weights. Follow INSTALL.md to install the runtime. |
| Empty prompt fails | Install the expected qwen3vl_8b text encoder or type a manual Retake prompt. This node does not fall back to Gemma for captions. LTX still needs Gemma for text encoding. |
| First key is away from real camera | Reset the first key to the source camera or start from a preset, then recheck the path. |
| ReDirector fails | Read retake.log in the job directory. Confirm successful worker exit, correct base config and fresh output. Keep error logs with settings when requesting help. |
| GPU memory failure | Avoid simultaneous heavy jobs. Current nodes unload ComfyUI models for subprocess work. Qualify the 480p baseline before higher-resolution retakes. Exact capacity remains environment-specific. |
| Wrong Photoreal pass or socket type | Use Photoreal Pass (retake), whose input is the Retake job JSON path. The older depth/anchor pass expects a prepared shot from another route. |
| LTX load failure or unexpected output | Match exact model filenames, custom-node classes and union parameters to the tested export. Check the 19B detailer separately on the 22B base. |
| CrossViewWarp or MoGeInference missing | CrossViewWarp: install ComfyUI-CrossViewWarp at the tested commit. MoGeInference / LoadMoGeModel: use a ComfyUI revision with core MoGe support. Missing weights: put MoGe-2 under models/geometry_estimation. Restart ComfyUI, or set Photoreal Pass to `union control`. |
| Fast motion smears a face | CrossView can melt a face during quick spins or fast arms. Try a smaller move, a different seed, or `union control` for that shot. |
| Editor freeze/lens change absent from final | Current Retake selects source indices uniformly and uses source intrinsics. See EDITOR.md; editor authoring support alone does not establish backend channel transfer. |
| Person, background or camera extent changes | Compare source, retake and final at matching scene times to locate the change. Try a smaller camera move. |
| Imported path looks wrong | Confirm cam data, real source-camera metadata and follow/locked space. A portable move still needs composition review on the new capture. |

When reporting timing problems, include decoded frame count, FPS, duration and crop for the source, retake and final.

Before sharing an issue example, remove tokens and private machine paths and confirm permission to share the video.
