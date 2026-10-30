# The Workflow

Take a video of one person through Capture -> Virtual Camera -> Retake -> **Photoreal Pass (retake)**. LTX 2.3 renders every final frame.

## Capture

Upload a video in Capture and trim it. Capture prepares the tracked person, source-camera solve and person-in-setting transform. See the example workflow for mode selection; 4DGS mode also trains a splat for the editor preview and reusable asset.

The source-camera solve uses GVHMR observations, body geometry and PnP in source_cam.py. The scene layer uses DA3 and body registration to place the source camera and person in a setting frame. These are estimates. They do not establish metrically exact reconstruction. The retake adapter estimates translation scale from a posed body extent using an assumed 1.7 m height.

Queueing the same video and trim can reuse preparation. Quick mode, 4DGS mode and mannequin mode have different preparation costs; see the Capture note in [the Reshoot 4D workflow](../workflows/Reshoot%204D.json).

## Your camera move

Connect Capture to Virtual Camera, open the 4D camera editor and start from a preset or add keys. Keep the first key on the real source camera. Save the move back to the node. The node provides the capture directory and camera-path file for Retake.

Begin with a modest arc or push. Large unseen-side moves are useful to explore in the editor but need separate output validation. See [Editor](EDITOR.md) for freeze, lens and timing limitations at the Retake boundary.

## Retake

Connect Virtual Camera's capture and path outputs to Retake. The implementation samples 81 source indices and 81 target-camera samples. Both trajectories use OpenCV camera-to-world matrices relative to their first pose, with scaled translations. The adapter carries the source camera's initial tilt/roll into the target. An initial-position/orientation check rejects a path too far from the source camera.

ReDirector uses the actual source video, the solved source path and the target path to reshoot the person and setting jointly. It runs in its own environment after ComfyUI unloads models. A non-empty manual prompt is used directly. If the prompt is empty, Qwen3-VL 8B describes the real first frame. There is no automatic Gemma fallback in this Retake node.

Start with 480p mode, 50 steps and CFG 5. Portrait modes are 480x832 or 720x1280; landscape swaps dimensions. Keep the seed and prompt with each result. See [Retake setup](REDIRECTOR.md) for runtime details.

## Photoreal Pass

Connect Retake's `retake` output to Photoreal Pass (retake). It is a job JSON path, not just an MP4 filename. The pass expands into LTX 2.3 nodes inside the same prompt.

The default pass is **crossview**: LTX 2.3 with Cseti's [CrossView-Warp IC-LoRA](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2) (Apache-2.0) and the [ComfyUI-CrossViewWarp](https://github.com/cseti007/ComfyUI-CrossViewWarp) node. MoGe-2 estimates the depth of the real video; CrossViewWarp warps the real video along your camera path. LTX starts from the retake partway down its schedule, with the warp and the real video as guides.

Nothing is set by hand per clip. The pass reads the camera keys from the retake job (orbit angles and distance about the person), the orbit centre from the capture (her depth in front of the source camera, in metres) and the field of view from the source camera's intrinsics.

| Setting | Current default / behavior |
| --- | --- |
| pass_type | `crossview` (default) or `union control`, the earlier pass |
| crossview_start | 0.7, where LTX starts on its schedule over the retake. 1.0 ignores the retake |
| Guides | Depth warp of the real video along the path, then the real video itself (IC-LoRA guides) |
| Detailer | On by default; the 19B detailer adapter in the full-size stage, on the 2.3 22B base |
| Final portrait / landscape | 1024x1792 / 1792x1024 |
| Public outputs | Image batch and FPS; use a video output node to save the final video |

`union control` keeps the earlier pass: the retake as video-to-video input from start strength 0.35, its CPU Canny edges as union control (strength 0.6, LoRA 0.7) and the real first frame as the start guide (0.9). The start, control and first-frame settings apply to that pass only.

Both passes use two synthesis stages: a half-size stage, an x2 latent upscale and a full-size refinement.

## Save and judge a result

Keep the source, raw retake and LTX final as a matching triple. Save prompt, seed, source trim, model hashes, path JSON and graph exports beside the result inventory. Compare face/hair, clothes, body motion, feet, static setting landmarks, camera extent, crop and timing while playing all three. Select examples on that evidence rather than sharpness in one still.

Open [Reshoot 4D.json](../workflows/Reshoot%204D.json) for the node connections and save-video step. Source footage credits are in [Credits](CREDITS.md).
