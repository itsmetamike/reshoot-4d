# Credits and provenance

Reshoot 4D combines a ComfyUI workflow and a 4D camera editor. itsmetamike's contribution is the node-pack integration, capture-to-setting/camera handoff, Virtual Camera authoring tools and the Retake-to-LTX workflow. Upstream projects provide the learned models and reconstruction systems.

| Project | Contribution |
| --- | --- |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Graph execution, model loaders and frontend extension surface. |
| [ReDirector](https://github.com/byeongjun-park/ReDirector) | Source-video and source/target-camera conditioned retakes. |
| [Wan2.1-Fun Control-Camera](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera) | Base video model used by ReDirector. |
| [LTX 2.3](https://huggingface.co/Lightricks/LTX-2.3) | Final video synthesis; union and detailer adapters retain their own release terms. |
| [CrossView-Warp IC-LoRA](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2) and [ComfyUI-CrossViewWarp](https://github.com/cseti007/ComfyUI-CrossViewWarp) by Cseti | The default Photoreal Pass: LTX 2.3 guided by a depth warp of the real video along the camera path. Training-asset provenance is in its [ATTRIBUTION.md](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2/blob/main/ATTRIBUTION.md). |
| [4DAnyone](https://github.com/ant-research/4DAnyone) | Generated multiview capture preparation. |
| [sam3dbx4danyone](https://github.com/OpsiClear-4DGS/sam3dbx4danyone) | Capture adaptation and training tooling. |
| [FreeTimeGsVanilla](https://github.com/OpsiClear-4DGS/FreeTimeGsVanilla), [gsplat](https://github.com/nerfstudio-project/gsplat) | Moving splat asset, training and preview rendering. |
| [GVHMR](https://github.com/zju3dv/GVHMR), [SMPL-X](https://smpl-x.is.tue.mpg.de/modellicense.html) | Body tracking/geometry used in current capture and camera registration. |
| [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) | Foreground preprocessing. |
| [Depth Anything 3 Base](https://huggingface.co/depth-anything/DA3-BASE), [MoGe](https://github.com/microsoft/MoGe) | Scene geometry and setting registration support; MoGe-2 also gives CrossView its depth. |
| [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Gemma 3](https://huggingface.co/google/gemma-3-12b-it) | Automatic captioning and LTX text encoding respectively. |
| Three.js / TransformControls | Rendering and transform controls in the 4D camera editor; notice retained in web/lib/THREE_LICENSE. |
| [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite), [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Video and graph helper nodes. |

Built with assistance from Claude Code and Codex. See [Models and licences](MODELS_AND_LICENSES.md) for code and model terms.

## Tested upstream revisions

The workflow was built and run against these revisions. Licence terms for each are in [Models and licences](MODELS_AND_LICENSES.md).

| Project | Revision | Licence |
| --- | --- | --- |
| ComfyUI | v0.37.0-9 (`e638023d`) | GPL-3.0 |
| ReDirector | `ede9779` | Apache-2.0 |
| 4DAnyone | `6dcf626` | Apache-2.0 |
| sam3dbx4danyone | `1356955` | AGPL-3.0 |
| FreeTimeGsVanilla | `42e2f9b` | AGPL-3.0 |
| gsplat | 1.5.3 | Apache-2.0 |
| GVHMR | `6ec3ca3` | Education, research and non-profit use only |
| SAM 3D Body | `b5c765a` | SAM License |
| ComfyUI-LTXVideo | `3bf3ca6` | LTX-2 Community License |
| ComfyUI-VideoHelperSuite | `4d907be` | GPL-3.0 |
| ComfyUI-KJNodes | `d3cfe21` | GPL-3.0 |
| three.js | bundled in web/lib | MIT, notice in web/lib/THREE_LICENSE |

## Citations

If you use Reshoot 4D in research, please cite the work it builds on.

```bibtex
@article{park2025redirector,
  title={ReDirector: Creating Any-Length Video Retakes with Rotary Camera Encoding},
  author={Park, Byeongjun and Kim, Byung-Hoon and Chung, Hyungjin and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2511.19827},
  year={2025}
}

@article{jin2026fdanyone,
  title={4DAnyone: Create Anyone in 4D from a Casual Monocular Video},
  author={Jin, Yudong and Xie, Tao and Zhang, Qihang and Shen, Zehong and Xu, Zhen and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei and Xu, Yinghao},
  journal={arXiv preprint arXiv:2608.20335},
  year={2026}
}

@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}

@article{yang2026sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Sagar, Anushka and Fan, Taosha and Park, Jinhyung and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Feiszli, Matt and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint arXiv:2602.15989},
  year={2026}
}
```

Also built on:

- LTX-2: Efficient Joint Audio-Visual Foundation Model. HaCohen et al., Lightricks. [arXiv:2601.03233](https://arxiv.org/abs/2601.03233)
- Wan: Open and Advanced Large-Scale Video Generative Models. Wan Team, Alibaba. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- FreeTimeGS: Free Gaussian Primitives at Anytime and Anywhere for Dynamic Scene Reconstruction. CVPR 2025. [arXiv:2506.05348](https://arxiv.org/abs/2506.05348)
- gsplat: An Open-Source Library for Gaussian Splatting. Ye et al., JMLR 2025. [arXiv:2409.06765](https://arxiv.org/abs/2409.06765)
- Depth Anything 3: Recovering the Visual Space from Any Views. ByteDance Seed. [arXiv:2511.10647](https://arxiv.org/abs/2511.10647)
- MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision. Wang et al., CVPR 2025. [arXiv:2410.19115](https://arxiv.org/abs/2410.19115)
- Bilateral Reference for High-Resolution Dichotomous Image Segmentation (BiRefNet). Zheng et al., CAAI AIR 2024. [arXiv:2401.03407](https://arxiv.org/abs/2401.03407)
- Expressive Body Capture: 3D Hands, Face, and Body from a Single Image (SMPL-X). Pavlakos et al., CVPR 2019. [arXiv:1904.05866](https://arxiv.org/abs/1904.05866)

## Example footage (Pexels)

Example footage is credited below with its source pages and the [Pexels licence](https://www.pexels.com/license/).

| Video in the examples | Pexels video | Creator |
| --- | --- | --- |
| window man | [30784702](https://www.pexels.com/video/man-smoking-by-window-in-evening-sunlight-30784702/) | Eren Ataselim |
| railing woman | [35285874](https://www.pexels.com/video/woman-posing-on-bridge-in-sunny-park-35285874/) | Anh Nguyen |
| green car woman | [16120740](https://www.pexels.com/video/a-woman-in-white-shorts-and-a-green-car-16120740/) | paashuu |
| orange top | [5385804](https://www.pexels.com/video/woman-dancing-outdoors-5385804/) | Polina Tankilevitch |
| bench man | [7251409](https://www.pexels.com/ca-es/video/ciutat-persona-edificis-direccio-7251409/) | MART PRODUCTION |
| runner tying laces | [7877153](https://www.pexels.com/video/a-woman-tying-her-shoelace-7877153/) | MART PRODUCTION |
| beanie man | [8118142](https://www.pexels.com/video/a-man-playing-drum-machine-8118142/) | Anna Pou |
| mic man | [8135170](https://www.pexels.com/video/man-is-talking-while-recording-using-condenser-microphone-8135170/) | ANTONI SHKRABA production |

## Demo video audio

| Part | Source | Licence |
| --- | --- | --- |
| Backing track | ["Lifted Positive Garage"](https://pixabay.com/music/upbeat-lifted-positive-garage-431116/) by 9jackjack8 | [Pixabay Content License](https://pixabay.com/service/license-summary/) |
| Walkthrough narration | [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS, voice af_heart, run locally | Apache-2.0 |
