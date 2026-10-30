"""Photoreal Pass, CrossView: LTX 2.3 with Cseti's CrossView-Warp IC-LoRA (Apache-2.0) finishes the ReDirector retake.

Everything comes from the capture and the retake job, nothing per clip:
  keys   the retake's target cameras (path.npz, metres, source camera frame 0) as orbit az/el/dist about the subject
  pivot  the subject's depth from the source camera at frame 0 (capture geometry, metres from her 1.7 m height)
  hfov   the source camera's intrinsics, after the crop to LTX's frame
MoGe-2 depth of the real source drives CrossViewWarp's depth warp along those keys. LTX stage 1 starts from the retake partway down
the schedule with the warp and the real source as IC-LoRA guides, then the x2 latent upscale and stage 2 with the source guide.
"""
import json, math
from pathlib import Path

import cv2
import numpy as np
import folder_paths

from . import ltx_graph as sg
from . import shot as S

LORA = "LTX2.3-22B_IC-LoRA-CrossView-Warp_v2_6000.safetensors"
MOGE = "moge_2_vitl_normal_fp16.safetensors"
SAMPLER = "euler_cfg_pp"


def _cover(img, w, h):
    s = max(w / img.shape[1], h / img.shape[0]); img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return img[y0:y0 + h, x0:x0 + w]


def camera(job, LW, LH):
    """Keyframes, pivot depth and horizontal FOV for CrossViewWarp, plus the real source frames at the retake's sample indices."""
    z = np.load(job["npz"]); td = Path(job["npz"]).parent.parent
    cap = S.Capture(str(td), zoom_views=False, scene=True)
    fs = z["frames"]; C0 = cap.pose(cap.source_cam, int(fs[0])); V = cap.geometry()
    c = np.linalg.inv(C0) @ np.append(V[int(fs[0])].mean(0), 1.0)
    pz = float(c[2] * z["metres_per_unit"])   # the orbit centre: her depth in front of the source camera
    keys = []
    for i in list(range(0, len(fs), 4)) + ([len(fs) - 1] if (len(fs) - 1) % 4 else []):
        # the target camera seen from the source camera at the same moment (the warp moves each source frame from its own camera);
        # cond and targ share frame 0, so for a tripod source this is targ itself
        rel = np.linalg.inv(z["cond"][i]) @ z["targ"][i]
        v = rel[:3, 3] - np.array([0.0, 0.0, pz]); d = float(np.linalg.norm(v)); u = v / d   # OpenCV: x right, y down, z forward
        keys.append({"f": i + 1, "az": round(math.degrees(math.atan2(u[0], -u[2])), 3), "el": round(math.degrees(math.asin(-u[1])), 3), "dist": round(d / pz, 4)})
    r = cv2.VideoCapture(str(z["video"])); src = []
    while True:
        ok, f = r.read()
        if not ok:
            break
        src.append(f)
    r.release()
    fh, fw = src[0].shape[:2]; fx = float(z["K"][0, 0]) * fw / float(z["W"])
    vis_w = LW / max(LW / fw, LH / fh)   # source pixels across LTX's frame after the cover crop
    hfov = math.degrees(2 * math.atan(vis_w / 2 / fx))
    pick = [src[min(int(i), len(src) - 1)] for i in fs]
    return keys, pz, hfov, pick


def build(job, LW, LH, fps, seed, start, detailer=True):
    """The CrossView graph (API format) and its decode node id."""
    keys, pz, hfov, frames = camera(job, LW, LH)
    inp = Path(folder_paths.get_input_directory()); name = f"{job['tag']}_cvsrc.mp4"
    w = cv2.VideoWriter(str(inp / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (LW, LH))
    for f in frames:
        w.write(_cover(f, LW, LH))
    w.release()
    FR = len(frames); g = sg.Graph(); o = sg.Graph.o
    load = lambda video, w_, h_, title: g.add("VHS_LoadVideo", title, video=video, force_rate=0, custom_width=w_, custom_height=h_, frame_load_cap=FR,
                                               skip_first_frames=0, select_every_nth=1, format="AnimateDiff")
    src = load(name, LW, LH, "Real source")
    moge = g.add("MoGeInference", "MoGe-2 depth", moge_model=o(g.add("LoadMoGeModel", "MoGe-2", model_name=MOGE)), image=o(src), resolution_level=2,
                 fov_x_degrees=round(hfov, 3), batch_size=4, force_projection=True, apply_mask=True, refine_steps=0)
    warp = g.add("CrossViewWarp", "Depth warp along the move", frames=o(src), azimuth=0.0, elevation=0.0, distance=1.0, hfov=round(hfov, 3), vertical_shift=0.0,
                 depth_ratio=6.0, smooth_depth=False, invert_depth=False, roll_lock=True, pivot_override=True, pivot_x=0.0, pivot_y=0.0, pivot_z=round(pz, 4),
                 use_keyframes=True, frame_count=0, keyframes=json.dumps(keys), interp_motion="smooth", interpolation="smooth", keep_source_aim=True,
                 moge_geometry=o(moge), preview_size=512, frame_index=1)
    unet = g.add("UNETLoader", "LTX model", unet_name=sg.MODELS["unet"], weight_dtype="default")
    model = o(g.add("LTXICLoRALoaderModelOnly", "CrossView IC-LoRA", model=o(unet), lora_name=LORA, strength_model=1.0))
    clip = o(g.add("DualCLIPLoader", "Gemma + projection", clip_name1=sg.MODELS["gemma"], clip_name2=sg.MODELS["proj"], type="ltxv", device="default"))
    vae = o(g.add("VAELoader", "LTX video VAE", vae_name=sg.MODELS["vae"]))
    avae = o(g.add("VAELoaderKJ", "LTX audio VAE", vae_name=sg.MODELS["avae"], device="main_device", weight_dtype="bf16"))
    cond = g.add("LTXVConditioning", "Conditioning", positive=o(g.add("CLIPTextEncode", "Prompt", text="Crossview.", clip=clip)),
                 negative=o(g.add("CLIPTextEncode", "Negative", text="text, subtitles, logo, watermark, signature", clip=clip)), frame_rate=float(fps))
    rt = load(job["tag"] + ".mp4", LW // 2, LH // 2, "Retake")
    lat = o(g.add("VAEEncode", "Retake latent", pixels=o(rt), vae=vae))
    s1 = ", ".join([f"{start:g}"] + [x.strip() for x in sg.S1_SIGMAS.split(",") if float(x) < start])

    def guide(P, N, lat, img, title):
        a = g.add("LTXAddVideoICLoRAGuide", title, positive=P, negative=N, vae=vae, latent=lat, image=img, frame_idx=0, strength=1.0,
                  latent_downscale_factor=1.0, crop="disabled", use_tiled_encode=False, tile_size=256, tile_overlap=64)
        return o(a, 0), o(a, 1), o(a, 2)

    def sample(P, N, av, sig, title, m=None):
        gd = g.add("CFGGuider", title + " guider", model=m or model, positive=P, negative=N, cfg=1.0)
        return g.add("SamplerCustomAdvanced", title, noise=o(g.add("RandomNoise", title + " seed", noise_seed=seed)), guider=o(gd),
                     sampler=o(g.add("KSamplerSelect", title + " sampler", sampler_name=SAMPLER)), sigmas=o(g.add("ManualSigmas", title + " sigmas", sigmas=sig)),
                     latent_image=av)

    P1, N1, l1 = guide(o(cond, 0), o(cond, 1), lat, o(warp, 0), "Warp guide")
    P1, N1, l1 = guide(P1, N1, l1, o(src), "Source guide")
    alat = o(g.add("LTXVEmptyLatentAudio", "Audio latent", audio_vae=avae, frames_number=FR, frame_rate=fps, batch_size=1))
    sm1 = sample(P1, N1, o(g.add("LTXVConcatAVLatent", "AV latent", video_latent=l1, audio_latent=alat)), s1, "Stage 1")
    sep1 = g.add("LTXVSeparateAVLatent", "Split AV", av_latent=o(sm1)); cr1 = g.add("LTXVCropGuides", "Drop guide tokens", positive=P1, negative=N1, latent=o(sep1, 0))
    up = o(g.add("LTXVLatentUpsampler", "x2 upscale", samples=o(cr1, 2), vae=vae,
                 upscale_model=o(g.add("LatentUpscaleModelLoader", "LTX x2 upscaler", model_name=sg.MODELS["upscaler"]))))
    P2, N2, l2 = guide(o(cr1, 0), o(cr1, 1), up, o(src), "Source guide (2)")
    sm2 = sample(P2, N2, o(g.add("LTXVConcatAVLatent", "AV latent (2)", video_latent=l2, audio_latent=o(sep1, 1))), sg.S2_SIGMAS, "Stage 2",
                 o(g.add("LoraLoaderModelOnly", "Detailer LoRA (stage 2)", model=model, lora_name=sg.MODELS["detailer"], strength_model=1.0)) if detailer else None)
    sep2 = g.add("LTXVSeparateAVLatent", "Split AV (2)", av_latent=o(sm2)); cr2 = g.add("LTXVCropGuides", "Drop guide tokens (2)", positive=P2, negative=N2, latent=o(sep2, 0))
    dec = g.add("VAEDecodeTiled", "Decode", samples=o(cr2, 2), vae=vae, tile_size=768, overlap=64, temporal_size=4096, temporal_overlap=4)
    return g.api(), dec, dict(keys=keys, pivot_z=pz, hfov=hfov)
