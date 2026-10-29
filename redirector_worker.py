"""ReDirector retake, run in ReDirector's own venv (torch 2.8 + DiffSynth) by the Retake node: the source clip re-shot along the
target camera path. Reads the job written by retake.py, writes the retake mp4, prints "STEP i n" lines for ComfyUI's progress bar.
usage: python redirector_worker.py <redirector dir> <job.json>"""
import json, os, sys
rd, job_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, rd); os.chdir(rd)
import numpy as np, torch, imageio
from PIL import Image
from einops import rearrange
from diffsynth.models import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from model_utils import adjust_to_ReDirector, save_video
from omegaconf import OmegaConf

job = json.loads(open(job_path).read()); cfg = OmegaConf.load("configs/base.yaml")
z = np.load(job["npz"]); Ht, Wt, steps = int(job["height"]), int(job["width"]), int(job["steps"])
fs = z["frames"]; K = z["K"]; W, H = int(z["W"]), int(z["H"]); N = len(fs)
r = imageio.get_reader(str(z["video"])); allf = [f for f in r]; r.close()
sc = max(Wt / W, Ht / H); nw, nh = round(W * sc), round(H * sc); x0, y0 = (nw - Wt) // 2, (nh - Ht) // 2
frames = [torch.from_numpy(np.asarray(Image.fromarray(allf[min(int(f), len(allf) - 1)]).resize((nw, nh), Image.BILINEAR)
                                      .crop((x0, y0, x0 + Wt, y0 + Ht)))).float().div(127.5).sub(1).permute(2, 0, 1) for f in fs]
video = rearrange(torch.stack(frames), "T C H W -> C T H W")
# intrinsics normalised to the cropped frame, principal point moved with the crop
fx, fy = K[0, 0] * sc / Wt, K[1, 1] * sc / Ht; cx, cy = (K[0, 2] * sc - x0) / Wt, (K[1, 2] * sc - y0) / Ht
intr = torch.tensor([0, fx, fy, cx, cy, 0, 0], dtype=torch.float32).unsqueeze(0).repeat(N, 1)
cam = lambda X: torch.cat([intr, rearrange(torch.as_tensor(X[:, :3], dtype=torch.float32), "b c d -> b (c d)")], 1)
pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda", redirect_common_files=False, model_configs=[
    ModelConfig(model_id=cfg.model_path, origin_file_pattern=p, offload_device="cpu", skip_download=True) for p in
    ["diffusion_pytorch_model*.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]],
    tokenizer_config=ModelConfig(model_id=cfg.model_path, origin_file_pattern="google/*", skip_download=True))
setattr(pipe, "dit", adjust_to_ReDirector(pipe.dit, cfg)[0])
sd = load_state_dict(cfg.eval.ckpt_path, torch_dtype=pipe.torch_dtype, device=pipe.device)
_, unexpected = pipe.dit.load_state_dict(sd, strict=False); assert not unexpected
pipe.enable_vram_management()


def progress(it):
    it = list(it)
    for i, x in enumerate(it):
        print(f"STEP {i} {len(it)}", flush=True); yield x
    print(f"STEP {len(it)} {len(it)}", flush=True)


v = pipe(prompt=job["prompt"], negative_prompt=cfg.eval.negative_prompt, first_frame=video.unsqueeze(0)[:, :, 0], cond_video=video.unsqueeze(0),
         traj=cam(z["targ"]).unsqueeze(0).cuda(), traj_cond=cam(z["cond"]).unsqueeze(0).cuda(), height=Ht, width=Wt, num_frames=N,
         cfg_scale=float(job["cfg"]), seed=int(job["seed"]), tiled=False, num_inference_steps=steps, progress_bar_cmd=progress)
save_video(v, job["out"], fps=N / float(z["seconds"]))
print("DONE", job["out"], "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 1), flush=True)
