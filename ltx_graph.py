"""Vendored from the author's INF Studio builder (ladder/studio/studio_graph.py) for DROPCAM 4D. Local change: spec["single_stage"]
renders at full resolution in one pass with the guides attached for the whole denoise (what restoration IC-LoRAs such as
Deblur expect; their stage 2 has no reference and drifts on identity), and spec["control"]["lora_name"] picks the IC-LoRA.
INF Studio graph builder: job spec -> ComfyUI API graph.   (single source of truth for the workflow tab and a future UI)

Modules (each a function adding nodes to a Graph and returning its outputs):
  loaders_ltx   LTX model/clip/vae/upscaler loaders
  refs          reference images -> resized guide frames
  ltx_render    two-stage LTX 2.3 render: t2v / i2v / start+end / anchors / control video / audio modes
Job spec keys are documented in DEFAULT_SPEC.
"""
import copy, json, math, os, time, urllib.request

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")   # standalone helpers only (queue/wait); nodes run in-process
HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SPEC = {
    "prompt": "", "negative": "pc game, console game, video game, cartoon, childish, ugly",
    "video": {"seconds": 5, "fps": 24, "width": 768, "height": 1344, "seed": 5501},
    # refs: list of {"image": file in ComfyUI/input, "at": seconds | "start" | "end", "strength": 0.8}
    "refs": [],
    # audio: mode generate (model invents audio from prompt) | custom (use file, lip-sync) | id_lora (voice ref)
    "audio": {"mode": "generate", "file": None, "id_guidance": 2.0, "id_ref_seconds": 5},
    # control: reference video as motion control through the LTX union IC-LoRA
    "control": {"video": None, "type": "canny", "strength": 0.8, "lora_strength": 0.7, "frames_skip": 0},
    "character_lora": {"name": None, "strength": 0.8},
    # stills: Qwen 2.1 + Lenovo LoRA image generation (character face as reference image); klein = optional detail upscale
    "stills": {"enabled": False, "prompt": "", "face": None, "lora": "lenovo_qwen21.safetensors", "lora_strength": 0.7,
               "negative": "text, letters, logo, watermark, cartoon, 3d render, plastic skin, heavy makeup, deformed hands, extra fingers, child",
               "width": 768, "height": 1344, "steps": 30, "cfg": 4.0, "seed": 5501, "klein": False, "save": True,
               "ingredients": [], "only": False},
    # identity: LTX-Best-Face-ID reference-to-video LoRA + reinforcer node. mode face (close-up image) | sheet (4-panel 1536x1024) | none
    "identity": {"mode": "none", "image": None, "face_lora": 1.0, "sheet_lora": 1.0, "base_with_sheet": 0.2, "identity_strength": 1.0,
                 "source_id": 2.0, "gating": "mask_soft", "placement": "i2v_safe", "prefix_ref_t2v": True},
    # charsheet: generate a character sheet (H3 turnaround video -> LTX-format 4-panel 1536x1024 sheet: face | front | side | back).
    # source = image filename (full-body works best) | "stills" (the generated Lenovo still). Frame numbers pick the views from the 124-frame turnaround.
    "charsheet": {"enabled": False, "source": None, "seed": 20260922, "only": False, "frames": {"face": 83, "front": 7, "side": 41, "back": 23}},
    # simple: describe it in a sentence -> Gemma expands it -> Lenovo still (always) -> LTX. look = standing description of the character.
    "simple": {"enabled": False, "description": "", "look": "a woman in her mid twenties with light freckles across her nose and cheeks, brown eyes, dark brown hair slicked back into a low bun",
               "face": None, "ingredients": [], "speech": None, "seed": 7, "ground_video_prompt_on_still": True,
               # layer switches: expand_still/expand_video = Gemma rewrites (off = description / spec prompt as-is); character = look text + face ref + persona
               # wording; style = selfie wording; raw = all four off (only the l3n0v0 trigger stays, the Lenovo LoRA needs it)
               "expand_still": True, "expand_video": True, "character": True, "style": True, "raw": False},
    # detail: LTX IC-LoRA detailer on the stage-2 (full-res) pass.  post: RTX video super-res (scale) + encode quality (crf, lower = better)
    "detail": {"lora": "ltx-2-19b-ic-lora-detailer.safetensors", "strength": 1.0},
    "post": {"vsr": None, "crf": 14},
    "prefix": "studio/ltx",
}

MODELS = {
    "face_id": "Best_FaceID_v1.0_LoRA.safetensors", "face_id_sheet": "Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors",
    "unet": "ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors",
    "gemma": "gemma_3_12B_it_fp4_mixed.safetensors", "proj": "ltx-2.3_text_projection_bf16.safetensors",
    "vae": "LTX23_video_vae_bf16.safetensors", "avae": "LTX23_audio_vae_bf16.safetensors",
    "upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "id_lora": "ltx-2.3-id-lora-celebvhq-3k.safetensors",
    "ic_lora": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "detailer": "ltx-2-19b-ic-lora-detailer.safetensors",
}
S1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
S2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
SAMPLER = "euler_ancestral_cfg_pp"


class Graph:
    def __init__(self):
        self.n = {}
        self._i = 0

    def add(self, cls, title=None, **inputs):
        self._i += 1
        nid = str(self._i)
        self.n[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title or cls}}
        return nid

    @staticmethod
    def o(nid, slot=0):
        return [nid, slot]

    def api(self):
        return json.loads(json.dumps(self.n))


def merge(d, base):
    out = copy.deepcopy(base)
    for k, v in (d or {}).items():
        out[k] = merge(v, out[k]) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def frames_for(seconds, fps):
    n = round(seconds * fps)                        # LTX needs 8k+1 frames: round up, the caller trims back to n
    return 8 * max(1, -(-(n - 1) // 8)) + 1


def check_dims(w, h, ic=False):
    m = 128 if ic else 64          # IC-LoRA control needs the half-res latent (w/64) divisible by 2
    assert w % m == 0 and h % m == 0, f"width/height must be multiples of {m}{' (control video on)' if ic else ''}: {w}x{h}"


# --------------------------------------------------------------------------- modules
def clean(g, anything, title="Free RAM/VRAM before next model"):
    """pass-through that unloads models + empties caches between model families. Uses KJNodes VRAM_Debug (NOT an output node):
    easy cleanGpuUsed is an output node, so it always executes and defeats lazy switches by dragging its whole upstream branch in."""
    return g.o(g.add("VRAM_Debug", title, empty_cache=True, gc_collect=True, unload_all_models=True, any_input=anything))


def stills_module(g, spec):
    """Qwen 2.1 + Lenovo LoRA still, optionally conditioned on the character face image; returns IMAGE output."""
    st = spec["stills"]
    unet = g.add("UNETLoader", "Qwen Image 2.1", unet_name="qwen_image_2.1_int8_convrot.safetensors", weight_dtype="default")
    lora = g.add("LoraLoaderModelOnly", "Lenovo LoRA", model=g.o(unet), lora_name=st["lora"], strength_model=st["lora_strength"])
    clip = g.add("CLIPLoader", "Qwen3-VL encoder", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default")
    vae = g.add("VAELoader", "Qwen VAE", vae_name="qwen_image_2.1_vae_bf16.safetensors")
    enc_in = dict(prompt=st["prompt"], negative_prompt=st["negative"], resolution=max(st["width"], st["height"]), clip=g.o(clip))
    ref = None
    if st.get("ingredients"):
        ref = collage(g, st.get("face"), st["ingredients"])
    elif st.get("face"):
        ref = g.o(g.add("LoadImage", "Character face", image=st["face"], upload="image"))
    if ref is not None:                      # exactly ONE reference image (two or more crash ComfyUI on this stack)
        enc_in["images.image_1"] = ref; enc_in["vae"] = g.o(vae)
    enc = g.add("TextEncodeQwenImage21", "Still prompt (+ references)", **enc_in)
    lat = g.add("EmptyLatentImage", "Still size", width=st["width"], height=st["height"], batch_size=1)
    ks = g.add("KSampler", "Still sample", model=g.o(lora), seed=st["seed"], steps=st["steps"], cfg=st["cfg"], sampler_name="euler", scheduler="simple",
               positive=g.o(enc, 0), negative=g.o(enc, 1), latent_image=g.o(lat), denoise=1.0)
    img = g.o(g.add("VAEDecode", "Decode still", samples=g.o(ks), vae=g.o(vae)))
    if st.get("save"):
        g.add("SaveImage", "Save still", images=img, filename_prefix=spec["prefix"] + "_still")
    if st.get("klein"):
        img = klein_module(g, clean(g, img, "Free before Klein"))
        if st.get("save"):
            g.add("SaveImage", "Save Klein still", images=img, filename_prefix=spec["prefix"] + "_still_klein")
    return img


def collage(g, face, ingredients):
    """Pack the character face (left) and ingredient images (stacked, right) into ONE reference image."""
    def load(name, title):
        return g.o(g.add("LoadImage", title, image=name, upload="image"))
    def fit(img, w, h, title):
        return g.o(g.add("ImageScale", title, image=img, upscale_method="lanczos", width=w, height=h, crop="center"))
    col = fit(load(ingredients[0], f"Ingredient 1: {ingredients[0]}"), 512, 384, "ingredient 1 panel")
    for k, ing in enumerate(ingredients[1:], start=2):
        nxt = fit(load(ing, f"Ingredient {k}: {ing}"), 512, 384, f"ingredient {k} panel")
        col = g.o(g.add("ImageStitch", f"stack ingredient {k}", image1=col, direction="down", match_image_size=False, spacing_width=0, spacing_color="white", image2=nxt))
    if len(ingredients) == 1:
        col = fit(load(ingredients[0], f"Ingredient 1: {ingredients[0]}"), 512, 768, "ingredient panel")
    else:
        col = fit(col, 512, 768, "ingredients column")
    if face is None:                         # character shaping off: ingredients only
        return col
    left = fit(load(face, "Character face"), 640, 768, "face panel")
    return g.o(g.add("ImageStitch", "face | ingredients", image1=left, direction="right", match_image_size=False, spacing_width=8, spacing_color="white", image2=col))


def gen_text(g, clip, prompt, title, max_length=400, ltx_template=False, image=None, seed=7):
    cls = "TextGenerateLTX2Prompt" if ltx_template else "TextGenerate"
    inp = dict(clip=clip, prompt=prompt, max_length=max_length, thinking=False, use_default_template=bool(ltx_template))
    inp.update({"sampling_mode": "on", "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64, "sampling_mode.top_p": 0.95,
                "sampling_mode.min_p": 0.05, "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": seed})
    if image is not None:
        inp["image"] = image
    return g.o(g.add(cls, title, **inp))


def strcat(g, a, b, delim=""):
    return g.o(g.add("StringConcatenate", "join text", string_a=a, string_b=b, delimiter=delim))


def ingredient_captions(g, spec, clip):
    """Ingredient labels supplied by the user (e.g. 'a camel wool double-breasted coat'). NOTE: Gemma/Qwen3-VL cannot caption images through TextGenerate
    (they hallucinate), so labels are required text, not auto-generated."""
    labels = [l for l in (spec["simple"].get("ingredient_labels") or []) if l]
    if not labels:
        return None
    return "; ".join(labels)


def layers(sp):
    """effective layer switches for simple mode (raw turns them all off)"""
    return {k: bool(sp.get(k, True)) and not sp.get("raw") for k in ("expand_still", "expand_video", "character", "style")}


def simple_still_prompt(g, spec, clip):
    """description -> expanded photo description -> full Lenovo prompt (trigger word + selfie style + character reference)"""
    sp = spec["simple"]; on = layers(sp)
    caps = ingredient_captions(g, spec, clip) if sp["ingredients"] else None
    if on["expand_still"]:
        instr = ("You write prompts for a photorealistic image model. Turn the idea below into ONE paragraph (under 90 words) describing "
                 + ("a candid iPhone selfie photo: " if on["style"] else "a photo: ")
                 + ("the setting, the lighting, what she is wearing, her expression and pose, and the camera framing. Keep the person exactly as described. " if on["character"]
                    else "the setting, the lighting, the subject, the pose and the camera framing. ")
                 + "No quotation marks, no lists, no preamble, output only the paragraph.")
        person = (f"\n\nPerson: {sp['look']}" if on["character"] else "\n") + f"\nIdea: {sp['description']}"
        if caps is not None:
            instr_in = instr + " The items she wears or uses are fixed facts, use them exactly as given and never change their colour or material: " + caps + "." + person
        else:
            instr_in = instr + person
        scene = gen_text(g, clip, instr_in, "Expand still description (Gemma)", max_length=300, seed=sp["seed"])
        scene = g.o(g.add("StringReplace", "tidy", string=scene, find="\n", replace=" "))
        g.add("PreviewAny", "Expanded still description", source=scene)
    else:
        scene = sp["description"]
    head = "l3n0v0, " + ("amateur photo, candid iPhone front-camera selfie, vertical 9:16, " if on["style"] else "")
    if on["character"]:
        head += f"the woman in the reference image (keep exactly her face, freckles, eyes, nose and lips), {sp['look']}. "
    if caps is not None:
        head = head + f"She is wearing or using: {caps} (shown in the {'right part of the ' if on['character'] else ''}reference image, reproduce exactly). "
    p = strcat(g, head, scene)
    tail = " The phone is held out at arm's length, natural skin texture with visible pores, slight wide-angle lens distortion." if on["style"] else ""
    p = strcat(g, p, tail)
    g.add("PreviewAny", "Final still prompt", source=p)
    return p, (scene if on["expand_still"] else None), caps


def simple_video_prompt(g, spec, clip, scene, caps):
    """expanded video prompt; grounded on the expanded still text (the enhancer cannot see the still image)"""
    sp = spec["simple"]; on = layers(sp)
    if not on["expand_video"]:               # Gemma video rewrite off: spec prompt as-is (falls back to the description)
        return spec["prompt"] or sp["description"]
    base = (f"{sp['look']}. " if on["character"] else "") + f"{sp['description']}."
    if caps:
        base += f" She wears or uses: {caps}."
    if sp.get("speech"):
        base += f' She says: "{sp["speech"]}"'
    elif on["character"]:
        base += " She talks to the camera in a natural, casual voice."
    base = ("A vertical iPhone front-camera selfie video, handheld, one continuous shot. " if on["style"] else "") + base
    text = strcat(g, base + " Visual context of the opening frame: ", scene) if scene is not None else base
    out = gen_text(g, clip, text, "Expand video prompt (Gemma, LTX template)", max_length=700, ltx_template=True, seed=sp["seed"])
    g.add("PreviewAny", "Expanded video prompt", source=out)
    return out


def charsheet_module(g, spec, ref_image):
    """Import the proven H3 character-sheet graph (MICW turnaround, 896x1184, 124 frames) and compose the LTX 4-panel sheet. Returns (sheet4 IMAGE, turnaround frames IMAGE)."""
    cs = spec["charsheet"]
    api = json.load(open(os.path.join(HERE, "charsheet_h3_api.json")))
    drop = {"51", "4264"} | {k for k in api if len(k) == 4 and k.startswith("90")}     # reference loader, preview video, old 3x3 sheet composer
    remap = lambda v: ["cs" + v[0], v[1]] if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in api and v[0] not in drop else v
    for k, node in api.items():
        if k in drop:
            continue
        ins = {a: remap(b) for a, b in node["inputs"].items()}
        for a, b in list(ins.items()):
            if isinstance(b, list) and len(b) == 2 and b[0] == "51":
                ins[a] = ref_image
        if node["class_type"] == "RandomNoise":
            ins["noise_seed"] = cs["seed"]
        if node["class_type"] == "VHS_VideoCombine":
            ins["filename_prefix"] = spec["prefix"] + "_turnaround"
        g.n["cs" + k] = {"class_type": node["class_type"], "inputs": ins, "_meta": {"title": "H3 sheet: " + node["class_type"]}}
    frames = ["cs253", 0]
    fr = cs["frames"]

    def pick(i, title):
        return g.o(g.add("ImageFromBatch", title, image=frames, batch_index=i, length=1))

    def cover(img, w, h, title):
        return g.o(g.add("ImageScale", title, image=img, upscale_method="lanczos", width=w, height=h, crop="center"))
    face = g.o(g.add("ImageScale", "face panel scale", image=pick(fr["face"], f"frame {fr['face']} (face)"), upscale_method="lanczos", width=384, height=507, crop="disabled"))
    canvas = g.o(g.add("EmptyImage", "white panel", width=384, height=1024, batch_size=1, color=16777215))
    p1 = g.o(g.add("ImageCompositeMasked", "face panel", destination=canvas, source=face, x=0, y=0, resize_source=False))
    p2 = cover(pick(fr["front"], f"frame {fr['front']} (front)"), 384, 1024, "front panel")
    half = g.o(g.add("ImageCrop", "left profile half", image=pick(fr["side"], f"frame {fr['side']} (side)"), width=448, height=1184, x=0, y=0))
    p3 = cover(half, 384, 1024, "side panel")
    p4 = cover(pick(fr["back"], f"frame {fr['back']} (back)"), 384, 1024, "back panel")
    row = g.o(g.add("ImageStitch", "face | front", image1=p1, direction="right", match_image_size=False, spacing_width=0, spacing_color="white", image2=p2))
    row = g.o(g.add("ImageStitch", "| side", image1=row, direction="right", match_image_size=False, spacing_width=0, spacing_color="white", image2=p3))
    sheet = g.o(g.add("ImageStitch", "| back", image1=row, direction="right", match_image_size=False, spacing_width=0, spacing_color="white", image2=p4))
    g.add("SaveImage", "Save LTX 4-panel sheet", images=sheet, filename_prefix=spec["prefix"] + "_sheet4")
    return sheet, frames


def klein_module(g, image):
    """Flux.2 Klein 9B detail pass (4 MP) with colour match; image in -> image out"""
    sh = g.add("ImageSharpen", "pre-sharpen", image=image, sharpen_radius=1, sigma=0.5, alpha=0.5)
    sc = g.add("ImageScaleToTotalPixels", "to 4 MP", image=g.o(sh), upscale_method="nearest-exact", megapixels=4.0, resolution_steps=2)
    unet = g.add("UNETLoader", "Flux.2 Klein 9B", unet_name="flux-2-klein-9b-fp8.safetensors", weight_dtype="default")
    clip = g.add("CLIPLoader", "Klein text encoder", clip_name="qwen_3_8b_fp8mixed.safetensors", type="flux2", device="default")
    vae = g.add("VAELoader", "Flux2 VAE", vae_name="flux2-vae.safetensors")
    pos = g.add("CLIPTextEncode", "Klein prompt", text="Upscale the image in high definition, restore realistic skin texture, remove plastic looking skin, keep the entire image content unchanged.", clip=g.o(clip))
    zero = g.add("ConditioningZeroOut", "negative", conditioning=g.o(pos))
    enc = g.add("VAEEncode", "Encode", pixels=g.o(sc), vae=g.o(vae))
    rp = g.add("ReferenceLatent", "ref (pos)", conditioning=g.o(pos), latent=g.o(enc))
    rn = g.add("ReferenceLatent", "ref (neg)", conditioning=g.o(zero), latent=g.o(enc))
    size = g.add("GetImageSize", "size", image=g.o(sc))
    emp = g.add("EmptyFlux2LatentImage", "latent", width=g.o(size, 0), height=g.o(size, 1), batch_size=1)
    sch = g.add("Flux2Scheduler", "2 steps", steps=2, width=g.o(size, 0), height=g.o(size, 1))
    gd = g.add("CFGGuider", "cfg 1", model=g.o(unet), positive=g.o(rp), negative=g.o(rn), cfg=1.0)
    sm = g.add("SamplerCustomAdvanced", "Klein sample", noise=g.o(g.add("RandomNoise", "seed", noise_seed=7)), guider=g.o(gd),
               sampler=g.o(g.add("KSamplerSelect", "euler", sampler_name="euler")), sigmas=g.o(sch), latent_image=g.o(emp))
    dec = g.add("VAEDecode", "Decode", samples=g.o(sm), vae=g.o(vae))
    cm = g.add("ColorMatch", "colour match", image_ref=g.o(sc), image_target=g.o(dec), method="mkl", strength=0.8, multithread=True)
    return g.o(g.add("ImageSharpen", "post-sharpen", image=g.o(cm), sharpen_radius=1, sigma=0.3, alpha=0.3))


def loaders_ltx(g, spec, audio_mode, use_ic):
    unet = g.add("UNETLoader", "LTX model", unet_name=MODELS["unet"], weight_dtype="default")
    model = g.o(unet)
    cl = spec["character_lora"]
    if cl.get("name"):
        model = g.o(g.add("LoraLoaderModelOnly", "Character LoRA", model=model, lora_name=cl["name"], strength_model=cl["strength"]))
    if audio_mode == "id_lora":
        model = g.o(g.add("LoraLoaderModelOnly", "ID-LoRA (voice)", model=model, lora_name=MODELS["id_lora"], strength_model=1.0))
    idm = spec["identity"]["mode"]
    if idm == "face":
        model = g.o(g.add("LoraLoaderModelOnly", "Best-Face-ID LoRA", model=model, lora_name=MODELS["face_id"], strength_model=spec["identity"]["face_lora"]))
    elif idm == "sheet":
        model = g.o(g.add("LoraLoaderModelOnly", "Best-Face-ID (base, low)", model=model, lora_name=MODELS["face_id"], strength_model=spec["identity"]["base_with_sheet"]))
        model = g.o(g.add("LoraLoaderModelOnly", "Best-Face-ID Character Sheet LoRA", model=model, lora_name=MODELS["face_id_sheet"], strength_model=spec["identity"]["sheet_lora"]))
    icp = None
    if use_ic:
        model = g.o(g.add("LoraLoaderModelOnly", "IC-LoRA union control", model=model, lora_name=spec["control"].get("lora_name") or MODELS["ic_lora"],
                          strength_model=spec["control"]["lora_strength"]))
        icp = g.o(g.add("GetICLoRAParameters", "IC-LoRA params", iclora_model=model))
    clip = g.o(g.add("DualCLIPLoader", "Gemma + projection", clip_name1=MODELS["gemma"], clip_name2=MODELS["proj"], type="ltxv", device="default"))
    vae = g.o(g.add("VAELoader", "LTX video VAE", vae_name=MODELS["vae"]))
    avae = g.o(g.add("VAELoaderKJ", "LTX audio VAE", vae_name=MODELS["avae"], device="main_device", weight_dtype="bf16"))
    ups = g.o(g.add("LatentUpscaleModelLoader", "LTX x2 upscaler", model_name=MODELS["upscaler"]))
    return dict(model=model, clip=clip, vae=vae, avae=avae, ups=ups, icp=icp)


def scaled_image(g, image_out, w, h, title):
    return g.o(g.add("ImageScale", title, image=image_out, upscale_method="lanczos", width=w, height=h, crop="center"))


def refs_module(g, spec, W, H, frames, still=None):
    """Return list of dicts {img, idx, strength} for every reference image. ref {"source": "stills"} uses the generated still."""
    out = []
    for r in spec["refs"]:
        if r.get("link"):   # an IMAGE output elsewhere in the graph (e.g. the previous chunk's last frame)
            at = r.get("at", "start")
            idx = 0 if at == "start" else (-1 if at == "end" else int(round(float(at) * spec["video"]["fps"])))
            out.append({"img": list(r["link"]), "idx": idx, "strength": r.get("strength", 0.8), "name": r.get("name", "linked image")})
            continue
        if r.get("source") == "stills":
            li = None
        else:
            li = g.add("LoadImage", f"Ref: {r['image']}", image=r["image"], upload="image")
        at = r.get("at", "start")
        idx = 0 if at == "start" else (-1 if at == "end" else int(round(float(at) * spec["video"]["fps"])))
        mask = None
        if r.get("mask"):   # where the reference holds real pixels: the guide only steers there (attention mask)
            mask = g.o(g.add("ImageToMask", f"Ref mask: {r['mask']}", image=g.o(g.add("LoadImage", f"Ref mask file: {r['mask']}", image=r["mask"], upload="image")), channel="red"))
        out.append({"img": still if li is None else g.o(li), "idx": idx, "strength": r.get("strength", 0.8), "name": r.get("image", "generated still"), "mask": mask})
    return out


def control_module(g, spec, W, H, frames):
    c = spec["control"]
    if not c.get("video"):
        return None
    vid = g.add("VHS_LoadVideo", "Control video", video=c["video"], force_rate=spec["video"]["fps"], custom_width=W, custom_height=H,
                frame_load_cap=frames, skip_first_frames=c.get("frames_skip", 0), select_every_nth=1, format="AnimateDiff")
    frames_out = g.o(vid, 0)
    if c["type"] == "canny":
        frames_out = g.o(g.add("Canny", "Canny control", image=frames_out, low_threshold=0.2, high_threshold=0.6))
    return frames_out


def audio_latent(g, spec, L, frames):
    a = spec["audio"]; fps = spec["video"]["fps"]
    if a["mode"] == "custom":
        la = g.add("LoadAudio", "Voice audio", audio=a["file"])
        tr = g.add("TrimAudioDuration", "Trim to length", audio=g.o(la), start_index=0.0, duration=float(spec["video"]["seconds"]))
        enc = g.add("LTXVAudioVAEEncode", "Encode audio", audio=g.o(tr), audio_vae=L["avae"])
        # frozen audio: noise mask 0 = keep exactly
        return g.o(g.add("SetLatentNoiseMask", "Freeze audio", samples=g.o(enc),
                         mask=g.o(g.add("SolidMask", "zero mask", value=0.0, width=W_DUMMY, height=H_DUMMY))))
    return g.o(g.add("LTXVEmptyLatentAudio", "Empty audio latent", frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=L["avae"]))


W_DUMMY = 64
H_DUMMY = 64


def ltx_render(g, spec, L, refs, control):
    v = spec["video"]; fps, secs = v["fps"], v["seconds"]; W, H = v["width"], v["height"]
    check_dims(W, H, bool(control is not None)); frames = frames_for(secs, fps)
    single = bool(spec.get("single_stage")); w2, h2 = (W, H) if single else (W // 2, H // 2)
    ptxt = spec.get("_prompt_link") or spec["prompt"]
    if isinstance(ptxt, str) and spec["identity"]["mode"] != "none" and spec["identity"].get("prefix_ref_t2v") and not ptxt.startswith("ref_t2v:"):
        ptxt = "ref_t2v: " + ptxt
    pos = g.add("CLIPTextEncode", "Prompt", text=ptxt, clip=L["clip"])
    neg = g.add("CLIPTextEncode", "Negative", text=spec["negative"], clip=L["clip"])
    cond = g.add("LTXVConditioning", "Frame-rate conditioning", positive=g.o(pos), negative=g.o(neg), frame_rate=float(fps))
    P, N = g.o(cond, 0), g.o(cond, 1)
    model = L["model"]
    if spec["audio"]["mode"] == "id_lora":
        ra = g.add("LoadAudio", "Voice reference", audio=spec["audio"]["file"])
        tr = g.add("TrimAudioDuration", "Reference 5 s", audio=g.o(ra), start_index=0.0, duration=float(spec["audio"].get("id_ref_seconds", 5)))
        rg = g.add("LTXVReferenceAudio", "ID-LoRA reference", model=model, positive=P, negative=N, reference_audio=g.o(tr),
                   audio_vae=L["avae"], identity_guidance_scale=spec["audio"].get("id_guidance", 2.0), start_percent=0.0, end_percent=1.0)
        model, P, N = g.o(rg, 0), g.o(rg, 1), g.o(rg, 2)

    # "reinforce" (DROPCAM): reference face through the reinforcer only, no Best-Face-ID LoRA, and every guide kept
    id_on = spec["identity"]["mode"] not in ("none", "reinforce")

    def add_guides(P, N, lat, W_, H_, stage_strength=None):
        for r in refs:
            im = scaled_image(g, r["img"], W_, H_, f"fit {r['name']} {W_}x{H_}")
            if id_on:
                # the identity node needs the first frame injected in place (no extra guide tokens); other anchors are unsupported with identity on
                if r["idx"] == 0:
                    lat = g.o(g.add("LTXVImgToVideoInplace", f"Start frame (in place) {r['name']}", vae=L["vae"], image=im, latent=lat,
                                    strength=r["strength"] if stage_strength is None else stage_strength, bypass=False))
                else:
                    print(f"WARN: anchor/end frame '{r['name']}'@{r['idx']} skipped: not supported together with identity mode")
                continue
            extra = {"attention_mask": r["mask"]} if r.get("mask") else {}
            ag = g.add("LTXVAddGuide", f"Guide {r['name']}@{r['idx']}", positive=P, negative=N, vae=L["vae"], latent=lat, image=im,
                       frame_idx=r["idx"], strength=r["strength"], **extra)
            P, N, lat = g.o(ag, 0), g.o(ag, 1), g.o(ag, 2)
        if control is not None:
            c = spec["control"]
            im = g.o(g.add("ImageScale", f"fit control {W_}x{H_}", image=control, upscale_method="lanczos", width=W_, height=H_, crop="center"))
            ag = g.add("LTXVAddGuide", "Control video guide", positive=P, negative=N, vae=L["vae"], latent=lat, image=im, frame_idx=0,
                       strength=c["strength"], iclora_parameters=L["icp"])
            P, N, lat = g.o(ag, 0), g.o(ag, 1), g.o(ag, 2)
        return P, N, lat

    idc = spec["identity"]; id_img = None
    if idc["mode"] != "none":
        id_img = g.o(g.add("LoadImage", "Identity reference (" + idc["mode"] + ")", image=idc["image"], upload="image"))

    def reinforce(m, target_latent, title):
        if id_img is None:
            return m
        return g.o(g.add("LTXFaceIdentityReinforcer", title, model=m, vae=L["vae"], reference_image=id_img, target_latent=target_latent,
                         identity_strength=idc["identity_strength"], face_padding=0.15, auto_face_crop=(idc["mode"] in ("face", "reinforce")), crop_zoom_factor=2.0,
                         spatial_gating=idc["gating"], placement_mode=idc["placement"], source_id=idc["source_id"], phase_scale=1.0, debug=False))

    # ---- stage 1 (half res)
    s1_sigmas = S1_SIGMAS
    if spec.get("source"):   # start from a rough render (DROPCAM: the splat) partway down the schedule instead of from noise
        src, st = spec["source"]["video"], float(spec["source"]["strength"])
        sv = g.add("VHS_LoadVideo", "Source video", video=src, force_rate=fps, custom_width=w2, custom_height=h2, frame_load_cap=frames,
                   skip_first_frames=0, select_every_nth=1, format="AnimateDiff")
        lat = g.o(g.add("VAEEncode", "Encode source", pixels=g.o(sv, 0), vae=L["vae"]))
        s1_sigmas = ", ".join([str(st)] + [x.strip() for x in S1_SIGMAS.split(",") if float(x) < st])
    else:
        lat = g.o(g.add("EmptyLTXVLatentVideo", "Empty video latent (half-res)", width=w2, height=h2, length=frames, batch_size=1))
    P1, N1, lat1 = add_guides(P, N, lat, w2, h2)
    model1 = reinforce(model, lat1, "Identity reinforcer (stage 1)")
    alat = audio_latent(g, spec, L, frames)
    cat1 = g.o(g.add("LTXVConcatAVLatent", "Concat A/V", video_latent=lat1, audio_latent=alat))
    guid1 = g.add("CFGGuider", "CFG 1 (stage 1)", model=model1, positive=P1, negative=N1, cfg=1.0)
    smp1 = g.add("SamplerCustomAdvanced", "Stage 1 sample", noise=g.o(g.add("RandomNoise", "Seed", noise_seed=v["seed"])), guider=g.o(guid1),
                 sampler=g.o(g.add("KSamplerSelect", "sampler", sampler_name=SAMPLER)),
                 sigmas=g.o(g.add("ManualSigmas", "Stage 1 sigmas", sigmas=s1_sigmas)), latent_image=cat1)
    sep1 = g.add("LTXVSeparateAVLatent", "Split A/V", av_latent=g.o(smp1))
    crop1 = g.add("LTXVCropGuides", "Drop guide tokens", positive=P1, negative=N1, latent=g.o(sep1, 0))
    if single:
        img = g.o(g.add("VAEDecodeTiled", "Decode video", samples=g.o(crop1, 2), vae=L["vae"], tile_size=768, overlap=64, temporal_size=4096, temporal_overlap=4))
        if spec.get("return_image"):
            return img[0]
        return g.add("VHS_VideoCombine", "Save video (h264)", images=img, frame_rate=float(fps), loop_count=0, filename_prefix=spec["prefix"],
                     format="video/h264-mp4", pix_fmt="yuv420p", crf=spec["post"].get("crf", 16), save_metadata=True, trim_to_audio=False,
                     pingpong=False, save_output=True)
    # ---- upscale
    up = g.o(g.add("LTXVLatentUpsampler", "x2 latent upscale", samples=g.o(crop1, 2), upscale_model=L["ups"], vae=L["vae"]))
    P2, N2, lat2 = add_guides(g.o(crop1, 0), g.o(crop1, 1), up, W, H, stage_strength=1.0)
    cat2 = g.o(g.add("LTXVConcatAVLatent", "Concat A/V (stage 2)", video_latent=lat2, audio_latent=g.o(sep1, 1)))
    model2_base = model
    if spec["detail"].get("lora"):
        model2_base = g.o(g.add("LoraLoaderModelOnly", "Detailer LoRA (stage 2)", model=model, lora_name=spec["detail"]["lora"], strength_model=spec["detail"]["strength"]))
    model2 = reinforce(model2_base, lat2, "Identity reinforcer (stage 2)")
    guid2 = g.add("CFGGuider", "CFG 1 (stage 2)", model=model2, positive=P2, negative=N2, cfg=1.0)
    smp2 = g.add("SamplerCustomAdvanced", "Stage 2 refine", noise=g.o(g.add("RandomNoise", "Seed 2", noise_seed=v["seed"])), guider=g.o(guid2),
                 sampler=g.o(g.add("KSamplerSelect", "sampler 2", sampler_name=SAMPLER)),
                 sigmas=g.o(g.add("ManualSigmas", "Stage 2 sigmas", sigmas=S2_SIGMAS)), latent_image=cat2)
    sep2 = g.add("LTXVSeparateAVLatent", "Split A/V (2)", av_latent=g.o(smp2))
    crop2 = g.add("LTXVCropGuides", "Drop guide tokens (2)", positive=P2, negative=N2, latent=g.o(sep2, 0))
    # ---- decode + save
    img = g.o(g.add("VAEDecodeTiled", "Decode video", samples=g.o(crop2, 2), vae=L["vae"], tile_size=768, overlap=64, temporal_size=4096, temporal_overlap=4))
    if spec.get("return_image"):   # frames only (node expansion): the caller chains, trims and saves
        return img[0]
    aud = g.o(g.add("LTXVAudioVAEDecode", "Decode audio", samples=g.o(sep2, 1), audio_vae=L["avae"]))
    post = spec["post"]
    if post.get("vsr"):
        img = g.o(g.add("RTXVideoSuperResolution", f"RTX super-res x{post['vsr']}", images=img, resize_type="scale by multiplier", quality="ULTRA", **{"resize_type.scale": float(post["vsr"])}))
    sv = g.add("VHS_VideoCombine", "Save video (h264)", images=img, frame_rate=float(fps), loop_count=0, filename_prefix=spec["prefix"], format="video/h264-mp4",
               pix_fmt="yuv420p", crf=post.get("crf", 16), save_metadata=True, trim_to_audio=False, pingpong=False, save_output=True, audio=aud)
    return sv


def build(spec_in):
    spec = merge(spec_in, DEFAULT_SPEC)
    g = Graph()
    v = spec["video"]; frames = frames_for(v["seconds"], v["fps"])
    use_ic = bool(spec["control"].get("video"))
    cs = spec["charsheet"]
    if cs["enabled"]:
        if cs["source"] in (None,):
            raise ValueError("charsheet.source must be an image file or 'stills'")
        if cs["source"] != "stills":
            charsheet_module(g, spec, g.o(g.add("LoadImage", "Character source", image=cs["source"], upload="image")))
            if cs.get("only"):
                return g.api()
    if spec["stills"]["enabled"] and spec["stills"].get("only"):
        stills_module(g, spec); return g.api()
    still = None
    sp = spec["simple"]
    L = None
    if sp["enabled"]:
        # text-to-video always goes through a Lenovo still: description -> Gemma expansion -> Qwen+Lenovo still -> expanded video prompt -> LTX
        L = loaders_ltx(g, spec, spec["audio"]["mode"], use_ic)
        st = spec["stills"]
        sprompt, scene, caps = simple_still_prompt(g, spec, L["clip"])
        st.update(enabled=True, face=sp["face"] if layers(sp)["character"] else None, ingredients=sp["ingredients"], prompt=clean(g, sprompt, "Free Gemma before Qwen"))
        still = clean(g, stills_module(g, spec), "Free Qwen before LTX")
        spec["_prompt_link"] = simple_video_prompt(g, spec, L["clip"], clean(g, scene, "order after still") if scene is not None else None, caps)
        if not any(r.get("at") == "start" for r in spec["refs"]):
            spec["refs"] = [{"source": "stills", "at": "start", "strength": 0.8}] + list(spec["refs"])
    elif spec["stills"]["enabled"]:
        still = clean(g, stills_module(g, spec), "Free Qwen before LTX")
    if cs["enabled"] and cs["source"] == "stills":
        if still is None:
            raise ValueError("charsheet.source='stills' needs simple mode or stills.enabled")
        sheet, _ = charsheet_module(g, spec, clean(g, still, "Free Qwen before H3"))
        if cs.get("only"):
            return g.api()
    if L is None:
        L = loaders_ltx(g, spec, spec["audio"]["mode"], use_ic)
    refs = refs_module(g, spec, v["width"], v["height"], frames, still)
    control = control_module(g, spec, v["width"], v["height"], frames)
    out = ltx_render(g, spec, L, refs, control)
    return (g.api(), out) if spec.get("return_image") else g.api()


# --------------------------------------------------------------------------- run helpers
def queue(graph):
    r = urllib.request.urlopen(urllib.request.Request(COMFY + "/prompt", json.dumps({"prompt": graph}).encode(), {"Content-Type": "application/json"}))
    return json.loads(r.read())["prompt_id"]


def wait(pid, timeout=900):
    t = time.time()
    while time.time() - t < timeout:
        try:
            h = json.load(urllib.request.urlopen(f"{COMFY}/history/{pid}"))
        except Exception:
            h = {}
        if pid in h:
            st = h[pid]["status"]
            err = [m[1].get("exception_message", "")[:300] for m in st["messages"] if m[0] == "execution_error"]
            return st["status_str"], round(time.time() - t), err, h[pid]["outputs"]
        time.sleep(4)
    return "timeout", timeout, [], {}


def free():
    urllib.request.urlopen(urllib.request.Request(COMFY + "/free", json.dumps({"unload_models": True, "free_memory": True}).encode(), {"Content-Type": "application/json"}))
