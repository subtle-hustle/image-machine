"""Image Machine for PC: an offline text-to-image generator.

Runs Stable Diffusion 1.5-family models locally with diffusers and a Gradio UI.
After the first model download, everything works without an internet connection.
"""

import gc
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
CUSTOM_DIR = MODELS_DIR / "custom"
OUTPUTS_DIR = ROOT / "outputs"
for d in (MODELS_DIR, CUSTOM_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from diffusers import (  # noqa: E402
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from PIL import Image, ImageFilter  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

ZIMAGE = "Z-Image Turbo (best quality, slower)"
BUILTIN_MODELS = {
    ZIMAGE: "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
    "DreamShaper 8 (all-rounder)": "Lykon/dreamshaper-8",
    "Realistic Vision 5.1 (photos)": "SG161222/Realistic_Vision_V5.1_noVAE",
    "Stable Diffusion 1.5 (original)": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}
DEFAULT_MODEL = ZIMAGE
SD_LAYOUT_MODEL = "DreamShaper 8 (all-rounder)"
ZIMAGE_STEPS = 8
# Not in the model list: the fast edit mode's dedicated inpainting model.
INPAINT_MODEL = "DreamShaper 8 inpainting"
INPAINT_REPO = "Lykon/dreamshaper-8-inpainting"

# Pixel sizes per shape. Z-Image gets a little more room; every shape costs about
# the same time (roughly 1.5 to 2 minutes per picture on a GTX 1660 Ti).
SIZES = {
    "Square": {"sd": (512, 512), "z": (640, 640)},
    "Portrait": {"sd": (512, 768), "z": (512, 768)},
    "Landscape": {"sd": (768, 512), "z": (768, 512)},
    "Tall": {"sd": (448, 832), "z": (448, 832)},
    "Wide": {"sd": (832, 448), "z": (832, 448)},
}

STYLES = {
    "None": ("", ""),
    "Photo": ("photograph, 35mm, natural light, sharp focus, highly detailed",
              "painting, drawing, illustration, cartoon, render"),
    "Cinematic": ("cinematic still, dramatic lighting, shallow depth of field, film grain, color graded",
                  "flat lighting, cartoon"),
    "Digital art": ("digital painting, concept art, vibrant colors, intricate detail, artstation",
                    "photo, blurry"),
    "Anime": ("anime style, clean lineart, cel shading, vivid colors",
              "photo, realistic, 3d render"),
    "Fantasy": ("epic fantasy art, magical atmosphere, volumetric light, ornate detail",
                "photo, modern"),
    "Watercolor": ("watercolor painting, soft washes, paper texture, delicate",
                   "photo, 3d render, harsh lines"),
    "3D render": ("3d render, octane, studio lighting, subsurface scattering, ultra detailed",
                  "painting, sketch, flat"),
}

BASE_NEGATIVE = ("lowres, blurry, worst quality, low quality, jpeg artifacts, deformed, "
                 "bad anatomy, extra fingers, extra limbs, watermark, signature")

# The model's built-in safety checker is off. This one guard stays: sexual
# content involving minors is illegal everywhere and is never generated.
_MINOR = re.compile(r"\b(child|children|kid|kids|minor|underage|teen|teenage|preteen|loli|shota|"
                    r"toddler|infant|baby|schoolgirl|schoolboy|\d{1,2}\s*(yo|y/o|year[s]?\s*old))\b", re.I)
_SEXUAL = re.compile(r"\b(nude|naked|nsfw|sex|sexual|porn|erotic|explicit|topless|lingerie|"
                     r"genital\w*|breasts?|nipples?)\b", re.I)


def blocked(text: str) -> bool:
    return bool(_MINOR.search(text) and _SEXUAL.search(text))


def pick_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        # GTX 16xx cards produce black images in half precision, so they run in full precision.
        half_ok = not re.search(r"GTX 16|GTX 10|MX\d", name)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        return "cuda", (torch.float16 if half_ok else torch.float32), f"{name} ({vram:.0f} GB)"
    return "cpu", torch.float32, "CPU (no NVIDIA GPU found, this will be slow)"


DEVICE, DTYPE, DEVICE_LABEL = pick_device()
# Z-Image overflows in fp16, and on GTX 16xx cards fp16 matmuls are slower than fp32 anyway.
ZDTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False) \
    else torch.float32
_pipe = None
_pipe_key = None


def model_choices():
    customs = sorted(p.name for p in CUSTOM_DIR.glob("*.safetensors"))
    return list(BUILTIN_MODELS) + [f"Custom: {c}" for c in customs]


HF = "https://huggingface.co"
WEIGHT_DIRS = ("unet", "vae", "text_encoder")


def _http_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _download(url, dest, size, report):
    """Plain resumable download. huggingface_hub's own downloader stalls on some networks."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    failures = 0
    # Fetch in 32 MB ranges: whole-file requests for multi-GB files can sit at 0 bytes on some PCs.
    while size is None or have < size:
        end = have + (32 << 20) - 1 if size is not None else ""
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 200:  # server ignored the range and sent the whole file
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        report(have)
            failures = 0
            if size is None or r.status == 200:
                break
        except OSError:
            failures += 1
            if failures > 8:
                break
            time.sleep(3)
            have = part.stat().st_size if part.exists() else 0
    if size is not None and part.stat().st_size != size:
        raise gr.Error("The model download was interrupted. Check your internet and try again.")
    part.replace(dest)


def fetch_model(repo, progress):
    """Download a diffusers model once into models/<name>. Returns (folder, variant)."""
    folder = MODELS_DIR / repo.split("/")[-1]
    marker = folder / ".complete"
    if marker.exists():
        return folder, marker.read_text().strip() or None
    try:
        info = _http_json(f"{HF}/api/models/{repo}?blobs=true")
    except OSError:
        raise gr.Error("This model isn't downloaded yet and there's no internet connection.") from None
    files = {s["rfilename"]: s.get("size") for s in info["siblings"]}
    variant = "fp16" if all(f"{d}/{'model' if d == 'text_encoder' else 'diffusion_pytorch_model'}.fp16.safetensors"
                            in files for d in WEIGHT_DIRS) else None
    want = []
    for name, size in files.items():
        top = name.split("/")[0]
        if "unet/config.json" not in files:  # not an SD 1.5 layout: take every model file
            if top != "assets" and not name.endswith((".md", ".gitattributes", ".png", ".jpg", ".webp")):
                want.append((name, size))
        elif name == "model_index.json" or (top in ("scheduler", "tokenizer", "feature_extractor")):
            want.append((name, size))
        elif top in WEIGHT_DIRS:
            if name.endswith("config.json"):
                want.append((name, size))
            elif name.endswith(".fp16.safetensors" if variant else ".safetensors") and \
                    (variant or name.count(".") == 1):
                want.append((name, size))
    total = sum(s or 0 for _, s in want) or 1
    done = 0
    for name, size in want:
        base = done
        _download(f"{HF}/{repo}/resolve/main/{name}", folder / name, size,
                  lambda n, base=base: progress((base + n) / total,
                                                desc=f"Downloading model once: {(base + n) / 2**30:.1f} of {total / 2**30:.1f} GB"))
        done += size or 0
    marker.write_text(variant or "")
    return folder, variant


def load_pipe(choice, progress):
    global _pipe, _pipe_key
    if _pipe is not None and _pipe_key == choice:
        return _pipe
    _pipe = None
    gc.collect()  # free the previous model's RAM before loading the next one
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    kwargs = dict(torch_dtype=DTYPE, safety_checker=None, feature_extractor=None,
                  requires_safety_checker=False)
    if choice.startswith("Custom: "):
        path = CUSTOM_DIR / choice[len("Custom: "):]
        # Single-file checkpoints borrow their layout from DreamShaper, so they also load offline.
        folder, _ = fetch_model(BUILTIN_MODELS[SD_LAYOUT_MODEL], progress)
        progress(0, desc="Loading model…")
        pipe = StableDiffusionPipeline.from_single_file(str(path), config=str(folder), **kwargs)
    elif choice == ZIMAGE:
        folder, _ = fetch_model(BUILTIN_MODELS[choice], progress)
        progress(0, desc="Loading model…")
        _pipe, _pipe_key = load_zimage(folder), choice
        return _pipe
    else:
        repo = INPAINT_REPO if choice == INPAINT_MODEL else BUILTIN_MODELS[choice]
        cls = StableDiffusionInpaintPipeline if choice == INPAINT_MODEL else StableDiffusionPipeline
        folder, variant = fetch_model(repo, progress)
        progress(0, desc="Loading model…")
        pipe = cls.from_pretrained(str(folder), variant=variant, **kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    pipe.set_progress_bar_config(disable=True)
    pipe.to(DEVICE)
    pipe.vae.enable_tiling()
    _pipe, _pipe_key = pipe, choice
    return pipe


def load_zimage(folder):
    """Z-Image Turbo in 4-bit: the text encoder and image model take turns on the GPU."""
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(str(folder), torch_dtype=ZDTYPE)
    for part in (pipe.transformer, pipe.text_encoder):
        for mod in part.modules():
            if hasattr(mod, "compute_dtype"):
                mod.compute_dtype = ZDTYPE
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(prompt, negative, style, size, model, steps, cfg, seed, count, progress=gr.Progress()):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Type a description of the image first.")
    if blocked(prompt):
        raise gr.Error("That prompt can't be generated.")

    progress(0, desc="Loading model…")
    pipe = load_pipe(model, progress)
    is_z = model == ZIMAGE
    style_pos, style_neg = STYLES.get(style, ("", ""))
    full_prompt = f"{prompt}, {style_pos}" if style_pos else prompt
    width, height = SIZES[size]["z" if is_z else "sd"]
    steps, count = int(steps), int(count)
    if is_z:  # a distilled model: fixed short schedule, no negative prompt
        full_negative, steps, cfg = "", ZIMAGE_STEPS, 0.0
    else:
        full_negative = ", ".join(x for x in (negative.strip(), style_neg, BASE_NEGATIVE) if x)
    base_seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2**31 - 1)

    images, started = [], time.time()
    for i in range(count):
        s = base_seed + i

        def on_step(_pipe, step, _t, kw, i=i):
            progress((i * steps + step + 1) / (count * steps),
                     desc=f"Image {i + 1} of {count}, step {step + 1}/{steps}")
            return kw

        extra = {} if is_z else {"negative_prompt": full_negative}
        img = pipe(full_prompt, width=width, height=height,
                   num_inference_steps=steps, guidance_scale=float(cfg),
                   generator=torch.Generator(DEVICE).manual_seed(s),
                   callback_on_step_end=on_step, **extra).images[0]
        meta = {"prompt": full_prompt, "negative": full_negative, "seed": s, "steps": steps,
                "cfg": cfg, "size": f"{width}x{height}", "model": model}
        info = PngInfo()
        info.add_text("parameters", json.dumps(meta))
        path = OUTPUTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{s}.png"
        img.save(path, pnginfo=info)
        images.append((str(path), f"seed {s}"))

    took = time.time() - started
    status = f"Done: {count} image{'s' if count > 1 else ''} in {took:.0f}s · seed {base_seed} · saved to outputs"
    return images, status, base_seed


EDIT_AREA = "Change the painted area"
EDIT_WHOLE = "Vary the whole picture"
EDIT_BEST = "Best (Z-Image, about 1.5 min)"
EDIT_FAST = "Fast (DreamShaper, about 20 s)"


def fit_size(img, area, multiple):
    """Scale to roughly `area` pixels, keeping the shape, in sizes the model accepts."""
    scale = (area / (img.width * img.height)) ** 0.5
    return (max(multiple, round(img.width * scale / multiple) * multiple),
            max(multiple, round(img.height * scale / multiple) * multiple))


def save_output(img, meta):
    info = PngInfo()
    info.add_text("parameters", json.dumps(meta))
    path = OUTPUTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{meta['seed']}.png"
    img.save(path, pnginfo=info)
    return str(path)


def edit(editor, prompt, mode, quality, strength, seed, progress=gr.Progress()):
    if not editor or editor.get("background") is None:
        raise gr.Error("Add a picture first: upload one, or press \"Edit this picture\" on the Create tab.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Describe what the picture should show.")
    if blocked(prompt):
        raise gr.Error("That prompt can't be generated.")
    original = editor["background"].convert("RGB")
    mask = None
    if mode == EDIT_AREA:
        mask = Image.new("L", original.size, 0)
        for layer in editor.get("layers") or []:
            mask = Image.composite(Image.new("L", original.size, 255), mask,
                                   layer.convert("RGBA").getchannel("A").point(lambda a: 255 if a > 0 else 0))
        if not mask.getbbox():
            raise gr.Error("Paint over the part you want changed first (or pick \"Vary the whole picture\").")

    best = quality == EDIT_BEST
    s = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2**31 - 1)
    gen = torch.Generator(DEVICE).manual_seed(s)
    started = time.time()
    progress(0, desc="Loading model…")
    if best:
        from diffusers import ZImageImg2ImgPipeline, ZImageInpaintPipeline

        base = load_pipe(ZIMAGE, progress)
        size = fit_size(original, 640 * 640, 16)
        cls = ZImageInpaintPipeline if mask else ZImageImg2ImgPipeline
        pipe = cls.from_pipe(base)
        pipe.set_progress_bar_config(disable=True)
        args = dict(num_inference_steps=ZIMAGE_STEPS, guidance_scale=0.0)
    else:
        size = fit_size(original, 512 * 512, 8)
        if mask:
            pipe = load_pipe(INPAINT_MODEL, progress)
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pipe(load_pipe(SD_LAYOUT_MODEL, progress))
            pipe.set_progress_bar_config(disable=True)
        args = dict(num_inference_steps=30, guidance_scale=7.0, negative_prompt=BASE_NEGATIVE)
    progress(0.1, desc="Painting…")
    image_in = original.resize(size, Image.LANCZOS)
    extra = {"mask_image": mask.resize(size, Image.LANCZOS)} if mask else {}
    if best or mask:  # SD img2img takes its size from the input picture instead
        extra.update(width=size[0], height=size[1])
    out = pipe(prompt, image=image_in, strength=float(strength), generator=gen, **extra, **args).images[0]
    out = out.resize(original.size, Image.LANCZOS)
    if mask:  # keep every unpainted pixel exactly as it was, with a soft seam
        soft = mask.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(6))
        out = Image.composite(out, original, soft)
    path = save_output(out, {"prompt": prompt, "seed": s, "mode": mode, "strength": strength,
                             "model": ZIMAGE if best else INPAINT_MODEL if mask else SD_LAYOUT_MODEL})
    took = time.time() - started
    return path, f"Done in {took:.0f}s · seed {s} · saved to outputs. Press \"Keep editing this\" to build on it."


def open_outputs():
    if sys.platform == "win32":
        os.startfile(OUTPUTS_DIR)  # noqa: S606
    else:
        subprocess.Popen(["xdg-open", str(OUTPUTS_DIR)])


def recent_images():
    files = sorted(OUTPUTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:24]
    return [str(p) for p in files]


CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#title {text-align: left; margin-bottom: 0;}
#title h1 {font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; margin: 0;}
#subtitle {opacity: .7; margin-top: 2px;}
#prompt textarea {font-size: 1.05rem;}
#go, #go2 {min-height: 52px; font-size: 1.05rem; font-weight: 600;}
#gallery {min-height: 560px;}
footer {display: none !important;}
"""

theme = gr.themes.Soft(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate",
                       font=["Segoe UI", "Inter", "system-ui", "sans-serif"])

FORCE_DARK = """() => { if (!document.body.classList.contains('dark')) document.body.classList.add('dark'); }"""

with gr.Blocks(title="Image Machine", theme=theme, css=CSS, js=FORCE_DARK) as demo:
    gr.Markdown("# Image Machine", elem_id="title")
    gr.Markdown(f"Offline image generator · running on {DEVICE_LABEL}", elem_id="subtitle")
    with gr.Tabs() as tabs:
        with gr.Tab("Create", id="create"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=340):
                    prompt = gr.Textbox(label="Describe your image", lines=4, elem_id="prompt",
                                        placeholder="A lighthouse on a cliff at sunset, crashing waves, golden light")
                    with gr.Row():
                        style = gr.Dropdown(list(STYLES), value="None", label="Style")
                        size = gr.Dropdown(list(SIZES), value="Square", label="Shape")
                    count = gr.Slider(1, 4, value=1, step=1, label="How many images")
                    go = gr.Button("Generate", variant="primary", elem_id="go")
                    with gr.Accordion("More options", open=False):
                        negative = gr.Textbox(label="Leave out (SD 1.5 models only)", lines=2,
                                              placeholder="things you don't want in the picture")
                        model = gr.Dropdown(model_choices(), value=DEFAULT_MODEL, label="Model",
                                            info="Z-Image follows prompts and writes text best. The SD 1.5 models are "
                                                 "much faster. Drop SD 1.5 .safetensors files into models\\custom to add your own.")
                        steps = gr.Slider(10, 50, value=25, step=1, label="Steps (SD 1.5 models only)")
                        cfg = gr.Slider(1, 15, value=7, step=0.5, label="Prompt strength (SD 1.5 models only)")
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                        with gr.Row():
                            refresh = gr.Button("Refresh model list", size="sm")
                            reuse = gr.Button("Reuse last seed", size="sm")
                with gr.Column(scale=6):
                    gallery = gr.Gallery(value=recent_images(), label="Results (click one to pick it)", columns=2,
                                         height=600, object_fit="contain", elem_id="gallery",
                                         show_download_button=True)
                    status = gr.Markdown("Ready.")
                    with gr.Row():
                        to_edit = gr.Button("Edit this picture", size="sm", variant="secondary")
                        folder = gr.Button("Open outputs folder", size="sm")

        with gr.Tab("Edit", id="edit"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    editor = gr.ImageEditor(
                        label="Your picture: paint over the part to change", type="pil", height=600,
                        brush=gr.Brush(colors=["#ff3b7f"], color_mode="fixed", default_size=40),
                        eraser=gr.Eraser(default_size=40), transforms=(), layers=False,
                        sources=("upload", "clipboard"))
                with gr.Column(scale=4, min_width=340):
                    edit_prompt = gr.Textbox(label="What should it show?", lines=3,
                                             placeholder="a red leather jacket")
                    edit_mode = gr.Radio([EDIT_AREA, EDIT_WHOLE], value=EDIT_AREA, label="What to change")
                    edit_quality = gr.Radio([EDIT_BEST, EDIT_FAST], value=EDIT_BEST, label="Quality")
                    edit_strength = gr.Slider(0.2, 1.0, value=1.0, step=0.05, label="How much to change",
                                              info="Low keeps more of the original, high redraws more")
                    edit_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    edit_go = gr.Button("Apply edit", variant="primary", elem_id="go2")
                    edit_result = gr.Image(label="Result", type="filepath", height=420,
                                           show_download_button=True)
                    edit_status = gr.Markdown("Upload a picture or send one from the Create tab.")
                    keep = gr.Button("Keep editing this", size="sm")

    last_seed = gr.State(-1)
    picked = gr.State(None)
    inputs = [prompt, negative, style, size, model, steps, cfg, seed, count]
    go.click(generate, inputs, [gallery, status, last_seed])
    prompt.submit(generate, inputs, [gallery, status, last_seed])
    reuse.click(lambda s: s, last_seed, seed)
    refresh.click(lambda: gr.update(choices=model_choices()), None, model)
    folder.click(open_outputs)

    def pick(evt: gr.SelectData):
        v = evt.value
        return v["image"]["path"] if isinstance(v, dict) and "image" in v else v

    def send_to_edit(path):
        path = path or next(iter(recent_images()), None)
        if not path:
            raise gr.Error("Make a picture first.")
        return {"background": path, "layers": [], "composite": path}, gr.Tabs(selected="edit")

    gallery.select(pick, None, picked)
    to_edit.click(send_to_edit, picked, [editor, tabs])
    edit_mode.change(lambda m: 1.0 if m == EDIT_AREA else 0.5, edit_mode, edit_strength)
    edit_go.click(edit, [editor, edit_prompt, edit_mode, edit_quality, edit_strength, edit_seed],
                  [edit_result, edit_status])
    keep.click(lambda p: send_to_edit(p)[0] if p else gr.update(), edit_result, editor)

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="127.0.0.1", inbrowser=True,
                                  allowed_paths=[str(OUTPUTS_DIR)])
