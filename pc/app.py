"""Image Machine for PC: an offline text-to-image generator.

Runs Stable Diffusion 1.5-family models locally with diffusers and a Gradio UI.
After the first model download, everything works without an internet connection.
"""

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
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

BUILTIN_MODELS = {
    "DreamShaper 8 (all-rounder)": "Lykon/dreamshaper-8",
    "Realistic Vision 5.1 (photos)": "SG161222/Realistic_Vision_V5.1_noVAE",
    "Stable Diffusion 1.5 (original)": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}
DEFAULT_MODEL = "DreamShaper 8 (all-rounder)"

SIZES = {
    "Square 512×512": (512, 512),
    "Portrait 512×768": (512, 768),
    "Landscape 768×512": (768, 512),
    "Tall 448×832": (448, 832),
    "Wide 832×448": (832, 448),
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
                 "bad anatomy, extra fingers, extra limbs, watermark, text, signature")

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
        if name == "model_index.json" or (top in ("scheduler", "tokenizer", "feature_extractor")):
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
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    kwargs = dict(torch_dtype=DTYPE, safety_checker=None, feature_extractor=None,
                  requires_safety_checker=False)
    if choice.startswith("Custom: "):
        path = CUSTOM_DIR / choice[len("Custom: "):]
        # Single-file checkpoints borrow their layout from DreamShaper, so they also load offline.
        folder, _ = fetch_model(BUILTIN_MODELS[DEFAULT_MODEL], progress)
        progress(0, desc="Loading model…")
        pipe = StableDiffusionPipeline.from_single_file(str(path), config=str(folder), **kwargs)
    else:
        folder, variant = fetch_model(BUILTIN_MODELS[choice], progress)
        progress(0, desc="Loading model…")
        pipe = StableDiffusionPipeline.from_pretrained(str(folder), variant=variant, **kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    pipe.set_progress_bar_config(disable=True)
    pipe.to(DEVICE)
    pipe.vae.enable_tiling()
    _pipe, _pipe_key = pipe, choice
    return pipe


def generate(prompt, negative, style, size, model, steps, cfg, seed, count, progress=gr.Progress()):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Type a description of the image first.")
    if blocked(prompt):
        raise gr.Error("That prompt can't be generated.")

    progress(0, desc="Loading model…")
    pipe = load_pipe(model, progress)
    style_pos, style_neg = STYLES.get(style, ("", ""))
    full_prompt = f"{prompt}, {style_pos}" if style_pos else prompt
    full_negative = ", ".join(x for x in (negative.strip(), style_neg, BASE_NEGATIVE) if x)
    width, height = SIZES[size]
    steps, count = int(steps), int(count)
    base_seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2**31 - 1)

    images, started = [], time.time()
    for i in range(count):
        s = base_seed + i

        def on_step(_pipe, step, _t, kw, i=i):
            progress((i * steps + step + 1) / (count * steps),
                     desc=f"Image {i + 1} of {count}, step {step + 1}/{steps}")
            return kw

        img = pipe(full_prompt, negative_prompt=full_negative, width=width, height=height,
                   num_inference_steps=steps, guidance_scale=float(cfg),
                   generator=torch.Generator(DEVICE).manual_seed(s),
                   callback_on_step_end=on_step).images[0]
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
#go {min-height: 52px; font-size: 1.05rem; font-weight: 600;}
#gallery {min-height: 560px;}
footer {display: none !important;}
"""

theme = gr.themes.Soft(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate",
                       font=["Segoe UI", "Inter", "system-ui", "sans-serif"])

FORCE_DARK = """() => { if (!document.body.classList.contains('dark')) document.body.classList.add('dark'); }"""

with gr.Blocks(title="Image Machine", theme=theme, css=CSS, js=FORCE_DARK) as demo:
    gr.Markdown("# Image Machine", elem_id="title")
    gr.Markdown(f"Offline image generator · running on {DEVICE_LABEL}", elem_id="subtitle")
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=340):
            prompt = gr.Textbox(label="Describe your image", lines=4, elem_id="prompt",
                                placeholder="A lighthouse on a cliff at sunset, crashing waves, golden light")
            with gr.Row():
                style = gr.Dropdown(list(STYLES), value="None", label="Style")
                size = gr.Dropdown(list(SIZES), value="Square 512×512", label="Shape")
            count = gr.Slider(1, 4, value=1, step=1, label="How many images")
            go = gr.Button("Generate", variant="primary", elem_id="go")
            with gr.Accordion("More options", open=False):
                negative = gr.Textbox(label="Leave out (negative prompt)", lines=2,
                                      placeholder="things you don't want in the picture")
                model = gr.Dropdown(model_choices(), value=DEFAULT_MODEL, label="Model",
                                    info="Drop .safetensors files into models\\custom to add your own")
                steps = gr.Slider(10, 50, value=25, step=1, label="Steps (more = slower, finer)")
                cfg = gr.Slider(1, 15, value=7, step=0.5, label="Prompt strength")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                with gr.Row():
                    refresh = gr.Button("Refresh model list", size="sm")
                    reuse = gr.Button("Reuse last seed", size="sm")
        with gr.Column(scale=6):
            gallery = gr.Gallery(value=recent_images(), label="Results", columns=2, height=600,
                                 object_fit="contain", elem_id="gallery", show_download_button=True)
            status = gr.Markdown("Ready.")
            folder = gr.Button("Open outputs folder", size="sm")

    last_seed = gr.State(-1)
    inputs = [prompt, negative, style, size, model, steps, cfg, seed, count]
    go.click(generate, inputs, [gallery, status, last_seed])
    prompt.submit(generate, inputs, [gallery, status, last_seed])
    reuse.click(lambda s: s, last_seed, seed)
    refresh.click(lambda: gr.update(choices=model_choices()), None, model)
    folder.click(open_outputs)

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="127.0.0.1", inbrowser=True,
                                  allowed_paths=[str(OUTPUTS_DIR)])
