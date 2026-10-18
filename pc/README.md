# Image Machine for PC

An offline text-to-image generator for Windows with an NVIDIA graphics card.
Type a description, press Generate, and the picture is made on your own PC.
Nothing is uploaded anywhere.

## Use it

Double-click **Start Image Machine.bat**. A black window opens and your browser
shows the app. Keep the black window open while you use it; close it to quit.

The first launch downloads Python and PyTorch (about 3 GB), and the first picture
with each model downloads that model (6.3 GB for Z-Image, about 2 GB for the others). After that it all works with no internet.

Pictures are saved in the `outputs` folder, with the prompt and seed stored inside
each PNG.

## Tips

- **Style** adds a look (photo, cinematic, anime…) without you typing it.
- **Shape** picks square, portrait or landscape.
- **Write prompts as full sentences** for Z-Image, and put any words you want in
  the picture in quotes, like `a sign that says "FRESH BREAD DAILY"`.
- **More options → Seed**: the same seed and prompt give the same picture. Use
  "Reuse last seed" to tweak a prompt while keeping the composition.
- **Models**: Z-Image Turbo is the default. It follows long, detailed prompts and
  spells words on signs correctly, but takes about 1.5 to 2 minutes per picture on
  this PC. DreamShaper 8 (about 16 seconds) is good for quick drafts, and Realistic
  Vision for photos. Steps, prompt strength and "leave out" only affect those SD 1.5 models.
  To add any Stable Diffusion 1.5 model from Civitai, put its `.safetensors` file
  in `models\custom` and press "Refresh model list".

## Editing a picture

Click a picture on the Create tab, press **Edit this picture**, or upload one on the
**Edit** tab.

- **Change the painted area**: brush over the part to change, then describe what
  should be there ("a red leather jacket"). Everything you didn't paint stays exactly
  as it was.
- **Vary the whole picture**: redraws everything while keeping the layout. Lower
  "How much to change" keeps more of the original.
- **Fast** (about 20 to 30 seconds) is best for painted-area edits. **Best** uses
  Z-Image (about 2 minutes) and works better when you describe the whole picture.
- **Keep editing this** sends the result back in, so you can make several edits in a row.

## About this PC

Built for a GTX 1660 Ti (6 GB). These cards make black images in half precision,
so the app runs in full precision automatically. Z-Image runs 4-bit compressed so
it fits in 6 GB.

The model's built-in safety filter is turned off. The app still refuses prompts
that sexualize minors.
