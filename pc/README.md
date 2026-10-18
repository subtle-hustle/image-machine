# Image Machine for PC

An offline text-to-image generator for Windows with an NVIDIA graphics card.
Type a description, press Generate, and the picture is made on your own PC.
Nothing is uploaded anywhere.

## Use it

Double-click **Start Image Machine.bat**. A black window opens and your browser
shows the app. Keep the black window open while you use it; close it to quit.

The first launch downloads Python and PyTorch (about 3 GB), and the first picture
downloads the model (about 2 GB). After that it all works with no internet.

Pictures are saved in the `outputs` folder, with the prompt and seed stored inside
each PNG.

## Tips

- **Style** adds a look (photo, cinematic, anime…) without you typing it.
- **Shape** picks square, portrait or landscape.
- **More options → Seed**: the same seed and prompt give the same picture. Use
  "Reuse last seed" to tweak a prompt while keeping the composition.
- **Models**: DreamShaper 8 is the default. Realistic Vision is better for photos.
  To add any Stable Diffusion 1.5 model from Civitai, put its `.safetensors` file
  in `models\custom` and press "Refresh model list".

## About this PC

Built for a GTX 1660 Ti (6 GB). These cards make black images in half precision,
so the app runs in full precision automatically. Expect roughly 20 to 40 seconds
per 512×512 picture at 25 steps.

The model's built-in safety filter is turned off. The app still refuses prompts
that sexualize minors.
