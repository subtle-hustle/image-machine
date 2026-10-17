# Image Machine

A private, offline image generator for iPhone and iPad. Type a prompt and get an image made entirely on your device by Apple's Core ML Stable Diffusion, running on the Neural Engine.

The app goes online exactly once, to download the model. After that you can put the phone in airplane mode and it keeps working.

## Tuned for iPhone 16 Pro Max

- **SDXL at 768×768** is the default on 8 GB phones (iPhone 15 Pro, all iPhone 16s). It's Apple's 4-bit SDXL build made specifically for iPhone, with the network split in two so it fits the Neural Engine. Phones with less RAM default to Stable Diffusion 2.1 at 512×512.
- **Neural Engine only.** Both models are Apple's `split_einsum` builds, laid out for the A-series Neural Engine, and the app runs them on Neural Engine plus CPU so weights never move to the GPU.
- **Whole model stays in memory.** The app asks iOS for its increased memory limit, and when the room is there it keeps every part of the model loaded, so the second image onward starts immediately. If memory is tight it falls back to loading one part at a time.
- **20-step DPM-Solver++** (with Karras spacing for SDXL) gets good results in far fewer steps than the classic 50.
- The first launch after download compiles the model for your exact chip. iOS caches that, so it's slow once and fast after.

## What you need

- A Mac with **Xcode 16 or newer** (free from the Mac App Store)
- An iPhone or iPad on **iOS 17 or newer**. SDXL needs 8 GB of RAM (iPhone 15 Pro or any iPhone 16); SD 2.1 runs on iPhone 12 and later.
- About **6 GB free** on the phone during setup for SDXL (3 GB for SD 2.1); the zip is deleted after unpacking.
- A free Apple ID is enough. With a free account the app stops opening after 7 days and you just build it again; a paid developer account ($99/yr) makes that a year.

## Build it onto your phone

1. Copy the `ImageMachine` folder to your Mac and double-click `ImageMachine.xcodeproj`.
2. Wait for Xcode to fetch the two packages (ml-stable-diffusion and ZIPFoundation). You'll see "Resolving packages" at the top of the window.
3. In the left sidebar, click the blue **ImageMachine** project, then the **ImageMachine** target, then **Signing & Capabilities**. Under **Team**, pick your Apple ID (Xcode → Settings → Accounts lets you add it if it isn't there).
   - If Xcode says your team doesn't support the **Increased Memory Limit** capability (this can happen with a free Apple ID), click the trash icon next to that capability on the same screen. The app still works; it just loads the model in smaller pieces.
4. Plug in your phone with a cable, unlock it, and tap **Trust** if asked.
5. On the phone, turn on **Settings → Privacy & Security → Developer Mode** and restart when it asks (iOS shows this option only after step 4).
6. At the top of Xcode, choose your phone as the run destination, then press **⌘R** (or the ▶ button).
7. The first time, the phone may say the developer is untrusted. Go to **Settings → General → VPN & Device Management**, tap your Apple ID, and tap **Trust**. Then open the app.

## First run

1. Pick a model (SDXL is preselected on iPhone 16 Pro Max), then on Wi-Fi tap **Download**. Keep the app open until it finishes; SDXL is a few GB.
2. The app unpacks the model and loads it onto the Neural Engine. The very first load can take a few minutes; later launches are much faster.
3. That's it. From here on, no internet is needed.

**No Wi-Fi on the phone at all?** Download the model zip on any computer:

- SDXL: `https://huggingface.co/apple/coreml-stable-diffusion-xl-base-ios/resolve/main/coreml-stable-diffusion-xl-base-ios_split_einsum_compiled.zip`
- SD 2.1: `https://huggingface.co/apple/coreml-stable-diffusion-2-1-base-palettized/resolve/main/coreml-stable-diffusion-2-1-base-palettized_split_einsum_v2_compiled.zip`

AirDrop it to the phone (it lands in Files), then tap **Import model .zip from Files** in the app.

## Using it

- **Prompt**: describe what you want. Style words help ("watercolor", "35mm photo", "isometric").
- **Things to avoid**: a negative prompt.
- **Steps**: 20 is a good default. More is slower and slightly more detailed. SD 2.1 should take seconds on an iPhone 16 Pro Max and SDXL somewhat longer; these are estimates, not measured.
- **Prompt strength**: how closely it follows your words. 7–8 is typical.
- **Seed**: leave blank for a new image each time, or tap **Reuse** to keep a composition and tweak the prompt.
- **Save to Photos** or **Share** the result.

SD 2.1 includes Stable Diffusion's standard safety filter; if it flags an image, the app tells you instead of showing it. Apple's SDXL build ships without one.

To switch models, tap **Delete model** at the bottom and download the other one.

## Troubleshooting

- **App closes during loading or generating**: iOS ran out of memory. Close other apps and try again. Without 8 GB of RAM, use SD 2.1; on 4 GB phones even that may not fit.
- **"Untrusted Developer"**: see step 7 above.
- **App stopped opening after a week**: free Apple ID builds expire after 7 days. Plug in and press ⌘R again; the downloaded model is kept.
- **Package errors in Xcode**: File → Packages → Reset Package Caches, then build again.

## Don't want to build it yourself?

**Draw Things** (free on the App Store) does the same thing, fully offline, with many more models and features. Image Machine is the small, simple version you own and can change.

## How it's put together

- `ModelManager.swift`: downloads or imports the model once, stores it in Application Support (excluded from iCloud backup), and loads the SDXL or SD 2.1 pipeline on the Neural Engine, fully resident when memory allows.
- `Generator.swift`: runs a text-to-image job with the DPM-Solver++ scheduler and reports step progress.
- `ImageMachine.entitlements`: requests iOS's increased memory limit.
- `ContentView.swift`: setup screen and the generate screen.
