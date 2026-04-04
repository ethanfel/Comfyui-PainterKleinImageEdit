# ComfyUI-PainterKleinImageEdit

An improved ComfyUI node for **FLUX.2 [klein]** (4B / 9B) image editing and text-to-image generation.

This is an updated version of the original PainterFluxImageEdit node, rewritten specifically for the FLUX.2 [klein] architecture with bug fixes and dynamic inputs.

---

## What's New vs Original

- **Bug fixes** — removed dead vision-token code that did nothing on FLUX.2 [klein]; fixed `noise_mask` sized to `width//8` (wrong for FLUX.2 [klein]'s 16× VAE)
- **Negative prompt** — dedicated input for negative conditioning
- **Dynamic inputs** — set `num_images` (1–10) and image slots appear/disappear automatically; no more manual mode dropdown
- **Correct inpainting canvas** — with `mask1` connected, the canvas is encoded `image1` so unmasked regions are preserved; without a mask the canvas is empty so the model generates the edit freely from the reference

---

## Features

- **Text-to-image** — leave all image slots unconnected
- **Single-image edit** — connect one reference image; the model edits it according to the prompt
- **Multi-image edit** — connect up to 10 reference images simultaneously
- **Inpainting** — connect a mask to `mask1` to target a specific region for regeneration

---

## Installation

Via ComfyUI Manager — search for `ComfyUI-PainterKleinImageEdit`.

Or clone manually into your `custom_nodes` directory:

```bash
git clone https://github.com/ethanfel/Comfyui-PainterKleinImageEdit.git
```

---

## Node Inputs

| Input | Type | Description |
|---|---|---|
| `clip` | CLIP | FLUX.2 [klein] CLIP (Qwen3 text encoder) |
| `prompt` | STRING | Positive text prompt |
| `negative_prompt` | STRING | Negative text prompt |
| `num_images` | INT 1–10 | Number of image slots to show |
| `batch_size` | INT 1–64 | Generation batch size |
| `width` | INT | Output canvas width |
| `height` | INT | Output canvas height |
| `vae` | VAE | FLUX.2 [klein] VAE |
| `image1`…`image10` | IMAGE | Reference images (slots shown based on `num_images`) |
| `mask1` | MASK | Optional inpainting mask for image1 |

## Node Outputs

| Output | Type | Description |
|---|---|---|
| `positive` | CONDITIONING | Positive conditioning with reference latents |
| `negative` | CONDITIONING | Negative conditioning with reference latents |
| `latent` | LATENT | Canvas latent (encoded image1 if inpainting, empty otherwise) |

---

## Usage

Connect the node outputs directly to a FLUX.2 [klein] sampler. It replaces the standard text encoder + VAE encode + reference latent node chain.

### Text-to-image
Connect `clip`, `vae`, `prompt` — leave all image slots empty.

### Single-image edit
Set `num_images` to 1, connect `image1`. Describe the desired edit in `prompt`.

### Inpainting
Connect `image1` + `mask1`. The masked region will be regenerated; the rest is preserved from the original image.

### Multi-image edit
Set `num_images` to the number of references, connect each image. The model handles positional encoding automatically.

---

## Notes

- Designed for **FLUX.2 [klein] 4B and 9B** only — not compatible with FLUX.1 or FLUX.1 Kontext
- Width and height should be multiples of 16
