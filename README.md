# ComfyUI-PainterKleinImageEdit

An improved ComfyUI node for **FLUX.2 [klein]** (4B / 9B) image editing and text-to-image generation.

This is an updated version of the original PainterFluxImageEdit node, rewritten specifically for the FLUX.2 [klein] architecture with bug fixes, new features, and dynamic inputs.

---

## What's New

- **Bug fixes** — removed dead vision-token code inherited from QwenImage that did nothing on FLUX.2 [klein]; fixed a latent output bug that caused the reference image to replace the generation canvas
- **Negative prompt** — dedicated input for negative conditioning
- **Dynamic inputs** — set `num_images` (1–10) and image/mask slots appear and disappear automatically; no more manual mode dropdown
- **Per-image masks** — every image slot has its own mask input; mask1 drives inpainting on the canvas, masks 2–N zero out regions of their reference image before encoding
- **Reference latents method** — expose `offset`, `index`, `uxo/uno`, `index_timestep_zero` for multi-image workflows (required for the 9B KV model)

---

## Features

- **Text-to-image** — leave all image slots unconnected
- **Single-image edit** — connect one reference image; the model edits it according to the prompt
- **Multi-image edit** — connect up to 10 reference images simultaneously
- **Inpainting** — connect a mask to `mask1` to target a specific region for regeneration
- **Reference image masking** — connect masks to `mask2`–`mask10` to focus the model on specific regions of each reference

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
| `num_images` | INT 1–10 | Number of image/mask slot pairs to show |
| `batch_size` | INT 1–64 | Generation batch size |
| `width` | INT | Output canvas width |
| `height` | INT | Output canvas height |
| `vae` | VAE | FLUX.2 [klein] VAE |
| `image1`…`image10` | IMAGE | Reference images (slots shown based on `num_images`) |
| `mask1` | MASK | Optional inpainting mask for image1 (noise_mask on canvas) |

## Node Outputs

| Output | Type | Description |
|---|---|---|
| `positive` | CONDITIONING | Positive conditioning with reference latents |
| `negative` | CONDITIONING | Negative conditioning with reference latents |
| `latent` | LATENT | Empty canvas latent (+ noise_mask if mask1 connected) |

---

## Usage

Connect the node outputs directly to a FLUX.2 [klein] sampler. It replaces the standard text encoder + VAE encode + reference latent node chain.

### Text-to-image
Connect `clip`, `vae`, `prompt` — leave all image slots empty.

### Single-image edit
Set `num_images` to 1, connect `image1`. Describe the desired edit in `prompt`.

### Inpainting
Connect `image1` + `mask1`. The masked region will be regenerated; the rest follows the reference.

### Multi-image edit
Set `num_images` to the number of references, connect each image. The model handles positional encoding automatically.

---

## Notes

- Designed for **FLUX.2 [klein] 4B and 9B** only — not compatible with FLUX.1 or FLUX.1 Kontext
- Image resolution is up to you — no forced rescaling
- Width and height should be multiples of 16
