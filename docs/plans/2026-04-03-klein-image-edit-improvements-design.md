# PainterFluxImageEdit — FLUX.2 [klein] Improvements Design

**Date:** 2026-04-03  
**Target model:** FLUX.2 [klein] 4B / 9B  
**Scope:** Bug fixes, UX improvements, dynamic inputs, full feature parity

---

## Context

Research revealed that the existing node contains dead code inherited from the QwenImage model family (vision token injection, 384×384 VL image resizing) that silently does nothing when connected to a FLUX.2 [klein] CLIP. It also has a latent output bug and missing features. This document specifies the corrected, improved design.

---

## Key Research Findings

- FLUX.2 [klein] uses `KleinTokenizer` which **ignores** the `images=` parameter — vision tokens do nothing
- Image editing signal flows entirely through `reference_latents` in the conditioning dict
- The output canvas must be an **empty latent** (zeros); reference latents must not replace it
- Multi-image workflows require a `reference_latents_method` key in conditioning
- FLUX.2 [klein] uses a 128-channel latent space with 16× spatial downsampling

---

## Architecture

Two files:

| File | Role |
|---|---|
| `PainterFluxImageEdit.py` | Rewritten Python node |
| `web/js/PainterFluxImageEdit.js` | New JS extension for dynamic input visibility |

`__init__.py` already sets `WEB_DIRECTORY = "./web/js"` — no change needed there.

---

## Inputs

### Required
| Name | Type | Details |
|---|---|---|
| `clip` | CLIP | |
| `prompt` | STRING multiline | positive text prompt |
| `negative_prompt` | STRING multiline | default `""` |
| `num_images` | INT 1–10 | controls JS visibility of image/mask slots |
| `batch_size` | INT 1–64 | |
| `width` | INT 512–4096 step 8 | output canvas width |
| `height` | INT 512–4096 step 8 | output canvas height |

### Optional
| Name | Type | Details |
|---|---|---|
| `vae` | VAE | required at runtime, optional in graph |
| `reference_latents_method` | dropdown | `"offset"`, `"index"`, `"uxo/uno"`, `"index_timestep_zero"` — only applied when num_images > 1 |
| `image1`…`image10` | IMAGE | shown/hidden by JS based on num_images |
| `mask1`…`mask10` | MASK | shown/hidden alongside each image slot |

The `mode` dropdown from the original node is **removed** — replaced by `num_images`.

---

## Outputs

| Name | Type |
|---|---|
| `positive` | CONDITIONING |
| `negative` | CONDITIONING |
| `latent` | LATENT |

---

## Data Flow

### Encoding loop (per image slot 1..num_images)

1. Skip if image not connected
2. If mask present: zero out masked regions of the image before encoding (`image * (1 - mask)`)
3. VAE-encode image → `ref_latent`
4. Append to `ref_latents` list

### Conditioning

```
positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
positive = conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)

negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
negative = conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

if num_images > 1:
    positive = conditioning_set_values(positive, {"reference_latents_method": method})
    negative = conditioning_set_values(negative, {"reference_latents_method": method})
```

No `images=` argument passed to `clip.tokenize()`. No vision token prefix in prompt.

### Output latent

```
empty_pixels = torch.zeros(1, height, width, 3)
latent = {"samples": vae.encode(empty_pixels)}
```

Reference latents are **never** assigned to `latent["samples"]`.

### Mask on image1 → noise_mask

`mask1`, if connected, is downsampled to latent space and stored as `latent["noise_mask"]` for inpainting — same behavior as original.

### Masks on images 2–10

Applied directly to the reference image before VAE encoding (zero out regions the model should ignore from that reference). Not stored as noise_mask.

### Batch size

Expand conditioning list and repeat latent samples/mask along batch dim.

---

## JS Dynamic Inputs

File: `web/js/PainterFluxImageEdit.js`

- Registers a ComfyUI extension via `app.registerExtension`
- On node creation and `num_images` widget change: add or remove `imageN` / `maskN` input slots to match the count
- On workflow load: restore slot count from serialized widget value
- Slot naming: `image1`/`mask1` … `image10`/`mask10` to match Python parameter names

---

## Bug Fixes Summary

| Bug | Fix |
|---|---|
| Dead vision token code (384×384 resize, `vl_images`, `image_prompt_prefix`) | Removed entirely |
| `latent["samples"] = ref_latents[0]` overrides empty canvas | Removed; empty latent always used |
| Mask only on image1 | Masks on all image slots |
| No `reference_latents_method` for multi-image | Added as optional input |
| No negative prompt control | Added `negative_prompt` input |
| Redundant `mode` dropdown | Replaced by `num_images` widget |
