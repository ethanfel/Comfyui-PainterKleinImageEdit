import node_helpers
import comfy.utils
import torch
import comfy.model_management


class PainterKleinImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
            },
            "optional": {
                "vae": ("VAE",),
                "reference_latents_method": (["offset", "index", "uxo/uno", "index_timestep_zero"], {
                    "tooltip": (
                        "Positional encoding applied to multi-image reference latents (only used when ≥2 images are connected).\n"
                        "• offset — recommended for most cases; adds a position offset per image.\n"
                        "• index — uses the raw image index as position.\n"
                        "• uxo/uno — alternates positive/negative offsets between images.\n"
                        "• index_timestep_zero — index encoding zeroed at the first timestep; recommended for the 9B KV model."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "FLUX.2 [klein] image editing — dynamic image/mask inputs via num_images widget"

    def encode(self, clip, prompt, negative_prompt, num_images, batch_size, width, height,
               vae=None, reference_latents_method="offset", **kwargs):

        if vae is None:
            raise RuntimeError("VAE is required. Please connect a VAE loader.")

        ref_latents = []
        pending_mask1 = None

        for i in range(1, num_images + 1):
            image = kwargs.get(f"image{i}")
            mask = kwargs.get(f"mask{i}")

            if image is None:
                continue

            img = image[:, :, :, :3]  # ensure RGB only

            if mask is not None:
                # Resize mask to image spatial dims, then zero out masked regions
                img_h, img_w = img.shape[1], img.shape[2]
                if mask.dim() == 2:
                    m = mask.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
                elif mask.dim() == 3:
                    m = mask.unsqueeze(1)                 # [B, 1, H, W]
                else:
                    m = mask

                m = m.float()
                if m.shape[2] != img_h or m.shape[3] != img_w:
                    m = comfy.utils.common_upscale(m, img_w, img_h, "area", "disabled")

                m = m.squeeze(1).unsqueeze(-1)  # [B, H, W, 1]
                img = img * (1.0 - m)

            ref_latent = vae.encode(img)
            ref_latents.append(ref_latent)

            # Store mask1 to compute noise_mask after canvas encoding (canvas dims needed)
            if i == 1 and mask is not None:
                pending_mask1 = mask

        # Encode prompts — FLUX.2 [klein] KleinTokenizer ignores images=, so we never pass it
        positive_tokens = clip.tokenize(prompt)
        positive = clip.encode_from_tokens_scheduled(positive_tokens)

        negative_tokens = clip.tokenize(negative_prompt)
        negative = clip.encode_from_tokens_scheduled(negative_tokens)

        if ref_latents:
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": ref_latents}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_latents": ref_latents}, append=True
            )

            if len(ref_latents) > 1:
                positive = node_helpers.conditioning_set_values(
                    positive, {"reference_latents_method": reference_latents_method}
                )
                negative = node_helpers.conditioning_set_values(
                    negative, {"reference_latents_method": reference_latents_method}
                )

        # Empty canvas — reference latents flow through conditioning only, never replace the canvas
        device = comfy.model_management.get_torch_device()
        empty_pixels = torch.zeros(1, height, width, 3, device=device)
        latent = {"samples": vae.encode(empty_pixels)}

        # Compute noise_mask now, using canvas latent spatial dims (not reference image dims)
        if pending_mask1 is not None:
            latent_h = latent["samples"].shape[2]
            latent_w = latent["samples"].shape[3]
            if pending_mask1.dim() == 2:
                ms = pending_mask1.unsqueeze(0).unsqueeze(0)
            else:
                ms = pending_mask1.unsqueeze(1)
            ms = comfy.utils.common_upscale(ms.float(), latent_w, latent_h, "area", "disabled")
            latent["noise_mask"] = ms.squeeze(1)

        if batch_size > 1:
            positive = positive * batch_size
            negative = negative * batch_size

            samples = latent["samples"]
            if samples.shape[0] != batch_size:
                latent["samples"] = samples.repeat(batch_size, *([1] * (samples.dim() - 1)))

            if "noise_mask" in latent and latent["noise_mask"].shape[0] == 1:
                latent["noise_mask"] = latent["noise_mask"].repeat(batch_size, 1, 1)

        return (positive, negative, latent)


NODE_CLASS_MAPPINGS = {
    "PainterKleinImageEdit": PainterKleinImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterKleinImageEdit": "Painter Klein Image Edit",
}
