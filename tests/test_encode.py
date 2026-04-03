import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call
import torch

# ---------------------------------------------------------------------------
# Mock all ComfyUI dependencies before importing the node
# ---------------------------------------------------------------------------
node_helpers_mock = types.ModuleType("node_helpers")

def _conditioning_set_values(conditioning, values={}, append=False):
    """Minimal real implementation for test assertions."""
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k, val in values.items():
            if append:
                old = n[1].get(k)
                if old is not None:
                    val = old + val
            n[1][k] = val
        c.append(n)
    return c

node_helpers_mock.conditioning_set_values = _conditioning_set_values
sys.modules["node_helpers"] = node_helpers_mock

comfy_mock = types.ModuleType("comfy")
comfy_utils_mock = types.ModuleType("comfy.utils")
comfy_mm_mock = types.ModuleType("comfy.model_management")

comfy_utils_mock.common_upscale = MagicMock(side_effect=lambda t, w, h, *a, **k: t)
comfy_mm_mock.get_torch_device = MagicMock(return_value=torch.device("cpu"))
comfy_mock.utils = comfy_utils_mock
comfy_mock.model_management = comfy_mm_mock

sys.modules["comfy"] = comfy_mock
sys.modules["comfy.utils"] = comfy_utils_mock
sys.modules["comfy.model_management"] = comfy_mm_mock

from PainterFluxImageEdit import PainterFluxImageEdit  # noqa: E402


def _make_clip(prompt_result=None):
    """Return a mock CLIP that produces deterministic conditioning."""
    clip = MagicMock()
    cond = prompt_result or [[torch.zeros(1, 77, 768), {}]]
    clip.tokenize.return_value = {"tokens": []}
    clip.encode_from_tokens_scheduled.return_value = cond
    return clip


def _make_vae(latent_shape=(1, 128, 64, 64)):
    """Return a mock VAE whose encode() returns a tensor of the given shape."""
    vae = MagicMock()
    vae.encode.return_value = torch.zeros(*latent_shape)
    return vae


def _make_image(h=512, w=512, b=1, c=3):
    """Return a fake ComfyUI IMAGE tensor [B, H, W, C] in [0, 1]."""
    return torch.rand(b, h, w, c)


def _make_mask(h=512, w=512, b=1):
    """Return a fake ComfyUI MASK tensor [B, H, W] in [0, 1]."""
    return torch.rand(b, h, w)


class TestInputTypes(unittest.TestCase):
    def test_num_images_replaces_mode(self):
        inputs = PainterFluxImageEdit.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("num_images", required)
        self.assertNotIn("mode", required)

    def test_negative_prompt_required(self):
        inputs = PainterFluxImageEdit.INPUT_TYPES()
        self.assertIn("negative_prompt", inputs["required"])

    def test_reference_latents_method_optional(self):
        inputs = PainterFluxImageEdit.INPUT_TYPES()
        self.assertIn("reference_latents_method", inputs["optional"])

    def test_no_image_slots_in_input_types(self):
        """Image slots are added dynamically by JS, not defined in INPUT_TYPES."""
        inputs = PainterFluxImageEdit.INPUT_TYPES()
        all_keys = list(inputs.get("required", {})) + list(inputs.get("optional", {}))
        image_keys = [k for k in all_keys if k.startswith("image") or k.startswith("mask")]
        self.assertEqual(image_keys, [], "image/mask slots must not be in INPUT_TYPES")


class TestEncodeNoImages(unittest.TestCase):
    def setUp(self):
        self.node = PainterFluxImageEdit()
        self.clip = _make_clip()
        self.vae = _make_vae()

    def test_returns_three_outputs(self):
        pos, neg, latent = self.node.encode(
            self.clip, "prompt", "", 1, 1, 512, 512, vae=self.vae
        )
        self.assertIsInstance(pos, list)
        self.assertIsInstance(neg, list)
        self.assertIn("samples", latent)

    def test_no_images_produces_no_reference_latents(self):
        pos, neg, latent = self.node.encode(
            self.clip, "prompt", "", 1, 1, 512, 512, vae=self.vae
        )
        for _, cond_dict in pos:
            self.assertNotIn("reference_latents", cond_dict)
        for _, cond_dict in neg:
            self.assertNotIn("reference_latents", cond_dict)

    def test_empty_canvas_latent(self):
        self.node.encode(self.clip, "prompt", "", 1, 1, 512, 512, vae=self.vae)
        calls = self.vae.encode.call_args_list
        zero_calls = [c for c in calls if c.args and torch.all(c.args[0] == 0)]
        self.assertGreater(len(zero_calls), 0, "Empty canvas latent must be encoded")

    def test_no_vision_tokens_in_prompt(self):
        self.node.encode(self.clip, "hello", "", 1, 1, 512, 512, vae=self.vae)
        tokenize_args = self.clip.tokenize.call_args_list
        for c in tokenize_args:
            self.assertNotIn("images", c.kwargs, "images= must not be passed to tokenize")
            prompt_arg = c.args[0] if c.args else ""
            self.assertNotIn("<|vision_start|>", prompt_arg)


class TestEncodeWithImages(unittest.TestCase):
    def setUp(self):
        self.node = PainterFluxImageEdit()
        self.clip = _make_clip([[torch.zeros(1, 77, 768), {}]])
        self.vae = _make_vae()

    def test_single_image_sets_reference_latents(self):
        img = _make_image()
        pos, neg, _ = self.node.encode(
            self.clip, "prompt", "", 1, 1, 512, 512,
            vae=self.vae, image1=img
        )
        pos_dict = pos[0][1]
        neg_dict = neg[0][1]
        self.assertIn("reference_latents", pos_dict)
        self.assertIn("reference_latents", neg_dict)
        self.assertEqual(len(pos_dict["reference_latents"]), 1)

    def test_output_latent_is_empty_not_ref(self):
        """latent['samples'] must be zeros, not the reference image latent."""
        self.vae.encode.side_effect = [
            torch.ones(1, 128, 32, 32),   # ref image encoding → ones
            torch.zeros(1, 128, 32, 32),  # empty canvas encoding → zeros
        ]
        img = _make_image()
        _, _, latent = self.node.encode(
            self.clip, "prompt", "", 1, 1, 256, 256,
            vae=self.vae, image1=img
        )
        self.assertTrue(torch.all(latent["samples"] == 0),
                        "Output latent must be empty canvas, not ref_latents[0]")

    def test_mask_applied_to_image_before_encoding(self):
        """When a mask is connected, the masked region must be zeroed in the image."""
        img = torch.ones(1, 64, 64, 3)
        mask = torch.ones(1, 64, 64)  # fully masked → image should become zeros

        captured = []
        def capture_encode(x):
            captured.append(x.clone())
            return torch.zeros(1, 128, 4, 4)
        self.vae.encode.side_effect = capture_encode

        self.node.encode(
            self.clip, "p", "", 1, 1, 64, 64,
            vae=self.vae, image1=img, mask1=mask
        )
        ref_input = captured[0]
        self.assertAlmostEqual(ref_input.max().item(), 0.0, places=4,
                               msg="Masked pixels in reference image must be zeroed")

    def test_multi_image_sets_reference_latents_method(self):
        img1, img2 = _make_image(), _make_image()
        pos, neg, _ = self.node.encode(
            self.clip, "p", "", 2, 1, 512, 512,
            vae=self.vae, image1=img1, image2=img2,
            reference_latents_method="index"
        )
        pos_dict = pos[0][1]
        neg_dict = neg[0][1]
        self.assertIn("reference_latents_method", pos_dict)
        self.assertIn("reference_latents_method", neg_dict)
        self.assertEqual(pos_dict["reference_latents_method"], "index")

    def test_single_image_no_reference_latents_method(self):
        img = _make_image()
        pos, neg, _ = self.node.encode(
            self.clip, "p", "", 1, 1, 512, 512,
            vae=self.vae, image1=img
        )
        self.assertNotIn("reference_latents_method", pos[0][1])

    def test_mask1_becomes_noise_mask(self):
        """noise_mask must exist and use the canvas latent spatial dims, not the ref image dims."""
        img = _make_image(32, 32)
        mask = _make_mask(32, 32)

        canvas_latent = torch.zeros(1, 128, 8, 8)
        ref_latent = torch.zeros(1, 128, 2, 2)
        self.vae.encode.side_effect = [ref_latent, canvas_latent]

        import comfy.utils as cu
        original_side_effect = cu.common_upscale.side_effect
        cu.common_upscale.side_effect = lambda t, w, h, *a, **k: torch.zeros(
            t.shape[0], t.shape[1], h, w
        )
        self.addCleanup(setattr, cu, 'common_upscale',
                        MagicMock(side_effect=lambda t, w, h, *a, **k: t))

        _, _, latent = self.node.encode(
            self.clip, "p", "", 1, 1, 128, 128,
            vae=self.vae, image1=img, mask1=mask
        )

        self.assertIn("noise_mask", latent)
        nm = latent["noise_mask"]
        self.assertEqual(nm.shape[-2], 8, "noise_mask height must match canvas latent height")
        self.assertEqual(nm.shape[-1], 8, "noise_mask width must match canvas latent width")

    def test_negative_prompt_encoded_separately(self):
        img = _make_image()
        self.node.encode(
            self.clip, "positive text", "negative text", 1, 1, 512, 512,
            vae=self.vae, image1=img
        )
        tokenize_calls = [c.args[0] for c in self.clip.tokenize.call_args_list]
        self.assertIn("positive text", tokenize_calls)
        self.assertIn("negative text", tokenize_calls)

    def test_reference_latents_uses_append(self):
        """reference_latents must accumulate across images, not replace."""
        img1, img2 = _make_image(), _make_image()
        pos, neg, _ = self.node.encode(
            self.clip, "p", "", 2, 1, 512, 512,
            vae=self.vae, image1=img1, image2=img2,
        )
        # Both reference latents must be present — append=True, not replace
        self.assertEqual(len(pos[0][1]["reference_latents"]), 2,
                         "Both reference latents must be in conditioning (append=True)")
        self.assertEqual(len(neg[0][1]["reference_latents"]), 2)

    def test_vae_required(self):
        with self.assertRaises(RuntimeError):
            self.node.encode(self.clip, "p", "", 1, 1, 512, 512, vae=None)


if __name__ == "__main__":
    unittest.main()
