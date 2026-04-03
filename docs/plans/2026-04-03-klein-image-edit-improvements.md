# PainterFluxImageEdit FLUX.2 Klein Improvements — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all correctness bugs and add negative prompt, per-image masks, reference_latents_method, and JS-driven dynamic image/mask slots to replace the broken mode dropdown.

**Architecture:** Python node uses `**kwargs` to receive dynamically-named image/mask inputs (`image1`, `mask1`, …). A ComfyUI JS extension listens to the `num_images` widget and adds/removes those input slots in real time. The dead vision-token code is removed entirely; reference latents flow through conditioning only; the output canvas is always an empty zeros latent.

**Tech Stack:** Python 3, PyTorch, ComfyUI node API (`node_helpers`, `comfy.utils`, `comfy.model_management`), ComfyUI JS extension API (`app.registerExtension`, LiteGraph slot API), pytest + `unittest.mock` for unit tests.

---

## Reference files

- Design doc: `docs/plans/2026-04-03-klein-image-edit-improvements-design.md`
- Node under test: `PainterFluxImageEdit.py`
- ComfyUI node_helpers: `/media/p5/Comfyui/node_helpers.py`
- Canonical reference latent node: `/media/p5/Comfyui/comfy_extras/nodes_edit_model.py`
- Canonical Qwen image edit node (multi-image pattern): `/media/p5/Comfyui/comfy_extras/nodes_qwen.py`

---

## Task 1: Set up test scaffold

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_encode.py`

**Step 1: Create the test directory and empty init**

```bash
mkdir -p tests
touch tests/__init__.py
```

**Step 2: Write the test scaffold with mocks**

Create `tests/test_encode.py`:

```python
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
        # conditioning dict must not have reference_latents when no images supplied
        for _, cond_dict in pos:
            self.assertNotIn("reference_latents", cond_dict)
        for _, cond_dict in neg:
            self.assertNotIn("reference_latents", cond_dict)

    def test_empty_canvas_latent(self):
        self.node.encode(self.clip, "prompt", "", 1, 1, 512, 512, vae=self.vae)
        # VAE must be called to produce the empty canvas (zeros input)
        calls = self.vae.encode.call_args_list
        # At least one call with an all-zeros tensor
        zero_calls = [c for c in calls if c.args and torch.all(c.args[0] == 0)]
        self.assertGreater(len(zero_calls), 0, "Empty canvas latent must be encoded")

    def test_no_vision_tokens_in_prompt(self):
        self.node.encode(self.clip, "hello", "", 1, 1, 512, 512, vae=self.vae)
        tokenize_args = self.clip.tokenize.call_args_list
        for c in tokenize_args:
            # images= kwarg must never be passed
            self.assertNotIn("images", c.kwargs, "images= must not be passed to tokenize")
            # prompt must not contain vision start token
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
        # First VAE call = reference image; pixels where mask=1 must be ~0
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
        # Not set for single image
        self.assertNotIn("reference_latents_method", pos[0][1])

    def test_mask1_becomes_noise_mask(self):
        img = _make_image(64, 64)
        mask = _make_mask(64, 64)
        _, _, latent = self.node.encode(
            self.clip, "p", "", 1, 1, 64, 64,
            vae=self.vae, image1=img, mask1=mask
        )
        self.assertIn("noise_mask", latent)

    def test_negative_prompt_encoded_separately(self):
        img = _make_image()
        self.node.encode(
            self.clip, "positive text", "negative text", 1, 1, 512, 512,
            vae=self.vae, image1=img
        )
        tokenize_calls = [c.args[0] for c in self.clip.tokenize.call_args_list]
        self.assertIn("positive text", tokenize_calls)
        self.assertIn("negative text", tokenize_calls)

    def test_vae_required(self):
        with self.assertRaises(RuntimeError):
            self.node.encode(self.clip, "p", "", 1, 1, 512, 512, vae=None)


if __name__ == "__main__":
    unittest.main()
```

**Step 3: Run tests to confirm they all fail (node not yet rewritten)**

```bash
cd /media/p5/Comfyui-PainterFluxImageEdit
python -m pytest tests/test_encode.py -v 2>&1 | head -60
```

Expected: multiple failures — `test_no_image_slots_in_input_types`, `test_num_images_replaces_mode`, `test_no_vision_tokens_in_prompt`, `test_output_latent_is_empty_not_ref`, etc.

**Step 4: Commit the failing tests**

```bash
git add tests/
git commit -m "test: add failing tests for Klein node improvements"
```

---

## Task 2: Rewrite PainterFluxImageEdit.py

**Files:**
- Modify: `PainterFluxImageEdit.py` (full rewrite)

**Step 1: Replace the file contents**

```python
import node_helpers
import comfy.utils
import torch
import comfy.model_management


class PainterFluxImageEdit:
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
                "reference_latents_method": (["offset", "index", "uxo/uno", "index_timestep_zero"],),
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
        noise_mask = None

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

                if m.shape[2] != img_h or m.shape[3] != img_w:
                    m = comfy.utils.common_upscale(m.float(), img_w, img_h, "area", "center")

                m = m.squeeze(1).unsqueeze(-1)  # [B, H, W, 1]
                img = img * (1.0 - m)

            ref_latent = vae.encode(img)
            ref_latents.append(ref_latent)

            # mask on slot 1 → noise_mask for the output canvas (inpainting)
            if i == 1 and mask is not None:
                raw_mask = kwargs.get("mask1")
                if raw_mask.dim() == 2:
                    ms = raw_mask.unsqueeze(0).unsqueeze(0)
                elif raw_mask.dim() == 3:
                    ms = raw_mask.unsqueeze(1)
                else:
                    ms = None

                if ms is not None:
                    latent_h = ref_latent.shape[2]
                    latent_w = ref_latent.shape[3]
                    ms = comfy.utils.common_upscale(ms.float(), latent_w, latent_h, "area", "center")
                    noise_mask = ms.squeeze(1)

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

            if num_images > 1:
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

        if noise_mask is not None:
            latent["noise_mask"] = noise_mask

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
    "PainterFluxImageEdit": PainterFluxImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterFluxImageEdit": "Painter Flux Image Edit",
}
```

**Step 2: Run tests — expect all to pass**

```bash
python -m pytest tests/test_encode.py -v
```

Expected output: all tests `PASSED`. If any fail, debug before continuing.

**Step 3: Commit**

```bash
git add PainterFluxImageEdit.py
git commit -m "feat: rewrite node for FLUX.2 klein — fix latent bug, remove dead vision token code, add negative prompt, per-image masks, reference_latents_method"
```

---

## Task 3: Create JS dynamic input extension

**Files:**
- Create: `web/js/PainterFluxImageEdit.js`

**Step 1: Create the directory**

```bash
mkdir -p web/js
```

**Step 2: Write the JS extension**

Create `web/js/PainterFluxImageEdit.js`:

```javascript
import { app } from "../../scripts/app.js";

/**
 * Syncs the imageN / maskN input slots on the node to match `count`.
 * Adds missing slots in order (image1, mask1, image2, mask2, …) and removes
 * slots beyond `count`. Preserves existing connections where possible.
 */
function syncSlots(node, count) {
    count = Math.max(1, Math.min(10, parseInt(count) || 1));

    const wanted = new Set();
    for (let i = 1; i <= count; i++) {
        wanted.add(`image${i}`);
        wanted.add(`mask${i}`);
    }

    // Remove unwanted dynamic inputs (iterate backwards to keep indices stable)
    const inputs = node.inputs || [];
    for (let i = inputs.length - 1; i >= 0; i--) {
        const name = inputs[i]?.name ?? "";
        if (/^(image|mask)\d+$/.test(name) && !wanted.has(name)) {
            node.removeInput(i);
        }
    }

    // Add missing inputs in order so they appear grouped (image1, mask1, image2, mask2, …)
    const existingNames = new Set((node.inputs || []).map((inp) => inp.name));
    for (let i = 1; i <= count; i++) {
        if (!existingNames.has(`image${i}`)) node.addInput(`image${i}`, "IMAGE");
        if (!existingNames.has(`mask${i}`)) node.addInput(`mask${i}`, "MASK");
    }

    node.setSize(node.computeSize());
}

app.registerExtension({
    name: "Painter.FluxImageEdit",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PainterFluxImageEdit") return;

        // Called when a new node is created (drag from menu)
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const widget = this.widgets?.find((w) => w.name === "num_images");
            if (widget) {
                syncSlots(this, widget.value);

                // Watch value changes from the user editing the widget
                const origCallback = widget.callback;
                widget.callback = (...args) => {
                    origCallback?.(...args);
                    syncSlots(this, args[0]);
                };
            }

            return result;
        };

        // Called when a node is restored from a saved workflow
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            onConfigure?.apply(this, arguments);
            const widget = this.widgets?.find((w) => w.name === "num_images");
            if (widget) syncSlots(this, widget.value);
        };
    },
});
```

**Step 3: Verify the file is in the right location**

```bash
ls web/js/
# Expected: PainterFluxImageEdit.js
```

**Step 4: Commit**

```bash
git add web/js/PainterFluxImageEdit.js
git commit -m "feat: add JS extension for dynamic image/mask slot visibility"
```

---

## Task 4: Manual verification in ComfyUI

**Step 1: Restart ComfyUI**

If ComfyUI is running, restart it so it picks up the new custom node version. Check the terminal for errors — a successful load looks like:

```
Import PainterFluxImageEdit OK
```

No `ImportError` or `SyntaxError` should appear.

**Step 2: Load the node in a workflow**

1. Open ComfyUI in browser
2. Right-click canvas → `Add Node` → `advanced/conditioning` → `Painter Flux Image Edit`
3. Confirm: no `mode` dropdown, `num_images` widget is present, `negative_prompt` text area is present

**Step 3: Test num_images=1**

Set `num_images` to 1. Confirm only `image1` and `mask1` slots appear.

**Step 4: Test num_images=3**

Change `num_images` to 3. Confirm `image1`/`mask1`, `image2`/`mask2`, `image3`/`mask3` all appear. Change back to 1 — confirm `image2`, `mask2`, `image3`, `mask3` disappear.

**Step 5: Test text-to-image (no images connected)**

Connect clip + VAE + prompt, run generation. Should produce a result without errors.

**Step 6: Test single-image edit**

Connect one image to `image1`, run. Should produce an edited result using the reference latent.

**Step 7: Test inpainting with mask1**

Connect image1 + mask1, run. The masked area should be the target of regeneration.

**Step 8: Test multi-image (num_images=2)**

Connect two different images, try `reference_latents_method = offset` and `index`. Both should run without error.

---

## Task 5: Commit and clean up

**Step 1: Run final test suite**

```bash
python -m pytest tests/test_encode.py -v
```

All tests must pass.

**Step 2: Final commit**

```bash
git add -A
git commit -m "chore: finalize FLUX.2 klein improvements — all tests passing"
```
