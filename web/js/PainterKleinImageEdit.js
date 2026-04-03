import { app } from "../../scripts/app.js";

/**
 * Syncs the imageN / maskN input slots on the node to match `count`.
 * Adds missing slots in order (image1, mask1, image2, mask2, …) and removes
 * slots beyond `count`. Preserves existing connections where possible.
 */
function syncSlots(node, count) {
    count = Math.max(1, Math.min(10, parseInt(count) || 1));

    // image1 + mask1 always present; images 2-N have no mask slot
    const wanted = new Set();
    wanted.add("image1");
    wanted.add("mask1");
    for (let i = 2; i <= count; i++) {
        wanted.add(`image${i}`);
    }

    // Remove unwanted dynamic inputs (iterate backwards to keep indices stable)
    const inputs = node.inputs || [];
    for (let i = inputs.length - 1; i >= 0; i--) {
        const name = inputs[i]?.name ?? "";
        if (/^(image|mask)\d+$/.test(name) && !wanted.has(name)) {
            node.removeInput(i);
        }
    }

    // Add missing inputs in order: image1, mask1, image2, image3, …
    const existingNames = new Set((node.inputs || []).map((inp) => inp.name));
    if (!existingNames.has("image1")) node.addInput("image1", "IMAGE");
    if (!existingNames.has("mask1"))  node.addInput("mask1",  "MASK");
    for (let i = 2; i <= count; i++) {
        if (!existingNames.has(`image${i}`)) node.addInput(`image${i}`, "IMAGE");
    }

    node.size = node.computeSize();
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "Painter.KleinImageEdit",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PainterKleinImageEdit") return;

        // Called when a new node is created (drag from menu)
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            const widget = this.widgets?.find((w) => w.name === "num_images");
            if (widget) {
                syncSlots(node, widget.value);

                // Intercept ALL value changes (drag, arrows, keyboard, programmatic)
                // via property descriptor — more reliable than wrapping widget.callback.
                let _value = widget.value;
                Object.defineProperty(widget, "value", {
                    get() { return _value; },
                    set(v) {
                        _value = v;
                        syncSlots(node, v);
                    },
                    configurable: true,
                });
            }

            return result;
        };

        // Called when a node is restored from a saved workflow.
        // First pass: sync slots from raw config BEFORE the base onConfigure so
        // LiteGraph finds the slots when it rewires saved connections.
        // Second pass: re-sync after the base restores widget values, so we use
        // the actual widget value rather than a hardcoded index.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            // Pre-pass: use raw widgets_values to add slots before link rewiring.
            // Find num_images by scanning for its position among widget names.
            const widgetNames = (this.widgets || []).map((w) => w.name);
            const idx = widgetNames.indexOf("num_images");
            const rawCount = idx !== -1 ? config?.widgets_values?.[idx] : config?.widgets_values?.[2];
            if (rawCount !== undefined) syncSlots(this, rawCount);
            onConfigure?.apply(this, arguments);
            // Post-pass: re-sync using the restored widget value (authoritative).
            const widget = this.widgets?.find((w) => w.name === "num_images");
            if (widget) syncSlots(this, widget.value);
        };
    },
});
