try:
    from .PainterKleinImageEdit import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Running outside of ComfyUI (e.g. during unit tests) — relative imports
    # are not available.  Expose empty mappings so pytest can load this file
    # without error.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "1.0.0"

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
