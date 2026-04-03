# conftest.py — pytest root configuration
# Prevent pytest from collecting the package __init__.py, which uses
# ComfyUI-style relative imports that only work when loaded by ComfyUI itself.
import sys
import types

collect_ignore = ["__init__.py"]

# Pre-register the package under its directory name so that pytest's import
# machinery does not try to load the real __init__.py (which uses relative
# imports that are only valid inside ComfyUI's loader).
_pkg_name = "Comfyui-PainterFluxImageEdit"
if _pkg_name not in sys.modules:
    _stub = types.ModuleType(_pkg_name)
    sys.modules[_pkg_name] = _stub
