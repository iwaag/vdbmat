"""Renderer-independent preprocessing for voxel material optics."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vbdmat")
except PackageNotFoundError:
    # Examples also run directly from a mounted source tree in optional native
    # renderer containers, where distribution metadata is intentionally absent.
    __version__ = "0.1.0"

__all__ = ["__version__"]
