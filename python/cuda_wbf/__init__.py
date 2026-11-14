"""Native Weighted Boxes Fusion."""

from ._native import available_backends, build_info
from .api import fuse

__all__ = ["available_backends", "build_info", "fuse"]
__version__ = "0.1.0"
