"""Vision and UI abstractions.

Status: PROPOSED. The four-view abstraction (pixels, UI/accessibility,
source-to-render, temporal) is typed and exercised in tests. No live
capture or model is loaded in v0.1.0.
"""

from __future__ import annotations

from .interface import (
    CaptureFrame,
    PixelView,
    SourceRenderMap,
    StructuredView,
    TemporalFrame,
    TemporalView,
    VisionSnapshot,
    make_pixel_view_from_bytes,
)

__all__ = [
    "CaptureFrame",
    "PixelView",
    "SourceRenderMap",
    "StructuredView",
    "TemporalFrame",
    "TemporalView",
    "VisionSnapshot",
    "make_pixel_view_from_bytes",
]
