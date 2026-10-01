"""Vision four-view abstraction.

References ``docs/agent-architecture/VISION_AND_UI.md``. The four views
(pixels, structured UI/accessibility, source-to-render mapping, temporal
frames) are exposed as Pydantic shapes so the agent can reason over
them uniformly.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core import Status


class PixelView(BaseModel):
    """A captured pixel image as a Pillow object."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    image: Any = None  # PIL.Image.Image
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    format: str = "PNG"
    status: Status = Status.PROPOSED


class StructuredView(BaseModel):
    """UI/accessibility tree as a flat list of nodes."""

    model_config = ConfigDict(extra="forbid")

    nodes: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    role_signature: str | None = None
    status: Status = Status.PROPOSED


class SourceRenderMap(BaseModel):
    """Bidirectional source ↔ UI element mapping."""

    model_config = ConfigDict(extra="forbid")

    pairs: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    status: Status = Status.PROPOSED


class TemporalFrame(BaseModel):
    """One entry in the temporal interaction trace."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    summary: str
    screenshot_ref: str | None = None


class TemporalView(BaseModel):
    """Ordered list of :class:`TemporalFrame`."""

    model_config = ConfigDict(extra="forbid")

    frames: tuple[TemporalFrame, ...] = Field(default_factory=tuple)
    status: Status = Status.PROPOSED


class VisionSnapshot(BaseModel):
    """A unified capture containing all four views."""

    model_config = ConfigDict(extra="forbid")

    pixels: PixelView | None = None
    structure: StructuredView | None = None
    mapping: SourceRenderMap | None = None
    temporal: TemporalView | None = None
    captured_at: float = 0.0


def make_pixel_view_from_bytes(data: bytes) -> PixelView:
    """Create a :class:`PixelView` from in-memory PNG/JPEG bytes."""
    from PIL import Image as PILImage

    img = PILImage.open(BytesIO(data))
    return PixelView(
        image=img,
        width=img.width,
        height=img.height,
        format=img.format or "PNG",
        status=Status.EXPERIMENTAL,
    )


# Renamed alias to avoid the unused-import warning when this file is
# imported for types only.
CaptureFrame = TemporalFrame


__all__ = [
    "PixelView",
    "StructuredView",
    "SourceRenderMap",
    "TemporalFrame",
    "TemporalView",
    "VisionSnapshot",
    "CaptureFrame",
    "make_pixel_view_from_bytes",
]
