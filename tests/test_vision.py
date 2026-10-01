"""Vision four-view tests."""

from __future__ import annotations

import struct

from oai2.core import Status
from oai2.vision import (
    PixelView,
    SourceRenderMap,
    StructuredView,
    TemporalFrame,
    TemporalView,
    VisionSnapshot,
    make_pixel_view_from_bytes,
)


def _png_1x1() -> bytes:
    """Return a minimal valid 1x1 transparent PNG."""
    # Pre-computed 1x1 transparent PNG bytes — no external generator needed.
    import base64

    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )


def test_pixel_view_from_bytes() -> None:
    pv = make_pixel_view_from_bytes(_png_1x1())
    assert isinstance(pv, PixelView)
    assert pv.width == 1 and pv.height == 1
    assert pv.status is Status.EXPERIMENTAL


def test_structured_view_role_signature() -> None:
    sv = StructuredView(
        nodes=({"role": "button"}, {"role": "text"}),
        role_signature="button,text",
    )
    assert sv.role_signature == "button,text"


def test_source_render_map_pairs() -> None:
    m = SourceRenderMap(pairs=(("btn.tsx:7", "Submit"),))
    assert m.pairs == (("btn.tsx:7", "Submit"),)


def test_temporal_view_preserves_order() -> None:
    tv = TemporalView(
        frames=(
            TemporalFrame(index=0, summary="frame 0"),
            TemporalFrame(index=1, summary="frame 1"),
        )
    )
    assert [f.summary for f in tv.frames] == ["frame 0", "frame 1"]


def test_vision_snapshot_optional_views() -> None:
    snap = VisionSnapshot(pixels=PixelView(width=0, height=0))
    assert snap.pixels is not None
    assert snap.structure is None
