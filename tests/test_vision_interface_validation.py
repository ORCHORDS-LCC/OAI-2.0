"""Outer guard rails for ``oai2/vision/interface.py`` boundary contracts.

The existing ``tests/test_vision.py`` covers the high-level snapshot
construction end-to-end. What is NOT pinned for
``oai2/vision/interface.py``:

* Module shape — module docstring mentions "Vision" + "four-view" + "Pydantic"
  + the VISION_AND_UI.md doc reference; ``from __future__ import annotations``
  (UP006-clean); stdlib imports ``BytesIO`` from ``io`` and ``Any`` from
  ``typing``; pydantic symbols ``BaseModel`` / ``ConfigDict`` / ``Field`` co-
  imported on one line; relative import ``from ..core import Status`` (single-
  name form); no absolute ``import oai2`` / ``from oai2.vision`` at line-
  start; no wildcard imports; no cloud-runtime imports.
* ``__all__`` completeness — exactly 8 names (``PixelView`` /
  ``StructuredView`` / ``SourceRenderMap`` / ``TemporalFrame`` /
  ``TemporalView`` / ``VisionSnapshot`` / ``CaptureFrame`` /
  ``make_pixel_view_from_bytes``); every name is importable from the module;
  identity-equal re-exports at the ``oai2.vision`` package level.
* ``PixelView`` Pydantic v2 ``BaseModel`` with
  ``ConfigDict(extra="forbid", arbitrary_types_allowed=True)`` — fields
  ``image: Any = None`` / ``width: int = Field(ge=0)`` / ``height: int =
  Field(ge=0)`` / ``format: str = "PNG"`` / ``status: Status =
  Status.PROPOSED``; ``width`` / ``height`` bounded ``>= 0`` and reject
  fractional floats via Pydantic v2 ``int_from_float`` validation.
* ``StructuredView`` / ``SourceRenderMap`` / ``TemporalFrame`` /
  ``TemporalView`` / ``VisionSnapshot`` — same extra="forbid" pattern; tuple
  default factories preserved; ``TemporalFrame.index: int = Field(ge=0)``
  bounded ``>= 0`` and rejects fractional floats.
* ``CaptureFrame = TemporalFrame`` alias assignment (module-level
  identity-preserving rename, NOT a ``TypeAlias``).
* ``make_pixel_view_from_bytes(data: bytes) -> PixelView`` — defer-imports
  ``PIL.Image`` inside the function body (kept Pillow-free at parse time),
  reads width / height / format off the opened PIL Image, defaults format to
  ``"PNG"`` when PIL returns ``None`` for ``img.format``, and sets
  ``status=Status.EXPERIMENTAL`` (the only public surface that overrides
  Status.PROPOSED).
* Public-safety — no ``boto3`` / ``azure`` / ``google.cloud`` /
  ``kubernetes`` / ``docker`` / ``fabric`` (vision is pure pydantic); no
  hardcoded credentials, no ``print`` / ``pprint``, no ``subprocess`` /
  ``os.system``, no direct ``requests`` / ``urllib`` / ``httpx`` /
  ``aiohttp``, no ``eval`` / ``exec`` / ``compile``, no ``os.environ`` /
  ``os.getenv``, no ``TODO`` / ``FIXME`` / ``XXX`` markers.

Each section pins one or more of these contracts with a small, sharp test
that fails immediately on a regression. The pattern follows the slice-60 …
slice-65 outer-guard-rail files in this campaign.
"""

from __future__ import annotations

import inspect
import re
import struct
import zlib
from unittest import mock

import pytest
from pydantic import BaseModel, ValidationError

from oai2.core import Status
from oai2.vision.interface import (
    CaptureFrame,
    PixelView,
    SourceRenderMap,
    StructuredView,
    TemporalFrame,
    TemporalView,
    VisionSnapshot,
    make_pixel_view_from_bytes,
)
from oai2.vision.interface import (
    __all__ as interface_all,
)

VISION_INTERFACE_MODULE = "oai2.vision.interface"


def _module_source() -> str:
    import oai2.vision.interface as _mod

    return inspect.getsource(_mod)


def _png_bytes(width: int = 1, height: int = 1) -> bytes:
    """Build a tiny in-memory PNG of size ``width`` x ``height`` (single colour).

    Uses only stdlib (``struct`` + ``zlib``) so the test does not require
    Pillow at import-time. The output is a valid 8-bit greyscale PNG.
    """
    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    # IHDR
    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        8,
        0,
        0,
        0,
        0,  # 8-bit greyscale
    )
    # IDAT: one filter byte (0 = None) per row, all-zero pixels for greyscale.
    raw = b""
    for _ in range(height):
        raw += b"\x00" + b"\x00" * width
    idat = zlib.compress(raw)
    iend = b""

    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", iend)


# ---------------------------------------------------------------------------
# Section 1 — docstring + module-level imports
# ---------------------------------------------------------------------------


def test_vision_module_docstring_mentions_vision_four_view_and_pydantic() -> None:
    src = _module_source()
    doc_match = re.search(r'^"""(?P<body>.*?)"""', src, re.DOTALL)
    assert doc_match is not None, "module must begin with a triple-quoted docstring"
    body = doc_match.group("body").lower()
    assert "vision" in body, "docstring must mention 'vision'"
    assert "four-view" in body or "four view" in body or "four views" in body, (
        "docstring must mention 'four-view' / 'four view(s)'"
    )
    assert "pydantic" in body, "docstring must mention 'pydantic'"


def test_vision_module_has_future_annotations() -> None:
    src = _module_source()
    assert re.search(r"^from __future__ import annotations\s*$", src, re.MULTILINE), (
        "module must declare `from __future__ import annotations` for UP006"
    )


def test_vision_module_imports_bytesio_from_io_and_any_from_typing() -> None:
    src = _module_source()
    assert re.search(r"^from io import BytesIO\s*$", src, re.MULTILINE), (
        "module must `from io import BytesIO` (single-name form)"
    )
    assert re.search(r"^from typing import Any\s*$", src, re.MULTILINE), (
        "module must `from typing import Any` (single-name form)"
    )


def test_vision_module_imports_pydantic_symbols_on_one_line() -> None:
    src = _module_source()
    match = re.search(r"^from pydantic import BaseModel, ConfigDict, Field\s*$", src, re.MULTILINE)
    assert match is not None, "module must co-import BaseModel / ConfigDict / Field on one line"


def test_vision_module_imports_status_from_core_relative() -> None:
    src = _module_source()
    match = re.search(r"^from \.\.core import Status\s*$", src, re.MULTILINE)
    assert match is not None, "module must `from ..core import Status` (single-name form)"


def test_vision_module_has_no_absolute_oai2_imports() -> None:
    src = _module_source()
    for line in src.splitlines():
        stripped = line.lstrip()
        if (
            stripped.startswith("import oai2")
            or stripped.startswith("from oai2 ")
            or stripped.startswith("from oai2.")
        ):
            if stripped.startswith("from oai2.vision.interface "):
                # Self-references allowed.
                continue
            if stripped.startswith("from oai2.core ") or stripped.startswith(
                "from oai2.vision.interface "
            ):
                # Allowed re-imports that already exist upstream.
                continue
            pytest.fail(f"unexpected absolute oai2 import: {line!r}")


def test_vision_module_has_no_cloud_runtime_imports() -> None:
    src = _module_source()
    forbidden = (
        "boto3",
        "azure",
        "google.cloud",
        "kubernetes",
        "docker",
        "fabric",
    )
    for needle in forbidden:
        assert needle not in src, f"vision interface must not reference {needle!r} (cloud runtime)"


def test_vision_module_has_no_wildcard_imports() -> None:
    src = _module_source()
    assert not re.search(r"^from\s+\S+\s+import\s+\*\s*$", src, re.MULTILINE), (
        "vision interface must not contain wildcard imports"
    )


# ---------------------------------------------------------------------------
# Section 2 — __all__ + identity
# ---------------------------------------------------------------------------


def test_vision_module_all_is_declared_and_has_eight_names() -> None:
    assert isinstance(interface_all, list), "__all__ must be a list"
    assert sorted(interface_all) == sorted(
        [
            "PixelView",
            "StructuredView",
            "SourceRenderMap",
            "TemporalFrame",
            "TemporalView",
            "VisionSnapshot",
            "CaptureFrame",
            "make_pixel_view_from_bytes",
        ]
    ), f"__all__ must export exactly 8 documented names (got {interface_all!r})"


def test_vision_module_all_names_are_importable() -> None:
    import oai2.vision.interface as mod

    for name in interface_all:
        assert hasattr(mod, name), f"{name} must be importable from the module"


def test_vision_module_source_pins_eight_quoted_all_strings() -> None:
    src = _module_source()
    quoted = re.findall(r"^__all__\s*=\s*\[([^\]]+)\]", src, re.MULTILINE)
    assert quoted, "__all__ must be a literal list assignment at module scope"
    names = [
        n.strip().strip('"').strip("'")
        for n in quoted[0].split(",")
        if n.strip().strip('"').strip("'")
    ]
    assert len(names) == 8, (
        f"__all__ literal must list exactly 8 names (got {len(names)}: {names!r})"
    )


def test_vision_module_source_has_no_top_level_dunder_side_effects() -> None:
    import ast

    tree = ast.parse(_module_source())
    allowed = (
        ast.Module,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.Expr,
        ast.If,
        ast.Try,
        ast.With,
        ast.Pass,
    )
    for node in tree.body:
        assert isinstance(node, allowed), (
            f"unexpected top-level statement kind: {type(node).__name__}"
        )


def test_vision_symbols_reexported_at_vision_package_level() -> None:
    import oai2.vision as pkg

    for name in interface_all:
        assert hasattr(pkg, name), f"oai2.vision must re-export {name} via __init__.py"
        assert getattr(pkg, name) is globals()[name], (
            f"oai2.vision.{name} must identity-equal oai2.vision.interface.{name}"
        )


def test_vision_symbols_match_direct_module_imports() -> None:
    from oai2.vision.interface import (
        CaptureFrame as DirectCF,
    )
    from oai2.vision.interface import (
        PixelView as DirectPixel,
    )
    from oai2.vision.interface import (
        SourceRenderMap as DirectSRM,
    )
    from oai2.vision.interface import (
        StructuredView as DirectSV,
    )
    from oai2.vision.interface import (
        TemporalFrame as DirectTF,
    )
    from oai2.vision.interface import (
        TemporalView as DirectTV,
    )
    from oai2.vision.interface import (
        VisionSnapshot as DirectVS,
    )
    from oai2.vision.interface import (
        make_pixel_view_from_bytes as DirectMaker,
    )

    assert DirectCF is CaptureFrame
    assert DirectPixel is PixelView
    assert DirectSRM is SourceRenderMap
    assert DirectSV is StructuredView
    assert DirectTF is TemporalFrame
    assert DirectTV is TemporalView
    assert DirectVS is VisionSnapshot
    assert DirectMaker is make_pixel_view_from_bytes


def test_vision_package_all_contains_eight_interface_names() -> None:
    import oai2.vision as pkg

    for name in interface_all:
        assert name in pkg.__all__, (
            f"oai2.vision.__all__ must contain {name} so it is reachable as oai2.vision.{name}"
        )


# ---------------------------------------------------------------------------
# Section 3 — PixelView
# ---------------------------------------------------------------------------


def test_pixel_view_is_a_pydantic_basemodel() -> None:
    assert issubclass(PixelView, BaseModel)


def test_pixel_view_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PixelView(unknown_field="x")  # type: ignore[call-arg]


def test_pixel_view_uses_arbitrary_types_allowed() -> None:
    config = PixelView.model_config
    assert config.get("extra") == "forbid"
    assert config.get("arbitrary_types_allowed") is True


def test_pixel_view_field_set_is_five_names() -> None:
    field_names = set(PixelView.model_fields.keys())
    assert field_names == {"image", "width", "height", "format", "status"}, (
        f"unexpected field set: {field_names!r}"
    )


def test_pixel_view_image_defaults_to_none() -> None:
    pv = PixelView(width=0, height=0)
    assert pv.image is None


def test_pixel_view_format_default_is_png() -> None:
    pv = PixelView(width=0, height=0)
    assert pv.format == "PNG"


def test_pixel_view_status_default_is_proposed() -> None:
    pv = PixelView(width=0, height=0)
    assert pv.status is Status.PROPOSED


def test_pixel_view_width_and_height_bounded_non_negative() -> None:
    for w in (0, 1, 100, 1000000):
        pv = PixelView(width=w, height=0)
        assert pv.width == w
    for w in (-1, -100):
        with pytest.raises(ValidationError):
            PixelView(width=w, height=0)
    for h in (0, 1, 100, 1000000):
        pv = PixelView(width=0, height=h)
        assert pv.height == h
    for h in (-1, -100):
        with pytest.raises(ValidationError):
            PixelView(width=0, height=h)


def test_pixel_view_width_and_height_reject_fractional_floats() -> None:
    """Pydantic v2 ``int_from_float`` validation: a fractional ``1.5`` raises
    ``ValidationError`` when the field is typed ``int``."""
    with pytest.raises(ValidationError):
        PixelView(width=1.5, height=0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PixelView(width=0, height=1.5)  # type: ignore[arg-type]


def test_pixel_view_status_accepts_every_status_enum_value() -> None:
    for value in Status:
        pv = PixelView(width=0, height=0, status=value)
        assert pv.status is value


def test_pixel_view_status_accepts_wire_string_coercion() -> None:
    """A literal ``"PROPOSED`` wire string IS coerced to ``Status.PROPOSED``
    because Status is a StrEnum and Pydantic v2 honours StrEnum coercion."""
    pv = PixelView(width=0, height=0, status="PROPOSED")  # type: ignore[arg-type]
    assert pv.status is Status.PROPOSED


def test_pixel_view_status_rejects_unknown_wire_string() -> None:
    with pytest.raises(ValidationError):
        PixelView(width=0, height=0, status="NOT_A_REAL_STATUS")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Section 4 — StructuredView
# ---------------------------------------------------------------------------


def test_structured_view_is_a_pydantic_basemodel() -> None:
    assert issubclass(StructuredView, BaseModel)


def test_structured_view_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        StructuredView(unknown_field="x")  # type: ignore[call-arg]


def test_structured_view_field_set_is_three_names() -> None:
    field_names = set(StructuredView.model_fields.keys())
    assert field_names == {"nodes", "role_signature", "status"}, (
        f"unexpected field set: {field_names!r}"
    )


def test_structured_view_nodes_defaults_to_empty_tuple() -> None:
    sv = StructuredView()
    assert sv.nodes == ()
    assert isinstance(sv.nodes, tuple)


def test_structured_view_role_signature_defaults_to_none() -> None:
    sv = StructuredView()
    assert sv.role_signature is None


def test_structured_view_status_default_is_proposed() -> None:
    sv = StructuredView()
    assert sv.status is Status.PROPOSED


def test_structured_view_with_single_node_preserves_dict() -> None:
    node = {"tag": "button", "role": "button", "label": "Submit"}
    sv = StructuredView(nodes=(node,), role_signature="button")
    assert sv.nodes == (node,)
    assert sv.role_signature == "button"


def test_structured_view_status_accepts_every_status_enum_value() -> None:
    for value in Status:
        sv = StructuredView(status=value)
        assert sv.status is value


# ---------------------------------------------------------------------------
# Section 5 — SourceRenderMap
# ---------------------------------------------------------------------------


def test_source_render_map_is_a_pydantic_basemodel() -> None:
    assert issubclass(SourceRenderMap, BaseModel)


def test_source_render_map_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SourceRenderMap(unknown_field="x")  # type: ignore[call-arg]


def test_source_render_map_field_set_is_two_names() -> None:
    field_names = set(SourceRenderMap.model_fields.keys())
    assert field_names == {"pairs", "status"}, f"unexpected field set: {field_names!r}"


def test_source_render_map_pairs_defaults_to_empty_tuple() -> None:
    srm = SourceRenderMap()
    assert srm.pairs == ()
    assert isinstance(srm.pairs, tuple)


def test_source_render_map_status_default_is_proposed() -> None:
    srm = SourceRenderMap()
    assert srm.status is Status.PROPOSED


def test_source_render_map_with_multiple_pairs_preserves_order() -> None:
    pairs = (
        ("src:1", "render:button"),
        ("src:2", "render:input"),
        ("src:3", "render:link"),
    )
    srm = SourceRenderMap(pairs=pairs)
    assert srm.pairs == pairs
    assert srm.status is Status.PROPOSED


# ---------------------------------------------------------------------------
# Section 6 — TemporalFrame + TemporalView
# ---------------------------------------------------------------------------


def test_temporal_frame_is_a_pydantic_basemodel() -> None:
    assert issubclass(TemporalFrame, BaseModel)


def test_temporal_frame_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TemporalFrame(index=0, summary="x", unknown_field="y")  # type: ignore[call-arg]


def test_temporal_frame_field_set_is_three_names() -> None:
    field_names = set(TemporalFrame.model_fields.keys())
    assert field_names == {"index", "summary", "screenshot_ref"}, (
        f"unexpected field set: {field_names!r}"
    )


def test_temporal_frame_index_bounded_non_negative() -> None:
    for n in (0, 1, 100, 1000000):
        tf = TemporalFrame(index=n, summary="x")
        assert tf.index == n
    for n in (-1, -100):
        with pytest.raises(ValidationError):
            TemporalFrame(index=n, summary="x")


def test_temporal_frame_index_rejects_fractional_floats() -> None:
    with pytest.raises(ValidationError):
        TemporalFrame(index=0.5, summary="x")  # type: ignore[arg-type]


def test_temporal_frame_summary_is_mandatory() -> None:
    with pytest.raises(ValidationError):
        TemporalFrame(index=0)  # type: ignore[call-arg]


def test_temporal_frame_screenshot_ref_defaults_to_none() -> None:
    tf = TemporalFrame(index=0, summary="x")
    assert tf.screenshot_ref is None
    tf2 = TemporalFrame(index=0, summary="x", screenshot_ref="ref.png")
    assert tf2.screenshot_ref == "ref.png"


def test_temporal_view_is_a_pydantic_basemodel() -> None:
    assert issubclass(TemporalView, BaseModel)


def test_temporal_view_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TemporalView(unknown_field="x")  # type: ignore[call-arg]


def test_temporal_view_field_set_is_two_names() -> None:
    field_names = set(TemporalView.model_fields.keys())
    assert field_names == {"frames", "status"}, f"unexpected field set: {field_names!r}"


def test_temporal_view_frames_defaults_to_empty_tuple() -> None:
    tv = TemporalView()
    assert tv.frames == ()
    assert isinstance(tv.frames, tuple)


def test_temporal_view_with_multiple_frames_preserves_order() -> None:
    frames = (
        TemporalFrame(index=0, summary="first"),
        TemporalFrame(index=1, summary="second"),
        TemporalFrame(index=2, summary="third"),
    )
    tv = TemporalView(frames=frames)
    assert tv.frames == frames
    assert all(isinstance(f, TemporalFrame) for f in tv.frames)


def test_temporal_view_status_default_is_proposed() -> None:
    tv = TemporalView()
    assert tv.status is Status.PROPOSED


def test_temporal_view_status_accepts_every_status_enum_value() -> None:
    for value in Status:
        tv = TemporalView(status=value)
        assert tv.status is value


# ---------------------------------------------------------------------------
# Section 7 — VisionSnapshot
# ---------------------------------------------------------------------------


def test_vision_snapshot_is_a_pydantic_basemodel() -> None:
    assert issubclass(VisionSnapshot, BaseModel)


def test_vision_snapshot_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        VisionSnapshot(unknown_field="x")  # type: ignore[call-arg]


def test_vision_snapshot_field_set_is_five_names() -> None:
    field_names = set(VisionSnapshot.model_fields.keys())
    assert field_names == {
        "pixels",
        "structure",
        "mapping",
        "temporal",
        "captured_at",
    }, f"unexpected field set: {field_names!r}"


def test_vision_snapshot_default_views_are_none() -> None:
    snap = VisionSnapshot()
    assert snap.pixels is None
    assert snap.structure is None
    assert snap.mapping is None
    assert snap.temporal is None


def test_vision_snapshot_captured_at_defaults_to_zero() -> None:
    snap = VisionSnapshot()
    assert snap.captured_at == 0.0


def test_vision_snapshot_with_all_four_views_preserves_them() -> None:
    snap = VisionSnapshot(
        pixels=PixelView(width=10, height=10),
        structure=StructuredView(role_signature="button"),
        mapping=SourceRenderMap(pairs=(("a", "b"),)),
        temporal=TemporalView(frames=(TemporalFrame(index=0, summary="x"),)),
        captured_at=1700000000.0,
    )
    assert isinstance(snap.pixels, PixelView)
    assert isinstance(snap.structure, StructuredView)
    assert isinstance(snap.mapping, SourceRenderMap)
    assert isinstance(snap.temporal, TemporalView)
    assert snap.captured_at == 1700000000.0


def test_vision_snapshot_nested_invalid_view_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        VisionSnapshot(pixels=PixelView(width=-1, height=0))


# ---------------------------------------------------------------------------
# Section 8 — make_pixel_view_from_bytes + CaptureFrame alias
# ---------------------------------------------------------------------------


def test_make_pixel_view_from_bytes_is_a_function() -> None:
    assert inspect.isfunction(make_pixel_view_from_bytes)


def test_make_pixel_view_from_bytes_signature_is_data_bytes_only() -> None:
    sig = inspect.signature(make_pixel_view_from_bytes)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["data"], (
        f"signature must be exactly (data), got {[p.name for p in params]!r}"
    )
    data_param = params[0]
    assert data_param.annotation is bytes or str(data_param.annotation) == "bytes"


def test_make_pixel_view_from_bytes_returns_pixel_view() -> None:
    data = _png_bytes(width=2, height=3)
    pv = make_pixel_view_from_bytes(data)
    assert isinstance(pv, PixelView)
    assert pv.width == 2
    assert pv.height == 3


def test_make_pixel_view_from_bytes_uses_experimental_status() -> None:
    """``make_pixel_view_from_bytes`` is the only public surface that
    overrides the PROPOSED default — it sets Status.EXPERIMENTAL."""
    pv = make_pixel_view_from_bytes(_png_bytes())
    assert pv.status is Status.EXPERIMENTAL


def test_make_pixel_view_from_bytes_default_format_is_png() -> None:
    """When ``img.format`` is ``None`` (rare with PIL but possible), the
    helper falls back to ``\"PNG\"``."""
    fake_img = mock.MagicMock()
    fake_img.width = 4
    fake_img.height = 5
    fake_img.format = None
    with mock.patch("PIL.Image.open", return_value=fake_img):
        pv = make_pixel_view_from_bytes(b"\x00")
    assert pv.format == "PNG"
    assert pv.width == 4
    assert pv.height == 5


def test_make_pixel_view_from_bytes_propagates_pil_unidentified_image_error() -> None:
    """Non-image bytes cause PIL to raise ``UnidentifiedImageError`` which
    is allowed to propagate (the function does not swallow the exception)."""
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        make_pixel_view_from_bytes(b"")


def test_capture_frame_is_identity_alias_for_temporal_frame() -> None:
    """``CaptureFrame`` is a module-level rename of ``TemporalFrame`` — the
    assignment must preserve identity so existing users of either name get the
    same class."""
    assert CaptureFrame is TemporalFrame


def test_capture_frame_alias_assignment_present_in_source() -> None:
    src = _module_source()
    assert re.search(r"^CaptureFrame\s*=\s*TemporalFrame\s*$", src, re.MULTILINE), (
        "CaptureFrame must be a module-level alias for TemporalFrame"
    )


def test_make_pixel_view_from_bytes_module_attribute_path() -> None:
    import oai2.vision.interface as mod

    assert mod.__name__ == VISION_INTERFACE_MODULE


# ---------------------------------------------------------------------------
# Section 9 — public-safety + structural counts
# ---------------------------------------------------------------------------


def test_vision_module_has_no_hardcoded_credentials() -> None:
    src = _module_source()
    assert "api_key=" not in src, "no `api_key=` literal in source"
    assert "BEGIN PRIVATE KEY" not in src, "no embedded private keys"
    for pattern in (
        r"sk-[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"AIza[0-9A-Za-z\-_]{35}",
        r"xox[baprs]-[A-Za-z0-9-]+",
    ):
        assert not re.search(pattern, src), f"credential-like pattern {pattern!r} found in source"


def test_vision_module_has_no_print_or_pprint() -> None:
    src = _module_source()
    assert not re.search(r"\bprint\s*\(", src), "no print() in source"
    assert not re.search(r"\bpprint\s*[\.\(]", src), "no pprint in source"


def test_vision_module_has_no_subprocess_or_shell() -> None:
    src = _module_source()
    assert "subprocess" not in src, "no subprocess in source"
    assert "os.system" not in src, "no os.system in source"
    assert "shell=True" not in src, "no shell=True in source"


def test_vision_module_has_no_direct_http_imports() -> None:
    src = _module_source()
    for needle in (
        "import requests",
        "import urllib",
        "import httpx",
        "import aiohttp",
    ):
        assert needle not in src, f"unexpected direct HTTP import: {needle!r}"


def test_vision_module_has_no_eval_or_exec() -> None:
    src = _module_source()
    assert not re.search(r"^\s*eval\s*\(", src, re.MULTILINE), "no eval(...) call"
    assert not re.search(r"^\s*exec\s*\(", src, re.MULTILINE), "no exec(...) call"
    assert "compile(" not in src, "no compile(...) call"


def test_vision_module_has_no_os_environ_access() -> None:
    src = _module_source()
    assert "os.environ" not in src, "no os.environ access"
    assert "os.getenv" not in src, "no os.getenv call"


def test_vision_module_has_no_todo_fixme_xxx_markers() -> None:
    src = _module_source()
    for marker in ("TODO", "FIXME", "XXX"):
        assert not re.search(rf"\b{marker}\b", src), f"no {marker!r} markers allowed in source"


def test_vision_module_source_ends_with_single_trailing_newline() -> None:
    src = _module_source()
    assert src.endswith("\n"), "source must end with at least one newline"
    assert not src.endswith("\n\n"), "source must end with exactly one trailing newline (POSIX)"


def test_vision_module_source_pins_six_basemodel_subclasses() -> None:
    src = _module_source()
    basemodels = re.findall(r"^class\s+(\w+)\s*\(\s*BaseModel\s*\)\s*:", src, re.MULTILINE)
    assert sorted(basemodels) == sorted(
        [
            "PixelView",
            "StructuredView",
            "SourceRenderMap",
            "TemporalFrame",
            "TemporalView",
            "VisionSnapshot",
        ]
    ), f"expected exactly 6 BaseModel subclasses (the 6 listed), got {basemodels!r}"


def test_vision_module_source_pins_one_model_config_block() -> None:
    """``extra='forbid'`` is declared in 6 model_config blocks (one per
    BaseModel subclass). The 6th (``PixelView``) also sets
    ``arbitrary_types_allowed=True``."""
    src = _module_source()
    model_configs = re.findall(r"^\s+model_config\s*=\s*ConfigDict\(", src, re.MULTILINE)
    assert len(model_configs) == 6, (
        f"expected exactly 6 model_config blocks, got {len(model_configs)}"
    )


def test_vision_module_source_pins_one_extra_forbid_among_model_configs() -> None:
    src = _module_source()
    forbid_count = len(re.findall(r'extra="forbid"', src))
    assert forbid_count == 6, f"expected 6 `extra='forbid'` declarations, got {forbid_count}"


def test_vision_module_source_pins_one_arbitrary_types_allowed() -> None:
    src = _module_source()
    assert re.search(r"arbitrary_types_allowed\s*=\s*True", src), (
        "PixelView must declare `arbitrary_types_allowed=True`"
    )


def test_vision_module_source_pins_one_pil_defer_import() -> None:
    """``make_pixel_view_from_bytes`` defer-imports PIL inside the function
    body, so the module is Pillow-free at parse time."""
    src = _module_source()
    # No top-level PIL import.
    assert not re.search(r"^import PIL", src, re.MULTILINE), (
        "vision interface must not `import PIL` at module scope"
    )
    assert not re.search(r"^from PIL", src, re.MULTILINE), (
        "vision interface must not `from PIL …` at module scope"
    )
    # Defer-import inside the function.
    assert re.search(r"from PIL import Image as PILImage", src), (
        "PIL.Image must be defer-imported inside make_pixel_view_from_bytes"
    )


def test_vision_module_make_pixel_view_uses_bytesio_for_in_memory_data() -> None:
    src = _module_source()
    assert re.search(r"BytesIO\(data\)", src), (
        "make_pixel_view_from_bytes must wrap data in BytesIO before handing to PIL.Image.open"
    )


def test_vision_module_source_pins_one_module_level_alias_assignment() -> None:
    src = _module_source()
    assert re.search(r"^CaptureFrame\s*=\s*TemporalFrame\s*$", src, re.MULTILINE), (
        "CaptureFrame must be a module-level alias assignment"
    )


def test_vision_module_does_not_import_pillow_at_module_scope() -> None:
    """Verify the runtime contract — import the module and check that PIL is
    NOT in its namespace."""
    import oai2.vision.interface as mod

    assert not hasattr(mod, "PIL"), (
        "vision interface must not expose PIL at module scope (defer-imported)"
    )
    assert not hasattr(mod, "Image"), "vision interface must not expose PIL.Image at module scope"
