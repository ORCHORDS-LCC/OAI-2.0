"""OAI-2.0 — large-capacity, dynamically activated, multimodal coding intelligence.

.. admonition:: Architecture target

    OAI-2.0 is a **large-capacity, dynamically activated, multimodal
    coding intelligence** — not a permanently tiny model plus RAG. The
    design principle is ``LARGE TOTAL INTELLIGENCE CAPACITY, SMALL
    DYNAMIC INSTANTANEOUS COMPUTE``:

    - Total specialist capacity: ~10–30B+ (research range, not fixed).
    - FURIOUS active compute: ~500M–1B.
    - NORMAL active compute: ~1–2B.
    - DEEP active compute: ~2–4B.
    - SWARM: multiple independent ~1–2B+ reasoning lanes.

    Architecture intelligence takes priority over raw tok/s. Any
    optimization that materially harms verified capability is rejected.
    See ``docs/agent-architecture/ARCHITECTURE_TARGET.md`` for the full
    target and ``docs/agent-architecture/REASONING_AND_SPEED.md`` for the
    speed strategy that does not sacrifice intelligence.

.. admonition:: Status

    The OAI-2.0 source tree is **EXPERIMENTAL / SCAFFOLD-ONLY** as of
    ``v0.1.0``. Modules here define stable public shapes, schemas, and
    public-facing abstractions. Behavior that touches live MLX inference,
    Cloudflare knowledge, or external tool execution is **PROPOSED** until
    a release of OAI-2.0 is marked ``IMPLEMENTED`` in
    ``docs/agent-architecture/``.

This package layout mirrors ``docs/agent-architecture/SYSTEM_ARCHITECTURE.md``:

- ``oai2.core`` — shared types and status markers
- ``oai2.protocols`` — tool and state schemas (host-facing)
- ``oai2.verification`` — evidence graph and claim-state machine
- ``oai2.knowledge`` — Cloudflare-backed external knowledge abstraction
- ``oai2.runtime`` — Apple Silicon (MLX) inference runtime
- ``oai2.reasoning`` — FURIOUS / NORMAL / DEEP / SWARM controllers
- ``oai2.vision`` — pixels / UI / source-render abstractions
- ``oai2.tools`` — tool dispatch and policy
- ``oai2.agents`` — multi-agent orchestration
- ``oai2.evals`` — evaluation entry points (deferred)
"""

from __future__ import annotations

from importlib import metadata as _metadata

__all__ = ["__version__"]

try:
    __version__ = _metadata.version("oai2")
except _metadata.PackageNotFoundError:
    # Source checkout where the package is not installed.
    __version__ = "0.1.0"
