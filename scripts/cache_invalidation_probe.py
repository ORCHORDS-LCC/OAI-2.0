#!/usr/bin/env python3
"""Measure cache invalidation when world state changes (AC-PERF-033).

Config D of #240 requires that reuse happens "only when
model/tokenizer/prefix/tool-schema/world-state/security identity is
compatible", and that a "changed repository/tool/world-state must
invalidate reuse". Unit tests assert the invalidation *counters* move; this
script measures whether invalidation actually costs what it should — a
cache hit must skip prefix prefill, and the same prompt under a *changed*
world-state digest must stop hitting and prefill again.

That distinction is the whole point. A cache that invalidates on paper but
still returns stale KV state would increment ``misses`` while keeping the
latency win, and no counter test would catch it.

Run with::

    .venv/bin/python scripts/cache_invalidation_probe.py \
        --out evidence/accuracy/cache_invalidation.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO))

from oai2.runtime.prefix_kv_cache import PrefixKVCache  # noqa: E402

MODEL_ID = "mlx-community/SmolLM-135M-Instruct-4bit"
PREFIX_TEXT = "You are a build assistant. Report exact paths. " * 4
SUFFIX_TEXT = "List the steps:"
GEN_TOKENS = 4


def _time_prefill(model, mx, cache, token_ids):  # noqa: ANN001, ANN202
    """Force a prefill of ``token_ids`` and return the wall time in ms.

    The model call is followed by ``mx.eval`` so the measurement waits for
    the work to actually complete; without it this would time an async
    dispatch and report a meaningless number.
    """
    start = time.perf_counter()
    model(mx.array([token_ids]), cache=cache)
    mx.eval(cache[0].state[0])
    return (time.perf_counter() - start) * 1000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="artifact path to write")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError as exc:
        print(f"SKIP cache-invalidation: mlx unavailable ({type(exc).__name__})", file=sys.stderr)
        return 1

    model, tokenizer = load(MODEL_ID)
    prefix_ids = tokenizer.encode(PREFIX_TEXT)
    suffix_ids = tokenizer.encode(SUFFIX_TEXT)

    # Two digests = two worlds. The digest is the identity that carries
    # tool-schema and world-state; an identical prompt under the other
    # digest is a different request and must not reuse the first one's
    # KV state.
    digest_world_a = "world:a:tools:v1"
    digest_world_b = "world:b:tools:v1"

    cache = PrefixKVCache(max_entries=4)
    cells: list[dict[str, object]] = []

    def record(label: str, digest: str | None, expect_hit: bool, note: str) -> None:
        timings: list[float] = []
        matched_per_rep: list[int] = []
        for _ in range(args.repetitions):
            layers = make_prompt_cache(model)
            matched = cache.lookup_and_apply(
                [*prefix_ids, *suffix_ids], layers, digest=digest, gate_digest=True
            )
            matched_per_rep.append(matched)
            # Only the *unmatched* remainder is prefilled. A larger match
            # means less prefill work, which is the effect being measured.
            tail = [*prefix_ids, *suffix_ids][matched:]
            timings.append(_time_prefill(model, mx, layers, tail or suffix_ids))
            # Store the state we just produced so the next repetition can hit.
            cache.store(prefix_ids, [layer.state for layer in layers], digest=digest)
        metrics = cache.metrics
        cells.append(
            {
                "label": label,
                "digest": digest,
                "expect_hit": expect_hit,
                # The first repetition is the one that must miss after a
                # world-state change: later repetitions legitimately hit,
                # because the first one already stored the new world's
                # state. Recording only the last would hide the miss.
                "matched_prefix_tokens": matched_per_rep[0],
                "matched_prefix_tokens_per_repetition": matched_per_rep,
                "prefix_tokens": len(prefix_ids),
                "prefill_ms_median": round(sorted(timings)[len(timings) // 2], 3),
                "prefill_ms_samples": [round(t, 3) for t in timings],
                "hits": metrics.hits,
                "misses": metrics.misses,
                "restores": metrics.restores,
                "invalidations": metrics.invalidations,
                "stored_entries": metrics.stored_entries,
                "note": note,
            }
        )
        cell = cells[-1]
        print(
            f"  {label:<34} matched(rep1)={cell['matched_prefix_tokens']:>3}/{cell['prefix_tokens']}  "
            f"prefill={cell['prefill_ms_median']:>8.2f} ms  hits={metrics.hits} misses={metrics.misses}"
        )

    print(f"model: {MODEL_ID}")
    print(f"prefix tokens: {len(prefix_ids)}   repetitions: {args.repetitions}\n")

    record("1. world A, first contact", digest_world_a, True, "cold: nothing stored yet")
    record("2. world A, unchanged", digest_world_a, True, "must hit: same world, same prefix")
    record(
        "3. world B, SAME prompt text",
        digest_world_b,
        False,
        "MUST NOT hit: identical text, changed world state",
    )
    record("4. world B, unchanged", digest_world_b, True, "must hit again under world B")

    by = {c["label"]: c for c in cells}
    a_unchanged = by["2. world A, unchanged"]
    b_changed = by["3. world B, SAME prompt text"]
    b_unchanged = by["4. world B, unchanged"]

    # The assertions that make this a measurement rather than a demo.
    assert a_unchanged["matched_prefix_tokens"] == len(prefix_ids), (
        "world A unchanged must reuse the whole prefix"
    )
    assert b_changed["matched_prefix_tokens"] == 0, (
        "changed world state must NOT reuse the identical prompt's KV state"
    )
    assert b_unchanged["matched_prefix_tokens"] == len(prefix_ids), (
        "world B must hit again once it is the current world"
    )

    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "requirement": "REQ-PERF-033 / #240 config D: world-state change must invalidate reuse",
        "model": MODEL_ID,
        "digest_model": "digest carries tool-schema + world-state; gate_digest restricts candidates",
        "repetitions": args.repetitions,
        "cells": cells,
        "verdict": {
            "hit_when_world_unchanged": True,
            "miss_when_world_changed": True,
            "rehit_under_new_world": True,
            "matched_tokens_unchanged_world": a_unchanged["matched_prefix_tokens"],
            "matched_tokens_changed_world": b_changed["matched_prefix_tokens"],
        },
    }
    print(
        "\nverdict: identical prompt text under a changed world digest matched "
        f"{b_changed['matched_prefix_tokens']} prefix tokens (expected 0)"
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
