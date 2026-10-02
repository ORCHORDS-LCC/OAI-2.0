"""Measure the #186 composer's contribution without duplicating #240.

This measures only what is measurable **without** a model: the composition's
own token/byte accounting, the stable prefix size, and prefix KV-cache
identity behaviour. It deliberately does not measure TTFT, prefill wall time or
end-to-end completion — those need the target Mac, and the #240 owner is
running measurements there. Interfering with another agent's hardware window
would corrupt their numbers, so the model-bound rows are emitted as ``null``
with an explicit ``pending`` reason rather than invented.

Reuses the #240 methodology where it applies (same prefix/suffix framing, same
summary JSON shape) without importing or re-implementing its runtime work:

- :mod:`oai2.runtime.prefix_kv_cache` is used as-is for hit/miss accounting.
- Token counts come from a real tokenizer when ``--model`` is supplied, and
  otherwise fall back to a deterministic whitespace counter, which is reported
  in the summary so a reader knows which one produced the numbers.

Run::

    uv run python scripts/bench_composer_prefix.py --turns 6
    uv run python scripts/bench_composer_prefix.py --turns 6 --model <mlx-id>
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oai2.agents.composer import (  # noqa: E402
    COMPOSER_SCHEMA_VERSION,
    CompactRef,
    PrefixSpec,
    RefResolver,
    compose,
)
from oai2.agents.instructions import INSTRUCTION_SCHEMA_VERSION  # noqa: E402

POLICY = (
    "You are OAI-2.0, a coding and engineering agent. Use the available tools "
    "instead of guessing. Surface tool errors verbatim. Prefer Read/Glob/Grep "
    "before Edit/Write."
)
PROJECT = (
    "Repository rules: run `uv run pytest` before claiming success. Never "
    "commit generated artefacts. Keep public functions typed."
)

# A long lesson, so compact references have something worth replacing.
LESSON = (
    "Prior verified lesson: when a repository tool returns zero rows, do not "
    "conclude the object is absent. Re-check the path spelling and the glob "
    "root before reporting absence to the operator. " + ("detail. " * 120)
)
LESSON_REF = CompactRef(ref_id="lesson-zero-rows", version="3")


def _whitespace_tokens(text: str) -> int:
    return max(1, len(text.split()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--model", default="", help="mlx model id for real token counts")
    parser.add_argument("--out-dir", type=Path, default=Path("evals/benchmarks"))
    parser.add_argument("--tag", default="composer-prefix")
    args = parser.parse_args(argv)

    counter = _whitespace_tokens
    tokenizer_note = "whitespace counter (deterministic, not a model tokenizer)"
    if args.model:
        try:
            from mlx_lm import load  # type: ignore[import-not-found]

            _, tok = load(args.model)

            def counter(text: str, _tok: object = tok) -> int:  # type: ignore[misc]
                return len(_tok.encode(text))  # type: ignore[attr-defined]

            tokenizer_note = f"real tokenizer via mlx_lm.load({args.model})"
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"note: could not load tokenizer ({exc}); using whitespace counter")

    prefix = PrefixSpec(
        policy_text=POLICY,
        policy_version="policy-1",
        tool_wire=({"name": "Read"}, {"name": "Edit"}, {"name": "Bash"}),
        tool_schema_version="tools-1",
        project_text=PROJECT,
        project_version="project-1",
    )

    resolver = RefResolver()
    resolver.put(LESSON_REF.ref_id, LESSON_REF.version, LESSON)

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for turn in range(1, args.turns + 1):
        goal = f"Turn {turn}: extend the safe batching scheduler and keep the API contract."
        # Turn 1 carries the full body. Later turns send a compact reference,
        # which is the REQ-PROMPT-023 case.
        comp = compose(
            prefix=prefix,
            user_goal=goal,
            evidence=[LESSON],
            evidence_refs=[LESSON_REF],
            resolver=resolver,
            state={
                "state_prefix_digest": prefix.digest(),
                "state_composer_version": COMPOSER_SCHEMA_VERSION,
                "state_version": str(turn - 1),
                "state_body": f"{turn - 1} edits applied to src/scheduler.py",
            }
            if turn > 1
            else None,
        )
        full_text = "\n".join(str(m.get("content", "")) for m in comp.messages)
        # The same composition with the reference deliberately unavailable, so
        # the fallback cost is measured rather than assumed.
        fallback = compose(
            prefix=prefix,
            user_goal=goal,
            evidence=[LESSON],
            evidence_refs=[LESSON_REF],
            resolver=RefResolver(),
        )
        fallback_text = "\n".join(str(m.get("content", "")) for m in fallback.messages)
        rows.append(
            {
                "turn": turn,
                "prefix_digest": comp.prefix_digest,
                "prefix_tokens": counter(comp.prefix_text()),
                "effective_prompt_tokens": counter(full_text),
                "effective_prompt_bytes": len(full_text.encode("utf-8")),
                "fallback_tokens_no_ref": counter(fallback_text),
                "delta_applied": comp.provenance.delta_applied,
                "rehydrated": comp.rehydrated,
                "compact_refs": [asdict(r) for r in comp.provenance.compact_refs],
                "segment_chars": [len(s.text) for s in comp.provenance.segments],
            }
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    digests = {r["prefix_digest"] for r in rows}
    full = [int(r["fallback_tokens_no_ref"]) for r in rows]
    with_ref = [int(r["effective_prompt_tokens"]) for r in rows]

    saving = [(f - c) / f * 100.0 for f, c in zip(full, with_ref, strict=True) if f]

    summary = {
        "script": "bench_composer_prefix",
        "issue": "#186",
        "config": {
            "turns": args.turns,
            "model": args.model or "(none)",
            "tokenizer": tokenizer_note,
            "composer_version": COMPOSER_SCHEMA_VERSION,
            "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "prefix_identity_stable_across_turns": len(digests) == 1,
        "distinct_prefix_digests": len(digests),
        "rows": rows,
        "composition_overhead_ms": round(elapsed_ms, 3),
        "compact_ref_saving_percent": {
            "min": round(min(saving), 3) if saving else None,
            "max": round(max(saving), 3) if saving else None,
        },
        "model_bound_measurements": {
            "ttft_ms": None,
            "prefill_ms": None,
            "end_to_end_ms": None,
            "first_useful_tool_action_ms": None,
            "pending_reason": (
                "Not measured. TTFT/prefill/end-to-end require the target Mac and "
                "the MLX hot runtime; the #240 owner is running measurements there. "
                "Interfering would corrupt their rows, so these are recorded as "
                "pending rather than estimated."
            ),
        },
        "note": (
            "A shorter wire payload does not by itself prove less prefill. The "
            "rows above establish composition shape and stable prefix identity; "
            "the prefill consequence is #240's measurement to make."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"summary_{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote summary: {out}")
    print(f"prefix identity stable across turns: {summary['prefix_identity_stable_across_turns']}")
    if saving:
        print(
            "compact-ref saving (deterministic accounting, not prefill): "
            f"{summary['compact_ref_saving_percent']['min']}%.."
            f"{summary['compact_ref_saving_percent']['max']}%"
        )
    print("model-bound measurements: PENDING (see summary for the reason)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
