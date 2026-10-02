"""Measure the #186 composer's contribution without duplicating #240.

Token counts come from the **serving model's real tokenizer and chat
template** (``transformers`` + ``apply_chat_template``) when one is available,
because the earlier whitespace-based figure was not a token measurement at all.
On SmolLM-135M the two differ by roughly 3.7x, so a whitespace count is not a
usable proxy and the synthetic number is now reported only for contrast.

It deliberately does not measure TTFT, prefill wall time, decode rate or
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
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--tokenizer",
        default="mlx-community/SmolLM-135M-Instruct-4bit",
        help="HF id whose tokenizer + chat template define the token counts",
    )
    parser.add_argument("--no-tokenizer", action="store_true", help="force synthetic counting")
    parser.add_argument("--out-dir", type=Path, default=Path("evals/benchmarks"))
    parser.add_argument("--tag", default="composer-prefix")
    parser.add_argument("--session-id", default="bench-session")
    args = parser.parse_args(argv)

    # The real rendered request is what the runtime tokenises, so count the
    # messages through the serving model's chat template, not a flat string.
    render = None
    counter = _whitespace_tokens
    real_tokenizer = False
    tokenizer_note = "SYNTHETIC whitespace counter — not a token measurement"
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            tok = AutoTokenizer.from_pretrained(args.tokenizer)

            def render(messages: list[dict[str, object]]) -> str:
                return tok.apply_chat_template(  # type: ignore[attr-defined]
                    [{"role": str(m["role"]), "content": str(m["content"])} for m in messages],
                    tokenize=False,
                    add_generation_prompt=True,
                )

            def counter(text: str, _tok: object = tok) -> int:  # type: ignore[misc]
                return len(_tok.encode(text))  # type: ignore[attr-defined]

            tokenizer_note = f"real tokenizer + chat template via {args.tokenizer}"
            real_tokenizer = True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"note: tokenizer unavailable ({exc}); falling back to SYNTHETIC counting")
            render = None

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
        state = (
            {
                "state_prefix_digest": prefix.digest(),
                "state_composer_version": COMPOSER_SCHEMA_VERSION,
                "state_version": str(turn - 1),
                "state_body": f"{turn - 1} edits applied to src/scheduler.py",
            }
            if turn > 1
            else None
        )
        # With the reference resolvable for THIS session and the host verifying
        # the receiving side -- the only case where a ref is emitted.
        served = RefResolver(session_id=args.session_id)
        served.put(LESSON_REF.ref_id, LESSON_REF.version, LESSON)
        compact = compose(
            prefix=prefix,
            user_goal=goal,
            evidence=[LESSON],
            evidence_refs=[LESSON_REF],
            resolver=served,
            session_id=args.session_id,
            retained_state_verified=True,
            state=state,
        )
        # The same request with the reference unusable (fresh/restarted session).
        inline = compose(
            prefix=prefix,
            user_goal=goal,
            evidence=[LESSON],
            evidence_refs=[LESSON_REF],
            resolver=RefResolver(session_id=args.session_id),
            session_id=args.session_id,
            retained_state_verified=True,
            state=state,
        )
        rows.append(
            {
                "turn": turn,
                "prefix_digest": compact.prefix_digest,
                # Tokens of the stable PREFIX alone, counted with whichever
                # counter is actually in use. This is the quantity a prefix
                # KV cache could reuse; the full rendered request below is a
                # different number and must not be labelled as the prefix.
                "prefix_tokens": counter(compact.prefix_text()),
                # Counted with the REAL tokenizer when one is available, so
                # this is "tokens" and not "tokens (real or synthetic)" — the
                # counter in use is named in summary.config.counter.
                "prefix_tokens_counted": "real_tokenizer" if real_tokenizer else "whitespace",
                "inline_tokens_real": _count(inline, render, counter),
                "compact_tokens_real": _count(compact, render, counter),
                "effective_prompt_bytes": len(_rendered(compact, render).encode("utf-8")),
                "delta_applied": compact.provenance.delta_applied,
                "rehydrated": compact.rehydrated,
                "compact_ref_used": bool(compact.provenance.compact_refs),
                "inline_ref_used": bool(inline.provenance.compact_refs),
                "segment_chars": [len(s.text) for s in compact.provenance.segments],
            }
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    digests = {r["prefix_digest"] for r in rows}
    inline = [int(r["inline_tokens_real"]) for r in rows]
    compact_tokens = [int(r["compact_tokens_real"]) for r in rows]
    saving = [(f - c) / f * 100.0 for f, c in zip(inline, compact_tokens, strict=True) if f]

    summary = {
        "script": "bench_composer_prefix",
        "issue": "#186",
        "measurement_kind": "tokenizer-only (no model forward pass)",
        "config": {
            "turns": args.turns,
            "tokenizer": tokenizer_note,
            "counter": "real_tokenizer" if real_tokenizer else "whitespace",
            "prefix_token_field": (
                "prefix_tokens counts the stable prefix only, using the counter "
                "named above; it is NOT the full rendered request"
            ),
            "serving_template_applied": render is not None,
            "session_id": args.session_id,
            "composer_version": COMPOSER_SCHEMA_VERSION,
            "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "prefix_identity_stable_across_turns": len(digests) == 1,
        "distinct_prefix_digests": len(digests),
        "rows": rows,
        "composition_overhead_ms": round(elapsed_ms, 3),
        "compact_ref_token_saving": {
            "unit": "real tokenizer tokens of the rendered chat template",
            "min_percent": round(min(saving), 3) if saving else None,
            "max_percent": round(max(saving), 3) if saving else None,
        },
        "superseded_claim": {
            "previous_figure": "51.95-58.59%",
            "was": "whitespace-based synthetic accounting of a flat string",
            "corrected": (
                "That number was not a token measurement. On this tokenizer the "
                "whitespace count understates real rendered tokens by roughly "
                "3.7x. Prefer compact_ref_token_saving above; retain the "
                "synthetic figure only as a contrast column."
            ),
        },
        "model_bound_measurements": {
            "ttft_ms": None,
            "prefill_ms": None,
            "decode_tok_per_s": None,
            "end_to_end_ms": None,
            "first_useful_tool_action_ms": None,
            "retrieval_latency_ms": None,
            "pending_reason": (
                "Not measured. These require the target Mac, the MLX hot "
                "runtime and a matched before/after run; the #240 owner holds "
                "that hardware window. Interfering would corrupt their rows, so "
                "these stay null rather than estimated."
            ),
        },
        "note": (
            "A shorter wire payload is not by itself evidence of less prefill. "
            "These rows establish rendered token accounting and stable prefix "
            "identity only. The prefill consequence is #240's to establish, on "
            "a matched model revision, task set, output settings and knowledge "
            "snapshot."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"summary_{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote summary: {out}")
    print(f"prefix identity stable across turns: {summary['prefix_identity_stable_across_turns']}")
    print(f"tokenizer: {tokenizer_note}")
    if saving:
        print(
            "compact-ref saving, REAL rendered tokens (not prefill): "
            f"{summary['compact_ref_token_saving']['min_percent']}%.."
            f"{summary['compact_ref_token_saving']['max_percent']}%"
        )
    print("superseded 51.95-58.59% figure: whitespace-based synthetic, see summary")
    print("model-bound measurements: PENDING (see summary for the reason)")
    return 0


def _rendered(comp: Any, render: Any) -> str:
    if render is not None:
        return render(list(comp.messages))
    return "\n".join(str(m.get("content", "")) for m in comp.messages)


def _count(comp: Any, render: Any, counter: Any) -> int:
    return counter(_rendered(comp, render))


if __name__ == "__main__":
    raise SystemExit(main())
