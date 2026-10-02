#!/usr/bin/env python3
"""Stage E: repeated-prefix prefill measurement.

Simulates the planned GPT-5.6 → OAI-2.0 → MiniMax training-loop shape:
the system prefix (teaching context, tool schemas, repository policy)
is stable across N exercises, and only the suffix (the exercise question)
changes. With a real KV cache the second-through-Nth prefill would be
amortized; without one, every request re-pays the prefix cost.

This script:
  1. Builds a stable teaching prefix (configurable token count).
  2. Submits N requests sharing the prefix + unique suffix to the
     canonical surface (one SessionCompatibilityKey so they batch).
  3. Records prefill_ms / decode_ms / decode_tps for each.
  4. Computes the "amortization" saving: how much prefill work would be
     saved if the prefix were KV-cached (i.e. subsequent prefill cost
     drops to 0).

Reports the result as JSON to stdout. Does NOT require GPT-5.6 or
MiniMax to be live — the model under test is oai-2.0 and the
"verifier" is the actual prefill-time measurement.

Run with::

    OAI2_GATEWAY_API_KEY=... uv run python scripts/bench_training_prefix.py \
        --base-url http://127.0.0.1:9100 \
        --prefix-tokens 1024 \
        --suffix-tokens 64 \
        --repetitions 8
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import httpx


def _build_prompt(target_words: int, *, prefix: bool) -> str:
    """Build a deterministic prompt of approximately ``target_words`` words."""
    base = (
        "You are being taught a stable repository policy. Read carefully and "
        "apply the same rules to every exercise. "
    )
    if target_words <= 0:
        return ""
    target_words = max(1, target_words)
    words = (base * (target_words // len(base.split()) + 2)).split()
    text = " ".join(words[:target_words])
    if not prefix:
        # Suffix gets a per-rep marker so each request has a unique
        # ``Exercise N:`` token.
        return f"\n\nExercise #{uuid.uuid4().hex[:6]}: answer the question."
    return text


@dataclass(slots=True)
class TrainingPrefixSample:
    rep: int
    request_id: str
    prefill_ms: float
    decode_ms: float
    decode_tokens_per_second: float | None
    generated_tokens: int | None
    end_to_end_ms: float
    accepted_speculative_tokens: int = 0  # not used; reserved for MTP future


async def _submit_and_drain(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    prefix_prompt: str,
    suffix_prompt: str,
    rep: int,
    max_tokens: int,
    shared_client: bool,
    shared_security_context: str | None,
) -> TrainingPrefixSample:
    session_id = f"stageE-sess-{rep}-{uuid.uuid4().hex[:6]}"
    request_id = f"stageE-req-{rep}-{uuid.uuid4().hex[:6]}"
    combined = prefix_prompt + suffix_prompt
    full_digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    submitted_at_ms = time.time() * 1000.0

    # Open a session.
    client_id = (
        "stageE-shared"
        if shared_client
        else f"stageE-client-{rep}-{uuid.uuid4().hex[:6]}"
    )
    r = await client.post(
        f"{base_url}/v1/sessions",
        json={"client_id": client_id, "session_id": session_id},
        timeout=300.0,
    )
    if r.status_code != 201:
        raise RuntimeError(f"open_session failed: {r.status_code} {r.text}")

    body = {
        "client_id": client_id,
        "session_id": session_id,
        "request_id": request_id,
        "prompt": combined,
        "model_id": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        "tokenizer_version": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        "prefix_digest": full_digest,  # same digest → same key for same combined prompt
        "tool_schema_version": "default",
        "world_state_version": "default",
    }
    if shared_security_context is not None:
        body["security_context"] = shared_security_context
    r = await client.post(
        f"{base_url}/v1/inference",
        json=body,
        timeout=300.0,
    )
    if r.status_code != 202:
        raise RuntimeError(f"submit_inference failed: {r.status_code} {r.text}")

    # Drain until our request returns.
    collected: dict[str, dict[str, object]] = {}
    for _ in range(64):
        r = await client.post(f"{base_url}/v1/inference/drain", timeout=300.0)
        if r.status_code != 200:
            raise RuntimeError(f"drain failed: {r.status_code} {r.text}")
        body = r.json()
        for result in body.get("results", []):
            rid = str(result.get("request_id", ""))
            if rid == request_id:
                collected[rid] = result
        if request_id in collected:
            break
    completed_at_ms = time.time() * 1000.0
    result = collected.get(request_id)
    if result is None:
        raise RuntimeError(f"no drain result for {request_id}")
    return TrainingPrefixSample(
        rep=rep,
        request_id=request_id,
        prefill_ms=float(result.get("prefill_ms") or 0.0),
        decode_ms=float(result.get("decode_ms") or 0.0),
        decode_tokens_per_second=result.get("decode_tokens_per_second"),
        generated_tokens=result.get("generated_tokens"),
        end_to_end_ms=completed_at_ms - submitted_at_ms,
    )


def _stat(values: Iterable[float]) -> dict[str, float | int | None]:
    nums = [v for v in values if v is not None]
    if not nums:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "stddev": None}
    n = len(nums)
    return {
        "count": n,
        "mean": statistics.fmean(nums),
        "median": statistics.median(nums),
        "min": min(nums),
        "max": max(nums),
        "stddev": statistics.pstdev(nums) if n > 1 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:9100")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")
    parser.add_argument("--prefix-tokens", type=int, default=1024,
                        help="Approximate prefix word count (stable across reps).")
    parser.add_argument("--suffix-tokens", type=int, default=64,
                        help="Approximate suffix word count (changes per rep).")
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--shared-client", action="store_true")
    parser.add_argument("--shared-security-context", default=None)
    args = parser.parse_args(argv)

    # Build the stable prefix ONCE.
    prefix_prompt = _build_prompt(args.prefix_tokens, prefix=True)
    samples: list[TrainingPrefixSample] = []

    async def run() -> None:
        async with httpx.AsyncClient() as client:
            for rep in range(args.repetitions):
                # New suffix per rep, so each exercise is distinct
                # but the prefix is identical.
                suffix_prompt = _build_prompt(args.suffix_tokens, prefix=False) + f" [rep={rep}]"
                sample = await _submit_and_drain(
                    client=client,
                    base_url=args.base_url,
                    prefix_prompt=prefix_prompt,
                    suffix_prompt=suffix_prompt,
                    rep=rep,
                    max_tokens=args.max_tokens,
                    shared_client=args.shared_client,
                    shared_security_context=args.shared_security_context,
                )
                samples.append(sample)

    asyncio.run(run())

    # Aggregate.
    prefill_stats = _stat([s.prefill_ms for s in samples])
    decode_stats = _stat([s.decode_ms for s in samples])
    tps_stats = _stat([s.decode_tokens_per_second for s in samples])
    e2e_stats = _stat([s.end_to_end_ms for s in samples])

    # Counterfactual: with a real KV cache the 2nd-through-Nth prefill
    # cost would be ~0 (only the suffix tokens would re-prefill).
    # The current implementation pays full prefill every time.
    total_prefill_ms = sum(s.prefill_ms for s in samples)
    if samples:
        first_prefill = samples[0].prefill_ms
        # Subsequent reps re-prefill the full combined prompt because
        # mlx_lm 0.32 has no stable KV handle. The 2nd-onward
        # prefill time is the model's actual re-prefill cost. The
        # counterfactual cache-only-prefix cost is the proportion of
        # the prompt that is the *prefix*; with a real KV cache, the
        # prefix is paid once and each subsequent call only pays for
        # the new suffix tokens.
        total_prompt_words = args.prefix_tokens + args.suffix_tokens
        prefix_share = args.prefix_tokens / total_prompt_words if total_prompt_words else 0.0
        subsequent_median_prefill = statistics.median(
            [s.prefill_ms for s in samples[1:]]
        ) if len(samples) > 1 else 0.0
        suffix_prefill_estimate = subsequent_median_prefill * (1.0 - prefix_share)
        # If the prefix were cached after the first request, the
        # remaining N-1 prefill costs would drop to suffix-only.
        counterfactual_total_prefill_ms = (
            first_prefill + max(0, len(samples) - 1) * suffix_prefill_estimate
        )
    else:
        first_prefill = 0.0
        suffix_prefill_estimate = 0.0
        counterfactual_total_prefill_ms = 0.0
    saved_ms = total_prefill_ms - counterfactual_total_prefill_ms
    saved_pct = (
        100.0 * saved_ms / total_prefill_ms if total_prefill_ms > 0 else 0.0
    )

    summary = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "prefix_words_target": args.prefix_tokens,
            "suffix_words_target": args.suffix_tokens,
            "repetitions": args.repetitions,
            "max_tokens": args.max_tokens,
            "shared_client": args.shared_client,
            "shared_security_context": args.shared_security_context,
        },
        "samples": [asdict(s) for s in samples],
        "aggregate": {
            "prefill_ms": prefill_stats,
            "decode_ms": decode_stats,
            "decode_tokens_per_second": tps_stats,
            "end_to_end_ms": e2e_stats,
        },
        "counterfactual_cache_savings": {
            "current_total_prefill_ms": total_prefill_ms,
            "first_request_prefill_ms": first_prefill,
            "median_subsequent_prefill_ms": suffix_prefill_estimate,
            "counterfactual_total_prefill_ms": counterfactual_total_prefill_ms,
            "saved_ms": saved_ms,
            "saved_pct": saved_pct,
        },
        "limitations": [
            "mlx_lm 0.32 does not expose stable KV handles; prefill is recomputed every request",
            "This measures the cost that WOULD be saved with a real prefix KV cache",
            "Suffix prefill estimate is the median of observed prefill minus the first (full-prefix) prefill",
        ],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
