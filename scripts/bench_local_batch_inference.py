#!/usr/bin/env python3
"""Concurrent load benchmark against the canonical local batch-inference app.

Uses the /v1/sessions, /v1/inference and /v1/inference/drain protocol
defined by ``oai2.runtime.service_binding``. Each agent opens its own
session, submits a request, and the drain call executes the underlying
``MLXHotRuntime`` for every queued request.

Use:

- 1 agent → baseline hot latency.
- N=2/4/8 agents → concurrency matrix for WI-PERF-003 / #240.

Run with:    uv run python scripts/bench_local_batch_inference.py [args]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx


@dataclass(slots=True)
class AgentSample:
    agent_id: int
    client_id: str
    session_id: str
    request_id: str
    prompt_label: str
    prompt_tokens: int
    generated_tokens: int
    queue_wait_ms: float
    prefill_ms: float
    decode_ms: float
    decode_tokens_per_second: float | None
    text_excerpt: str
    end_to_end_ms: float
    submitted_at_ms: float
    completed_at_ms: float
    notes: list[str] = field(default_factory=list)


def _build_prompt(target_tokens: int) -> tuple[str, int]:
    """Mirror ``scripts/bench.py._build_prompt`` for matched workloads."""
    base = (
        "Summarize the following Python module. Focus on public functions, "
        "their inputs and outputs, error conditions, and any side effects. "
        "Be concise and accurate. "
    )
    if target_tokens <= 0:
        return "", 0
    target_words = max(1, int(target_tokens / 1.3))
    words = (base * (target_words // len(base.split()) + 2)).split()
    text = " ".join(words[:target_words])
    return text, len(text.split())


def _stat(values: Iterable[float | None]) -> dict[str, float | int | None]:
    nums = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
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


def _parse_note_ms(notes: list[str], key: str) -> float:
    prefix = f"{key}="
    for note in notes:
        if note.startswith(prefix):
            try:
                return float(note[len(prefix):]) * 1000.0
            except ValueError:
                return 0.0
    return 0.0


def _parse_note_float(notes: list[str], key: str) -> float | None:
    prefix = f"{key}="
    for note in notes:
        if note.startswith(prefix):
            try:
                return float(note[len(prefix):])
            except ValueError:
                return None
    return None


async def _submit_one(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    agent_id: int,
    prompt: str,
    model_id: str,
    unique_prompts: bool,
) -> tuple[str, str, str, str, int, float]:
    """Open session + submit one request; return identifiers for later matching."""
    client_id = f"bench-client-{agent_id}-{uuid.uuid4().hex[:6]}"
    session_id = f"bench-sess-{agent_id}-{uuid.uuid4().hex[:6]}"
    request_id = f"bench-req-{agent_id}-{uuid.uuid4().hex[:6]}"
    effective_prompt = (
        f"{prompt}\n\n[agent_id={agent_id}]" if unique_prompts else prompt
    )
    prefix_digest = hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest()
    submitted_at_ms = time.time() * 1000.0

    r = await client.post(
        f"{base_url}/v1/sessions",
        json={"client_id": client_id, "session_id": session_id},
        timeout=300.0,
    )
    if r.status_code != 201:
        raise RuntimeError(f"open_session failed: {r.status_code} {r.text}")
    r = await client.post(
        f"{base_url}/v1/inference",
        json={
            "client_id": client_id,
            "session_id": session_id,
            "request_id": request_id,
            "prompt": effective_prompt,
            "model_id": model_id,
            "tokenizer_version": model_id,
            "prefix_digest": prefix_digest,
            "tool_schema_version": "default",
            "world_state_version": "default",
        },
        timeout=300.0,
    )
    if r.status_code != 202:
        raise RuntimeError(f"submit_inference failed: {r.status_code} {r.text}")
    return (
        request_id,
        client_id,
        session_id,
        effective_prompt,
        len(effective_prompt.split()),
        submitted_at_ms,
    )


async def _drain_all(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    expected_request_ids: set[str],
    max_drains: int = 64,
) -> dict[str, dict[str, object]]:
    """Drain repeatedly until every expected request_id is collected.

    Each :func:`BatchInferenceSurface.drain` call pops one
    compatibility-exact batch (up to ``max_batch_size``). With
    N distinct compatibility keys, we need at least N drain calls
    to collect all N results; this loops until the set of outstanding
    request_ids is empty or ``max_drains`` is exhausted.
    """
    collected: dict[str, dict[str, object]] = {}
    for _ in range(max_drains):
        if not expected_request_ids - collected.keys():
            break
        r = await client.post(f"{base_url}/v1/inference/drain", timeout=300.0)
        if r.status_code != 200:
            raise RuntimeError(f"drain failed: {r.status_code} {r.text}")
        body = r.json()
        for result in body.get("results", []):
            rid = str(result.get("request_id", ""))
            if rid in expected_request_ids and rid not in collected:
                collected[rid] = result
    return collected


async def _concurrent_agents(
    *,
    base_url: str,
    n_agents: int,
    prompt: str,
    prompt_label: str,
    model_id: str,
    max_tokens: int,
    unique_prompts: bool,
) -> list[AgentSample]:
    async with httpx.AsyncClient() as client:
        # Phase 1: open + submit all agents in parallel.
        submissions = await asyncio.gather(
            *[
                _submit_one(
                    client=client,
                    base_url=base_url,
                    agent_id=agent_id,
                    prompt=prompt,
                    model_id=model_id,
                    unique_prompts=unique_prompts,
                )
                for agent_id in range(n_agents)
            ]
        )
        request_ids = {sub[0] for sub in submissions}
        # Phase 2: drain until every expected request has a result.
        collected = await _drain_all(
            client=client,
            base_url=base_url,
            expected_request_ids=request_ids,
        )

    # Phase 3: assemble per-agent samples from collected results.
    samples: list[AgentSample] = []
    for sub in submissions:
        (
            request_id,
            client_id,
            session_id,
            effective_prompt,
            prompt_tokens,
            submitted_at_ms,
        ) = sub
        agent_id = int(request_id.split("-")[2])
        result = collected.get(request_id)
        completed_at_ms = time.time() * 1000.0
        if result is None:
            raise RuntimeError(f"no drain result for {request_id} after collecting {len(collected)}/{len(request_ids)}")
        text = str(result.get("text", ""))
        queue_wait_ms = float(result.get("queue_wait_ms", 0.0))
        samples.append(
            AgentSample(
                agent_id=agent_id,
                client_id=client_id,
                session_id=session_id,
                request_id=request_id,
                prompt_label=prompt_label,
                prompt_tokens=prompt_tokens,
                generated_tokens=len(text.split()),
                queue_wait_ms=queue_wait_ms,
                prefill_ms=0.0,
                decode_ms=0.0,
                decode_tokens_per_second=None,
                text_excerpt=text[:120],
                end_to_end_ms=completed_at_ms - submitted_at_ms,
                submitted_at_ms=submitted_at_ms,
                completed_at_ms=completed_at_ms,
                notes=[],
            )
        )
    return samples


async def _concurrent_agents(
    *,
    base_url: str,
    n_agents: int,
    prompt: str,
    prompt_label: str,
    model_id: str,
    max_tokens: int,
    unique_prompts: bool,
) -> list[AgentSample]:
    async with httpx.AsyncClient() as client:
        # Phase 1: open + submit all agents in parallel.
        submissions = await asyncio.gather(
            *[
                _submit_one(
                    client=client,
                    base_url=base_url,
                    agent_id=agent_id,
                    prompt=prompt,
                    model_id=model_id,
                    unique_prompts=unique_prompts,
                )
                for agent_id in range(n_agents)
            ]
        )
        request_ids = {sub[0] for sub in submissions}
        # Phase 2: drain until every expected request has a result.
        collected = await _drain_all(
            client=client,
            base_url=base_url,
            expected_request_ids=request_ids,
        )

    # Phase 3: assemble per-agent samples from collected results.
    samples: list[AgentSample] = []
    for sub in submissions:
        (
            request_id,
            client_id,
            session_id,
            effective_prompt,
            prompt_tokens,
            submitted_at_ms,
        ) = sub
        agent_id = int(request_id.split("-")[2])
        result = collected.get(request_id)
        completed_at_ms = time.time() * 1000.0
        if result is None:
            raise RuntimeError(
                f"no drain result for {request_id} after collecting "
                f"{len(collected)}/{len(request_ids)}"
            )
        text = str(result.get("text", ""))
        queue_wait_ms = float(result.get("queue_wait_ms", 0.0))
        samples.append(
            AgentSample(
                agent_id=agent_id,
                client_id=client_id,
                session_id=session_id,
                request_id=request_id,
                prompt_label=prompt_label,
                prompt_tokens=prompt_tokens,
                generated_tokens=len(text.split()),
                queue_wait_ms=queue_wait_ms,
                prefill_ms=0.0,
                decode_ms=0.0,
                decode_tokens_per_second=None,
                text_excerpt=text[:120],
                end_to_end_ms=completed_at_ms - submitted_at_ms,
                submitted_at_ms=submitted_at_ms,
                completed_at_ms=completed_at_ms,
                notes=[],
            )
        )
    return samples


def _aggregate(samples: list[AgentSample]) -> dict[str, dict[str, float | int | None]]:
    metric_names = [
        "end_to_end_ms",
        "queue_wait_ms",
        "generated_tokens",
    ]
    out: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        values = [getattr(s, name) for s in samples]
        out[name] = _stat(values)
    return out


def _print_table(n_agents: int, samples: list[AgentSample], agg: dict[str, dict[str, float | int | None]]) -> None:
    print(f"\n=== agents={n_agents}  |  samples={len(samples)} ===")
    for s in sorted(samples, key=lambda s: s.agent_id):
        print(
            f"  agent {s.agent_id:>2}: "
            f"e2e={s.end_to_end_ms:7.1f}ms "
            f"queue_wait={s.queue_wait_ms:6.1f}ms "
            f"gen={s.generated_tokens:4d}tok"
        )
    print(f"  AGG: {agg}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:9100")
    parser.add_argument(
        "--n-agents",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="One or more concurrency levels (default: 1 2 4 8).",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        help="Model id the server loaded (must match what is wired into the surface).",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[128, 1024, 4096],
        help="Approximate prompt token budgets (mirrors scripts/bench.py).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Independent run blocks per (n_agents, prompt_tokens) cell.",
    )
    parser.add_argument(
        "--unique-prompts",
        action="store_true",
        help="Suffix each agent's prompt with its agent id so the prefix digest differs.",
    )
    parser.add_argument("--tag", default="after-hot-canonical")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evals" / "benchmarks",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "tag": args.tag,
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "unique_prompts": args.unique_prompts,
        "cells": [],
    }

    for budget in args.prompt_tokens:
        prompt, word_count = _build_prompt(budget)
        prompt_label = f"~{budget}tokens ({word_count}words)"
        for n_agents in args.n_agents:
            cell_samples: list[AgentSample] = []
            for rep in range(args.repetitions):
                print(
                    f"--- n_agents={n_agents} prompt={prompt_label} rep={rep + 1}/{args.repetitions} ---",
                    file=sys.stderr,
                )
                samples = asyncio.run(
                    _concurrent_agents(
                        base_url=args.base_url,
                        n_agents=n_agents,
                        prompt=prompt,
                        prompt_label=prompt_label,
                        model_id=args.model,
                        max_tokens=args.max_tokens,
                        unique_prompts=args.unique_prompts,
                    )
                )
                cell_samples.extend(samples)
            agg = _aggregate(cell_samples)
            _print_table(n_agents, cell_samples, agg)
            summary["cells"].append(  # type: ignore[attr-defined]
                {
                    "prompt_label": prompt_label,
                    "n_agents": n_agents,
                    "samples": [asdict(s) for s in cell_samples],
                    "aggregate": agg,
                }
            )

    out_path = out_dir / f"summary_{args.tag}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote summary: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
