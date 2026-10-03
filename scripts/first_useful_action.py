#!/usr/bin/env python3
"""Time to first *verified-correct* output — the "first useful action" column.

AC-PERF-034 asks the report to quantify TTFT, **first useful action**,
prefill and end-to-end task time "without mislabeling them as decode-speed
gains". Three of the four are already measured. This is the fourth.

Why it is not TTFT
------------------

TTFT is time to first byte. A model can emit its first token in under 10 ms
and still be wrong, and at the measured 4/12 verified-correct baseline most
first tokens are worthless. Earlier work on this issue made the same point
about ``days in a week``: 23.9 ms to a response at a 13% verified-correct
rate, implying ~183 ms expected time to first *correct* response. That
figure was never measured by a harness, only reasoned about.

So this script measures the quantity directly:

    for each case, issue attempts until one satisfies the case's oracle,
    recording TTFT and wall time for every attempt, and report the
    cumulative time at which a verified-correct output first exists.

Everything is judged by the suite's own regex oracle — the same
machine-checkable predicate used for the 4/12 accuracy baseline. No model
or human grades the output, so "useful" here means "provably matches the
expected answer", not "looks plausible".

Run with::

    .venv/bin/python scripts/first_useful_action.py --max-attempts 8 \
        --out evals/benchmarks/llamacpp_production_1ba5283/first_useful_action.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from oai2.evals.deterministic_hard import (  # noqa: E402
    DERIVED_ANSWERS,
    deterministic_hard_suite,
)

BASE_URL = "http://127.0.0.1:8851"
MODEL = "smollm2-1.7b-q4km"
SEED = 20261004


def stream_once(prompt: str, max_tokens: int, timeout: float) -> dict[str, object]:
    """One streaming attempt. Returns TTFT, total time, text and server timings.

    TTFT is measured to the first chunk carrying actual ``content``. The
    first SSE frame is often an empty role delta; timing that would flatter
    the number by a whole chunk boundary, and the whole point of this
    measurement is not to flatter it.
    """
    import urllib.request

    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": SEED,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft: float | None = None
    parts: list[str] = []
    timings: dict[str, object] = {}
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk.get("timings"), dict):
                timings = chunk["timings"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    parts.append(content)
    return {
        "ttft_seconds": None if ttft is None else round(ttft, 6),
        "total_seconds": round(time.perf_counter() - start, 6),
        "text": "".join(parts),
        "server_predicted_per_second": timings.get("predicted_per_second"),
        "server_prompt_ms": timings.get("prompt_ms"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    suite = deterministic_hard_suite()
    results: list[dict[str, object]] = []

    print(f"model: {MODEL}   max attempts/case: {args.max_attempts}\n")
    print(f"{'case':<24}{'attempts':>9}{'to-correct s':>14}{'ttft s':>9}{'answer':>12}")
    print("-" * 70)

    for case in suite.cases:
        expected = DERIVED_ANSWERS[case.case_id]
        attempts: list[dict[str, object]] = []
        cumulative = 0.0
        first_correct_at: float | None = None
        for n in range(1, args.max_attempts + 1):
            try:
                r = stream_once(case.prompt, args.max_tokens, args.timeout)
            except Exception as exc:  # noqa: BLE001 - a failure is data here
                attempts.append({"attempt": n, "error": f"{type(exc).__name__}: {exc}"})
                continue
            text = r.pop("text")
            ok = any(re.search(p, text) for p in case.expected_patterns)
            cumulative += float(r["total_seconds"])
            attempts.append({**r, "attempt": n, "verified_correct": ok, "text": text.strip()[:120]})
            if ok:
                first_correct_at = cumulative
                break

        # With temperature 0.0 the serving path is deterministic, so a
        # repeated request reproduces the same output. Recording whether
        # that held keeps "attempts to first correct" honest: if every
        # attempt is byte-identical the retry loop explored no
        # distribution, and the case either succeeds on attempt 1 or never.
        attempt_texts = [a.get("text") for a in attempts if "text" in a]
        deterministic = len(set(attempt_texts)) <= 1

        results.append(
            {
                "case_id": case.case_id,
                "expected_answer": expected,
                "attempts_used": len(attempts),
                "attempts_byte_identical": deterministic,
                "verified_correct": first_correct_at is not None,
                "seconds_to_first_correct": (
                    None if first_correct_at is None else round(first_correct_at, 4)
                ),
                "attempts": attempts,
            }
        )
        r = results[-1]
        last = attempts[-1] if attempts else {}
        print(
            f"{case.case_id:<24}{r['attempts_used']:>9}"
            f"{(f'{r["seconds_to_first_correct"]:.3f}' if r['seconds_to_first_correct'] is not None else 'none'):>14}"
            f"{(f'{last.get("ttft_seconds"):.4f}' if isinstance(last.get('ttft_seconds'), float) else 'n/a'):>9}"
            f"{(expected[:11] if r['verified_correct'] else 'MISS'):>12}"
        )

    solved = [r for r in results if r["verified_correct"]]
    unsolved = [r for r in results if not r["verified_correct"]]
    times = sorted(float(r["seconds_to_first_correct"]) for r in solved)
    ttfts = [
        float(a["ttft_seconds"])
        for r in results
        for a in r["attempts"]
        if isinstance(a.get("ttft_seconds"), float)
    ]
    median_correct = times[len(times) // 2] if times else None
    median_ttft = sorted(ttfts)[len(ttfts) // 2] if ttfts else None

    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "requirement": "AC-PERF-034: first useful action, distinct from TTFT",
        "definition": (
            "cumulative wall time until an attempt's output satisfies the case's "
            "regex oracle; TTFT is reported alongside and is NOT this metric"
        ),
        "endpoint": BASE_URL,
        "model": MODEL,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": SEED, "max_tokens": args.max_tokens},
        "max_attempts": args.max_attempts,
        "suite_id": suite.suite_id,
        "summary": {
            "cases": len(results),
            "reached_verified_correct": len(solved),
            "never_reached_within_attempt_cap": [r["case_id"] for r in unsolved],
            "median_seconds_to_first_correct": (
                None if median_correct is None else round(median_correct, 4)
            ),
            "median_ttft_seconds": None if median_ttft is None else round(median_ttft, 6),
            "ttft_understates_by_x": (
                None
                if (median_correct is None or median_ttft in (None, 0))
                else round(median_correct / median_ttft, 1)
            ),
            # The sharper statement: for a case never answered correctly
            # there is no finite time to a useful action, so TTFT does not
            # merely overstate those cases by a factor -- it reports a time
            # for an outcome that never occurs.
            "cases_with_no_finite_time_to_correct": len(unsolved),
            "all_attempts_byte_identical": all(bool(r["attempts_byte_identical"]) for r in results),
            "retry_note": (
                "serving is deterministic (temperature 0.0, fixed seed) so repeated attempts "
                "reproduce the same output; the retry loop bounds work, it does not explore "
                "a distribution. A case that is wrong once is wrong every time, which is why "
                "retrying cannot rescue the failures."
            ),
        },
        "cases": results,
    }

    print(
        f"\nreached verified-correct within {args.max_attempts} attempts: "
        f"{len(solved)}/{len(results)}"
    )
    if unsolved:
        print(f"never reached: {', '.join(str(r['case_id']) for r in unsolved)}")
    if median_correct is not None and median_ttft:
        print(
            f"median TTFT {median_ttft * 1000:.2f} ms vs median time-to-first-correct "
            f"{median_correct:.3f} s  -> TTFT understates by ~{median_correct / median_ttft:.0f}x"
        )
    if unsolved:
        print(
            f"\nFor the other {len(unsolved)} cases there is NO finite time to a correct "
            f"answer, so the understatement is not a\nfactor but unbounded: TTFT reports a "
            f"time for an outcome that never occurs."
        )
    if artifact["summary"]["all_attempts_byte_identical"]:
        print(
            "\nAll attempts byte-identical: serving is deterministic (temperature 0.0, fixed "
            "seed), so retrying\ncannot convert a wrong answer into a right one. The failures "
            "are confident, not unlucky."
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
