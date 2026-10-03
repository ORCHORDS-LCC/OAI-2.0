#!/usr/bin/env python3
"""Matched A/B measurement of what the fleet worker actually contributes.

This harness answers one narrow question with numbers instead of adjectives:

    For ONE pinned revision of this repository, running ONE identical bounded
    command, is it better to run the work on this Mac or to hand it to the
    cluster worker `p50` -- and *where does the time actually go*?

It runs two cells over the same useful work:

  Cell A  mac-local  -- prepare a checkout of the pinned sha, then run the
                       verification command locally.
  Cell B  fleet-p50  -- enqueue the same command to the cluster control plane
                       and let the worker node execute it.

Design commitments that are enforced in code, not just documented:

  * MATCHED WORK IS PROVEN FROM EXECUTION EVIDENCE.  Cell B's identity is
    recovered from what the worker actually executed -- the `argv` and
    `profile` the node wrote into each entry of the job's `checks[]` block --
    and from the profile list the node advertises.  It is never copied from a
    local variable.  A result that carries no executable identity FAILS the
    gate as "remote identity unproven".
  * COLD vs WARM IS OBSERVED, NEVER ASSUMED.  A rep is labelled from the
    preparation it actually paid, not from its index.  The worker makes a
    disposable clone per job, so in practice every remote rep pays preparation
    and no remote rep is warm; the local cell reuses one checkout after the
    first rep.  Both facts are read from evidence, not assumed either way.
  * PHASE DURATIONS HAVE A DOCUMENTED CLOCK DOMAIN.  Enqueue/grant/finish come
    from the cluster's own event timestamps; local timers stay monotonic for
    observation only.  Combining two instants from different clock domains is
    refused outright.  A plausibility ceiling filters impossible values; it is
    NOT validation and a value below the ceiling proves nothing about
    correctness.
  * HOST SAMPLING IS TIED TO THE REAL WINDOW and names its host.  Mac load
    averages describe the Mac running this harness.  They are not, and are
    never reported as, node p50 CPU utilisation.
  * Correctness gates run BEFORE any statistic is computed.  A job that did
    not succeed, that resolved to a different revision, that ran on the wrong
    node, or whose remote identity could not be proven makes the process exit
    non-zero and the statistics are marked void.
  * The node is a VERIFICATION-capacity worker.  `assert_verification_scoped`
    scans every rendered byte for generative-performance vocabulary and fails
    the run, so this script structurally cannot emit such a claim.
  * The only network peer is the cluster control plane.  Cell A runs a fixed
    argv with no URL and no model/service flag; that is checked, not assumed.

Stdlib only. See `--help` for the full CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "deploy" / "worker" / "profiles.json"
DEFAULT_CHECK = "python-unittest"
DEFAULT_CLUSTER_URL = "http://127.0.0.1:10534"
DEFAULT_QPIPE = "/Users/orchords/.local/share/q-pipe/qpipe-venv/bin/qpipe"
TOKEN_ENV = "QPIPE_CLUSTER_TOKEN"
URL_ENV = "QPIPE_CLUSTER_URL"

EXIT_OK = 0
EXIT_GATE_FAILURE = 1
EXIT_CONFIG_ERROR = 2

TERMINAL_STATES = {
    "succeeded",
    "success",
    "ok",
    "done",
    "complete",
    "completed",
    "finished",
    "failed",
    "failure",
    "error",
    "cancelled",
    "canceled",
    "aborted",
    "timeout",
    "timed_out",
    "expired",
}
SUCCESS_STATES = {"succeeded", "success", "ok", "done", "complete", "completed", "finished"}

FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")

# ---------------------------------------------------------------------------
# Structural guard: this report may never carry a generative-performance claim.
#
# The harness measures test-execution capacity. Anything in this list is a
# claim about token rates, accelerator silicon, generative quality, or a
# comparative speed claim, which is a different system entirely and is not
# measured here. The guard is checked against the fully rendered report and the
# JSON payload, so a future edit that introduces such a claim fails the run
# instead of shipping a wrong number. Words are matched on word boundaries,
# case-insensitively.
# ---------------------------------------------------------------------------
FORBIDDEN_CLAIM_TERMS: tuple[str, ...] = (
    "inference",
    "inferences",
    "token",
    "tokens",
    "tok/s",
    "tokens/s",
    "tokens per second",
    "prompt tokens",
    "completion tokens",
    "generated tokens",
    "ttft",
    "time to first token",
    "inter-token",
    "prefill",
    "prefill throughput",
    "decode",
    "decoder",
    "gpu",
    "gpus",
    "cuda",
    "nvidia",
    "rocm",
    "vram",
    "accelerator",
    "vllm",
    "llama",
    "qwen",
    "gguf",
    "generative",
    "generation",
    "model server",
    "model serving",
    "model throughput",
    "llm",
    "llms",
    "batching throughput",
    "request rate",
    "requests per second",
    "req/s",
    "rps",
    "qps",
    "it/s",
    "words per second",
    "samples/s",
    "sampling rate",
    # Comparative-performance classes. A speed or rate claim about the node is
    # exactly the kind of unmeasured statement this harness must never print.
    "throughput",
    "speedup",
    "speed-up",
    "faster than",
    "times faster",
)

_CLAIM_GUARD_ERROR = (
    "blocked: the rendered report contained generative-performance vocabulary, "
    "which this harness does not and cannot measure"
)


class InferenceClaimError(RuntimeError):
    """Raised when a rendered report would carry an unmeasurable claim."""


class ConfigError(RuntimeError):
    """Raised for a usage/environment problem; maps to exit code 2."""


class ClockDomainError(RuntimeError):
    """Raised when two instants from different clock domains are combined."""


def assert_verification_scoped(text: str) -> str:
    """Fail closed if `text` carries generative-performance vocabulary.

    This is the mechanism that makes an unmeasured claim impossible to print
    rather than merely discouraged. It is applied to the human report and to
    the serialised JSON.
    """
    lowered = text.lower()
    for term in FORBIDDEN_CLAIM_TERMS:
        pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            raise InferenceClaimError(
                f"{_CLAIM_GUARD_ERROR} (matched vocabulary class #{FORBIDDEN_CLAIM_TERMS.index(term)})"
            )
    return text


# Ports that would indicate a model server, coordinator or gateway had been
# addressed. The harness refuses to point at any of them.
FORBIDDEN_PORTS = {
    11434,  # ollama
    1234,  # lm studio
    8000,  # generic model server
    8001,
    8080,  # coordinator / gateway
    8787,  # cloudflare-style gateway
    5000,
    3000,
}


def validate_cluster_url(url: str) -> str:
    """Reject any control-plane address that is not plainly the cluster."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ConfigError(f"cluster url must be http(s), got {parsed.scheme!r}")
    if not parsed.hostname:
        raise ConfigError(f"cluster url has no host: {url!r}")
    if parsed.port in FORBIDDEN_PORTS:
        raise ConfigError(
            f"refusing to use port {parsed.port}: this harness only talks to the "
            "cluster control plane, never to a model server, coordinator or gateway"
        )
    return url.rstrip("/")


def assert_local_argv_offline(argv: Sequence[str]) -> list[str]:
    """Cell A must be a fixed, local, non-networked argv.

    This is how the harness keeps itself honest about not sending foreground
    traffic anywhere: the Mac cell is a checkout plus a fixed command, and any
    argument that looks like a service address or a model/service switch is a
    hard error rather than a silent run.
    """
    for argument in argv:
        lowered = argument.lower()
        if "://" in lowered:
            raise ConfigError(
                f"local cell argv must not contain a network address: {argument!r}"
            )
        for forbidden in ("--model", "--base-url", "--api-key", "--host", "curl", "wget"):
            if forbidden in lowered:
                raise ConfigError(
                    f"local cell argv must not reference a service or model switch: {argument!r}"
                )
    return list(argv)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile. Documented because with n=5, p95 is the max."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
    return ordered[rank - 1]


def distribution(values: Sequence[float]) -> dict[str, Any]:
    """Summarise a sample. `n` is the sample count and is always present."""
    if not values:
        return {
            "n": 0,
            "min_s": None,
            "median_s": None,
            "p95_s": None,
            "max_s": None,
            "mean_s": None,
            "p95_method": "nearest-rank",
        }
    return {
        "n": len(values),
        "min_s": round(min(values), 4),
        "median_s": round(statistics.median(values), 4),
        "p95_s": round(percentile(values, 0.95) or 0.0, 4),
        "max_s": round(max(values), 4),
        "mean_s": round(statistics.fmean(values), 4),
        "p95_method": "nearest-rank",
    }


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def first_key(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def argv_digest(argv: Sequence[str]) -> str:
    """SHA-256 over the joined argv. Used to compare a LOCAL workload against a
    REMOTE one that was recovered from execution evidence."""
    import hashlib

    return hashlib.sha256("\x1f".join(str(a) for a in argv).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Clock domains (D3)
#
# A phase duration is only meaningful if both of its endpoints came from the
# same clock. Every instant this harness consumes is wrapped in a Stamp that
# names its domain, and `verify_same_clock` refuses to subtract across
# domains. The report prints the domain next to every duration so a reader can
# see which clock produced it.
# ---------------------------------------------------------------------------

CLUSTER_EVENT_CLOCK = "cluster.event.created_at"
CLUSTER_JOB_FIELD_CLOCK = "cluster.job.field"
WORKER_DURATION_CLOCK = "worker.monotonic.duration_ms"
LOCAL_MONOTONIC_CLOCK = "local.monotonic"
LOCAL_WALL_CLOCK = "local.wall"


@dataclass(frozen=True)
class Stamp:
    """One instant, tagged with the clock domain it was read from."""

    value: float
    clock: str
    source: str

    def describe(self) -> str:
        return f"{self.clock} via {self.source}={self.value:.4f}"


def verify_same_clock(start: Stamp, end: Stamp) -> str:
    """Explicit assertion that two instants share one clock domain.

    Raises ClockDomainError when they do not. This is an assertion about
    provenance, not a plausibility test: no magnitude can excuse a subtraction
    across two different clocks.
    """
    if start.clock != end.clock:
        raise ClockDomainError(
            f"cannot subtract {start.describe()} from {end.describe()}: "
            f"different clock domains ({start.clock} vs {end.clock})"
        )
    return start.clock


# A queue or execution phase that is negative, non-finite or larger than a day
# did not come from a duration arithmetic over a single clock. Such a value is
# filtered out rather than published.
#
# THIS IS A PLAUSIBILITY FILTER, NOT VALIDATION. Passing it means only that the
# number is not obviously impossible. It is NOT evidence that the job succeeded,
# that it ran the right code, or that the two cells ran matched work. Those are
# separate correctness gates, and they are what decide the exit code.
MAX_PLAUSIBLE_PHASE_S = 86_400.0
PLAUSIBILITY_CEILING_NOTE = (
    f"plausibility filter only: rejects negative, non-finite or >{MAX_PLAUSIBLE_PHASE_S:.0f}s "
    "values; a value that survives this filter is NOT validated and proves nothing "
    "about correctness"
)


def plausible_duration(value: Any) -> float | None:
    """Plausibility filter for a duration in seconds. Not a correctness check."""
    seconds = as_float(value)
    if seconds is None:
        return None
    if not math.isfinite(seconds):
        return None
    if seconds < 0.0 or seconds > MAX_PLAUSIBLE_PHASE_S:
        return None
    return round(seconds, 4)


@dataclass
class PhaseDuration:
    """A phase duration with its provenance attached and nothing hidden.

    `seconds` is None when the phase could not be established. `reason` then
    says why, and `same_clock` records whether the clock-domain assertion was
    satisfied, so the report can never show a duration without saying where it
    came from.
    """

    seconds: float | None
    clock_domain: str
    start_source: str
    end_source: str
    same_clock: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "clock_domain": self.clock_domain,
            "start_source": self.start_source,
            "end_source": self.end_source,
            "same_clock_domain": self.same_clock,
            "unavailable_reason": self.reason,
            "plausibility_filter": PLAUSIBILITY_CEILING_NOTE,
        }


def phase_duration(start: Stamp | None, end: Stamp | None) -> PhaseDuration:
    """Build a phase duration from two instants of one clock domain.

    Order of refusal, and the order is the point:
      1. a missing instant            -> unavailable, never guessed
      2. two different clock domains  -> refused by assertion (ClockDomainError)
      3. an impossible magnitude      -> dropped by the plausibility filter
    """
    if start is None or end is None:
        missing = "start" if start is None else "end"
        return PhaseDuration(
            seconds=None,
            clock_domain=(start or end).clock if (start or end) else "unknown",
            start_source=start.source if start else "<missing>",
            end_source=end.source if end else "<missing>",
            same_clock=False,
            reason=f"no {missing} instant reported by the cluster",
        )
    try:
        clock = verify_same_clock(start, end)
    except ClockDomainError as exc:
        return PhaseDuration(
            seconds=None,
            clock_domain=f"MIXED:{start.clock}+{end.clock}",
            start_source=start.source,
            end_source=end.source,
            same_clock=False,
            reason=str(exc),
        )
    raw = end.value - start.value
    seconds = plausible_duration(raw)
    return PhaseDuration(
        seconds=seconds,
        clock_domain=clock,
        start_source=start.source,
        end_source=end.source,
        same_clock=True,
        reason=(
            ""
            if seconds is not None
            else f"value {raw:.4f}s dropped by the plausibility filter; not a validation failure"
        ),
    )


# ---------------------------------------------------------------------------
# Best-effort host resource sampling. Never fatal, always host-labelled.
#
# Everything measured here describes THIS Mac, the host running the harness.
# It says nothing about node p50. That distinction is carried in the data
# itself, not just in prose.
# ---------------------------------------------------------------------------

LOCAL_HOST_LABEL = f"mac-local (this Mac, {platform.system()} {platform.machine()})"
REMOTE_HOST_LABEL = "fleet-p50 (the cluster worker node; NOT sampled by this harness)"


def _total_memory_bytes() -> int | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0:
            return int(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


_TOTAL_MEMORY_BYTES: int | None = None
_TOTAL_MEMORY_RESOLVED = False


def total_memory_bytes() -> int | None:
    global _TOTAL_MEMORY_BYTES, _TOTAL_MEMORY_RESOLVED
    if not _TOTAL_MEMORY_RESOLVED:
        _TOTAL_MEMORY_BYTES = _total_memory_bytes()
        _TOTAL_MEMORY_RESOLVED = True
    return _TOTAL_MEMORY_BYTES


def memory_pressure_percent() -> float | None:
    """Best-effort used-memory percentage. Any failure yields None."""
    try:
        completed = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    page_size = 4096
    free_pages: float | None = None
    inactive_pages: float | None = None
    speculative_pages: float | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("Mach Virtual Memory Statistics"):
            match = re.search(r"page size of (\d+) bytes", line)
            if match:
                page_size = int(match.group(1))
        for label, target in (
            ("Pages free", "free"),
            ("Pages inactive", "inactive"),
            ("Pages speculative", "speculative"),
        ):
            if line.startswith(label):
                digits = re.sub(r"[^0-9]", "", line.split(":", 1)[-1])
                if not digits:
                    continue
                if target == "free":
                    free_pages = float(digits)
                elif target == "inactive":
                    inactive_pages = float(digits)
                else:
                    speculative_pages = float(digits)
    total = total_memory_bytes()
    if total is None or free_pages is None:
        return None
    reclaimable = (inactive_pages or 0.0) + (speculative_pages or 0.0)
    used_bytes = max(0.0, total - (free_pages + reclaimable) * page_size)
    return round(100.0 * used_bytes / total, 2) if total else None


@dataclass
class HostSample:
    t: float
    load1: float | None
    load5: float | None
    load15: float | None
    memory_used_pct: float | None


class HostSampler:
    """Low-rate sampling thread for the LOCAL host only.

    Three states are distinguished and never conflated:
      * never-ran  -- start() was never called, so no sample is missing because
                      of a platform limit; the run simply did not sample.
      * unsupported-- start() ran, and this metric raised or was unavailable on
                      this platform (e.g. vm_stat is macOS-only).
      * supported  -- start() ran and the metric produced at least one value.
    A sampler failure is recorded, never raised.
    """

    def __init__(self, interval: float) -> None:
        self.interval = max(0.2, interval)
        self.samples: list[HostSample] = []
        self.notes: list[str] = []
        self.started_ever = False
        self.stopped = False
        self.metric_unsupported: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _take(self) -> HostSample:
        load1 = load5 = load15 = None
        try:
            load1, load5, load15 = os.getloadavg()
        except (OSError, AttributeError) as exc:  # pragma: no cover - platform
            self.metric_unsupported.setdefault("load_average", f"{type(exc).__name__}")
            if "load average unavailable" not in " ".join(self.notes):
                self.notes.append(f"load average unavailable: {type(exc).__name__}")
        memory = None
        try:
            memory = memory_pressure_percent()
        except Exception as exc:  # noqa: BLE001 - sampling must never be fatal
            self.metric_unsupported.setdefault("memory_used_pct", f"{type(exc).__name__}")
            if "memory metric unavailable" not in " ".join(self.notes):
                self.notes.append(f"memory metric unavailable: {type(exc).__name__}")
        return HostSample(time.monotonic(), load1, load5, load15, memory)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._take())
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self.started_ever:
            raise RuntimeError("sampler already started; it covers one real window")
        self.started_ever = True
        self.samples.append(self._take())
        self._thread = threading.Thread(target=self._loop, name="host-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.started_ever:
            self.notes.append("sampler was never started: no local resource sample was taken")
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 1.0)
        self.samples.append(self._take())
        self.stopped = True

    def state(self) -> str:
        if not self.started_ever:
            return "never-ran"
        return "stopped" if self.stopped else "running"

    def metric_status(self, metric: str) -> dict[str, Any]:
        """Distinguish 'unsupported here' from 'the sampler never ran'."""
        if not self.started_ever:
            return {
                "metric": metric,
                "host": LOCAL_HOST_LABEL,
                "status": "never-ran",
                "reason": "the sampler was never started for the real measurement window",
            }
        if metric in self.metric_unsupported:
            return {
                "metric": metric,
                "host": LOCAL_HOST_LABEL,
                "status": "unsupported",
                "reason": f"not available on this platform: {self.metric_unsupported[metric]}",
            }
        if metric == "load_average":
            observed = any(s.load1 is not None for s in self.samples)
        else:
            observed = any(s.memory_used_pct is not None for s in self.samples)
        return {
            "metric": metric,
            "host": LOCAL_HOST_LABEL,
            "status": "supported" if observed else "no-observations",
            "reason": "" if observed else "sampler ran but the metric produced no value",
        }

    def summary(self, window: tuple[float, float] | None = None) -> dict[str, Any]:
        if not self.started_ever:
            return {
                "sample_count": 0,
                "load1_mean": None,
                "load1_max": None,
                "memory_used_pct_max": None,
                "memory_used_pct_mean": None,
                "total_memory_bytes": None,
                "sampler_state": "never-ran",
                "host": LOCAL_HOST_LABEL,
            }
        chosen = self.samples
        if window is not None:
            start, end = window
            chosen = [s for s in self.samples if start <= s.t <= end]
        loads = [s.load1 for s in chosen if s.load1 is not None]
        memory = [s.memory_used_pct for s in chosen if s.memory_used_pct is not None]
        return {
            "sample_count": len(chosen),
            "load1_mean": round(statistics.fmean(loads), 3) if loads else None,
            "load1_max": round(max(loads), 3) if loads else None,
            "memory_used_pct_max": max(memory) if memory else None,
            "memory_used_pct_mean": round(statistics.fmean(memory), 2) if memory else None,
            "total_memory_bytes": total_memory_bytes(),
            "sampler_state": self.state(),
            "host": LOCAL_HOST_LABEL,
        }


# ---------------------------------------------------------------------------
# Remote identity recovered from execution evidence (D1)
#
# The worker writes, for every check it actually ran, the exact argv it
# executed and the profile name it ran it under. That is the only trustworthy
# statement of what the remote side did. Anything derived from a local
# variable describes the local side and is therefore not evidence of a match.
# ---------------------------------------------------------------------------

REMOTE_PREPARATION_STEPS = ("git-clone", "git-fetch", "git-checkout")


@dataclass
class RemoteIdentity:
    """What the remote side actually proved about the work it executed."""

    proven: bool
    detail: str
    profile: str | None = None
    advertised: list[str] = field(default_factory=list)
    observed_argv: list[str] | None = None
    observed_argv_sha256: str | None = None
    local_argv_sha256: str | None = None
    local_profile: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "proven": self.proven,
            "detail": self.detail,
            "remote_profile": self.profile,
            "remote_advertised_profiles": self.advertised,
            "remote_argv": self.observed_argv,
            "remote_argv_sha256": self.observed_argv_sha256,
            "local_profile": self.local_profile,
            "local_argv_sha256": self.local_argv_sha256,
        }


def _result_block(record: dict[str, Any]) -> dict[str, Any]:
    nested = record.get("result")
    return nested if isinstance(nested, dict) else record


def observed_remote_preparation(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Read the preparation the worker actually performed, from its evidence.

    The worker records git-clone / git-fetch / git-checkout steps with
    `duration_ms` for every job. When those steps are present, remote
    preparation is observed rather than inferred; when they are absent, the
    harness says so instead of assuming the node amortised anything.
    """
    evidence = result.get("evidence")
    steps = [
        item
        for item in (evidence if isinstance(evidence, list) else [])
        if isinstance(item, dict) and str(item.get("step") or "") in REMOTE_PREPARATION_STEPS
    ]
    millis = 0.0
    counted = 0
    for item in steps:
        value = as_float(item.get("duration_ms"))
        if value is not None and value >= 0:
            millis += value
            counted += 1
    return {
        "observed": bool(steps),
        "seconds": round(millis / 1000.0, 4) if counted else None,
        "steps": [str(item.get("step")) for item in steps],
        "clock_domain": WORKER_DURATION_CLOCK,
        "note": (
            "preparation steps present in worker evidence; this rep paid preparation"
            if steps
            else "no preparation steps in worker evidence; remote preparation is unobserved "
            "for this rep and is not assumed to be zero"
        ),
    }


def recover_remote_identity(
    result: dict[str, Any],
    local_argv: Sequence[str],
    local_profile: str,
    advertised_capabilities: Sequence[str] | None,
) -> RemoteIdentity:
    """Prove cell B ran the pinned local workload, from remote evidence only.

    Sources, in order of trust:
      1. the `checks[]` entries the worker wrote, which carry the exact `argv`
         it executed and the `profile` it ran it under;
      2. the `verification:<profile>` capability the node advertises.
    If the result carries no executable identity the identity is UNPROVEN and
    the caller must fail the gate. There is deliberately no local fallback.
    """
    local_digest = argv_digest(local_argv)
    advertised = [str(c) for c in (advertised_capabilities or [])]
    required_capability = f"verification:{local_profile}"

    checks = result.get("checks")
    if not isinstance(checks, list) or not checks:
        return RemoteIdentity(
            proven=False,
            detail=(
                "remote identity unproven: the job result carries no checks[] entries, so "
                "there is no record of what the node actually executed"
            ),
            advertised=advertised,
            local_argv_sha256=local_digest,
            local_profile=local_profile,
        )
    if required_capability not in advertised:
        return RemoteIdentity(
            proven=False,
            detail=(
                f"remote identity unproven: node does not advertise {required_capability!r}; "
                f"advertised={advertised}"
            ),
            advertised=advertised,
            local_argv_sha256=local_digest,
            local_profile=local_profile,
        )

    matching = [c for c in checks if isinstance(c, dict) and str(c.get("profile") or "") == local_profile]
    if not matching:
        observed = sorted({str(c.get("profile")) for c in checks if isinstance(c, dict)})
        return RemoteIdentity(
            proven=False,
            detail=(
                f"remote identity unproven: no checks[] entry ran profile {local_profile!r}; "
                f"the node ran {observed}"
            ),
            advertised=advertised,
            local_argv_sha256=local_digest,
            local_profile=local_profile,
        )
    if len(matching) > 1:
        return RemoteIdentity(
            proven=False,
            detail=(
                f"remote identity unproven: profile {local_profile!r} ran {len(matching)} times in "
                "one job, so the executed work is not the single pinned workload"
            ),
            advertised=advertised,
            local_argv_sha256=local_digest,
            local_profile=local_profile,
        )

    entry = matching[0]
    remote_argv = entry.get("argv")
    if not isinstance(remote_argv, list) or not remote_argv:
        return RemoteIdentity(
            proven=False,
            detail=(
                "remote identity unproven: the checks[] entry for profile "
                f"{local_profile!r} carries no argv, so the executed command is unknown"
            ),
            advertised=advertised,
            local_argv_sha256=local_digest,
            local_profile=local_profile,
        )

    remote_argv = [str(a) for a in remote_argv]
    remote_digest = argv_digest(remote_argv)
    base = dict(
        profile=local_profile,
        advertised=advertised,
        observed_argv=remote_argv,
        observed_argv_sha256=remote_digest,
        local_argv_sha256=local_digest,
        local_profile=local_profile,
    )
    if remote_digest != local_digest:
        return RemoteIdentity(
            proven=False,
            detail=(
                "remote identity MISMATCH: node executed "
                f"{shlex.join(remote_argv)} (sha256 {remote_digest[:16]}) but the pinned local "
                f"workload is {shlex.join(local_argv)} (sha256 {local_digest[:16]})"
            ),
            **base,
        )
    return RemoteIdentity(
        proven=True,
        detail=(
            f"node executed profile {local_profile!r} with argv sha256 {remote_digest[:16]}, "
            f"identical to the pinned local workload; node advertises {required_capability!r}"
        ),
        **base,
    )


def check_block_failed(result: dict[str, Any]) -> tuple[bool, str]:
    """True when any executed check reported a non-successful outcome.

    The job's top-level `ok` is the worker's own summary; this reads the
    individual check entries so a failure cannot be hidden behind an `ok`.
    """
    checks = result.get("checks")
    if not isinstance(checks, list) or not checks:
        return True, "no checks[] entries were reported"
    for entry in checks:
        if not isinstance(entry, dict):
            return True, "a checks[] entry was not a record"
        if bool(entry.get("timed_out")):
            return True, f"profile {entry.get('profile')!r} timed out"
        if bool(entry.get("canceled")):
            return True, f"profile {entry.get('profile')!r} was canceled"
        if int(as_float(entry.get("returncode")) or 0) != 0:
            return True, f"profile {entry.get('profile')!r} exited {entry.get('returncode')}"
    return False, f"{len(checks)} check(s) executed with exit code 0"


# ---------------------------------------------------------------------------
# Cluster access
# ---------------------------------------------------------------------------


class ClusterClient:
    """Read-only HTTP access to the cluster control plane, plus one enqueue.

    Only the control-plane origin passed on the command line is ever contacted.
    The bearer token is read from the environment, never from argv, never from
    a file, and is never included in any message this script prints.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        timeout: float = 15.0,
    ) -> None:
        self.base_url = validate_cluster_url(base_url)
        self._auth_header = auth_header
        self._auth_scheme = auth_scheme
        self._token = token
        self.timeout = timeout

    def _request(self, path: str) -> Any:
        request = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        value = f"{self._auth_scheme} {self._token}".strip()
        request.add_header(self._auth_header, value)
        request.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"cluster GET {path} failed: HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cluster GET {path} failed: {exc.reason}") from None
        except (TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"cluster GET {path} failed: {type(exc).__name__}") from None

    def job(self, job_id: int) -> dict[str, Any]:
        payload = self._request(f"/v1/cluster/jobs/{job_id}")
        if isinstance(payload, dict):
            for key in ("job", "result", "data"):
                if isinstance(payload.get(key), dict):
                    return {**payload, **payload[key]}
        if not isinstance(payload, dict):
            raise RuntimeError(f"cluster job {job_id} returned {type(payload).__name__}")
        return payload

    def events(self, job_id: int, limit: int = 200) -> list[dict[str, Any]]:
        payload = self._request(f"/v1/cluster/events?job_id={job_id}&limit={limit}")
        if isinstance(payload, list):
            return [e for e in payload if isinstance(e, dict)]
        if isinstance(payload, dict):
            for key in ("events", "data", "items"):
                if isinstance(payload.get(key), list):
                    return [e for e in payload[key] if isinstance(e, dict)]
        return []

    def nodes(self) -> list[dict[str, Any]]:
        payload = self._request("/v1/cluster/nodes")
        if isinstance(payload, list):
            return [n for n in payload if isinstance(n, dict)]
        if isinstance(payload, dict):
            for key in ("nodes", "data", "items"):
                if isinstance(payload.get(key), list):
                    return [n for n in payload[key] if isinstance(n, dict)]
        return []


def resolve_full_sha(repo_url: str, ref: str) -> dict[str, Any]:
    """Resolve `ref` to a FULL 40-character sha before any measurement.

    Acceptance evidence must be exact identity. A prefix is not acceptance, so
    the harness refuses to start unless it holds a full immutable sha, and it
    records whether that sha was independently confirmed to be published.
    """
    git = shutil.which("git")
    if git is None:
        raise ConfigError("git is not on PATH; the requested ref cannot be resolved to a full sha")
    if not ref.strip():
        raise ConfigError("--ref must not be empty")

    def ls_remote(*patterns: str) -> list[tuple[str, str]]:
        completed = subprocess.run(
            [git, "ls-remote", "--", repo_url, *patterns],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            return []
        found = []
        for line in completed.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and FULL_SHA_RE.fullmatch(parts[0]):
                found.append((parts[0].lower(), parts[1]))
        return found

    candidates: list[tuple[str, str]] = []
    if FULL_SHA_RE.fullmatch(ref.strip().lower()):
        sha = ref.strip().lower()
        published = any(entry[0] == sha for entry in ls_remote())
        return {
            "requested_ref": ref,
            "full_sha": sha,
            "ref_name": None,
            "reachability_confirmed": published,
            "note": (
                "requested ref is already a full immutable sha"
                if published
                else "requested ref is a full immutable sha; it was not found among the "
                "published refs, so reachability was not independently confirmed"
            ),
        }
    for pattern in (ref, f"refs/heads/{ref}", f"refs/tags/{ref}", f"refs/pull/{ref}/head"):
        candidates = ls_remote(pattern)
        if candidates:
            break
    if not candidates:
        raise ConfigError(
            f"could not resolve ref {ref!r} to a sha on the remote; a branch or tag name is "
            "required, or a full 40-character sha"
        )
    sha, ref_name = candidates[0]
    if len({entry[0] for entry in candidates}) > 1:
        raise ConfigError(f"ref {ref!r} is ambiguous on the remote: {candidates}")
    return {
        "requested_ref": ref,
        "full_sha": sha,
        "ref_name": ref_name,
        "reachability_confirmed": True,
        "note": f"resolved {ref!r} to the full immutable sha",
    }


def run_enqueue(
    qpipe: str,
    repo_url: str,
    ref: str,
    check: str,
    cluster_url: str,
    dry_run: bool = False,
) -> int:
    """Enqueue one verification job via the qpipe CLI. Returns the job id."""
    if dry_run:
        raise ConfigError("refusing to enqueue during a dry run")
    argv = [
        qpipe,
        "cluster",
        "enqueue-verification",
        "--repo-url",
        repo_url,
        "--ref",
        ref,
        "--check",
        check,
        "--url",
        cluster_url,
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {completed.returncode}"
        raise RuntimeError(f"enqueue failed: {tail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        match = re.search(r'"job_id"\s*:\s*(\d+)', completed.stdout)
        if not match:
            raise RuntimeError("enqueue returned no parsable job id") from None
        return int(match.group(1))
    if not payload.get("ok"):
        raise RuntimeError(f"enqueue reported ok=false: {payload!r}")
    return int(payload["job_id"])


def wait_for_job(
    client: ClusterClient,
    job_id: int,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    """Poll until the job reaches a terminal state. Returns the job record."""
    deadline = time.monotonic() + timeout
    while True:
        record = client.job(job_id)
        state = str(first_key(record, "state", "status") or "unknown").lower()
        if state in TERMINAL_STATES:
            return record
        if time.monotonic() > deadline:
            raise RuntimeError(f"job {job_id} did not reach a terminal state in {timeout}s")
        time.sleep(poll_interval)


def cluster_stamps(
    record: dict[str, Any],
    events: Sequence[dict[str, Any]],
) -> dict[str, Stamp | None]:
    """Recover enqueue/grant/finish instants from the CLUSTER's own timestamps.

    Events are the primary source. The job record is a documented fallback and
    is still a cluster-side clock, so the two never mix domains. A local clock
    is never used for any of these: that is precisely the defect that once
    published an epoch value as a duration.
    """
    stamps: dict[str, Stamp | None] = {"enqueue": None, "grant": None, "finish": None}
    for event in events:
        name = str(
            first_key(event, "event", "type", "name", "kind", "state", "status") or ""
        ).lower()
        moment = as_float(
            first_key(
                event,
                "created_at",
                "ts",
                "timestamp",
                "time",
                "at",
                "occurred_at",
            )
        )
        if moment is None or not name:
            continue
        if "enqueue" in name or "submit" in name:
            stamps["enqueue"] = stamps["enqueue"] or Stamp(moment, CLUSTER_EVENT_CLOCK, f"event:{name}")
        if any(token in name for token in ("lease", "grant", "assign", "accept", "start")):
            stamps["grant"] = stamps["grant"] or Stamp(moment, CLUSTER_EVENT_CLOCK, f"event:{name}")
        if any(
            token in name
            for token in ("succeed", "finish", "complete", "done", "result", "fail", "cancel", "end")
        ):
            stamps["finish"] = Stamp(moment, CLUSTER_EVENT_CLOCK, f"event:{name}")
    fallbacks = {
        "enqueue": ("created_at", "enqueued_at", "submitted_at"),
        "grant": ("started_at", "granted_at", "leased_at"),
        "finish": ("finished_at", "completed_at", "ended_at"),
    }
    for key, names in fallbacks.items():
        if stamps[key] is None:
            moment = as_float(first_key(record, *names))
            if moment is not None:
                stamps[key] = Stamp(moment, CLUSTER_JOB_FIELD_CLOCK, f"job.{names[0]}")
    return stamps


# ---------------------------------------------------------------------------
# Cell A: local execution
# ---------------------------------------------------------------------------


def shallow_prepare(repo_url: str, ref: str, destination: Path) -> float:
    """Materialise `ref` into `destination`. Returns preparation seconds.

    Preparation seconds come from a local monotonic timer. That is an
    observation of this Mac, not a cluster clock, and it is never subtracted
    from a cluster timestamp.
    """
    git = shutil.which("git")
    if git is None:
        raise ConfigError("git is not on PATH; cell A cannot prepare a checkout")
    is_commit = bool(FULL_SHA_RE.fullmatch(ref.strip().lower()))
    if is_commit:
        steps = (
            [git, "init", "--quiet", str(destination)],
            [git, "-C", str(destination), "remote", "add", "origin", repo_url],
            [
                git,
                "-C",
                str(destination),
                "fetch",
                "--depth",
                "1",
                "--quiet",
                "origin",
                ref,
            ],
            [git, "-C", str(destination), "checkout", "--quiet", "FETCH_HEAD"],
        )
    else:
        steps = (
            [
                git,
                "clone",
                "--depth",
                "1",
                "--quiet",
                "--branch",
                ref,
                repo_url,
                str(destination),
            ],
        )
    started = time.monotonic()
    for step in steps:
        completed = subprocess.run(step, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            tail = detail[-1] if detail else f"exit {completed.returncode}"
            raise RuntimeError(f"cell A preparation failed at {step[1]}: {tail}")
    return time.monotonic() - started


def checkout_revision(tree: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    completed = subprocess.run(
        [git, "-C", str(tree), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


# ---------------------------------------------------------------------------
# Phase labelling (D2)
#
# A rep is labelled from the preparation state it OBSERVED. The rep index is
# not an input to this decision and must never become one.
# ---------------------------------------------------------------------------

PHASE_COLD = "cold"
PHASE_WARM = "warm"
PHASES = (PHASE_COLD, PHASE_WARM)


def classify_phase(*, tree_reused: bool) -> str:
    """Label a rep from observed preparation, never from its index.

    `tree_reused` True means the rep ran against a checkout that already
    existed, so no preparation was paid in this rep. False means the rep
    materialised its own preparation. The caller must set that flag from what
    actually happened, not from the rep's position in the run.
    """
    return PHASE_WARM if tree_reused else PHASE_COLD


# ---------------------------------------------------------------------------
# Measurement records
# ---------------------------------------------------------------------------


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass
class CellResult:
    name: str
    reps: list[dict[str, Any]] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def failed_gates(self) -> list[Gate]:
        return [g for g in self.gates if not g.passed]

    def timings(self, phase: str | None = None) -> list[float]:
        chosen = self.reps if phase is None else [r for r in self.reps if r["phase"] == phase]
        return [r["total_s"] for r in chosen if isinstance(r.get("total_s"), (int, float))]

    def useful_completed(self) -> int:
        return sum(1 for r in self.reps if r.get("useful") is True)


def revision_identical(resolved: str | None, pinned_sha: str) -> bool:
    """Exact identity only. A prefix is NOT acceptance.

    A short ref could name a different commit than the one the worker actually
    checked out, so acceptance evidence must be the full sha on both sides.
    """
    if not resolved:
        return False
    return str(resolved).strip().lower() == str(pinned_sha).strip().lower()


def judge_record(
    record: dict[str, Any],
    pinned_sha: str,
    expected_node: str,
) -> tuple[bool, str]:
    """Single source of truth for correctness gating of one remote observation."""
    if not bool(record.get("ok")):
        return False, f"job did not succeed (ok={record.get('ok')!r})"
    resolved = record.get("resolved_revision")
    if not resolved:
        return False, "job reported no resolved_revision"
    if not revision_identical(str(resolved), pinned_sha):
        return False, (
            f"resolved_revision {resolved} is not exactly the pinned sha {pinned_sha}; "
            "a prefix match is not acceptance"
        )
    node = record.get("node_id")
    if not node:
        return False, "job reported no node_id, so execution location is unproven"
    if str(node) != expected_node:
        return False, f"executed on node {node}, expected {expected_node}"
    return True, f"ok on {node} at exactly {resolved}"


def detail_ok(record: dict[str, Any], revision: str | None, pinned_sha: str) -> str:
    if not record["ok"]:
        return "local command returned a non-zero exit code"
    if not revision:
        return "could not read the prepared checkout revision"
    if not revision_identical(revision, pinned_sha):
        return f"checkout at {revision} is not exactly the pinned sha {pinned_sha}"
    return f"ok at exactly {revision}"


def resolve_local_argv(check: str, override: str | None) -> tuple[list[str], str]:
    """Cell A runs exactly the argv the worker profile defines."""
    if override:
        return assert_local_argv_offline(shlex.split(override)), "--local-argv override"
    if PROFILES_PATH.is_file():
        try:
            profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))["profiles"]
        except (json.JSONDecodeError, KeyError, OSError):
            profiles = {}
        if isinstance(profiles, dict) and isinstance(profiles.get(check), dict):
            argv = profiles[check].get("argv")
            if isinstance(argv, list) and argv:
                return (
                    assert_local_argv_offline([str(a) for a in argv]),
                    f"deploy/worker/profiles.json:{check}",
                )
    return (
        assert_local_argv_offline(["python3", "-m", "unittest", "-v", "test_app"]),
        "built-in default (profiles.json unreadable)",
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _dist_line(label: str, stats: dict[str, Any]) -> str:
    """Every distribution is printed with its sample count attached."""
    if stats["n"] == 0:
        return f"  {label:22}: n=0 (no observations)"
    return (
        f"  {label:22}: n={stats['n']} "
        f"min={stats['min_s']} median={stats['median_s']} "
        f"p95={stats['p95_s']} max={stats['max_s']} mean={stats['mean_s']}"
    )


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("WORKER CONTRIBUTION -- MATCHED A/B AT ONE PINNED REVISION")
    add("=" * 78)
    config = payload["config"]
    add(f"generated_utc        : {payload['generated_at_utc']}")
    add(f"requested_ref        : {config['ref']}")
    add(f"pinned_full_sha      : {config['pinned_sha']}")
    add(f"ref_resolution       : {config['ref_resolution']['note']}")
    add(f"check_profile        : {config['check']}")
    add(f"expected_node        : {config['expected_node']}")
    add(f"node_role            : {payload['worker_role']} (test-execution capacity only)")
    add(f"reps_per_cell        : {config['reps']}")
    add("percentile_method    : nearest-rank (at small n, p95 IS the max)")
    add("")

    gates = payload["gates"]
    add(f"CORRECTNESS GATES: {gates['status']}")
    for gate in payload["gate_detail"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        add(f"  [{mark}] {gate['name']}: {gate['detail']}")
    add("")

    identity = payload["work_identity"]
    add("--- MATCHED-WORK EVIDENCE (remote side, not a local copy) ---")
    add(f"  local_workload      : {identity['local_argv']}")
    add(f"  local_argv_sha256   : {identity['local_argv_sha256']}")
    add(f"  remote_source       : {identity['remote_source']}")
    add(f"  remote_argv         : {identity['remote_argv']}")
    add(f"  remote_argv_sha256  : {identity['remote_argv_sha256']}")
    add(f"  remote_profiles     : {identity['remote_advertised_profiles']}")
    add(f"  identity_proven     : {identity['proven']}")
    add(f"  identity_detail     : {identity['detail']}")
    add("")

    for key, label in (("cell_a", "CELL A -- mac-local"), ("cell_b", "CELL B -- fleet-p50")):
        cell = payload[key]
        add(f"--- {label} ---")
        add(f"  work                : {cell['work_description']}")
        add(f"  correct             : {cell['correct']}")
        add(f"  useful_jobs_done    : {cell['completed_useful_jobs']}")
        rate = cell["completed_useful_jobs_per_minute"]
        add(f"  useful_jobs_per_min : {rate if rate is not None else 'n/a'}")
        add("  latency distributions (sample count shown for every one):")
        for phase in (*PHASES, "all"):
            add(_dist_line(f"total_s[{phase}]", cell["latency_s"][phase]))
        add("  phase durations by observed preparation state:")
        for phase in (*PHASES, "all"):
            split = cell["phase_split_s"][phase]
            add(
                f"  {phase:21}: n={split['n']} preparation={split['preparation_s']} "
                f"queue={split['queue_s']} execution={split['execution_s']} "
                f"(clock={split['clock_domain']})"
            )
        add(f"  phase_labelling     : {cell['phase_labelling']}")
        add("  per-rep records:")
        for rep in cell["reps"]:
            bits = [
                f"    rep {rep['rep']} phase={rep['phase']} (observed: {rep['phase_basis']})",
                f"job={rep.get('job_id')}",
                f"node={rep.get('node_id')}",
                f"state={rep.get('state')}",
                f"ok={rep['ok']}",
                f"requested_ref={rep.get('requested_ref')}",
                f"resolved={rep.get('resolved_revision')}",
                f"prep={rep.get('preparation_s')}",
                f"queue={rep.get('enqueue_to_grant_s')}",
                f"exec={rep.get('execution_s')}",
                f"obs_overhead={rep.get('observation_overhead_s')}",
            ]
            add("  ".join(bits))
            add(
                f"      resource samples in this window: n={rep.get('sample_count', 0)} "
                f"host={rep.get('sample_host')} "
                f"load1_mean={rep.get('load1_mean')} mem_max={rep.get('memory_used_pct_max')}"
            )
        for note in cell["notes"]:
            add(f"  note: {note}")
        add("")

    add("--- FOUR DISTINCT COMPARISONS (only the SUPPORTED ones were measured) ---")
    for item in payload["comparisons"]:
        verdict = "SUPPORTED" if item["supported"] else "NOT SUPPORTED"
        add(f"  ({item['id']}) {item['title']}")
        add(f"      verdict          : {verdict}")
        add(f"      basis            : {item['basis']}")
        add(f"      result           : {item['result']}")
        add(f"      sample_count     : {item['n']}")
        add(f"      caveat           : {item['caveat']}")
    add("")

    add("--- HOST RESOURCE USE (best effort, never fatal) ---")
    host = payload["host_resources"]
    add(f"  sampler_state       : {host['sampler_state']}")
    add(f"  sampling_interval_s : {host['sampling_interval_s']}")
    add(f"  measured_host       : {host['measured_host']}")
    for metric in host["metrics"]:
        add(
            f"  metric {metric['metric']:16}: status={metric['status']} "
            f"host={metric['host']} {metric['reason']}"
        )
    add(f"  {host['unmeasured_host_note']}")
    add(f"  overall             : {host['overall']}")
    add(f"  notes               : {host['notes'] or ['none']}")
    add("")
    add(f"evidence_caveat      : {payload['evidence_caveat']}")
    add(f"scope_note           : {payload['scope_note']}")
    add("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test / dry run. Performs no cluster contact of any kind.
# ---------------------------------------------------------------------------


def _sample_check_result(
    profile: str = "python-unittest",
    argv: Sequence[str] | None = None,
    returncode: int = 0,
    timed_out: bool = False,
    canceled: bool = False,
) -> dict[str, Any]:
    return {
        "argv": list(argv if argv is not None else ["python3", "-m", "unittest", "-v", "test_app"]),
        "returncode": returncode,
        "duration_ms": 1234,
        "timed_out": timed_out,
        "canceled": canceled,
        "stdout_tail": "OK",
        "stderr_tail": "",
        "profile": profile,
    }


def synthetic_payload(args: argparse.Namespace) -> dict[str, Any]:
    """A complete, obviously-fake payload used to exercise the renderer."""

    def rep(index: int, node: str | None, phase: str, basis: str) -> dict[str, Any]:
        return {
            "rep": index,
            "phase": phase,
            "phase_basis": basis,
            "ok": True,
            "useful": True,
            "total_s": 1.0 + index,
            "execution_s": 0.9 + index,
            "preparation_s": 0.5 if node is None and index == 0 else (0.0 if node is None else 1.1),
            "job_id": None if node is None else 100 + index,
            "node_id": node,
            "state": "succeeded" if node else "local",
            "requested_ref": args.ref,
            "resolved_revision": args.ref,
            "enqueue_to_grant_s": None if node is None else 0.7,
            "grant_to_finish_s": None if node is None else 2.5,
            "observation_overhead_s": 0.02,
            "sample_count": 3 if node else 2,
            "sample_host": LOCAL_HOST_LABEL,
            "load1_mean": 1.5,
            "memory_used_pct_max": 62.0,
        }

    def cell(name: str, node: str | None, labelling: str) -> dict[str, Any]:
        # The synthetic cell A reuses its tree after rep 0; the synthetic cell B
        # pays preparation in every rep because the worker clones per job.
        if node is None:
            phases = [(PHASE_COLD, "reps[0]: materialised a fresh checkout")] + [
                (PHASE_WARM, f"reps[{i}]: reused the existing checkout") for i in range(1, 3)
            ]
        else:
            phases = [
                (PHASE_COLD, f"reps[{i}]: worker evidence shows git-clone+git-fetch+git-checkout")
                for i in range(3)
            ]
        reps = [rep(i, node, phase, basis) for i, (phase, basis) in enumerate(phases)]
        result = CellResult(name=name, reps=reps)
        return {
            "cell": name,
            "work_description": "python3 -m unittest -v test_app",
            "work_argv_sha256": "0" * 64,
            "correct": True,
            "completed_useful_jobs": result.useful_completed(),
            "completed_useful_jobs_per_minute": 12.0,
            "latency_s": {
                PHASE_COLD: distribution(result.timings(PHASE_COLD)),
                PHASE_WARM: distribution(result.timings(PHASE_WARM)),
                "all": distribution(result.timings()),
            },
            "phase_split_s": {
                phase: {
                    "n": len([r for r in reps if r["phase"] == phase]),
                    "preparation_s": 0.5,
                    "queue_s": 0.7 if node else None,
                    "execution_s": 2.5 if node else None,
                    "clock_domain": CLUSTER_EVENT_CLOCK if node else LOCAL_MONOTONIC_CLOCK,
                }
                for phase in (*PHASES, "all")
            },
            "phase_labelling": labelling,
            "reps": reps,
            "notes": ["synthetic dry-run record"],
        }

    return {
        "schema": "oai2.worker_contribution/2",
        "generated_at_utc": utc_now(),
        "worker_role": "verification",
        "capacity_class": "verification-execution-only",
        "evidence_caveat": (
            "4 warm observations are not strong tail-latency evidence. With n=4 the nearest-rank "
            "p95 is simply the slowest of four samples, and no claim about a tail should be "
            "drawn from it."
        ),
        "scope_note": (
            "Both cells ran one identical bounded verification command at one pinned revision. "
            "The node is measured as a test-execution worker only. This report makes no claim "
            "about hardware speed, service capacity or output quality."
        ),
        "config": {
            "reps": args.reps,
            "ref": args.ref,
            "pinned_sha": "2983c5ccbfeb4364c41d73ff2aecb19d438939da",
            "ref_resolution": {
                "requested_ref": args.ref,
                "full_sha": "2983c5ccbfeb4364c41d73ff2aecb19d438939da",
                "ref_name": "refs/heads/main",
                "reachability_confirmed": True,
                "note": "synthetic dry-run record",
            },
            "repo_url": args.repo_url,
            "check": args.check,
            "expected_node": args.node,
            "cluster_url": args.cluster_url,
            "poll_interval_s": args.poll_interval,
            "timeout_s": args.timeout,
            "sample_interval_s": args.sample_interval,
        },
        "gates": {"status": "PASS", "exit_code": 0, "failure_count": 0},
        "gate_detail": [
            {"name": "ref_resolved_to_full_sha", "passed": True, "detail": "synthetic"},
            {"name": "cell_a_work_is_matched", "passed": True, "detail": "synthetic"},
            {"name": "cell_b_remote_identity_is_proven", "passed": True, "detail": "synthetic"},
        ],
        "work_identity": {
            "local_argv": "python3 -m unittest -v test_app",
            "local_argv_sha256": "0" * 64,
            "remote_source": "job result checks[] argv + node advertised capabilities",
            "remote_argv": "python3 -m unittest -v test_app",
            "remote_argv_sha256": "0" * 64,
            "remote_advertised_profiles": ["verification", "verification:python-unittest"],
            "proven": True,
            "detail": "synthetic dry-run record",
        },
        "cell_a": cell(
            "A",
            None,
            "observed: rep 0 materialised a fresh checkout, later reps reused it",
        ),
        "cell_b": cell(
            "B",
            args.node,
            "observed: every rep carries worker git-clone/git-fetch/git-checkout evidence",
        ),
        "comparisons": [
            {"id": "a", "title": "as-is", "supported": True, "basis": "synthetic", "result": "synthetic", "n": 3, "caveat": "synthetic"},
            {"id": "b", "title": "matched fresh preparation", "supported": True, "basis": "synthetic", "result": "synthetic", "n": 1, "caveat": "synthetic"},
            {"id": "c", "title": "matched prepared execution", "supported": False, "basis": "synthetic", "result": "synthetic", "n": 0, "caveat": "synthetic"},
            {"id": "d", "title": "useful verification capacity", "supported": True, "basis": "synthetic", "result": "synthetic", "n": 3, "caveat": "synthetic"},
        ],
        "host_resources": {
            "sampling_interval_s": args.sample_interval,
            "sampler_state": "stopped",
            "measured_host": LOCAL_HOST_LABEL,
            "metrics": [
                {"metric": "load_average", "host": LOCAL_HOST_LABEL, "status": "supported", "reason": ""},
                {"metric": "memory_used_pct", "host": LOCAL_HOST_LABEL, "status": "supported", "reason": ""},
            ],
            "unmeasured_host_note": (
                f"{REMOTE_HOST_LABEL}: no metric for this host was sampled; local figures above "
                "must never be read as this node's figures"
            ),
            "overall": {"sample_count": 0, "load1_mean": None, "load1_max": None},
            "notes": ["synthetic dry-run record"],
        },
    }


def run_self_test(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append((name, condition, detail))

    pinned = "2983c5ccbfeb4364c41d73ff2aecb19d438939da"
    local_argv, source = resolve_local_argv(args.check, args.local_argv)
    local_digest = argv_digest(local_argv)
    capabilities = ["verification", "verification:smoke", "verification:pass-probe", "verification:fail-probe", f"verification:{args.check}"]

    # 1. The claim guard fires on every forbidden vocabulary class.
    rejected = 0
    for term in FORBIDDEN_CLAIM_TERMS:
        probe = f"observed value 42 {term} 99"
        try:
            assert_verification_scoped(probe)
        except InferenceClaimError:
            rejected += 1
    check(
        "claim_guard_rejects_all_vocabulary",
        rejected == len(FORBIDDEN_CLAIM_TERMS),
        f"{rejected}/{len(FORBIDDEN_CLAIM_TERMS)} planted claims rejected",
    )
    try:
        assert_verification_scoped("job 7 ok on node p50, wall 4.5s, useful")
        guard_passes = True
    except InferenceClaimError:
        guard_passes = False
    check("claim_guard_allows_verification_wording", guard_passes, "clean report passes guard")

    # 2. Offline argv enforcement.
    try:
        assert_local_argv_offline(local_argv)
        argv_ok = True
    except ConfigError:
        argv_ok = False
    check("local_argv_accepts_verification_profile", argv_ok, f"verification profile allowed: {shlex.join(local_argv)}")
    blocked = 0
    for probe in (
        ["curl", "http://127.0.0.1:8000/v1/chat"],
        ["python3", "-m", "qpipe", "serve", "--model", "x.gguf"],
        ["python3", "bench.py", "--base-url", "http://127.0.0.1:1234"],
    ):
        try:
            assert_local_argv_offline(probe)
        except ConfigError:
            blocked += 1
    check("local_argv_blocks_service_traffic", blocked == 3, f"{blocked}/3 probes blocked")

    # 3. Cluster URL enforcement refuses non-control-plane ports.
    refused = 0
    for probe in ("http://127.0.0.1:8000", "http://127.0.0.1:1234", "http://127.0.0.1:8787"):
        try:
            validate_cluster_url(probe)
        except ConfigError:
            refused += 1
    check("cluster_url_refuses_service_ports", refused == 3, f"{refused}/3 refused")
    try:
        validate_cluster_url(args.cluster_url)
        url_ok = True
    except ConfigError:
        url_ok = False
    check("cluster_url_accepted", url_ok, args.cluster_url)

    # 4. Dry run cannot enqueue.
    try:
        run_enqueue("qpipe", "u", "r", "c", args.cluster_url, dry_run=True)
        enqueue_blocked = False
    except ConfigError:
        enqueue_blocked = True
    check("dry_run_cannot_enqueue", enqueue_blocked, "enqueue refuses under dry run")

    # 5. Percentile / distribution maths.
    dist = distribution([1.0, 2.0, 3.0, 4.0, 5.0])
    check(
        "distribution_maths",
        dist["min_s"] == 1.0 and dist["median_s"] == 3.0 and dist["p95_s"] == 5.0 and dist["n"] == 5,
        "min/median/p95/max correct for n=5 (p95==max by nearest-rank)",
    )
    check("distribution_empty_is_safe", distribution([])["n"] == 0, "empty input safe")

    # 6. Phase labelling is driven by observed preparation, not by the index.
    index_first_reused = classify_phase(tree_reused=True)
    index_later_prepared = classify_phase(tree_reused=False)
    check(
        "phase_label_ignores_rep_index",
        index_first_reused == PHASE_WARM and index_later_prepared == PHASE_COLD,
        "a first rep that reused a tree is warm; a later rep that materialised preparation is cold",
    )
    check(
        "remote_preparation_read_from_evidence",
        observed_remote_preparation(
            {
                "evidence": [
                    {"step": "git-clone", "duration_ms": 900, "returncode": 0},
                    {"step": "git-fetch", "duration_ms": 300, "returncode": 0},
                    {"step": "git-checkout", "duration_ms": 100, "returncode": 0},
                ]
            }
        )["seconds"]
        == 1.3,
        "remote preparation is summed from the worker's own evidence steps",
    )
    check(
        "remote_reports_cold_when_it_clones",
        classify_phase(
            tree_reused=not observed_remote_preparation(
                {"evidence": [{"step": "git-clone", "duration_ms": 10}]}
            )["observed"]
        )
        == PHASE_COLD,
        "a remote rep whose evidence shows a clone is cold, every time",
    )

    # 7. Revision identity must be exact, never a prefix.
    passed, detail = judge_record(
        {"ok": True, "node_id": "p50", "resolved_revision": pinned}, pinned, "p50"
    )
    check("gate_accepts_exact_pinned_sha", passed, detail)
    check(
        "gate_rejects_prefix_only_match",
        not judge_record(
            {"ok": True, "node_id": "p50", "resolved_revision": pinned[:9]}, pinned, "p50"
        )[0],
        "a prefix of the pinned sha is not acceptance",
    )
    check(
        "gate_rejects_wrong_resolved_revision",
        not judge_record(
            {"ok": True, "node_id": "p50", "resolved_revision": "deadbeef" + "0" * 32}, pinned, "p50"
        )[0],
        "a different resolved revision is refused",
    )
    for name, record, node in (
        ("failed_job", {"ok": False, "node_id": "p50", "resolved_revision": pinned}, "p50"),
        ("missing_node", {"ok": True, "node_id": None, "resolved_revision": pinned}, "p50"),
        ("wrong_node", {"ok": True, "node_id": "p49", "resolved_revision": pinned}, "p50"),
    ):
        passed, detail = judge_record(record, pinned, node)
        check(f"gate_rejects_{name}", not passed, detail)

    # 8. Remote identity comes from execution evidence (D1). Every negative case
    #    below must FAIL the gate; none of them may pass by comparing a local
    #    value with itself.
    matching = {
        "ok": True,
        "node_id": "p50",
        "resolved_revision": pinned,
        "checks": [_sample_check_result(args.check, local_argv)],
    }
    good_identity = recover_remote_identity(matching, local_argv, args.check, capabilities)
    check("remote_identity_proven_from_evidence", good_identity.proven, good_identity.detail)
    check(
        "remote_identity_digest_is_the_evidence_not_the_local_copy",
        good_identity.observed_argv_sha256 == local_digest
        and good_identity.observed_argv == [str(a) for a in local_argv],
        "the recovered digest came from the node's own checks[] argv",
    )
    negatives = (
        (
            "remote_identity_rejects_mismatched_argv",
            {**matching, "checks": [_sample_check_result(args.check, ["pytest", "-q", "test_app"])]},
        ),
        (
            "remote_identity_rejects_mismatched_profile",
            {**matching, "checks": [_sample_check_result("smoke", local_argv)]},
        ),
        (
            "remote_identity_rejects_missing_argv",
            {**matching, "checks": [{**_sample_check_result(args.check), "argv": []}]},
        ),
        (
            "remote_identity_rejects_no_checks_block",
            {k: v for k, v in matching.items() if k != "checks"},
        ),
        (
            "remote_identity_rejects_unadvertised_profile",
            {**matching, "checks": [_sample_check_result("pass-probe", local_argv)]},
        ),
    )
    for name, result in negatives:
        identity = recover_remote_identity(result, local_argv, args.check, capabilities)
        check(name, not identity.proven, identity.detail)
    no_capability = recover_remote_identity(matching, local_argv, args.check, ["verification"])
    check(
        "remote_identity_rejects_node_without_profile_capability",
        not no_capability.proven,
        no_capability.detail,
    )

    # 9. A check that failed is refused even when ok was not consulted.
    for label, entry in (
        ("failed_check_exit_code", _sample_check_result(args.check, local_argv, returncode=3)),
        ("failed_check_timeout", _sample_check_result(args.check, local_argv, timed_out=True)),
        ("failed_check_canceled", _sample_check_result(args.check, local_argv, canceled=True)),
    ):
        bad = {**matching, "checks": [entry]}
        failed, why = check_block_failed(bad)
        check(f"gate_rejects_{label}", failed, why)

    # 10. Clock domains (D3). A phase duration is built from two instants of one
    #     clock, and crossing a clock is refused by assertion.
    cluster_enq = Stamp(1_791_000_000.0, CLUSTER_EVENT_CLOCK, "event:job_enqueued")
    cluster_grant = Stamp(1_791_000_000.5, CLUSTER_EVENT_CLOCK, "event:job_leased")
    cluster_finish = Stamp(1_791_000_003.0, CLUSTER_EVENT_CLOCK, "event:job_succeeded")
    local_mono = Stamp(40_487.24, LOCAL_MONOTONIC_CLOCK, "local rep timer")
    queue = phase_duration(cluster_enq, cluster_grant)
    run_phase = phase_duration(cluster_grant, cluster_finish)
    check(
        "cluster_phase_duration_is_built",
        queue.seconds == 0.5 and run_phase.seconds == 2.5 and queue.same_clock,
        f"queue={queue.seconds}s run={run_phase.seconds}s from {queue.clock_domain}",
    )
    mixed = phase_duration(local_mono, cluster_grant)
    check(
        "mixed_clock_refused_by_assertion",
        mixed.seconds is None and not mixed.same_clock,
        mixed.reason,
    )
    try:
        verify_same_clock(local_mono, cluster_grant)
        raised = False
    except ClockDomainError:
        raised = True
    check("clock_domain_assertion_raises_on_mismatch", raised, "verify_same_clock raises across domains")
    missing = phase_duration(cluster_enq, None)
    check(
        "missing_instant_is_unavailable_not_guessed",
        missing.seconds is None and "no end instant" in missing.reason,
        missing.reason,
    )
    check(
        "plausibility_filter_is_not_validation",
        phase_duration(cluster_enq, Stamp(1_791_000_000.0 + 999_999.0, CLUSTER_EVENT_CLOCK, "event:job_succeeded")).seconds is None,
        "an impossible magnitude is filtered out and the report says so instead of passing it",
    )
    check(
        "plausibility_filter_rejects_mixed_clock_arithmetic",
        plausible_duration(1_791_001_913.31 - 40_487.24) is None,
        "a monotonic-minus-epoch subtraction is dropped by the filter",
    )
    check(
        "plausibility_filter_accepts_real_values",
        plausible_duration(1.6966) == 1.6966 and plausible_duration(0.0) == 0.0,
        "credible durations survive the filter unchanged",
    )
    check(
        "plausibility_filter_rejects_negative_and_nonfinite",
        plausible_duration(-1.0) is None
        and plausible_duration(float("nan")) is None
        and plausible_duration(float("inf")) is None
        and plausible_duration(None) is None,
        "negative, non-finite and missing values are all dropped",
    )

    # 11. Sampler states distinguish unsupported from never-ran (D4).
    never = HostSampler(interval=0.2)
    never.summary()
    check(
        "sampler_reports_never_ran",
        never.state() == "never-ran" and never.metric_status("load_average")["status"] == "never-ran",
        "an unstarted sampler says never-ran, which is not the same as unsupported",
    )
    sampler = HostSampler(interval=0.2)
    sampler.start()
    time.sleep(0.3)
    sampler.stop()
    summary = sampler.summary()
    check(
        "host_sampling_non_fatal",
        summary["sample_count"] > 0 and summary["sampler_state"] == "stopped",
        f"{summary['sample_count']} samples on {LOCAL_HOST_LABEL}",
    )
    check(
        "sampler_labels_its_host",
        summary["host"] == LOCAL_HOST_LABEL and all(
            m["host"] == LOCAL_HOST_LABEL for m in (sampler.metric_status("load_average"),)
        ),
        "every local metric names the Mac it describes, never the node",
    )
    for metric in ("load_average", "memory_used_pct"):
        status = sampler.metric_status(metric)
        check(
            f"metric_status_is_explicit_{metric}",
            status["status"] in {"supported", "unsupported", "no-observations"},
            f"{metric}: {status['status']} {status['reason']}",
        )

    # 12. Report rendering of a full synthetic payload survives the claim guard.
    payload = synthetic_payload(args)
    rendered = render_report(payload)
    try:
        assert_verification_scoped(rendered)
        assert_verification_scoped(json.dumps(payload))
        report_ok, report_detail = True, "synthetic report passes claim guard"
    except InferenceClaimError as exc:
        report_ok, report_detail = False, str(exc)
    check("synthetic_report_claim_free", report_ok, report_detail)
    check(
        "synthetic_report_shows_sample_counts",
        rendered.count("n=") >= 10 and "n=0 (no observations)" in rendered,
        "every rendered distribution carries its sample count",
    )
    check(
        "synthetic_report_states_tail_caveat",
        "4 warm observations are not strong tail-latency evidence" in rendered,
        "the report states plainly that 4 warm observations are not tail evidence",
    )
    check(
        "synthetic_report_marks_unsupported_comparison",
        "NOT SUPPORTED" in rendered and "(c)" in rendered,
        "a comparison the architecture cannot support is labelled, not inferred",
    )

    print("=" * 78)
    print("DRY RUN / SELF-TEST -- no cluster call, no enqueue, no remote contact")
    print("=" * 78)
    print("Resolved plan (what a real run would do):")
    print(f"  ref resolution : git ls-remote {args.repo_url} {args.ref} -> full 40-char sha")
    print(f"  cell A prepare : git materialise {args.repo_url} @ pinned sha (shallow)")
    print(f"  cell A work    : {shlex.join(local_argv)}   (from {source})")
    print(f"  cell A digest  : {local_digest[:16]}")
    print(
        f"  cell B enqueue : qpipe cluster enqueue-verification --check {args.check} "
        f"--ref <pinned sha> --url {args.cluster_url}"
    )
    print("  cell B observe : GET /v1/cluster/jobs/<id>, /v1/cluster/events, /v1/cluster/nodes")
    print("  cell B identity: recovered from the job's checks[] argv + advertised capabilities")
    print("  sampler        : started at the beginning of the real window, stopped at the end")
    print(f"  reps           : {args.reps} per cell; phase is labelled from observed preparation")
    print(f"  expected node  : {args.node}")
    print(f"  token source   : ${TOKEN_ENV} from the environment (never printed)")
    print("")
    failed = 0
    for name, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            failed += 1
        print(f"  [{mark}] {name}: {detail}")
    print("")
    print(f"{len(checks) - failed}/{len(checks)} self-test checks passed")
    print("=" * 78)
    if failed:
        print("DRY RUN FAILED")
        return EXIT_GATE_FAILURE
    print("DRY RUN OK -- harness is sound; no measurement was performed.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Real measurement
# ---------------------------------------------------------------------------


def measure(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cluster_url = validate_cluster_url(args.cluster_url)
    local_argv, argv_source = resolve_local_argv(args.check, args.local_argv)
    local_digest = argv_digest(local_argv)
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise ConfigError(
            f"{TOKEN_ENV} is not set in this environment; the harness fails closed "
            "rather than running unauthenticated"
        )

    # The requested ref is resolved to a full immutable sha BEFORE any work is
    # enqueued, and acceptance is exact identity against that sha.
    resolution = resolve_full_sha(args.repo_url, args.ref)
    pinned_sha = resolution["full_sha"]
    if not FULL_SHA_RE.fullmatch(pinned_sha):
        raise ConfigError(f"ref resolution did not produce a full 40-character sha: {pinned_sha!r}")

    client = ClusterClient(cluster_url, token, args.auth_header, args.auth_scheme)
    gates: list[Gate] = [
        Gate(
            "ref_resolved_to_full_sha",
            True,
            f"{resolution['requested_ref']} -> {pinned_sha} ({resolution['note']})",
        )
    ]

    node_facts: dict[str, Any] = {"expected_node": args.node, "observed": None}
    advertised_capabilities: list[str] = []
    try:
        for node in client.nodes():
            node_id = str(first_key(node, "node_id", "id", "name") or "")
            if node_id == args.node:
                observed = {
                    "node_id": node_id,
                    "state": first_key(node, "state", "status"),
                    "capabilities": first_key(node, "capabilities"),
                    "metadata": first_key(node, "metadata"),
                }
                node_facts["observed"] = observed
                capabilities = observed.get("capabilities")
                advertised_capabilities = [str(c) for c in capabilities] if isinstance(capabilities, list) else []
                break
        gates.append(
            Gate(
                "expected_node_advertises_verification",
                bool(node_facts["observed"] and "verification" in advertised_capabilities),
                f"node {args.node} capabilities: {advertised_capabilities or 'NOT FOUND'}",
            )
        )
        gates.append(
            Gate(
                "expected_node_advertises_the_pinned_profile",
                f"verification:{args.check}" in advertised_capabilities,
                f"node {args.node} must advertise verification:{args.check}; "
                f"advertised={advertised_capabilities or 'NOT FOUND'}",
            )
        )
    except RuntimeError as exc:
        gates.append(Gate("cluster_nodes_reachable", False, str(exc)))

    # Host sampling covers the real measurement window and nothing else.
    sampler = HostSampler(args.sample_interval)
    sampler.start()
    windows: list[dict[str, Any]] = []
    try:
        # ---------------- Cell A ----------------
        cell_a = CellResult(name="A", notes=[f"argv source: {argv_source}"])
        workdir = Path(tempfile.mkdtemp(prefix="oai2-cellA-"))
        tree = workdir / "tree"
        tree_reused = False
        try:
            for index in range(args.reps):
                rep_start = time.monotonic()
                prep = 0.0
                # The preparation state is READ BEFORE this rep possibly
                # materialises a tree, so the label below describes what this rep
                # actually did. The rep index is never consulted.
                reused_existing_tree = tree_reused
                if not tree_reused:
                    shutil.rmtree(tree, ignore_errors=True)
                    prep = shallow_prepare(args.repo_url, pinned_sha, tree)
                    tree_reused = True
                    basis = f"rep {index}: materialised a fresh checkout (preparation paid here)"
                else:
                    basis = f"rep {index}: reused the checkout materialised earlier (no preparation)"
                phase = classify_phase(tree_reused=reused_existing_tree)
                started = time.monotonic()
                completed = subprocess.run(
                    local_argv,
                    cwd=tree,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                execution = time.monotonic() - started
                total = prep + execution
                revision = checkout_revision(tree)
                record = {"ok": completed.returncode == 0, "resolved_revision": revision}
                passed = record["ok"] and revision_identical(revision, pinned_sha)
                detail = detail_ok(record, revision, pinned_sha)
                rep_end = time.monotonic()
                windows.append({"cell": "A", "rep": index, "start": rep_start, "end": rep_end})
                cell_a.reps.append(
                    {
                        "rep": index,
                        "phase": phase,
                        "phase_basis": basis,
                        "ok": record["ok"],
                        "useful": passed,
                        "total_s": round(total, 4),
                        "preparation_s": round(prep, 4),
                        "execution_s": round(execution, 4),
                        "requested_ref": args.ref,
                        "resolved_revision": revision,
                        "job_id": None,
                        "node_id": None,
                        "state": "local",
                        "exit_code": completed.returncode,
                        "enqueue_to_grant_s": None,
                        "grant_to_finish_s": None,
                        "observation_overhead_s": round(rep_end - rep_start, 4),
                        "clock_domains": {
                            "preparation_s": LOCAL_MONOTONIC_CLOCK,
                            "execution_s": LOCAL_MONOTONIC_CLOCK,
                            "observation_overhead_s": LOCAL_MONOTONIC_CLOCK,
                        },
                    }
                )
                cell_a.gates.append(Gate(f"cell_a_rep{index}_{phase}", passed, detail))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        # ---------------- Cell B ----------------
        cell_b = CellResult(name="B")
        identity: RemoteIdentity | None = None
        for index in range(args.reps):
            rep_start = time.monotonic()
            try:
                job_id = run_enqueue(args.qpipe, args.repo_url, pinned_sha, args.check, cluster_url)
                record = wait_for_job(client, job_id, args.timeout, args.poll_interval)
                events = client.events(job_id)
            except (RuntimeError, ConfigError) as exc:
                rep_end = time.monotonic()
                windows.append({"cell": "B", "rep": index, "start": rep_start, "end": rep_end})
                cell_b.reps.append(
                    {
                        "rep": index,
                        "phase": classify_phase(tree_reused=False),
                        "phase_basis": f"rep {index}: no job ran, so preparation is unobserved",
                        "ok": False,
                        "useful": False,
                        "total_s": round(rep_end - rep_start, 4),
                        "preparation_s": None,
                        "execution_s": None,
                        "requested_ref": args.ref,
                        "resolved_revision": None,
                        "job_id": None,
                        "node_id": None,
                        "state": "enqueue_or_poll_error",
                        "enqueue_to_grant_s": None,
                        "grant_to_finish_s": None,
                        "observation_overhead_s": round(rep_end - rep_start, 4),
                        "error": str(exc),
                    }
                )
                cell_b.gates.append(Gate(f"cell_b_rep{index}_error", False, str(exc)))
                continue
            rep_end = time.monotonic()
            windows.append({"cell": "B", "rep": index, "start": rep_start, "end": rep_end})

            stamps = cluster_stamps(record, events)
            queue_phase = phase_duration(stamps.get("enqueue"), stamps.get("grant"))
            exec_phase = phase_duration(stamps.get("grant"), stamps.get("finish"))
            result = _result_block(record)
            preparation = observed_remote_preparation(result)
            state = str(first_key(record, "state", "status") or "unknown").lower()
            observed = {
                "ok": bool(first_key(result, "ok")),
                "node_id": first_key(record, "node_id", "node"),
                "resolved_revision": first_key(result, "resolved_revision", "revision", "commit"),
            }
            passed, detail = judge_record(observed, pinned_sha, args.node)
            checks_failed, checks_detail = check_block_failed(result)
            rep_identity = recover_remote_identity(
                result, local_argv, args.check, advertised_capabilities
            )
            if identity is None:
                identity = rep_identity
            else:
                # Every rep must prove the same identity, not just the first.
                if rep_identity.observed_argv_sha256 != identity.observed_argv_sha256:
                    passed = False
                    detail = f"{detail}; remote argv changed between reps"
            cell_b.reps.append(
                {
                    "rep": index,
                    "phase": classify_phase(tree_reused=not preparation["observed"]),
                    "phase_basis": (
                        f"rep {index}: worker evidence shows {preparation['steps']}"
                        if preparation["observed"]
                        else f"rep {index}: no preparation steps in worker evidence"
                    ),
                    "ok": observed["ok"],
                    "useful": passed and rep_identity.proven and not checks_failed,
                    "total_s": round(rep_end - rep_start, 4),
                    "preparation_s": preparation["seconds"],
                    "preparation_steps": preparation["steps"],
                    "execution_s": exec_phase.seconds,
                    "requested_ref": args.ref,
                    "pinned_sha": pinned_sha,
                    "resolved_revision": observed["resolved_revision"],
                    "job_id": job_id,
                    "node_id": observed["node_id"],
                    "state": state,
                    "enqueue_to_grant_s": queue_phase.seconds,
                    "grant_to_finish_s": exec_phase.seconds,
                    "observation_overhead_s": round(rep_end - rep_start, 4),
                    "clock_domains": {
                        "preparation_s": preparation["clock_domain"],
                        "enqueue_to_grant_s": queue_phase.clock_domain,
                        "grant_to_finish_s": exec_phase.clock_domain,
                        "observation_overhead_s": LOCAL_MONOTONIC_CLOCK,
                    },
                    "phase_detail": {
                        "queue": queue_phase.as_dict(),
                        "execution": exec_phase.as_dict(),
                    },
                    "checks_ok": result.get("checks") if isinstance(result.get("checks"), list) else None,
                    "checks_detail": checks_detail,
                    "evidence_count": (
                        len(result["evidence"]) if isinstance(result.get("evidence"), list) else None
                    ),
                    "remote_identity": rep_identity.as_dict(),
                }
            )
            cell_b.gates.append(Gate(f"cell_b_rep{index}", passed and not checks_failed, f"{detail}; {checks_detail}"))
            if not rep_identity.proven:
                cell_b.gates.append(
                    Gate(f"cell_b_rep{index}_remote_identity", False, rep_identity.detail)
                )
            for label, phase in (("queue", queue_phase), ("execution", exec_phase)):
                if phase.seconds is None:
                    cell_b.notes.append(
                        f"rep {index}: {label} duration unavailable -- {phase.reason}"
                    )
    finally:
        sampler.stop()

    # The matched-work gate is now decided by remote evidence alone.
    if identity is None:
        gates.append(
            Gate(
                "cell_a_work_is_matched",
                False,
                "remote identity unproven: no cell B result carried an executable identity, so "
                "the two cells cannot be shown to have run the same work",
            )
        )
    else:
        gates.append(
            Gate(
                "cell_a_work_is_matched",
                identity.proven,
                f"{identity.detail}; local workload digest {local_digest[:16]} from {argv_source}",
            )
        )

    # Attach the resource samples taken during each rep.
    for window in windows:
        target = cell_a if window["cell"] == "A" else cell_b
        summary = sampler.summary((window["start"], window["end"]))
        for rep in target.reps:
            if rep["rep"] == window["rep"] and rep.get("sample_host") is None:
                rep.update(
                    {
                        "sample_count": summary["sample_count"],
                        "sample_host": summary["host"],
                        "load1_mean": summary["load1_mean"],
                        "load1_max": summary["load1_max"],
                        "memory_used_pct_max": summary["memory_used_pct_max"],
                    }
                )

    # ---------------- Assemble ----------------
    gates.extend(cell_a.gates)
    gates.extend(cell_b.gates)
    failures = [g for g in gates if not g.passed]
    status = "FAIL" if failures else "PASS"
    exit_code = EXIT_GATE_FAILURE if failures else EXIT_OK

    def split(result: CellResult) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for phase in (*PHASES, "all"):
            chosen = result.reps if phase == "all" else [r for r in result.reps if r["phase"] == phase]
            preps = [r["preparation_s"] for r in chosen if r.get("preparation_s") is not None]
            queues = [r["enqueue_to_grant_s"] for r in chosen if r.get("enqueue_to_grant_s") is not None]
            execs = [r["grant_to_finish_s"] for r in chosen if r.get("grant_to_finish_s") is not None]
            clocks = {r.get("clock_domains", {}).get("grant_to_finish_s") for r in chosen}
            clocks.discard(None)
            out[phase] = {
                "n": len(chosen),
                "preparation_s": round(statistics.fmean(preps), 4) if preps else None,
                "queue_s": round(statistics.fmean(queues), 4) if queues else None,
                "execution_s": round(statistics.fmean(execs), 4) if execs else None,
                "clock_domain": sorted(clocks)[0] if clocks else "unavailable",
            }
        return out

    def latency(result: CellResult) -> dict[str, Any]:
        return {
            PHASE_COLD: distribution(result.timings(PHASE_COLD)),
            PHASE_WARM: distribution(result.timings(PHASE_WARM)),
            "all": distribution(result.timings()),
        }

    def rate(result: CellResult) -> float | None:
        totals = result.timings()
        if not totals:
            return None
        total = sum(totals)
        if total <= 0.0:
            # Every observation rounded to zero, so there is no elapsed time to
            # divide by. Report the rate as unavailable rather than dividing.
            return None
        return round(60.0 * len(totals) / total, 3)

    a_latency = latency(cell_a)
    b_latency = latency(cell_b)
    a_split = split(cell_a)
    b_split = split(cell_b)

    warm_observations = a_latency[PHASE_WARM]["n"]
    comparisons = [
        {
            "id": "a",
            "title": "as-is local versus remote user-visible latency",
            "supported": a_latency["all"]["n"] > 0 and b_latency["all"]["n"] > 0,
            "basis": (
                "every rep end to end as a user would wait: local checkout+command versus "
                "enqueue+queue+node execution"
            ),
            "result": (
                f"local median {a_latency['all']['median_s']}s (n={a_latency['all']['n']}) versus "
                f"remote median {b_latency['all']['median_s']}s (n={b_latency['all']['n']})"
                if a_latency["all"]["n"] and b_latency["all"]["n"]
                else "not available: a cell produced no observations"
            ),
            "n": min(a_latency["all"]["n"], b_latency["all"]["n"]),
            "caveat": (
                f"this mixes preparation and execution; only {warm_observations} warm local "
                "observation(s) exist, and 4 warm observations are not strong tail-latency evidence"
            ),
        },
        {
            "id": "b",
            "title": "matched fresh-preparation",
            "supported": a_latency[PHASE_COLD]["n"] > 0 and b_latency[PHASE_COLD]["n"] > 0,
            "basis": "reps that both cells observed paying their own preparation",
            "result": (
                f"local preparation+run median {a_latency[PHASE_COLD]['median_s']}s (n="
                f"{a_latency[PHASE_COLD]['n']}) versus remote preparation+run median "
                f"{b_latency[PHASE_COLD]['median_s']}s (n={b_latency[PHASE_COLD]['n']})"
                if a_latency[PHASE_COLD]["n"] and b_latency[PHASE_COLD]["n"]
                else "not available: a cell produced no fresh-preparation observation"
            ),
            "n": min(a_latency[PHASE_COLD]["n"], b_latency[PHASE_COLD]["n"]),
            "caveat": "n is small because only the first local rep pays preparation",
        },
        {
            "id": "c",
            "title": "matched prepared-execution (work already on the node)",
            "supported": b_latency[PHASE_WARM]["n"] > 0 and a_latency[PHASE_WARM]["n"] > 0,
            "basis": (
                "reps that both cells observed reusing an already-prepared tree; the worker makes "
                "a disposable clone per job, so it exposes no such rep"
            ),
            "result": (
                "measured"
                if b_latency[PHASE_WARM]["n"] > 0 and a_latency[PHASE_WARM]["n"] > 0
                else "NOT SUPPORTED by the current worker architecture: the worker materialises a "
                "disposable clone per job (its own evidence carries git-clone, git-fetch and "
                "git-checkout for every job), so no remote rep reuses a prepared tree and the "
                "matched prepared-execution comparison cannot be made. It is left unmeasured rather "
                "than approximated."
            ),
            "n": min(a_latency[PHASE_WARM]["n"], b_latency[PHASE_WARM]["n"]),
            "caveat": "would require a worker mode that reuses one checkout across jobs",
        },
        {
            "id": "d",
            "title": "useful verification capacity under representative approved work",
            "supported": cell_b.useful_completed() > 0,
            "basis": (
                f"jobs that passed every correctness gate on the expected node at exactly the "
                f"pinned sha, running the approved profile {args.check!r} -- one bounded "
                "verification command, not a representative workload sample"
            ),
            "result": (
                f"{cell_b.useful_completed()} of {len(cell_b.reps)} remote reps completed useful "
                f"verified work; {cell_a.useful_completed()} of {len(cell_a.reps)} local reps did"
            ),
            "n": cell_b.useful_completed(),
            "caveat": (
                "one approved profile at one revision is not a representative workload, and the "
                "node advertises max_leases 1, so this bounds serial capacity only"
            ),
        },
    ]

    host_resources = {
        "sampling_interval_s": args.sample_interval,
        "sampler_state": sampler.state(),
        "measured_host": LOCAL_HOST_LABEL,
        "metrics": [sampler.metric_status("load_average"), sampler.metric_status("memory_used_pct")],
        "unmeasured_host_note": (
            f"{REMOTE_HOST_LABEL}: no metric for this host was sampled, so the local figures above "
            "must never be read as this node's figures"
        ),
        "overall": sampler.summary(),
        "per_rep": [
            {
                "cell": w["cell"],
                "rep": w["rep"],
                "phase": next(
                    (r["phase"] for r in (cell_a.reps if w["cell"] == "A" else cell_b.reps) if r["rep"] == w["rep"]),
                    None,
                ),
                "host": LOCAL_HOST_LABEL,
                "sample_count": sampler.summary((w["start"], w["end"]))["sample_count"],
            }
            for w in windows
        ],
        "notes": sampler.notes
        + [
            "sampling is best effort: an unavailable metric is reported as null and never fails "
            "the run",
            "a metric reported as unsupported means this platform cannot provide it; a sampler "
            "reported as never-ran means no sample was taken at all -- these are not the same",
        ],
    }

    payload: dict[str, Any] = {
        "schema": "oai2.worker_contribution/2",
        "generated_at_utc": utc_now(),
        "worker_role": "verification",
        "capacity_class": "verification-execution-only",
        "evidence_caveat": (
            f"{warm_observations} warm local observation(s) in this run. 4 warm observations are "
            "not strong tail-latency evidence: with nearest-rank percentiles the p95 of four "
            "samples is simply the slowest of the four, so no tail conclusion may be drawn from it."
        ),
        "scope_note": (
            "Both cells ran one identical bounded verification command at one pinned revision, "
            "with cell B's identity recovered from the node's own execution evidence. The node is "
            "measured as a test-execution worker only. This report makes no claim about hardware "
            "speed, service capacity or output quality."
        ),
        "config": {
            "reps": args.reps,
            "ref": args.ref,
            "pinned_sha": pinned_sha,
            "ref_resolution": resolution,
            "repo_url": args.repo_url,
            "check": args.check,
            "expected_node": args.node,
            "cluster_url": cluster_url,
            "poll_interval_s": args.poll_interval,
            "timeout_s": args.timeout,
            "sample_interval_s": args.sample_interval,
        },
        "node_facts": node_facts,
        "gates": {"status": status, "exit_code": exit_code, "failure_count": len(failures)},
        "gate_detail": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in gates],
        "work_identity": (identity or RemoteIdentity(proven=False, detail="no cell B result")).as_dict()
        | {
            "local_argv": shlex.join(local_argv),
            "local_argv_sha256": local_digest,
            "remote_source": "the job result's checks[] argv and the node's advertised capabilities",
        },
        "cell_a": {
            "cell": "A",
            "name": "mac-local",
            "work_description": shlex.join(local_argv),
            "work_argv": local_argv,
            "work_argv_sha256": local_digest,
            "correct": all(r["ok"] for r in cell_a.reps) and bool(cell_a.reps),
            "completed_useful_jobs": cell_a.useful_completed(),
            "completed_useful_jobs_per_minute": rate(cell_a),
            "latency_s": a_latency,
            "preparation_s": {
                "cold": a_split[PHASE_COLD]["preparation_s"],
                "warm": a_split[PHASE_WARM]["preparation_s"],
                "note": (
                    "measured from a local monotonic timer; warm reps reuse the prepared tree, so "
                    "preparation is absent by observation rather than assumed to be zero"
                ),
            },
            "phase_split_s": a_split,
            "phase_labelling": (
                "observed per rep: a rep that materialised a checkout is cold, a rep that reused "
                "one is warm; the rep index is not an input"
            ),
            "reps": cell_a.reps,
            "notes": cell_a.notes,
        },
        "cell_b": {
            "cell": "B",
            "name": "fleet-p50",
            "work_description": shlex.join(local_argv),
            "work_argv": local_argv,
            "work_argv_sha256": local_digest,
            "correct": all(r["ok"] for r in cell_b.reps) and bool(cell_b.reps),
            "completed_useful_jobs": cell_b.useful_completed(),
            "completed_useful_jobs_per_minute": rate(cell_b),
            "latency_s": b_latency,
            "phase_split_s": b_split,
            "preparation_s": {
                "cold": b_split[PHASE_COLD]["preparation_s"],
                "warm": b_split[PHASE_WARM]["preparation_s"],
                "note": (
                    "summed from the worker's own git-clone/git-fetch/git-checkout evidence steps; "
                    "the worker makes a disposable clone per job, so in practice every remote rep "
                    "pays preparation and none is warm"
                ),
            },
            "phase_labelling": (
                "observed per rep from the worker's preparation evidence; a rep whose evidence "
                "carries clone/fetch/checkout steps is cold regardless of its index"
            ),
            "reps": cell_b.reps,
            "notes": cell_b.notes,
        },
        "comparisons": comparisons,
        "host_resources": host_resources,
    }
    return payload, exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measure_worker_contribution.py",
        description=(
            "Matched A/B measurement of fleet-worker contribution for one pinned "
            "revision: run the same bounded verification command on this Mac versus "
            "on the cluster worker node. The requested ref is first resolved to a full "
            "40-character sha, cell B's identity is recovered from the node's own "
            "execution evidence, and each rep's cold/warm label comes from observed "
            "preparation rather than from its index."
        ),
        epilog=(
            "This harness measures test-execution capacity only. It will refuse to "
            "print any generative-performance claim. It contacts nothing except the "
            "cluster control plane and the git remote for ref resolution. Exit 0 = all "
            "gates passed, 1 = correctness gate failure, 2 = configuration error."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reps", type=int, default=5, help="repetitions per cell; phase is labelled from observed preparation")
    parser.add_argument("--ref", default="HEAD", help="pinned revision; resolved to a full 40-character sha before any job is enqueued")
    parser.add_argument("--repo-url", required=True, help="repository the worker clones and the Mac checks out")
    parser.add_argument("--check", default=DEFAULT_CHECK, help="worker check profile; also defines cell A's argv")
    parser.add_argument("--node", default="p50", help="the node the fleet job is required to execute on")
    parser.add_argument("--json-out", type=Path, help="write the machine-readable report to this path")
    parser.add_argument(
        "--cluster-url",
        default=os.environ.get(URL_ENV, DEFAULT_CLUSTER_URL),
        help=f"cluster control plane (or ${URL_ENV})",
    )
    parser.add_argument("--qpipe", default=DEFAULT_QPIPE, help="path to the qpipe CLI")
    parser.add_argument("--poll-interval", type=float, default=0.2, help="job poll interval in seconds")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-job wall-clock timeout in seconds")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="local host resource sampling interval in seconds")
    parser.add_argument("--auth-header", default="Authorization", help="HTTP header carrying the cluster token")
    parser.add_argument("--auth-scheme", default="Bearer", help="scheme prefix for the cluster token")
    parser.add_argument(
        "--local-argv",
        default=None,
        help="override cell A argv; by default it is read from deploy/worker/profiles.json so both cells match",
    )
    parser.add_argument(
        "--dry-run",
        "--self-test",
        dest="dry_run",
        action="store_true",
        help="validate config, run internal self-tests and print the plan; makes zero cluster calls",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the human-readable report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    if args.reps == 1 and not args.dry_run:
        print(
            "warning: --reps 1 yields a single fresh-preparation observation and no warm "
            "replay; use --reps 2 or more for a meaningful result",
            file=sys.stderr,
        )

    if args.dry_run:
        return run_self_test(args)

    try:
        payload, exit_code = measure(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except RuntimeError as exc:
        print(f"measurement error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        rendered = assert_verification_scoped(render_report(payload))
        serialised = assert_verification_scoped(json.dumps(payload, indent=2, sort_keys=True))
    except InferenceClaimError as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_GATE_FAILURE

    if not args.quiet:
        print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialised + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"\njson report: {args.json_out}")

    gates = payload["gates"]
    if gates["status"] != "PASS":
        print(
            f"CORRECTNESS GATES FAILED ({gates['failure_count']} of "
            f"{len(payload['gate_detail'])}); statistics above are VOID.",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
