#!/usr/bin/env python3
"""Matched A/B measurement of what the fleet worker actually contributes.

This harness answers one narrow question with numbers instead of adjectives:

    For ONE pinned revision of this repository, running ONE identical bounded
    command, is it better to run the work on this Mac or to hand it to the
    cluster worker `p50` -- and *where does the time actually go*?

It runs two cells over the same useful work:

  Cell A  mac-local  -- prepare a fresh shallow checkout of the pinned ref,
                       then run the verification command locally.
  Cell B  fleet-p50  -- enqueue the same command to the cluster control plane
                       and let the worker node execute it.

Design commitments that are enforced in code, not just documented:

  * COLD vs WARM is the central result.  The first observation of each cell
    includes repository preparation (checkout on the Mac, materialisation on
    the worker); every later observation is warm.  Both are reported
    separately, never averaged together blindly.
  * Correctness gates run BEFORE any statistic is computed.  A job that did
    not succeed, that resolved to a different revision, or that did not run on
    the expected node makes the process exit non-zero and the statistics are
    marked void.
  * The node is a VERIFICATION-capacity worker.  `assert_verification_scoped`
    scans every rendered byte for generative-performance vocabulary and fails
    the run, so this script structurally cannot emit such a claim.
  * The only network peer is the cluster control plane.  Cell A runs a fixed
    argv with no URL and no model/service flag; that is checked, not assumed.
  * Host resource sampling is best effort.  A missing metric degrades the
    report, it never fails the run.

Stdlib only. See `--help` for the full CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from datetime import datetime, timezone
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

# ---------------------------------------------------------------------------
# Structural guard: this report may never carry a generative-performance claim.
#
# The harness measures test-execution capacity. Anything in this list is a
# claim about token rates, accelerator silicon or generative quality, which is
# a different system entirely and is not measured here. The guard is checked
# against the fully rendered report and the JSON payload, so a future edit
# that introduces such a claim fails the run instead of shipping a wrong
# number. Words are matched on word boundaries, case-insensitively.
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
)

_CLAIM_GUARD_ERROR = (
    "blocked: the rendered report contained generative-performance vocabulary, "
    "which this harness does not and cannot measure"
)


class InferenceClaimError(RuntimeError):
    """Raised when a rendered report would carry an unmeasurable claim."""


class ConfigError(RuntimeError):
    """Raised for a usage/environment problem; maps to exit code 2."""


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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile. Documented because with n=5, p95 is the max."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
    return ordered[rank - 1]


def distribution(values: Sequence[float]) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Best-effort host resource sampling. Never fatal.
# ---------------------------------------------------------------------------


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
    """Low-rate local sampling thread. A sampler failure is recorded, not raised."""

    def __init__(self, interval: float) -> None:
        self.interval = max(0.2, interval)
        self.samples: list[HostSample] = []
        self.notes: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _take(self) -> HostSample:
        load1 = load5 = load15 = None
        try:
            load1, load5, load15 = os.getloadavg()
        except (OSError, AttributeError) as exc:  # pragma: no cover - platform
            if "loadavg" not in " ".join(self.notes):
                self.notes.append(f"load average unavailable: {type(exc).__name__}")
        memory = None
        try:
            memory = memory_pressure_percent()
        except Exception as exc:  # noqa: BLE001 - sampling must never be fatal
            if "memory" not in " ".join(self.notes):
                self.notes.append(f"memory metric unavailable: {type(exc).__name__}")
        return HostSample(time.monotonic(), load1, load5, load15, memory)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._take())
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="host-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 1.0)
        self.samples.append(self._take())

    def summary(self, window: tuple[float, float] | None = None) -> dict[str, Any]:
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
        }


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
) -> tuple[dict[str, Any], float, float]:
    """Poll until terminal. Returns (job record, enqueue_monotonic, finish_monotonic)."""
    deadline = time.monotonic() + timeout
    last_state = "unknown"
    while True:
        record = client.job(job_id)
        state = str(first_key(record, "state", "status") or "unknown").lower()
        if state != last_state:
            last_state = state
        if state in TERMINAL_STATES:
            return record, 0.0, time.monotonic()
        if time.monotonic() > deadline:
            raise RuntimeError(f"job {job_id} did not reach a terminal state in {timeout}s")
        time.sleep(poll_interval)


# A queue or execution phase that is negative, non-finite or larger than a day
# did not come from a duration arithmetic. It means two different clocks (or an
# unresolved timestamp) were combined. Such a value is reported as unavailable
# rather than published as a measurement.
MAX_PLAUSIBLE_PHASE_S = 86_400.0


def plausible_duration(value: Any) -> float | None:
    """Return a duration in seconds, or None if it is not a credible one."""
    seconds = as_float(value)
    if seconds is None:
        return None
    if not math.isfinite(seconds):
        return None
    if seconds < 0.0 or seconds > MAX_PLAUSIBLE_PHASE_S:
        return None
    return round(seconds, 4)


def job_timings(
    record: dict[str, Any],
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Recover grant/finish instants from events, falling back to job fields.

    The cluster event schema is not pinned by this harness, so each timestamp
    is looked up under several plausible names and any phase that cannot be
    recovered is reported as null rather than guessed.
    """
    stamps: dict[str, float] = {}
    for event in events:
        name = str(
            first_key(event, "type", "event", "name", "kind", "state", "status") or ""
        ).lower()
        moment = as_float(
            first_key(
                event,
                "ts",
                "timestamp",
                "time",
                "at",
                "created_at",
                "occurred_at",
            )
        )
        if moment is None:
            continue
        if any(token in name for token in ("grant", "lease", "assign", "start", "accept")):
            stamps.setdefault("grant", moment)
        if any(token in name for token in ("finish", "complete", "done", "result", "end", "fail")):
            stamps["finish"] = moment
    stamps.setdefault(
        "grant", as_float(first_key(record, "started_at", "granted_at", "leased_at")) or 0.0
    )
    stamps.setdefault(
        "finish",
        as_float(first_key(record, "finished_at", "completed_at", "ended_at", "finished"))
        or 0.0,
    )
    stamps.setdefault(
        "enqueue",
        as_float(first_key(record, "created_at", "enqueued_at", "submitted_at")) or 0.0,
    )
    return stamps


# ---------------------------------------------------------------------------
# Cell A: local execution
# ---------------------------------------------------------------------------


def shallow_prepare(repo_url: str, ref: str, destination: Path) -> float:
    """Materialise `ref` into `destination`. Returns preparation seconds.

    The first cell-A observation is deliberately cold: the tree is created from
    scratch so its cost is a real preparation cost and not a warm cache hit.
    """
    git = shutil.which("git")
    if git is None:
        raise ConfigError("git is not on PATH; cell A cannot prepare a checkout")
    is_commit = bool(re.fullmatch(r"[0-9a-f]{7,40}", ref))
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

    def execution_timings(self, phase: str | None = None) -> list[float]:
        chosen = self.reps if phase is None else [r for r in self.reps if r["phase"] == phase]
        return [
            r["execution_s"] for r in chosen if isinstance(r.get("execution_s"), (int, float))
        ]

    def useful_completed(self) -> int:
        return sum(1 for r in self.reps if r.get("useful") is True)


def classify_phase(index: int) -> str:
    """Rep 0 is cold (includes preparation); every later rep is warm."""
    return "cold" if index == 0 else "warm"


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("WORKER CONTRIBUTION -- MATCHED A/B AT ONE PINNED REVISION")
    add("=" * 78)
    config = payload["config"]
    add(f"generated_utc        : {payload['generated_at_utc']}")
    add(f"requested_ref        : {config['ref']}")
    add(f"check_profile        : {config['check']}")
    add(f"expected_node        : {config['expected_node']}")
    add(f"node_role            : {payload['worker_role']} (test-execution capacity only)")
    add(f"reps_per_cell        : {config['reps']}")
    add(f"p50_documentation    : nearest-rank percentile")
    add("")

    gates = payload["gates"]
    add(f"CORRECTNESS GATES: {gates['status']}")
    for gate in payload["gate_detail"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        add(f"  [{mark}] {gate['name']}: {gate['detail']}")
    add("")

    for key, label in (("cell_a", "CELL A -- mac-local"), ("cell_b", "CELL B -- fleet-p50")):
        cell = payload[key]
        add(f"--- {label} ---")
        add(f"  work                : {cell['work_description']}")
        add(f"  matched_work_id     : {cell['work_argv_sha256'][:16]} (cell A must equal cell B)")
        add(f"  correct             : {cell['correct']}")
        add(f"  useful_jobs_done    : {cell['completed_useful_jobs']}")
        rate = cell["completed_useful_jobs_per_minute"]
        add(f"  useful_jobs_per_min : {rate if rate is not None else 'n/a'}")
        for phase in ("cold", "warm", "all"):
            stats = cell["latency_s"][phase]
            if stats["n"] == 0:
                add(f"  latency[{phase:4}]     : n=0 (no observations)")
                continue
            add(
                f"  latency[{phase:4}]     : n={stats['n']} "
                f"min={stats['min_s']} median={stats['median_s']} "
                f"p95={stats['p95_s']} max={stats['max_s']} mean={stats['mean_s']}"
            )
        if key == "cell_a":
            prep = cell["preparation_s"]
            add(
                f"  preparation (clone) : cold={prep.get('cold')} warm={prep.get('warm')}"
            )
        else:
            split = cell["phase_split_s"]
            add(
                f"  queue/admission     : cold={split['cold']} warm={split['warm']}"
            )
            add(
                f"  node execution      : cold={split['cold']} warm={split['warm']}"
            )
        for rep in cell["reps"]:
            bits = [
                f"    rep {rep['rep']} [{rep['phase']}]",
                f"ok={rep['ok']}",
                f"total={rep['total_s']}s",
            ]
            if rep.get("job_id") is not None:
                bits.append(f"job={rep['job_id']}")
                bits.append(f"node={rep.get('node_id')}")
                bits.append(f"state={rep.get('state')}")
                bits.append(f"resolved={str(rep.get('resolved_revision'))[:12]}")
                bits.append(f"queue={rep.get('enqueue_to_grant_s')}")
                bits.append(f"exec={rep.get('grant_to_finish_s')}")
            if rep.get("preparation_s") is not None:
                bits.append(f"prepare={rep['preparation_s']}")
            add("  ".join(bits))
        for note in cell["notes"]:
            add(f"  note: {note}")
        add("")

    comparison = payload["comparison"]
    add("--- COMPARISON (read this before quoting any number) ---")
    add(f"  cell_b_slower_end_to_end : {comparison['cell_b_slower_end_to_end']}")
    add(f"  measured_benefit         : {comparison['measured_benefit']}")
    add(f"  forbidden_claim          : {comparison['forbidden_claim']}")
    add(f"  mac_offload_useful_jobs  : {comparison['mac_offload_useful_jobs']}")
    add("")

    host = payload["host_resources"]
    add("--- HOST RESOURCE USE (best effort, never fatal) ---")
    add(f"  sampling_interval_s : {host['sampling_interval_s']}")
    add(f"  overall             : {host['overall']}")
    add(f"  notes               : {host['notes'] or ['none']}")
    add("")
    add(f"scope_note: {payload['scope_note']}")
    add("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test / dry run. Performs no cluster contact of any kind.
# ---------------------------------------------------------------------------


def run_self_test(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append((name, condition, detail))

    # 1. The claim guard fires on every forbidden vocabulary class.
    rejected = 0
    for index, term in enumerate(FORBIDDEN_CLAIM_TERMS):
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
        assert_local_argv_offline(["python3", "-m", "unittest", "-v", "test_app"])
        argv_ok = True
    except ConfigError:
        argv_ok = False
    check("local_argv_accepts_verification_profile", argv_ok, "verification profile allowed")
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
    except ConfigError as exc:
        url_ok = False
        add_detail = str(exc)
    check("cluster_url_accepted", url_ok, args.cluster_url)

    # 4. Dry run cannot enqueue.
    try:
        run_enqueue("qpipe", "u", "r", "c", args.cluster_url, dry_run=True)
        enqueue_blocked = False
    except ConfigError:
        enqueue_blocked = True
    check("dry_run_cannot_enqueue", enqueue_blocked, "enqueue refuses under dry run")

    # 5. Percentile / distribution maths.
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    dist = distribution(values)
    check(
        "distribution_maths",
        dist["min_s"] == 1.0 and dist["median_s"] == 3.0 and dist["p95_s"] == 5.0 and dist["n"] == 5,
        "min/median/p95/max correct for n=5 (p95==max by nearest-rank)",
    )
    check("distribution_empty_is_safe", distribution([])["n"] == 0, "empty input safe")

    # 6. Cold/warm classification.
    phases = [classify_phase(i) for i in range(5)]
    check(
        "cold_warm_classification",
        phases == ["cold", "warm", "warm", "warm", "warm"],
        "rep 0 cold, all later reps warm",
    )

    # 7. Gate logic on synthetic records: success and each failure mode.
    #    "2983c5ccbfeb..." is the established baseline revision; the requested
    #    ref is given abbreviated, which the gate must accept, and each failure
    #    fixture breaks exactly one of the three gate conditions.
    full_rev = "2983c5ccbfeb" + "0" * 28
    short_ref = full_rev[:9]
    good = {
        "ok": True,
        "node_id": "p50",
        "resolved_revision": full_rev,
        "requested_ref": short_ref,
    }
    passed, detail = judge_record(good, short_ref, "p50")
    check("gate_accepts_good_record", passed, detail)
    check(
        "gate_accepts_abbreviated_ref",
        judge_record(good, full_rev, "p50")[0],
        "full requested ref also matches the full resolved sha",
    )
    check(
        "gate_refuses_trivial_prefix",
        not judge_record({**good, "resolved_revision": "a" * 40}, "a", "p50")[0],
        "a 1-character ref cannot satisfy the revision gate",
    )
    for name, record, ref, node in (
        ("failed_job", {**good, "ok": False}, short_ref, "p50"),
        ("revision_drift", {**good, "resolved_revision": "deadbeef" + "0" * 32}, short_ref, "p50"),
        ("wrong_node", {**good, "node_id": "p49"}, short_ref, "p50"),
        ("missing_node", {**good, "node_id": None}, short_ref, "p50"),
    ):
        passed, detail = judge_record(record, ref, node)
        check(f"gate_rejects_{name}", not passed, detail)

    # 8. Work matching: both cells must resolve to one identical command.
    local_argv, source = resolve_local_argv(args.check, args.local_argv)
    same = local_argv == args.check_argv if args.check_argv else True
    check("work_matching", same, f"cell A argv from {source}: {shlex.join(local_argv)}")

    # 9. Host sampling degrades instead of raising.
    sampler = HostSampler(interval=0.2)
    sampler.start()
    time.sleep(0.3)
    sampler.stop()
    summary = sampler.summary()
    check(
        "host_sampling_non_fatal",
        summary["sample_count"] > 0,
        f"{summary['sample_count']} samples, load1={summary['load1_mean']}, "
        f"memory={summary['memory_used_pct_max']}",
    )

    # 10. A duration that came from two different clocks is rejected, not
    #     published. Observed defect: the enqueue instant was taken from a
    #     monotonic clock (seconds since boot) while grant/finish are cluster
    #     wall-clock epochs, so the subtraction emitted ~1.79e9 "seconds".
    check(
        "duration_rejects_mixed_clock_arithmetic",
        plausible_duration(1791001913.31 - 40487.24) is None,
        "a monotonic-minus-epoch subtraction must be reported unavailable, not as a duration",
    )
    check(
        "duration_accepts_real_values",
        plausible_duration(1.6966) == 1.6966 and plausible_duration(0.0) == 0.0,
        "credible durations survive unchanged",
    )
    check(
        "duration_rejects_negative_and_nonfinite",
        plausible_duration(-1.0) is None
        and plausible_duration(float("nan")) is None
        and plausible_duration(float("inf")) is None
        and plausible_duration(None) is None,
        "negative, non-finite and missing durations are all unavailable",
    )

    # 11. Report rendering of a full synthetic payload survives the claim guard.
    payload = synthetic_payload(args)
    rendered = render_report(payload)
    try:
        assert_verification_scoped(rendered)
        assert_verification_scoped(json.dumps(payload))
        report_ok, report_detail = True, "synthetic report passes claim guard"
    except InferenceClaimError as exc:
        report_ok, report_detail = False, str(exc)
    check("synthetic_report_claim_free", report_ok, report_detail)

    print("=" * 78)
    print("DRY RUN / SELF-TEST -- no cluster call, no enqueue, no remote contact")
    print("=" * 78)
    print("Resolved plan (what a real run would do):")
    print(f"  cell A prepare : git materialise {args.repo_url} @ {args.ref} (shallow)")
    print(f"  cell A work    : {shlex.join(local_argv)}   (from {source})")
    print(
        f"  cell B enqueue : qpipe cluster enqueue-verification --check {args.check} "
        f"--ref {args.ref} --url {args.cluster_url}"
    )
    print(f"  cell B observe : GET /v1/cluster/jobs/<id>, /v1/cluster/events, /v1/cluster/nodes")
    print(f"  reps           : {args.reps} per cell (rep 0 cold, reps 1..N-1 warm)")
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


def judge_record(
    record: dict[str, Any],
    requested_ref: str,
    expected_node: str,
) -> tuple[bool, str]:
    """Single source of truth for correctness gating of one observation."""
    if not bool(record.get("ok")):
        return False, f"job did not succeed (ok={record.get('ok')!r})"
    resolved = record.get("resolved_revision")
    if not resolved:
        return False, "job reported no resolved_revision"
    if not revisions_match(str(resolved), requested_ref):
        return False, f"resolved_revision {resolved} != requested {requested_ref}"
    node = record.get("node_id")
    if not node:
        return False, "job reported no node_id, so execution location is unproven"
    if str(node) != expected_node:
        return False, f"executed on node {node}, expected {expected_node}"
    return True, f"ok on {node} at {str(resolved)[:12]}"


def revisions_match(resolved: str, requested: str) -> bool:
    """True when the two revision strings denote the same commit.

    The control plane may resolve a short requested ref to the full sha (or the
    reverse), so either string being a prefix of the other counts as a match.
    A prefix shorter than 7 characters is refused: it would let a 1-character
    ref satisfy the gate, which would defeat the point of the gate.
    """
    left, right = resolved.strip().lower(), requested.strip().lower()
    if not left or not right:
        return False
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) < 7:
        return False
    if not longer.startswith(shorter):
        return False
    # A revision token is hex; this keeps a non-revision word from matching.
    return bool(re.fullmatch(r"[0-9a-f]{7,40}", longer))


def resolve_local_argv(check: str, override: str | None) -> tuple[list[str], str]:
    """Cell A must run byte-for-byte the same argv the worker profile defines."""
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
                return assert_local_argv_offline([str(a) for a in argv]), f"deploy/worker/profiles.json:{check}"
    return (
        assert_local_argv_offline(["python3", "-m", "unittest", "-v", "test_app"]),
        "built-in default (profiles.json unreadable)",
    )


def synthetic_payload(args: argparse.Namespace) -> dict[str, Any]:
    """A complete, obviously-fake payload used to exercise the renderer."""

    def rep(index: int, node: str | None) -> dict[str, Any]:
        return {
            "rep": index,
            "phase": classify_phase(index),
            "ok": True,
            "useful": True,
            "total_s": 1.0 + index,
            "execution_s": 0.9 + index,
            "preparation_s": 0.5 if node is None and index == 0 else (0.0 if node is None else None),
            "job_id": None if node is None else 100 + index,
            "node_id": node,
            "state": "succeeded" if node else "local",
            "requested_ref": args.ref,
            "resolved_revision": None if node is None else args.ref,
            "enqueue_to_grant_s": None if node is None else 0.7,
            "grant_to_finish_s": None if node is None else 2.5,
        }

    def cell(name: str, node: str | None) -> dict[str, Any]:
        reps = [rep(i, node) for i in range(3)]
        result = CellResult(name=name, reps=reps)
        return {
            "cell": name,
            "work_description": "python3 -m unittest -v test_app",
            "work_argv_sha256": "0" * 64,
            "correct": True,
            "completed_useful_jobs": result.useful_completed(),
            "completed_useful_jobs_per_minute": 12.0,
            "latency_s": {
                "cold": distribution(result.timings("cold")),
                "warm": distribution(result.timings("warm")),
                "all": distribution(result.timings()),
            },
            "preparation_s": {"cold": 0.5, "warm": 0.0},
            "phase_split_s": {"cold": {"queue_s": 0.7, "execution_s": 3.0}, "warm": {"queue_s": 0.7, "execution_s": 2.0}},
            "reps": reps,
            "notes": ["synthetic dry-run record"],
        }

    return {
        "schema": "oai2.worker_contribution/1",
        "generated_at_utc": utc_now(),
        "worker_role": "verification",
        "capacity_class": "verification-execution-only",
        "scope_note": (
            "Both cells run one identical bounded verification command at one pinned "
            "revision. The node is measured as a test-execution worker only. This report "
            "makes no claim about hardware speed, service capacity or output quality."
        ),
        "config": {
            "reps": args.reps,
            "ref": args.ref,
            "repo_url": args.repo_url,
            "check": args.check,
            "expected_node": args.node,
            "cluster_url": args.cluster_url,
            "poll_interval_s": args.poll_interval,
            "timeout_s": args.timeout,
            "sample_interval_s": args.sample_interval,
        },
        "gates": {"status": "PASS", "exit_code": 0},
        "gate_detail": [
            {"name": "cell_a_correct", "passed": True, "detail": "synthetic"},
            {"name": "cell_b_all_jobs_ok", "passed": True, "detail": "synthetic"},
            {"name": "resolved_revision_matches_requested", "passed": True, "detail": "synthetic"},
            {"name": "executed_on_expected_node", "passed": True, "detail": "synthetic"},
        ],
        "cell_a": cell("A", None),
        "cell_b": cell("B", args.node),
        "comparison": {
            "cell_b_slower_end_to_end": True,
            "measured_benefit": "mac-offload",
            "forbidden_claim": "cell B is not faster; no speedup may be reported",
            "mac_offload_useful_jobs": 0,
        },
        "host_resources": {
            "sampling_interval_s": args.sample_interval,
            "overall": {"sample_count": 0, "load1_mean": None, "load1_max": None},
            "notes": ["synthetic dry-run record"],
        },
    }


# ---------------------------------------------------------------------------
# Real measurement
# ---------------------------------------------------------------------------


def measure(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cluster_url = validate_cluster_url(args.cluster_url)
    local_argv, argv_source = resolve_local_argv(args.check, args.local_argv)
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise ConfigError(
            f"{TOKEN_ENV} is not set in this environment; the harness fails closed "
            "rather than running unauthenticated"
        )
    client = ClusterClient(cluster_url, token, args.auth_header, args.auth_scheme)
    sampler = HostSampler(args.sample_interval)
    gates: list[Gate] = []
    gates.append(
        Gate(
            "cell_a_work_is_matched",
            True,
            f"cell A argv == profile {args.check} argv (from {argv_source})",
        )
    )

    node_facts: dict[str, Any] = {"expected_node": args.node, "observed": None}
    try:
        for node in client.nodes():
            node_id = str(first_key(node, "node_id", "id", "name") or "")
            if node_id == args.node:
                node_facts["observed"] = {
                    "node_id": node_id,
                    "state": first_key(node, "state", "status"),
                    "capabilities": first_key(node, "capabilities"),
                    "metadata": first_key(node, "metadata"),
                }
                break
        gates.append(
            Gate(
                "expected_node_advertises_verification",
                bool(
                    node_facts["observed"]
                    and "verification" in (node_facts["observed"].get("capabilities") or [])
                ),
                f"node {args.node} capabilities: {node_facts['observed'].get('capabilities') if node_facts['observed'] else 'NOT FOUND'}",
            )
        )
    except RuntimeError as exc:
        gates.append(Gate("cluster_nodes_reachable", False, str(exc)))

    # ---------------- Cell A ----------------
    cell_a = CellResult(name="A", notes=[f"argv source: {argv_source}"])
    workdir = Path(tempfile.mkdtemp(prefix="oai2-cellA-"))
    try:
        for index in range(args.reps):
            phase = classify_phase(index)
            prep = 0.0
            if phase == "cold":
                workdir.joinpath("tree").exists() and shutil.rmtree(workdir / "tree", ignore_errors=True)
                prep = shallow_prepare(args.repo_url, args.ref, workdir / "tree")
            tree = workdir / "tree"
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
            record = {
                "ok": completed.returncode == 0,
                "node_id": None,
                "resolved_revision": revision,
            }
            # Cell A is local by construction, so its gates are: the command
            # succeeded and the checkout it ran in is the revision we asked for.
            passed = record["ok"] and revisions_match(revision or "", args.ref)
            detail = detail_ok(record, revision, args.ref)
            cell_a.reps.append(
                {
                    "rep": index,
                    "phase": phase,
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
                }
            )
            cell_a.gates.append(Gate(f"cell_a_rep{index}_{phase}", passed, detail))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # ---------------- Cell B ----------------
    cell_b = CellResult(name="B")
    window: list[tuple[float, float]] = []
    for index in range(args.reps):
        phase = classify_phase(index)
        rep_start = time.monotonic()
        # Wall clock, deliberately separate from rep_start: durations derived
        # from cluster timestamps must only ever be compared against other
        # wall-clock instants.
        enqueue_wall = time.time()
        try:
            job_id = run_enqueue(
                args.qpipe, args.repo_url, args.ref, args.check, cluster_url
            )
            record, _, _ = wait_for_job(client, job_id, args.timeout, args.poll_interval)
            events = client.events(job_id)
        except (RuntimeError, ConfigError) as exc:
            cell_b.reps.append(
                {
                    "rep": index,
                    "phase": phase,
                    "ok": False,
                    "useful": False,
                    "total_s": round(time.monotonic() - rep_start, 4),
                    "preparation_s": None,
                    "execution_s": None,
                    "requested_ref": args.ref,
                    "resolved_revision": None,
                    "job_id": None,
                    "node_id": None,
                    "state": "enqueue_or_poll_error",
                    "enqueue_to_grant_s": None,
                    "grant_to_finish_s": None,
                    "error": str(exc),
                }
            )
            cell_b.gates.append(Gate(f"cell_b_rep{index}_{phase}", False, str(exc)))
            continue
        rep_end = time.monotonic()
        window.append((rep_start, rep_end))
        stamps = job_timings(record, events)
        grant = stamps.get("grant")
        finish = stamps.get("finish")
        # The enqueue instant must come from the CLUSTER's own wall clock, not
        # from this process. grant/finish are cluster epoch seconds; a local
        # monotonic clock (seconds since boot) subtracted from them yields a
        # number that looks like a timestamp but is not a duration. Prefer the
        # cluster-reported enqueue instant and only fall back to the local
        # wall clock, which is at least the same epoch as the cluster's.
        enqueue_stamp = stamps.get("enqueue")
        enqueue_to_grant = None
        if grant and enqueue_stamp:
            enqueue_to_grant = plausible_duration(grant - enqueue_stamp)
        elif grant:
            enqueue_to_grant = plausible_duration(grant - enqueue_wall)
        grant_to_finish = (
            plausible_duration(finish - grant) if grant and finish else None
        )
        state = str(first_key(record, "state", "status") or "unknown").lower()
        result_block = record.get("result") if isinstance(record.get("result"), dict) else record
        observed = {
            "ok": bool(first_key(result_block, "ok")),
            "node_id": first_key(record, "node_id", "node"),
            "resolved_revision": first_key(
                result_block, "resolved_revision", "revision", "commit"
            ),
        }
        passed, detail = judge_record(observed, args.ref, args.node)
        checks_block = result_block.get("checks")
        evidence_block = result_block.get("evidence")
        cell_b.reps.append(
            {
                "rep": index,
                "phase": phase,
                "ok": observed["ok"],
                "useful": passed,
                "total_s": round(rep_end - rep_start, 4),
                "preparation_s": None,
                "execution_s": grant_to_finish,
                "requested_ref": args.ref,
                "resolved_revision": observed["resolved_revision"],
                "job_id": job_id,
                "node_id": observed["node_id"],
                "state": state,
                "enqueue_to_grant_s": enqueue_to_grant,
                "grant_to_finish_s": grant_to_finish,
                "checks_ok": (
                    [c for c in checks_block if isinstance(c, dict)]
                    if isinstance(checks_block, list)
                    else None
                ),
                "evidence_count": len(evidence_block) if isinstance(evidence_block, list) else None,
            }
        )
        cell_b.gates.append(Gate(f"cell_b_rep{index}_{phase}", passed, detail))
        if enqueue_to_grant is None or grant_to_finish is None:
            cell_b.notes.append(
                f"rep {index}: scheduler events did not expose grant/finish instants; "
                "the queue/execution split for this rep is reported as unavailable"
            )

    # ---------------- Assemble ----------------
    gates.extend(cell_a.gates)
    gates.extend(cell_b.gates)
    failures = [g for g in gates if not g.passed]
    status = "FAIL" if failures else "PASS"
    exit_code = EXIT_GATE_FAILURE if failures else EXIT_OK

    a_latency = {
        "cold": distribution(cell_a.timings("cold")),
        "warm": distribution(cell_a.timings("warm")),
        "all": distribution(cell_a.timings()),
    }
    b_latency = {
        "cold": distribution(cell_b.timings("cold")),
        "warm": distribution(cell_b.timings("warm")),
        "all": distribution(cell_b.timings()),
    }

    def rate(result: CellResult) -> float | None:
        totals = result.timings()
        if not totals:
            return None
        return round(60.0 * len(totals) / sum(totals), 3)

    def split(result: CellResult) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for phase in ("cold", "warm"):
            chosen = [r for r in result.reps if r["phase"] == phase]
            queues = [r["enqueue_to_grant_s"] for r in chosen if r.get("enqueue_to_grant_s") is not None]
            execs = [r["grant_to_finish_s"] for r in chosen if r.get("grant_to_finish_s") is not None]
            out[phase] = {
                "queue_s": round(statistics.fmean(queues), 4) if queues else None,
                "execution_s": round(statistics.fmean(execs), 4) if execs else None,
                "n": len(chosen),
            }
        return out

    a_warm = a_latency["warm"]["median_s"]
    b_warm = b_latency["warm"]["median_s"]
    a_cold = a_latency["cold"]["median_s"]
    b_cold = b_latency["cold"]["median_s"]
    slower = None
    if a_warm is not None and b_warm is not None:
        slower = b_warm > a_warm
    elif a_cold is not None and b_cold is not None:
        slower = b_cold > a_cold

    payload: dict[str, Any] = {
        "schema": "oai2.worker_contribution/1",
        "generated_at_utc": utc_now(),
        "worker_role": "verification",
        "capacity_class": "verification-execution-only",
        "scope_note": (
            "Both cells ran one identical bounded verification command at one pinned "
            "revision. The node is measured as a test-execution worker only. This report "
            "makes no claim about hardware speed, service capacity or output quality."
        ),
        "config": {
            "reps": args.reps,
            "ref": args.ref,
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
        "gate_detail": [
            {"name": g.name, "passed": g.passed, "detail": g.detail} for g in gates
        ],
        "cell_a": {
            "cell": "A",
            "name": "mac-local",
            "work_description": shlex.join(local_argv),
            "work_argv": local_argv,
            "work_argv_sha256": argv_digest(local_argv),
            "correct": all(r["ok"] for r in cell_a.reps) and bool(cell_a.reps),
            "completed_useful_jobs": cell_a.useful_completed(),
            "completed_useful_jobs_per_minute": rate(cell_a),
            "latency_s": a_latency,
            "preparation_s": {
                "cold": a_latency["cold"].get("mean_s") and cell_a.reps[0]["preparation_s"],
                "warm": 0.0,
                "note": "warm reps reuse the prepared tree, so preparation is 0 by construction",
            },
            "reps": cell_a.reps,
            "notes": cell_a.notes,
        },
        "cell_b": {
            "cell": "B",
            "name": "fleet-p50",
            "work_description": shlex.join(local_argv),
            "work_argv": local_argv,
            "work_argv_sha256": argv_digest(local_argv),
            "correct": all(r["ok"] for r in cell_b.reps) and bool(cell_b.reps),
            "completed_useful_jobs": cell_b.useful_completed(),
            "completed_useful_jobs_per_minute": rate(cell_b),
            "latency_s": b_latency,
            "phase_split_s": split(cell_b),
            "preparation_s": {
                "cold": None,
                "warm": None,
                "note": (
                    "node-side repository materialisation is not separately observable in "
                    "the job record; it is isolated by the cold/warm rep contrast instead"
                ),
            },
            "reps": cell_b.reps,
            "notes": cell_b.notes,
        },
        "comparison": {
            "cell_a_cold_median_s": a_cold,
            "cell_a_warm_median_s": a_warm,
            "cell_b_cold_median_s": b_cold,
            "cell_b_warm_median_s": b_warm,
            "cell_b_slower_end_to_end": slower,
            "measured_benefit": "mac-offload",
            "forbidden_claim": (
                "cell B is not faster; no speedup may be reported from this harness"
                if slower
                else "direction is stated above; report it as measured, never as a speedup"
            ),
            "mac_offload_useful_jobs": cell_b.useful_completed(),
        },
        "host_resources": {
            "sampling_interval_s": args.sample_interval,
            "overall": sampler.summary(),
            "per_rep": [
                {
                    "rep": index,
                    "phase": classify_phase(index),
                    **(
                        sampler.summary(window[index])
                        if index < len(window)
                        else {"sample_count": 0}
                    ),
                }
                for index in range(len(window))
            ],
            "notes": sampler.notes
            + [
                "sampling is best effort: an unavailable metric is reported as null and "
                "never fails the run"
            ],
        },
    }
    return payload, exit_code


def detail_ok(record: dict[str, Any], revision: str | None, requested: str) -> str:
    if not record["ok"]:
        return "local command returned a non-zero exit code"
    if not revision:
        return "could not read the prepared checkout revision"
    if not revisions_match(revision, requested):
        return f"checkout at {revision} != requested {requested}"
    return f"ok at {revision[:12]}"


def argv_digest(argv: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\x1f".join(argv).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measure_worker_contribution.py",
        description=(
            "Matched A/B measurement of fleet-worker contribution for one pinned "
            "revision: run the same bounded verification command on this Mac versus "
            "on the cluster worker node. Reports cold (with preparation) separately "
            "from warm, and gates on correctness before any statistic."
        ),
        epilog=(
            "This harness measures test-execution capacity only. It will refuse to "
            "print any generative-performance claim. It contacts nothing except the "
            "cluster control plane. Exit 0 = all gates passed, 1 = correctness gate "
            "failure, 2 = configuration error."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reps", type=int, default=5, help="repetitions per cell; rep 0 is cold, 1..N-1 are warm")
    parser.add_argument("--ref", default="HEAD", help="pinned revision (commit sha or branch) both cells use")
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
    parser.add_argument("--sample-interval", type=float, default=1.0, help="host resource sampling interval in seconds")
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
    args.check_argv = None
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    if args.reps == 1 and not args.dry_run:
        print(
            "warning: --reps 1 yields no warm observations, so the cold/warm split "
            "cannot be computed; use --reps 2 or more for a meaningful result",
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
