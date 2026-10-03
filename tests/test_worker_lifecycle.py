"""Regression suite for the fleet-worker lifecycle probes (master #241, issue #256).

Pure stdlib, no network, no docker, no ssh, no live host. Run it with:

    python3 -m unittest tests.test_worker_lifecycle -v

=============================================================================
TEST-ISOLATION INCIDENT -- recorded here so it cannot recur quietly
=============================================================================
WHAT HAPPENED
    A validation of the worker supervisor that was SUPPOSED to be isolated
    passed ``--container oai2-worker`` with owner intent ``stop``. That is the
    LIVE production worker container on the P50 host. The new supervisor
    correctly honoured the intent it was given and STOPPED THE LIVE PRODUCTION
    WORKER CONTAINER. The run then "recovered" only because the already-running
    supervisor happened to restart the container. That was luck, not a control.
    A test that can reach production is worse than no test: it reports a green
    result while having done damage.

WHY THE "ISOLATED" TEST WAS NOT ISOLATED
    Three concrete holes, all in this module, none of them the script's fault:

    1. The fake environment PREPENDED the stub directory to the INHERITED
       ``PATH``. Stubbed or not, the real ``docker`` and the real ``qpipe``
       were still one PATH entry away. Any command the stubs did not shadow
       reached the developer's real machine.
    2. The suite's own default target WAS the production name. ``CONTAINER``
       defaults to ``oai2-worker`` in both shipped probes, and this module
       reused that string as its fixture constant, so "isolated" runs were
       addressed at production by default and nothing anywhere refused them.
    3. There was no isolation layer. Isolation was a claim in this docstring,
       enforced by nothing. The stub logs made scope violations visible AFTER
       the fact; they could not prevent one.

WHAT NOW PREVENTS IT
    ``assert_isolated_env`` (below) is the guard. A test's PATH is a single
    entry -- the test's own temp stub dir -- and the guard raises
    ``IsolationViolation`` if any real container runtime, service manager, host
    escape hatch or q-pipe CLI resolves from it, if any PATH entry lies outside
    the temp dir, or if a live cluster endpoint was inherited. The guard runs
    in ``setUpModule`` (so a suite that cannot detect live infrastructure fails
    before any test body) and again on every environment handed to a
    subprocess, so a test that mutates its own PATH cannot slip past it.
    ``isolated_target`` is the target layer: it refuses the production
    container name and any state dir outside the test's temp root, and the only
    way past that is the explicit, default-off ``OAI2_LIVE_TESTS=1`` opt-in.
    ``FakeCluster.run`` is the ONLY way a probe is launched, and it routes every
    invocation through both layers -- so a default run physically cannot name
    the production container, because it is given the synthetic name
    explicitly rather than inheriting the script's production default.
    The regression tests at the bottom of this file pin each of those claims.

=============================================================================

How the scripts are exercised
-----------------------------
The two probes are treated as TEXT (source-level contracts) and as PROGRAMS
(their real behaviour, driven against INJECTED FAKE dependencies). A temporary
directory provides:

  * a stub ``docker`` that serves a synthetic ``/proc`` tree, canned
    ``inspect`` state, a simulated ``restart`` and a simulated entrypoint
    respawn loop. It runs the probe's own embedded python payload with the real
    interpreter, so the shipped matching logic executes for real.
  * stubs for ``systemctl``, ``wsl`` and ``qpipe`` that log the call and fail.
    They exist so that a probe reaching outside the worker scope fails loudly
    instead of quietly reaching a real host.

That temp directory is the test's ENTIRE ``PATH``. Alongside the stubs it
carries symlinks to a declared list of real, non-infrastructure tools the
probes genuinely need (bash, sh, env, cat, date, grep, sed, tr, head, cut,
dirname, python3). They are the only real files in there, and none of them can
name a container, a service or a node.

No process is ever signalled: the probe's signal path only switches to
"simulated" when its proc root is not ``/proc``.

Defect map -- every test name below carries the id it pins:

  R1  self-counting: the probe matched the substrings "worker" and "verify" in
      /proc/<pid>/cmdline, so its own ``python3 -c`` source matched and a single
      live worker was reported as "count: 2".
  R2  no identity: the pid/start time of the process about to be killed was
      never recorded, so a recycled pid could pass as the original.
  R3  false pass on a no-op kill: "killed pids: NONE" still exited 0.
  R4  no replacement requirement: a fixed sleep then print-whatever.
  R5  no deadline and no exit status: always exited 0.
  R6  wrong evidence: a ``docker logs`` read plus a printed count as acceptance.
  R7  no regression test: nothing proved the probe could not count itself.
  L1  lifecycle-probe used ``ps``/``pkill``, which do not exist in this image.
  L2  a failed worker lookup was a printed line, not a failure.
  L3  no exit status on the lifecycle probe.
  L5  lifecycle cases were not individually asserted.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
PRB = REPO / "deploy" / "worker" / "process-recovery-probe.sh"
LP = REPO / "deploy" / "worker" / "lifecycle-probe.sh"
ENTRYPOINT = REPO / "deploy" / "worker" / "entrypoint.sh"

# The worker command line exactly as deploy/worker/entrypoint.sh launches it.
WORKER_C_SCRIPT = "from qpipe.cli import main; main()"
WORKER_ARGV = [
    "python3",
    "-c",
    WORKER_C_SCRIPT,
    "worker",
    "verify",
    "--url",
    "http://192.168.100.39:10534",
    "--node-id",
    "p50",
]

# ===========================================================================
# TEST ISOLATION -- enforced, not asserted in a docstring
# ===========================================================================
# See the TEST-ISOLATION INCIDENT block in this module's docstring for the run
# that stopped the live production worker container, and for why the previous
# "isolation" did not prevent it. Everything below is the replacement.

# The only way to reach live infrastructure: an explicit, default-off opt-in.
# Nothing in this module uses it to launch anything.
LIVE_ENV_VAR = "OAI2_LIVE_TESTS"
LIVE_OPT_IN = "1"

# The real production worker container. It is spelled out HERE so the isolation
# layer can refuse it, and nowhere else in this module: no default test may be
# addressed at it. Both shipped probes default --container to this same string,
# which is precisely why every launch below passes the synthetic name
# explicitly instead of inheriting the script default.
PRODUCTION_CONTAINER = "oai2-worker"

# The name every test in this module targets by default.
SYNTHETIC_CONTAINER = "oai2-test-worker"

# The fixture constant the tests already use. Deliberately NOT the production
# name: the old value of this constant was the incident.
CONTAINER_NAME = SYNTHETIC_CONTAINER

# The infrastructure that must be unreachable from a test. A test's PATH must
# resolve these to nothing but the temp stub dir.
INFRA_BINARIES = (
    # container runtimes -- the incident's owner intent was `stop` against the
    # production container, which is a `docker <name> stop`.
    "docker", "podman", "nerdctl", "ctr", "crictl",
    # service managers
    "systemctl", "service", "launchctl",
    # host escape hatches
    "wsl", "wslhost", "nsenter", "chroot",
    # the q-pipe CLI / cluster controller
    "qpipe",
    # remote shells
    "ssh", "scp", "rsync", "mosh",
)

# Must be shadowed by a stub in the isolated bin dir. A missing stub would be a
# hole even if nothing resolves to a real binary, so the guard requires these
# to be present as stubs.
REQUIRED_STUBS = ("docker", "systemctl", "wsl", "qpipe")

# Additional stubs this suite writes itself, for tools that are POSIX rather
# than infrastructure but whose behaviour a test needs to be real and bounded.
# `rm` is here because worker-supervisor.sh's `clear` path unlinks the owner
# sentinel, and that path has to be tested for real. The ambient `rm` cannot be
# trusted to that role: on a managed developer machine it may be a wrapper that
# refuses to delete outside the workspace, which would make the test fail for a
# reason that has nothing to do with the supervisor. The stub below performs a
# genuine unlink, and only ever inside this cluster's own temp root.
SUITE_STUBS = ("rm",)

# Real, non-infrastructure executables the worker scripts legitimately need.
# These are symlinked into the isolated bin dir, where they are the only entries
# that are not stubs this suite writes itself. None of them can name a
# container, a service, a namespace or a node: they are POSIX filesystem and
# text utilities, scoped by the tests to their own temp root. The set is
# derived from what the shipped scripts actually invoke -- a script that called
# something absent from here would fail with a confusing 127, so the list is a
# declaration of what the isolation layer considers non-infrastructure, not a
# convenience.
BENIGN_TOOLS = (
    "bash", "sh", "env", "cat", "date", "grep", "sed", "tr", "head", "cut",
    "dirname", "python3",
    # required by worker-supervisor.sh: state-dir creation, the atomic
    # intent/health update (write temp, chmod, rename, and unlink the temp or
    # the sentinel on the `clear` path), log rotation, the poll sleep, and the
    # uid the log reports. `rm` is here because the script uses it, and its
    # reach is bounded by the same temp root as every other tool.
    "mkdir", "mv", "sleep", "id", "chmod", "wc", "touch",
)

# Inherited environment that would point a test at live infrastructure even
# with a clean PATH. Stripped from every test env and refused by the guard.
CLUSTER_ENV_VARS = (
    "QPIPE_CLUSTER_URL", "QPIPE_CLUSTER_TOKEN", "QPIPE_NODE_ID",
    "QPIPE_WORK_ROOT", "QPIPE_PROFILES", "QPIPE_HEARTBEAT",
    "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY", "DOCKER_API_VERSION",
)

# Known production state directories. Named so the refusal message can be
# specific rather than merely "outside the temp root".
PRODUCTION_STATE_DIRS = (
    "/var/lib/worker", "/var/lib/oai2", "/srv/oai2", "/opt/oai2", "/etc/oai2",
    "/run/worker", "/var/run/worker",
)


class IsolationViolation(AssertionError):
    """A test tried to reach live infrastructure. Always a loud test failure.

    Subclasses AssertionError on purpose: it must surface as a FAILING test
    with the message attached, never as a skip, a warning or a silent fallback.
    """


def live_mode_enabled() -> bool:
    """True only for an explicit ``OAI2_LIVE_TESTS=1``. Default: off."""
    return os.environ.get(LIVE_ENV_VAR) == LIVE_OPT_IN


def resolve_in_path(name: str, path: str) -> str | None:
    """The executable ``name`` resolves to in ``path``, as a realpath, or None.

    Empty PATH entries are skipped rather than treated as the current
    directory, so a stray ``PATH=:/usr/bin`` cannot hide an entry from the
    guard.
    """
    for directory in path.split(os.pathsep):
        if not directory:
            continue
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.realpath(candidate)
    return None


def is_within(path: Path | str, root: Path | str) -> bool:
    """True if ``path`` is ``root`` or lives under it (after resolution)."""
    try:
        resolved = Path(path).resolve()
        base = Path(root).resolve()
    except OSError:
        return False
    return resolved == base or base in resolved.parents


def assert_isolated_env(env: dict[str, str], *, bin_dir: Path, label: str) -> None:
    """The guard. Raises IsolationViolation if a test's env can reach live infra.

    Enforced properties, all of them structural rather than best-effort:

    1. ``PATH`` is a single entry, and it is the test's own temp stub dir.
    2. Every infrastructure binary resolves either to a stub INSIDE that dir
       (realpath included, so a symlink to the real ``docker`` is rejected) or
       to nothing at all.
    3. ``docker``/``systemctl``/``wsl``/``qpipe`` are present as stubs, so a
       probe reaching out gets a loud failure instead of a silent no-op.
    4. No live cluster endpoint or docker context was inherited.
    """
    path = env.get("PATH", "")
    entries = [entry for entry in path.split(os.pathsep) if entry]
    stub_root = Path(bin_dir).resolve()
    problems: list[str] = []

    if not entries:
        problems.append("PATH is empty, so nothing is pinned to the stub dir")
    for entry in entries:
        if not is_within(entry, stub_root):
            problems.append(
                f"PATH entry {entry!r} is outside the isolated bin dir {stub_root}, so a real "
                "executable on this host is reachable"
            )

    for name in INFRA_BINARIES:
        found = resolve_in_path(name, path)
        if found is None:
            continue
        if not is_within(found, stub_root):
            problems.append(
                f"infrastructure executable {name!r} resolves to {found}, which is a REAL "
                f"binary and not a stub in {stub_root}"
            )
    for name in REQUIRED_STUBS:
        if resolve_in_path(name, path) is None:
            problems.append(
                f"no stub for {name!r}: a probe reaching for it would get a silent "
                "'command not found' instead of a loud, logged failure"
            )

    for name in CLUSTER_ENV_VARS:
        if name in env:
            problems.append(
                f"{name} is set in the test env ({env[name]!r}): a test must not inherit a live "
                "cluster endpoint or docker context"
            )

    if problems:
        raise IsolationViolation(
            "test isolation violated ({}):\n  - {}".format(label, "\n  - ".join(problems))
        )


class IsolatedTarget(NamedTuple):
    """An authorised target: a non-production container and a scratch state dir."""

    container: str
    state_dir: Path
    live: bool


def isolated_target(
    container: str = SYNTHETIC_CONTAINER,
    state_dir: Path | str | None = None,
    *,
    root: Path,
    live: bool = False,
) -> IsolatedTarget:
    """Authorise a (container, state dir) pair, or refuse it loudly.

    In isolated mode (the default, and the only mode any test here uses):

    * the production container name is refused outright;
    * a state dir outside the test's own temp root is refused, and the known
      production state dirs are refused by name with a specific message.

    ``live=True`` is the single documented way past either refusal, and it is
    refused too unless the caller has set ``OAI2_LIVE_TESTS=1``. The opt-in is
    therefore required, and it is off by default.
    """
    if live and not live_mode_enabled():
        raise IsolationViolation(
            f"live mode refused: it requires {LIVE_ENV_VAR}={LIVE_OPT_IN}, which is not set. Isolated "
            "mode is the default and the only mode this suite runs in."
        )

    if not live:
        if container == PRODUCTION_CONTAINER:
            raise IsolationViolation(
                f"isolated mode REFUSES the production container {container!r}. This is the "
                "exact target that was stopped in the test-isolation incident "
                f"(see this module's docstring). A test must use {SYNTHETIC_CONTAINER!r}, or opt in "
                f"with {LIVE_ENV_VAR}={LIVE_OPT_IN}."
            )
        if state_dir is not None:
            candidate = str(state_dir)
            for production in PRODUCTION_STATE_DIRS:
                if candidate == production or candidate.startswith(production + "/"):
                    raise IsolationViolation(
                        f"isolated mode REFUSES the production state dir {candidate!r}. A test "
                        f"state dir must live under its own temp root {root}."
                    )
            if Path(state_dir).is_absolute() and not is_within(state_dir, root):
                raise IsolationViolation(
                    f"isolated mode REFUSES the state dir {candidate!r}: it is outside the "
                    f"test's own temp root {root}, so it is not a scratch dir."
                )

    return IsolatedTarget(container, Path(state_dir) if state_dir is not None else None, live)


def require_benign_tools_present() -> None:
    """Fail loudly here rather than mysteriously in every probe run.

    The isolated bin dir is the stub dir PLUS symlinks to these real tools. A
    host missing one cannot be isolated properly, and the suite must say so
    once, clearly, instead of producing mysterious exit 127s.
    """
    ambient = os.environ.get("PATH", "")
    missing = [name for name in BENIGN_TOOLS if resolve_in_path(name, ambient) is None]
    if missing:
        raise IsolationViolation(
            "cannot build an isolated bin dir: these real, non-infrastructure "
            "tools are absent from this host's PATH: {}".format(", ".join(missing))
        )


def enforce_module_isolation() -> None:
    """The module-level guard. Runs from setUpModule, before any test body.

    A suite that cannot DETECT live infrastructure must not be trusted to be
    isolated, so this first proves the guard bites. Then it checks the layer's
    own defaults: the default target must be a legal non-production target, the
    production container must be refused, and the live gate must be shut unless
    the operator opened it on purpose.
    """
    require_benign_tools_present()

    # 1. The guard must reject a PATH with a real infrastructure binary on it.
    #    Built synthetically so the check is identical on a host with no docker
    #    and a host with one. If this ever stops raising, the suite is not
    #    isolated and must not be believed.
    with tempfile.TemporaryDirectory() as scratch:
        hostile_bin = Path(scratch) / "hostile-bin"
        hostile_bin.mkdir()
        decoy = hostile_bin / "docker"
        decoy.write_text("#!/bin/sh\nexit 0\n")
        decoy.chmod(0o755)
        try:
            assert_isolated_env(
                {"PATH": str(hostile_bin)}, bin_dir=Path(scratch) / "elsewhere",
                label="module self-test",
            )
        except IsolationViolation:
            pass
        else:  # pragma: no cover - the guard is broken, which is the failure
            raise IsolationViolation(
                "the isolation guard accepted a PATH resolving a real `docker`. "
                "This suite cannot prove it is isolated, so it must not be run."
            )

    # 2. The default target must be authorised, and must not be production.
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        default_target = isolated_target(root=root)
        if default_target.container == PRODUCTION_CONTAINER:  # pragma: no cover
            raise IsolationViolation("the default test target is the production container")
        if default_target.live:  # pragma: no cover
            raise IsolationViolation("the default test target is a live target")
        for container, state_dir in (
            (PRODUCTION_CONTAINER, root / "scratch"),
            (SYNTHETIC_CONTAINER, "/var/lib/worker"),
        ):
            try:
                isolated_target(container, state_dir, root=root)
            except IsolationViolation:
                continue
            raise IsolationViolation(  # pragma: no cover
                f"the isolation layer did not refuse container={container!r} state_dir={state_dir!r}"
            )

    # 3. The live gate must be shut unless the operator opened it on purpose.
    #    Nothing here launches anything either way; this only records intent.
    if not live_mode_enabled():
        with tempfile.TemporaryDirectory() as scratch:
            try:
                isolated_target(PRODUCTION_CONTAINER, root=Path(scratch), live=True)
            except IsolationViolation:
                pass
            else:  # pragma: no cover
                raise IsolationViolation(
                    f"live mode is open without {LIVE_ENV_VAR}={LIVE_OPT_IN}"
                )


def setUpModule() -> None:
    """Module-level guard: the suite fails before any test if it is not isolated."""
    enforce_module_isolation()


WORKER_PID = 100
WORKER_PGID = 100
SELF_PID = 4242          # the probe helper's own pid inside the synthetic tree
SELF_PGID = 4242
GHOST_PID = 4300         # a non-self pid that still carries the probe marker

BTIME = 1_600_000_000
HZ = int(os.sysconf("SC_CLK_TCK"))

# The exit code contract of process-recovery-probe.sh, mirrored here so the
# tests fail if the script and its documented contract drift apart.
EX_OK = 0
EX_USAGE = 2
EX_DOCKER = 3
EX_NOT_FOUND = 4
EX_KILL = 5
EX_DEADLINE = 6
EX_AMBIGUOUS = 7

_HELPER_RE = re.compile(
    r"PRB_HELPER_SOURCE=\$\(cat <<'PRB_HELPER_EOF'\n(.*?)\nPRB_HELPER_EOF\n\)",
    re.DOTALL,
)


def helper_source() -> str:
    """The probe's embedded python payload, taken from the shipped script."""
    match = _HELPER_RE.search(PRB.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"could not extract PRB_HELPER_SOURCE from {PRB}")
    return match.group(1)


def legacy_rule_matches(cmdline: str) -> bool:
    """The PRE-FIX matching rule, verbatim from the old process-recovery-probe.sh.

    It matched any /proc/<pid>/cmdline containing both words. The probe's own
    `python3 -c` source contains both, which is the R1 self-counting bug. This
    function exists so a regression test can show the contrast: it must still
    count the probe (2) while the new rule does not (1).
    """
    return "worker" in cmdline and "verify" in cmdline


def starttime_for_age(age_seconds: float, now: float | None = None) -> int:
    """A /proc/<pid>/stat start time (field 22) for a process of that age."""
    now = time.time() if now is None else now
    return int((now - age_seconds - BTIME) * HZ)


def stat_line(pid: int, pgrp: int, starttime: int) -> str:
    """A /proc/<pid>/stat line. Field 3 ("state") is the first token after the
    last ')', so field 5 (pgrp) is index 2 and field 22 (starttime) is index 19."""
    rest = ["S", "1", str(pgrp), "1"] + ["0"] * 15 + [str(starttime)]
    assert len(rest) == 20
    joined_rest = " ".join(rest)
    return f"{pid} (python3) {joined_rest}\n"


def write_entry(proc: Path, pid: int, argv: list[str], pgrp: int, starttime: int) -> None:
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("cmdline").write_bytes(
        b"\0".join(token.encode() for token in argv) + b"\0"
    )
    directory.joinpath("stat").write_text(stat_line(pid, pgrp, starttime))


def joined(argv: list[str]) -> str:
    return " ".join(argv)


def code_lines(path: Path) -> str:
    """The script's SHELL code: comments and every heredoc payload removed.

    Contracts are asserted against code, not against the words a header comment,
    a usage text or a python docstring happens to use while describing them.
    """
    kept = []
    pending_end: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if pending_end is not None:
            if stripped == pending_end:
                pending_end = None
            continue
        opening = re.search(r"<<'([A-Za-z_][A-Za-z0-9_]*)'\s*$", stripped)
        if opening is not None:
            pending_end = opening.group(1)
            continue
        if stripped.startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def docker_verbs(calls: list[str]) -> set[str]:
    """The docker subcommands actually invoked, read from the stub's call log.

    This is ground truth from a real run, so a prose mention in a comment, a
    usage text or a message string cannot influence it.
    """
    verbs = set()
    for call in calls:
        if call.startswith("FORBIDDEN"):
            verbs.add("forbidden:" + call.split()[1])
        elif call:
            verbs.add(call.split()[0])
    return verbs


def parse_records(output: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse the selector's MATCH / REJECT lines into dicts."""
    matches: list[dict[str, str]] = []
    rejects: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == "MATCH":
            matches.append({k: v for k, v in (f.split("=", 1) for f in fields[1:])})
        elif fields and fields[0] == "REJECT":
            rejects.append({k: v for k, v in (f.split("=", 1) for f in fields[1:])})
    return matches, rejects


def select_field(output: str, name: str) -> str:
    for line in output.splitlines():
        if line.startswith("PRB_SELECT "):
            for token in line.split()[1:]:
                key, _, value = token.partition("=")
                if key == name:
                    return value
    raise AssertionError(f"no PRB_SELECT {name}= in output:\n{output}")


# ---------------------------------------------------------------------------
# Fake docker CLI
# ---------------------------------------------------------------------------
DOCKER_STUB = r'''#!/usr/bin/env python3
"""Fake docker CLI. Serves a synthetic /proc tree; never touches a runtime."""
import json
import os
import pathlib
import subprocess
import sys

STATE = pathlib.Path(os.environ["FAKE_DOCKER_STATE"])
PROC = STATE / "proc"
# No fallback: a missing container name must be loud, and the fallback must not
# be the production container this suite refuses to target.
CONTAINER = os.environ["FAKE_CONTAINER"]

argv = sys.argv[1:]
verb = argv[0] if argv else ""
mode = os.environ.get("FAKE_DOCKER_MODE", "ok")


def log_call():
    """One line per invocation. The embedded python payload is a single argv
    element that contains newlines, so it must not be echoed verbatim or the
    log stops being a list of calls."""
    parts = [verb]
    for token in argv[1:]:
        if token == "-c" or "\n" in token or len(token) > 200:
            parts.append("<payload>")
        else:
            parts.append(token)
    with open(STATE / "calls.log", "a") as handle:
        handle.write(" ".join(parts) + "\n")


log_call()


def state(name, default=""):
    path = STATE / name
    return path.read_text().strip() if path.exists() else default


def fail(which, code=1):
    if mode in ("fail-all", "fail-" + which):
        sys.stderr.write("fake docker: injected %s failure\n" % which)
        sys.exit(int(os.environ.get("FAKE_DOCKER_RC", str(code))))


def require_container():
    if not (STATE / "container").exists():
        sys.stderr.write("Error: No such container: %s\n" % CONTAINER)
        sys.exit(1)
    if not (STATE / "running").exists():
        sys.stderr.write("Error: container %s is not running\n" % CONTAINER)
        sys.exit(1)


def write_entry(pid, tokens, starttime, pgrp):
    directory = PROC / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("cmdline").write_bytes(
        b"\0".join(str(t).encode() for t in tokens) + b"\0")
    rest = ["S", "1", str(pgrp), "1"] + ["0"] * 15 + [str(starttime)]
    directory.joinpath("stat").write_text("%d (python3) %s\n" % (pid, " ".join(rest)))


def load_argv(name, fallback):
    path = STATE / name
    if path.exists():
        return json.loads(path.read_text())
    path = STATE / fallback
    return json.loads(path.read_text()) if path.exists() else ["sh"]


def clear_proc():
    for child in PROC.iterdir():
        if child.is_dir():
            for item in child.iterdir():
                item.unlink()
            child.rmdir()


def maybe_respawn():
    """Stands in for the entrypoint's supervised loop: a fresh worker appears
    once the current one is gone. The test controls whether, when and with
    which identity that happens."""
    if not (STATE / "respawn").exists():
        return
    current = int(state("current_worker_pid", "100"))
    if (PROC / str(current)).exists():
        return
    new_pid = int(state("respawn_pid", str(current + 7)))
    if (STATE / "bump_restarts_on_respawn").exists():
        n = int(state("restarts", "0")) + 1
        (STATE / "restarts").write_text(str(n))
    write_entry(
        new_pid,
        load_argv("respawn_argv", "worker_argv"),
        int(state("respawn_starttime", "1")),
        int(state("respawn_pgid", str(new_pid))),
    )
    (STATE / "current_worker_pid").write_text(str(new_pid))


def bump_restart():
    n = int(state("restart_count", "0")) + 1
    (STATE / "restart_count").write_text(str(n))
    (STATE / "restarts").write_text(str(n))
    if not (STATE / "restart_keep_startedat").exists():
        (STATE / "started").write_text("2020-01-01T00:00:%02dZ" % n)
    new_pid = int(state("restart_pid", str(int(state("current_worker_pid", "100")) + 1)))
    clear_proc()
    write_entry(
        new_pid,
        load_argv("restart_argv", "worker_argv"),
        int(state("restart_starttime", "0")),
        int(state("restart_pgid", state("current_worker_pid", "100"))),
    )
    (STATE / "current_worker_pid").write_text(str(new_pid))


if verb == "inspect":
    fail("inspect")
    if not (STATE / "container").exists():
        sys.stderr.write("Error: No such container: %s\n" % CONTAINER)
        sys.exit(1)
    values = {
        "{{.State.Running}}": "true" if (STATE / "running").exists() else "false",
        "{{.State.StartedAt}}": state("started", "2020-01-01T00:00:00Z"),
        "{{.RestartCount}}": state("restarts", "0"),
        "{{.State.ExitCode}}": state("exitcode", "0"),
    }
    fmt = argv[-1]
    if fmt not in values:
        sys.stderr.write("fake docker: unsupported format %r\n" % fmt)
        sys.exit(64)
    print(values[fmt])
    sys.exit(0)

if verb == "logs":
    fail("logs")
    print("fake docker: a log line that proves nothing")
    sys.exit(0)

if verb == "restart":
    fail("restart")
    if not (STATE / "container").exists():
        sys.stderr.write("Error: No such container: %s\n" % CONTAINER)
        sys.exit(1)
    if not (STATE / "restart_noop").exists():
        bump_restart()
    sys.exit(0)

if verb == "exec":
    fail("exec")
    require_container()
    maybe_respawn()
    rest = argv[1:]
    if not rest or rest[0] != CONTAINER:
        sys.stderr.write("Error: No such container: %s\n" % (rest[0] if rest else ""))
        sys.exit(1)
    rest = rest[1:]
    if "-c" not in rest:
        sys.stderr.write("fake docker: only `exec <c> python3 -c <script>` supported\n")
        sys.exit(64)
    index = rest.index("-c")
    env = dict(os.environ)
    env["OAI2_PRB_PROC_ROOT"] = str(PROC)
    env["OAI2_PRB_SELF_PID"] = state("self_pid", "1")
    env["OAI2_PRB_SELF_PGID"] = state("self_pgid", "1")
    env["OAI2_PRB_SIMULATE_KILL"] = "1"
    sys.exit(subprocess.call([sys.executable, "-c", rest[index + 1]] + rest[index + 2:], env=env))

sys.stderr.write("fake docker: refusing verb %r; a worker probe is container-scoped\n" % verb)
sys.exit(64)
'''

FORBIDDEN_STUB = (
    "#!/usr/bin/env python3\n"
    "import os, pathlib, sys\n"
    'state = pathlib.Path(os.environ["FAKE_DOCKER_STATE"])\n'
    'with open(state / "calls.log", "a") as handle:\n'
    '    handle.write("FORBIDDEN @@TOOL@@ " + " ".join(sys.argv[1:]) + "\\n")\n'
    'sys.stderr.write("@@TOOL@@ is out of scope for a worker-scoped probe\\n")\n'
    "sys.exit(1)\n"
)

# A real unlink, bounded to this cluster's own temp root. Deliberately NOT a
# symlink to the host's `rm`: see SUITE_STUBS for why the ambient one cannot be
# trusted to fill this role.
RM_STUB = r'''#!/usr/bin/env python3
"""rm, restricted to this cluster's temp root. Never deletes anything else."""
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(os.environ["FAKE_CLUSTER_ROOT"]).resolve()

targets = []
for arg in sys.argv[1:]:
    if arg.startswith("-"):
        continue
    targets.append(arg)

if not targets:
    sys.stderr.write("rm: refusing to run with no target\n")
    sys.exit(1)

for raw in targets:
    path = pathlib.Path(raw)
    try:
        resolved = path.resolve()
    except OSError:
        sys.stderr.write("rm: cannot resolve %s\n" % raw)
        sys.exit(1)
    if resolved != ROOT and ROOT not in resolved.parents:
        sys.stderr.write(
            "rm: refusing to remove %s: outside this cluster's root %s\n"
            % (resolved, ROOT)
        )
        sys.exit(1)

for raw in targets:
    path = pathlib.Path(raw)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass  # -f semantics: a missing target is not an error
        except IsADirectoryError:
            sys.stderr.write("rm: %s is a directory\n" % raw)
            sys.exit(1)
sys.exit(0)
'''


class FakeCluster:
    """A temporary tree with stub executables and a synthetic /proc.

    Everything a test can reach is inside ``root``, a temp directory: the stub
    bin dir, the synthetic /proc, and the scratch state dir. The test's PATH is
    that stub dir and nothing else, and every launch goes through the guard.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = root / "state"
        self.proc = self.state / "proc"
        self.bin = root / "bin"
        self.scratch_state = root / "scratch-state"
        self.state.mkdir(parents=True)
        self.proc.mkdir(parents=True)
        self.bin.mkdir(parents=True)
        self.docker_env: dict[str, str] = {}

        # The scratch state dir this cluster is allowed to use. Authorised
        # through the isolation layer, so a production state dir cannot be
        # substituted for it. `live` is False and is not a constructor
        # argument: no test in this module can opt a cluster into live mode.
        self.scratch_state.mkdir(parents=True, exist_ok=True)
        self.target = isolated_target(
            SYNTHETIC_CONTAINER, self.scratch_state, root=root, live=False
        )

        # The real /proc/stat carries `btime`, which is how a process start time
        # (stat field 22) is turned into an age. Without it, age is unprovable
        # and the probe must refuse the process.
        self.proc.joinpath("stat").write_text(
            f"cpu  1 2 3 4\nintr 1\nbtime {BTIME}\nprocesses 1\n"
        )
        self.state.joinpath("calls.log").write_text("")
        self.state.joinpath("container").touch()
        self.state.joinpath("running").touch()
        self.state.joinpath("started").write_text("2020-01-01T00:00:00Z")
        self.state.joinpath("restarts").write_text("0")
        self.state.joinpath("self_pid").write_text(str(SELF_PID))
        self.state.joinpath("self_pgid").write_text(str(SELF_PGID))
        self.state.joinpath("worker_argv").write_text(json.dumps(WORKER_ARGV))
        self.state.joinpath("current_worker_pid").write_text(str(WORKER_PID))
        self.state.joinpath("worker_pid").write_text(str(WORKER_PID))

        self._install("docker", DOCKER_STUB)
        for tool in ("systemctl", "wsl", "qpipe"):
            self._install(tool, FORBIDDEN_STUB.replace("@@TOOL@@", tool))
        for tool in SUITE_STUBS:
            self._install(tool, RM_STUB)
        self._link_benign_tools()
        self.assert_isolated()

        # The intended worker, old enough that a replacement is unmistakable.
        # The start time is recorded here so tests can reuse the EXACT value the
        # tree holds, rather than recomputing it and risking a 1-tick drift.
        self.worker_starttime = starttime_for_age(3600)
        write_entry(
            self.proc,
            WORKER_PID,
            WORKER_ARGV,
            WORKER_PGID,
            self.worker_starttime,
        )
        # The entrypoint's supervised loop will bring a fresh one back.
        self.enable_respawn(pid=WORKER_PID + 7, age=2.0)
        # A container restart is a separate, later event, so its worker must be
        # younger than the one the supervised loop produces afterwards. Start
        # times must therefore increase across: initial worker -> restart worker
        # -> process-level respawn.
        self.write("restart_starttime", str(starttime_for_age(4)))

    # --- construction helpers ---
    def _install(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text(body)
        path.chmod(0o755)

    def _link_benign_tools(self) -> None:
        """Symlink the real, non-infrastructure tools the probes need.

        The bin dir is the test's ENTIRE PATH, so the interpreter and the
        handful of POSIX utilities the scripts call have to be in there. They
        are symlinks to the host's real copies -- the only real files in the
        dir, and none of them can name a container, a service or a node. The
        guard resolves symlinks, so a symlink out to the real ``docker`` is
        rejected rather than trusted.
        """
        ambient = os.environ.get("PATH", "")
        for name in BENIGN_TOOLS:
            source = resolve_in_path(name, ambient)
            if source is None:
                raise IsolationViolation(
                    f"cannot build the isolated bin dir: {name!r} is absent from this "
                    "host's PATH"
                )
            link = self.bin / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(source)

    def enable_respawn(self, pid: int, age: float = 1.0, pgrp: int | None = None) -> None:
        self.state.joinpath("respawn").touch()
        self.state.joinpath("respawn_pid").write_text(str(pid))
        # `pgrp` is the parameter; this line previously read an undefined
        # `pgid`. The conditional short-circuited on `pgrp is None`, so the
        # single in-suite caller never touched the bad name and the suite
        # stayed green -- passing `pgrp` at all raised NameError.
        self.state.joinpath("respawn_pgid").write_text(str(pgrp if pgrp is not None else pid))
        self.state.joinpath("respawn_starttime").write_text(str(starttime_for_age(age)))

    def disable_respawn(self) -> None:
        self.state.joinpath("respawn").unlink(missing_ok=True)

    def plant_probe_self(self, pid: int = SELF_PID, pgrp: int = SELF_PGID) -> None:
        """Put the probe's OWN `python3 -c <source>` into the tree, which is the
        exact live condition behind R1."""
        write_entry(self.proc, pid, ["python3", "-c", helper_source()], pgrp, starttime_for_age(3))

    def remove_worker(self) -> None:
        shutil_path = self.proc / str(WORKER_PID)
        if shutil_path.exists():
            for item in shutil_path.iterdir():
                item.unlink()
            shutil_path.rmdir()

    def touch(self, name: str) -> None:
        self.state.joinpath(name).touch()

    def write(self, name: str, value: str) -> None:
        self.state.joinpath(name).write_text(value)

    def calls(self) -> list[str]:
        return [line for line in self.state.joinpath("calls.log").read_text().splitlines() if line]

    def env(self) -> dict[str, str]:
        """The environment a test's subprocess sees. Guarded, not merely careful."""
        env = self._raw_env()
        self.assert_isolated(env)
        return env

    def _raw_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in CLUSTER_ENV_VARS}
        env["PATH"] = str(self.bin)
        env["FAKE_DOCKER_STATE"] = str(self.state)
        env["FAKE_CLUSTER_ROOT"] = str(self.root)
        env["FAKE_CONTAINER"] = self.target.container
        env["OAI2_TEST_SCRATCH_STATE"] = str(self.target.state_dir)
        env.pop("OAI2_PRB_PROC_ROOT", None)
        env.pop("OAI2_PRB_SELF_PID", None)
        env.pop("OAI2_PRB_SELF_PGID", None)
        env.pop("OAI2_PRB_SIMULATE_KILL", None)
        env.update(self.docker_env)
        return env

    def assert_isolated(self, env: dict[str, str] | None = None) -> None:
        """The guard, applied to this cluster. Raises on any live-reachable PATH.

        Called at construction, on every ``env()``, and again immediately
        before every launch, so a test that edits the environment (including via
        ``extra_env``) cannot get a subprocess past the guard.
        """
        assert_isolated_env(
            self._raw_env() if env is None else env,
            bin_dir=self.bin,
            label=f"FakeCluster {self.root}",
        )

    def authorize(self, args: tuple[str, ...]) -> list[str]:
        """The target layer, applied to one launch's arguments.

        Refuses an explicit production ``--container``, and supplies the
        synthetic one when a test omits it, so no run can inherit either
        probe's production default. Then re-checks the authorised target
        through ``isolated_target`` so there is exactly one place that decides.
        """
        argv = list(args)
        for index, token in enumerate(argv):
            if token != "--container":
                continue
            if index + 1 >= len(argv):
                raise IsolationViolation("--container with no value")
            requested = argv[index + 1]
            if requested != self.target.container:
                raise IsolationViolation(
                    f"refusing --container {requested!r}: this isolated cluster may only "
                    f"target {self.target.container!r}. {PRODUCTION_CONTAINER!r} is the production container that the "
                    "test-isolation incident stopped; it is never a test target."
                )
        if "--container" not in argv:
            argv += ["--container", self.target.container]
        isolated_target(
            self.target.container, self.target.state_dir, root=self.root, live=False
        )
        return argv

    def run(self, script: Path, *args: str, extra_env: dict[str, str] | None = None,
            timeout: int = 120) -> subprocess.CompletedProcess[str]:
        argv = self.authorize(args)
        env = self.env()
        if extra_env:
            env.update(extra_env)
        self.assert_isolated(env)   # after extra_env, not before
        return subprocess.run(
            ["bash", str(script), *argv],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def run_selector(self, extra_env: dict[str, str] | None = None,
                     timeout: int = 60) -> subprocess.CompletedProcess[str]:
        """Run ONLY the matching criterion against the synthetic tree. No docker,
        no signals, no restarts.

        The probe's own identity inside the synthetic tree is injected through
        the documented OAI2_PRB_SELF_PID / OAI2_PRB_SELF_PGID seam, because a
        local helper run would otherwise report the real test process's pid.
        """
        env = {
            "OAI2_PRB_SELF_PID": self.state.joinpath("self_pid").read_text().strip(),
            "OAI2_PRB_SELF_PGID": self.state.joinpath("self_pgid").read_text().strip(),
        }
        if extra_env:
            env.update(extra_env)
        return self.run(
            PRB,
            "--selftest-proc-root",
            str(self.proc),
            extra_env=env,
            timeout=timeout,
        )


class _ClusterCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cluster = FakeCluster(Path(self._tmp.name))

    def output(self, result: subprocess.CompletedProcess[str]) -> str:
        return result.stdout + result.stderr

    def fresh_cluster(self) -> FakeCluster:
        """An independent isolated cluster, for a test that must drive two."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return FakeCluster(Path(tmp.name))


# ===========================================================================
# ISOLATION -- the regressions that pin the test-isolation incident closed
# ===========================================================================
# Each test name states the requirement it enforces. Read them together with
# the TEST-ISOLATION INCIDENT block in this module's docstring: every one of
# them describes a way the incident could happen again.
class TestIsolationEnforcement(_ClusterCase):
    # --- requirement 1: default runs cannot reach live infrastructure ---

    def test_iso_r1_every_worker_script_launch_uses_a_single_entry_stub_only_path(self) -> None:
        """Req 1: a real run's PATH is the test's own temp stub dir, alone.

        The old env PREPENDED the stub dir to the inherited PATH, so the real
        `docker` and the real `qpipe` stayed reachable. This asserts the
        replacement on a real launch, not on a constructed dict.
        """
        result = self.cluster.run(PRB, "--deadline", "10")
        self.assertEqual(result.returncode, EX_OK, self.output(result))

        env = self.cluster.env()
        self.assertEqual(
            env["PATH"].split(os.pathsep),
            [str(self.cluster.bin)],
            "a test PATH must be the stub dir and nothing else",
        )
        self.assertTrue(
            is_within(self.cluster.bin, self.cluster.root),
            "the stub dir must live inside the test's own temp root",
        )
        # Stubs are the only thing on the PATH that can name a container.
        for name in REQUIRED_STUBS:
            resolved = resolve_in_path(name, env["PATH"])
            self.assertIsNotNone(resolved, f"{name} must be stubbed")
            self.assertTrue(
                is_within(resolved, self.cluster.bin),
                f"{name} must resolve to the stub dir, not to {resolved}",
            )
        # Everything else in the class has no stub and must not resolve at all.
        for name in INFRA_BINARIES:
            if name in REQUIRED_STUBS:
                continue
            with self.subTest(binary=name):
                self.assertIsNone(resolve_in_path(name, env["PATH"]), name)
        # The bin dir holds nothing but declared stubs and declared benign
        # tools. Every symlink is a tool on the allowlist; every regular file is
        # a stub. A real infrastructure binary cannot hide in here.
        for entry in sorted(self.cluster.bin.iterdir()):
            if entry.is_symlink():
                self.assertIn(
                    entry.name, BENIGN_TOOLS,
                    f"undeclared symlink in the isolated bin dir: {entry.name}",
                )
            else:
                self.assertIn(
                    entry.name, REQUIRED_STUBS + SUITE_STUBS,
                    f"undeclared real file in the isolated bin dir: {entry.name}",
                )

    def test_iso_r2_default_test_target_is_never_the_production_container(self) -> None:
        """Req 1: no default run targets the container the incident stopped.

        Read from the stub's call log, i.e. from what really ran, across BOTH
        probes, including the launches that pass no --container at all and so
        would otherwise inherit each script's production default.
        """
        recovery = self.cluster.run(PRB, "--deadline", "10")
        self.assertEqual(recovery.returncode, EX_OK, self.output(recovery))

        # A second, independent cluster for the other probe: the two probes
        # mutate the same synthetic respawn/restart state, so they must not
        # share a cluster.
        lifecycle_cluster = self.fresh_cluster()
        lifecycle = lifecycle_cluster.run(
            LP, "--deadline-a", "5", "--deadline-b", "5", "--poll-interval", "1"
        )
        self.assertEqual(lifecycle.returncode, 0, self.output(lifecycle))

        calls = self.cluster.calls() + lifecycle_cluster.calls()
        self.assertTrue(calls, "the log must be non-empty for this to prove anything")
        for call in calls:
            self.assertNotIn(
                PRODUCTION_CONTAINER,
                call,
                f"a default run addressed the production container: {call}",
            )
            self.assertIn(SYNTHETIC_CONTAINER, call, f"expected the synthetic target: {call}")
        self.assertNotEqual(SYNTHETIC_CONTAINER, PRODUCTION_CONTAINER)
        self.assertFalse(self.cluster.target.live, "a default cluster must not be a live target")

    def test_iso_r3_every_launch_uses_a_scratch_state_dir_inside_the_temp_root(self) -> None:
        """Req 1: the state dir is a scratch dir, under the test's temp root.

        A production-looking state dir is not a scratch dir, and passing one is
        how an isolated run ends up writing into /var/lib/worker.
        """
        env = self.cluster.env()
        scratch = Path(env["OAI2_TEST_SCRATCH_STATE"])
        self.assertEqual(scratch, self.cluster.scratch_state)
        self.assertTrue(scratch.is_dir(), "the scratch state dir must exist")
        self.assertTrue(
            is_within(scratch, self.cluster.root),
            f"the scratch state dir must be inside the test's own temp root, got {scratch}",
        )
        for production in PRODUCTION_STATE_DIRS:
            self.assertFalse(
                is_within(scratch, production),
                f"the scratch state dir must not be {production}",
            )
        self.assertEqual(
            self.cluster.target.state_dir, self.cluster.scratch_state,
            "the authorised target must be the cluster's own scratch dir",
        )

    def test_iso_r4_a_test_env_carrying_a_live_cluster_endpoint_is_refused(self) -> None:
        """Req 1: a clean PATH is not enough; the endpoint must not be inherited.

        Simulates the developer's shell leaking a real control plane or docker
        context into a test. The guard must refuse it rather than let a probe
        with a stub PATH still address a live cluster.
        """
        for variable, value in (
            ("QPIPE_CLUSTER_URL", "http://192.168.100.39:10534"),
            ("QPIPE_CLUSTER_TOKEN", "production-token"),
            ("DOCKER_HOST", "tcp://192.168.100.39:2375"),
        ):
            with self.subTest(variable=variable):
                env = self.cluster._raw_env()
                self.assertNotIn(variable, env, "the inherited value must be stripped")
                env[variable] = value
                with self.assertRaises(IsolationViolation) as caught:
                    self.cluster.assert_isolated(env)
                self.assertIn(variable, str(caught.exception))

    # --- requirement 2: the production target is rejected, loudly ---

    def test_iso_r5_isolated_mode_refuses_the_production_container_name(self) -> None:
        """Req 2: the isolation layer REFUSES the real production container.

        The exact name from the incident. The refusal must be an exception
        naming the target, not a warning and not a silent substitution.
        """
        with self.assertRaises(IsolationViolation) as caught:
            isolated_target(PRODUCTION_CONTAINER, self.cluster.scratch_state, root=self.cluster.root)
        message = str(caught.exception)
        self.assertIn(PRODUCTION_CONTAINER, message, "the refusal must name the target")
        self.assertIn("REFUSES", message)
        self.assertTrue(
            issubclass(IsolationViolation, AssertionError),
            "an isolation breach must surface as a failing test, never a skip",
        )

    def test_iso_r6_isolated_mode_refuses_a_production_state_dir(self) -> None:
        """Req 2: a production-looking state dir is refused even for a safe name."""
        for state_dir in ("/var/lib/worker", "/var/lib/worker/root", "/srv/oai2/state"):
            with self.subTest(state_dir=state_dir):
                with self.assertRaises(IsolationViolation) as caught:
                    isolated_target(
                        SYNTHETIC_CONTAINER, state_dir, root=self.cluster.root
                    )
                self.assertIn(state_dir, str(caught.exception))
        # And a state dir that is merely absolute is refused too: only a dir
        # under the test's own temp root is a scratch dir.
        with self.assertRaises(IsolationViolation):
            isolated_target(SYNTHETIC_CONTAINER, "/tmp/somewhere-else", root=self.cluster.root)

    def test_iso_r7_a_refused_production_target_executes_nothing_and_fails_loudly(self) -> None:
        """Req 2: the refusal happens BEFORE anything runs.

        This is the incident's shape exactly -- a launch addressed at
        `oai2-worker` -- run through the only launch path this suite has. It
        must raise, and the stub's call log must stay empty, proving no probe
        and no docker call was made on the way to the refusal.
        """
        self.assertEqual(self.cluster.calls(), [], "precondition: nothing has run yet")
        with self.assertRaises(IsolationViolation) as caught:
            self.cluster.run(PRB, "--deadline", "3", "--container", PRODUCTION_CONTAINER)
        self.assertIn(PRODUCTION_CONTAINER, str(caught.exception))
        self.assertEqual(
            self.cluster.calls(),
            [],
            "a refused target must not have executed the probe or any docker verb",
        )
        # The same refusal for the other probe, and for an empty value.
        with self.assertRaises(IsolationViolation):
            self.cluster.run(LP, "--deadline-a", "3", "--deadline-b", "3",
                             "--container", PRODUCTION_CONTAINER)
        with self.assertRaises(IsolationViolation):
            self.cluster.run(PRB, "--deadline", "3", "--container")
        self.assertEqual(self.cluster.calls(), [], "still nothing may have run")

    def test_iso_r8_production_container_is_reachable_only_via_the_default_off_opt_in(self) -> None:
        """Req 2: the genuine production name needs the explicit, default-off opt-in.

        OAI2_LIVE_TESTS unset (the default) refuses it twice over: as a target,
        and as a request for live mode. Only OAI2_LIVE_TESTS=1 admits it, and
        even then it authorises a name -- it never launches anything, because
        no test in this module passes live=True to a launch path.
        """
        self.assertFalse(
            live_mode_enabled(),
            f"live mode must be OFF by default; the suite is running with {LIVE_ENV_VAR} set",
        )
        with self.assertRaises(IsolationViolation):
            isolated_target(PRODUCTION_CONTAINER, self.cluster.scratch_state, root=self.cluster.root)
        with self.assertRaises(IsolationViolation) as caught:
            isolated_target(
                PRODUCTION_CONTAINER, self.cluster.scratch_state,
                root=self.cluster.root, live=True,
            )
        self.assertIn(LIVE_ENV_VAR, str(caught.exception))

        previous = os.environ.pop(LIVE_ENV_VAR, None)
        try:
            os.environ[LIVE_ENV_VAR] = LIVE_OPT_IN
            self.assertTrue(live_mode_enabled())
            target = isolated_target(
                PRODUCTION_CONTAINER, self.cluster.scratch_state,
                root=self.cluster.root, live=True,
            )
            self.assertEqual(target.container, PRODUCTION_CONTAINER)
            self.assertTrue(target.live)
        finally:
            os.environ.pop(LIVE_ENV_VAR, None)
            if previous is not None:
                os.environ[LIVE_ENV_VAR] = previous

        # Opting in changes what the layer AUTHORISES, never what a default
        # test launches: the cluster's own target stays non-production.
        self.assertNotEqual(self.cluster.target.container, PRODUCTION_CONTAINER)
        self.assertFalse(self.cluster.target.live)

    # --- requirement 3: the guard trips on a stub-less environment ---

    def test_iso_r9_guard_trips_on_a_stubless_path_resolving_a_real_docker(self) -> None:
        """Req 3: a PATH with a real `docker` on it is rejected, and never run.

        Builds the stub-less environment the old suite effectively had, points
        PATH at a `docker` that would leave a trace if it ever executed, and
        asserts the guard raises. Then it demonstrates the hazard is real (that
        same PATH really would execute it) and that the guard is what stops the
        suite from doing so.
        """
        hostile = Path(self._tmp.name) / "hostile-bin"
        hostile.mkdir()
        trace = Path(self._tmp.name) / "real-docker-ran"
        decoy = hostile / "docker"
        decoy.write_text(f"#!/bin/sh\n: > '{trace}'\nexit 0\n")
        decoy.chmod(0o755)

        # The hazard is real: without the guard this PATH executes `docker`.
        subprocess.run(["docker", "ps"], env={"PATH": str(hostile)},
                       capture_output=True, text=True, check=False)
        self.assertTrue(trace.exists(), "precondition: this PATH really is live")

        trace.unlink()
        with self.assertRaises(IsolationViolation) as caught:
            self.cluster.assert_isolated({"PATH": str(hostile)})
        message = str(caught.exception)
        self.assertIn("docker", message, "the guard must name the binary it caught")
        self.assertFalse(trace.exists(), "the guard must stop execution, not merely warn")

    def test_iso_r10_guard_trips_on_this_hosts_real_path(self) -> None:
        """Req 3: the guard also rejects the host's real PATH.

        This is the hole itself: the suite used to inherit this PATH. Whatever
        infrastructure this host really has installed, the guard must catch it
        if it is ever handed to a test.
        """
        ambient = os.environ.get("PATH", "")
        present = [name for name in INFRA_BINARIES if resolve_in_path(name, ambient)]
        if not present:
            self.skipTest(
                "this host has no infrastructure binary on PATH, so there is "
                "nothing for the guard to catch here"
            )
        with self.assertRaises(IsolationViolation) as caught:
            self.cluster.assert_isolated({"PATH": ambient})
        message = str(caught.exception)
        for name in present:
            self.assertIn(name, message, f"the guard must name {name}")

    def test_iso_r11_module_level_guard_runs_before_any_test_and_is_satisfiable(self) -> None:
        """Req 3: the module-level guard is real, not a no-op that always raises.

        It is what setUpModule runs, so it must pass on a healthy host. That it
        is satisfiable AND that it rejects the hostile cases above (enforced in
        enforce_module_isolation itself, and by the other tests here) is what
        makes it a control rather than decoration.
        """
        enforce_module_isolation()   # must not raise
        self.assertEqual(
            self.cluster.authorize(("--deadline", "10")),
            ["--deadline", "10", "--container", SYNTHETIC_CONTAINER],
            "an unsupplied container must be filled in with the synthetic name",
        )
        # A cluster can be built and used only because the guard passed.
        self.assertEqual(
            self.cluster.run(PRB, "--deadline", "10").returncode, EX_OK,
            "an isolated run must still work after the guard passed",
        )


# ===========================================================================
# R1 / R2 / R7 -- the matching and exclusion rule itself
# ===========================================================================
class MatchingRuleTests(_ClusterCase):
    """The shipped selector, run against a synthetic /proc tree.

    These tests target the rule rather than the whole script, so a revert to
    loose substring matching fails here with a precise diagnosis.
    """

    def test_r1_negative_regression_legacy_rule_counted_the_probe_and_new_rule_does_not(self) -> None:
        """R1. The live bug: one real worker, probe reported "count: 2".

        The tree holds the real worker AND the probe's own `python3 -c
        <source>`. The pre-fix rule must still count 2 (that is the bug being
        pinned), and the new rule must select exactly the real worker.
        """
        source = helper_source()
        self.cluster.plant_probe_self()

        self.assertIn("worker", source, "the payload must still contain the words R1 tripped on")
        self.assertIn("verify", source)

        worker_cmdline = joined(WORKER_ARGV)
        probe_cmdline = joined(["python3", "-c", source])

        # The OLD rule, reproduced. If this ever stops counting 2, the fixture
        # no longer reproduces the observed live failure and the test is void.
        legacy_hits = [
            cmdline
            for cmdline in (worker_cmdline, probe_cmdline)
            if legacy_rule_matches(cmdline)
        ]
        self.assertEqual(
            len(legacy_hits), 2, "fixture must reproduce the live 'count: 2' self-counting"
        )

        # The NEW rule, executed by the shipped script.
        result = self.cluster.run_selector()
        self.assertEqual(result.returncode, 0, self.output(result))
        matches, rejects = parse_records(self.output(result))

        self.assertEqual(
            len(matches), 1, f"R1: the probe must not count itself. output:\n{self.output(result)}"
        )
        self.assertEqual(matches[0]["pid"], str(WORKER_PID))
        self.assertEqual(select_field(self.output(result), "selected"), "1")

        rejected = {r["pid"]: r["reason"] for r in rejects}
        self.assertIn(str(SELF_PID), rejected, "the probe's own process must be listed as rejected")
        self.assertTrue(
            rejected[str(SELF_PID)].startswith("self-"),
            f"R1: the probe's own process must be rejected by a self exclusion, got {rejected[str(SELF_PID)]!r}",
        )

    def test_r1_marker_layer_rejects_a_non_self_pid_carrying_the_probe_marker(self) -> None:
        """R1, exclusion layer (c)/(d) in isolation: the marker string, not the
        pid and not the process group."""
        # A process that is neither the self pid nor in the self process group,
        # but whose -c script carries the probe's own marker.
        marker = "OAI2_PRB_SELF_HELPER_7f3c1a9d"
        self.cluster.write("self_pid", "9999")
        self.cluster.write("self_pgid", "9999")
        write_entry(
            self.cluster.proc,
            GHOST_PID,
            ["python3", "-c", f"import os  # {marker}\nprint('worker verify')\n"],
            GHOST_PID,
            starttime_for_age(2),
        )

        result = self.cluster.run_selector()
        self.assertEqual(result.returncode, 0, self.output(result))
        matches, rejects = parse_records(self.output(result))
        self.assertEqual(len(matches), 1, "only the real worker may match")
        self.assertEqual(matches[0]["pid"], str(WORKER_PID))
        reasons = {r["pid"]: r["reason"] for r in rejects}
        self.assertEqual(
            reasons.get(str(GHOST_PID)),
            "self-marker-in-argv",
            f"R1: the marker exclusion must fire before the argv anchor, got {reasons!r}",
        )

    def test_r1_self_pgid_layer_rejects_a_real_signature_in_the_probe_process_group(self) -> None:
        """R1, exclusion layer (b) in isolation: a byte-for-byte copy of the real
        worker signature that merely lives in the probe's process group."""
        self.cluster.write("self_pid", "9999")  # the tree holds no pid 9999
        write_entry(
            self.cluster.proc,
            GHOST_PID,
            WORKER_ARGV,               # exactly the worker signature
            SELF_PGID,                 # ... but in the probe's own process group
            starttime_for_age(2),
        )

        result = self.cluster.run_selector()
        self.assertEqual(result.returncode, 0, self.output(result))
        matches, rejects = parse_records(self.output(result))
        self.assertEqual(len(matches), 1, "only one worker may match")
        reasons = {r["pid"]: r["reason"] for r in rejects}
        self.assertEqual(
            reasons.get(str(GHOST_PID)),
            "self-pgid",
            "R1: a same-signature process in the probe's process group must be excluded",
        )

    def test_r1_self_pid_layer_rejects_the_probe_own_pid(self) -> None:
        """R1, exclusion layer (a) in isolation: the probe's own pid carrying the
        real signature, in an unrelated process group."""
        self.cluster.write("self_pgid", "9999")
        write_entry(
            self.cluster.proc,
            SELF_PID,
            WORKER_ARGV,
            GHOST_PID,                 # deliberately not the self process group
            starttime_for_age(2),
        )

        result = self.cluster.run_selector()
        self.assertEqual(result.returncode, 0, self.output(result))
        matches, rejects = parse_records(self.output(result))
        self.assertEqual(len(matches), 1)
        reasons = {r["pid"]: r["reason"] for r in rejects}
        self.assertEqual(reasons.get(str(SELF_PID)), "self-pid")

    def test_r1_signature_is_anchored_and_rejects_near_misses(self) -> None:
        """R1 precision: the rule matches the real command line and only it."""
        self.cluster.remove_worker()
        self.cluster.write("self_pid", "9999")
        self.cluster.write("self_pgid", "9999")

        sloppy = list(WORKER_ARGV)          # reformatted but semantically identical
        sloppy[2] = "from  qpipe.cli import main ;  main()"
        cases = [
            (WORKER_ARGV, True, "the exact entrypoint command line"),
            (sloppy, True, "whitespace-reformatted worker script"),
            (["python3", "-m", "qpipe.cli", "worker", "verify", "--node-id", "p50"],
             False, "module invocation instead of -c"),
            (["python3", "-c", WORKER_C_SCRIPT, "worker", "monitor", "--node-id", "p50"],
             False, "different subcommand"),
            (["python3", "-c", "from qpipe.cli import main", "worker", "verify", "--node-id", "p50"],
             False, "no entry call"),
            (["python3", "-c", WORKER_C_SCRIPT + "; setup()", "worker", "verify", "--node-id", "p50"],
             False, "extra work in the -c script"),
            (["python3", "-c", WORKER_C_SCRIPT, "worker", "verify", "--url", "http://x"],
             False, "no --node-id, so not the node worker"),
            (["sh", "-c", WORKER_C_SCRIPT + " worker verify --node-id p50"],
             False, "not a python interpreter"),
        ]
        for argv, expected, why in cases:
            with self.subTest(why=why):
                self.cluster.remove_worker()
                for child in [c for c in self.cluster.proc.iterdir() if c.is_dir()]:
                    for item in child.iterdir():
                        item.unlink()
                    child.rmdir()
                write_entry(
                    self.cluster.proc, 300, argv, 300, starttime_for_age(5)
                )
                result = self.cluster.run_selector()
                self.assertEqual(result.returncode, 0, self.output(result))
                matches, _ = parse_records(self.output(result))
                self.assertEqual(
                    len(matches),
                    1 if expected else 0,
                    f"R1: {why} -- matched={len(matches)}\n{self.output(result)}",
                )

    def test_r2_selector_reports_starttime_from_stat_field_22(self) -> None:
        """R2: identity is pid AND start time, and the start time really is
        field 22 of /proc/<pid>/stat (index 19 after the comm's ')')."""
        expected = 987654321
        self.cluster.write("self_pid", "9999")
        self.cluster.write("self_pgid", "9999")
        self.cluster.remove_worker()
        write_entry(self.cluster.proc, WORKER_PID, WORKER_ARGV, WORKER_PGID, expected)

        result = self.cluster.run_selector()
        self.assertEqual(result.returncode, 0, self.output(result))
        matches, _ = parse_records(self.output(result))
        self.assertEqual(len(matches), 1, self.output(result))
        self.assertEqual(matches[0]["starttime"], str(expected))
        self.assertEqual(
            select_field(self.output(result), "target_starttime"), str(expected)
        )

    def test_r2_selector_refuses_to_match_a_process_whose_stat_is_unreadable(self) -> None:
        """R2: an identity that cannot be read cannot be asserted. The selector
        rejects it instead of guessing."""
        self.cluster.write("self_pid", "9999")
        self.cluster.write("self_pgid", "9999")
        self.cluster.remove_worker()
        directory = self.cluster.proc / str(WORKER_PID)
        directory.mkdir()
        directory.joinpath("cmdline").write_bytes(
            b"\0".join(t.encode() for t in WORKER_ARGV) + b"\0"
        )
        # no stat file

        result = self.cluster.run_selector()
        self.assertEqual(result.returncode, 0, self.output(result))
        matches, rejects = parse_records(self.output(result))
        self.assertEqual(len(matches), 0, "an unprovable identity must not match")
        self.assertEqual(rejects[0]["reason"], "no-stat-identity-unprovable")

    def test_r7_ambiguity_is_reported_not_guessed(self) -> None:
        """R7: two genuine workers must be reported as 2, so the caller can
        refuse to choose."""
        self.cluster.write("self_pid", "9999")
        self.cluster.write("self_pgid", "9999")
        self.cluster.remove_worker()
        write_entry(self.cluster.proc, 100, WORKER_ARGV, 100, starttime_for_age(3600))
        write_entry(self.cluster.proc, 101, WORKER_ARGV, 101, starttime_for_age(2))

        result = self.cluster.run_selector()
        self.assertEqual(result.returncode, 0, self.output(result))
        self.assertEqual(select_field(self.output(result), "selected"), "2")
        self.assertEqual(select_field(self.output(result), "target_pid"), "-1")


# ===========================================================================
# R3 / R4 / R5 / R6 -- process-recovery-probe.sh, end to end against stubs
# ===========================================================================
class ProcessRecoveryProbeTests(_ClusterCase):
    def test_r3_no_intended_process_found_is_a_failure(self) -> None:
        """R3: the old script printed 'killed pids: NONE' and exited 0."""
        self.cluster.remove_worker()
        self.cluster.disable_respawn()   # nothing may appear, so nothing is found
        result = self.cluster.run(PRB, "--deadline", "3")
        self.assertEqual(
            result.returncode,
            EX_NOT_FOUND,
            f"R3: no intended worker must be a failure. output:\n{self.output(result)}",
        )
        self.assertIn("outcome=FAILED", self.output(result))
        self.assertNotIn("killed pids: NONE", self.output(result))

    def test_r3_kill_targets_only_the_intended_worker(self) -> None:
        """R3: a decoy that merely mentions 'worker' must survive."""
        decoy_pid = 101
        write_entry(
            self.cluster.proc,
            decoy_pid,
            ["sh", "-c", "tail -f /var/log/worker.log"],
            WORKER_PGID,
            starttime_for_age(500),
        )
        self.cluster.plant_probe_self()

        result = self.cluster.run(PRB, "--deadline", "10", "--container", CONTAINER_NAME)
        self.assertEqual(result.returncode, EX_OK, self.output(result))

        self.assertTrue(
            (self.cluster.proc / str(decoy_pid) / "cmdline").exists(),
            "R3: the decoy process must not be signalled",
        )
        self.assertFalse(
            (self.cluster.proc / str(WORKER_PID)).exists(),
            "R3: the intended worker must be gone",
        )

    def test_r3_docker_exec_failure_is_a_failure(self) -> None:
        """R3/R5: a failing docker command must not read as a pass."""
        self.cluster.docker_env["FAKE_DOCKER_MODE"] = "fail-exec"
        result = self.cluster.run(PRB, "--deadline", "3")
        self.assertEqual(result.returncode, EX_DOCKER, self.output(result))
        self.assertIn("outcome=FAILED", self.output(result))

    def test_r3_docker_inspect_failure_is_a_failure(self) -> None:
        """R3/R5: same for the preflight inspect."""
        self.cluster.docker_env["FAKE_DOCKER_MODE"] = "fail-inspect"
        result = self.cluster.run(PRB, "--deadline", "3")
        self.assertEqual(result.returncode, EX_DOCKER, self.output(result))

    def test_r3_container_not_running_is_a_failure(self) -> None:
        """R3/R5: a stopped container cannot host a process-level proof."""
        self.cluster.state.joinpath("running").unlink()
        result = self.cluster.run(PRB, "--deadline", "3")
        self.assertEqual(result.returncode, EX_DOCKER, self.output(result))
        self.assertIn("is not running", self.output(result))

    def test_r4_no_replacement_within_the_deadline_is_a_failure(self) -> None:
        """R4: the old script slept 25s and printed whatever it saw."""
        self.cluster.disable_respawn()
        started = time.monotonic()
        result = self.cluster.run(PRB, "--deadline", "3", "--poll-interval", "1")
        elapsed = time.monotonic() - started

        self.assertEqual(
            result.returncode, EX_DEADLINE, f"R4: no replacement must fail. output:\n{self.output(result)}"
        )
        self.assertIn("no replacement within", self.output(result))
        self.assertLess(elapsed, 30, f"R5: the wait must be bounded, took {elapsed:.1f}s")

    def test_r5_exceeding_the_deadline_is_a_failure_with_a_shell_deadline_of_one(self) -> None:
        """R5: even a one-second deadline must be enforced and reported."""
        self.cluster.disable_respawn()
        started = time.monotonic()
        result = self.cluster.run(PRB, "--deadline", "1", "--poll-interval", "1")
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, EX_DEADLINE, self.output(result))
        self.assertLess(elapsed, 20, f"a 1s deadline must not take {elapsed:.1f}s")

    def test_r4_replacement_with_a_reused_pid_and_a_new_start_time_is_rejected_by_default(self) -> None:
        """R2/R4: a same-pid replacement is only acceptable when the caller opts
        in; the new start time is what actually proves it is a new process."""
        self.cluster.write("respawn_starttime", str(starttime_for_age(1)))
        self.cluster.write("respawn_pid", str(WORKER_PID))  # the very same pid
        result = self.cluster.run(PRB, "--deadline", "3")
        self.assertEqual(result.returncode, EX_DEADLINE, self.output(result))
        self.assertIn(
            "pid-reused-and-strict-new-pid-required", self.output(result)
        )

    def test_r4_replacement_with_a_reused_pid_is_accepted_when_opted_in(self) -> None:
        """R4: --allow-pid-reuse accepts a same-pid replacement, but only with a
        strictly newer start time."""
        self.cluster.write("respawn_starttime", str(starttime_for_age(1)))
        self.cluster.write("respawn_pid", str(WORKER_PID))
        result = self.cluster.run(PRB, "--deadline", "5", "--allow-pid-reuse")
        self.assertEqual(result.returncode, EX_OK, self.output(result))
        self.assertIn("outcome=PROVEN-RECOVERY", self.output(result))

    def test_r4_replacement_with_a_stale_start_time_is_never_accepted(self) -> None:
        """R2: a process that reappears with the ORIGINAL start time is not a
        replacement, and must never be accepted."""
        self.cluster.write("respawn_starttime", str(self.cluster.worker_starttime))
        self.cluster.write("respawn_pid", str(WORKER_PID))
        result = self.cluster.run(PRB, "--deadline", "3", "--allow-pid-reuse")
        self.assertNotEqual(result.returncode, EX_OK, self.output(result))
        self.assertIn(
            "still alive with the original start time", self.output(result)
        )

    def test_r5_exit_zero_only_on_a_fully_proven_recovery(self) -> None:
        """R5/R6: 0 requires every criterion to be a real, passed check."""
        self.cluster.plant_probe_self()
        result = self.cluster.run(PRB, "--deadline", "10")
        out = self.output(result)
        self.assertEqual(result.returncode, EX_OK, out)
        for criterion in (
            "criterion container-running",
            "criterion identify",
            "criterion terminate",
            "criterion replacement",
            "criterion container-still-up",
        ):
            self.assertIn(f"[PASS] {criterion}", out)
        self.assertIn("outcome=PROVEN-RECOVERY", out)
        self.assertNotIn("[FAIL]", out)

    def test_r5_container_down_after_recovery_is_a_failure(self) -> None:
        """R5: the final container check is a real assertion, not a print."""
        self.cluster.touch("restart_noop")
        self.cluster.state.joinpath("running").unlink()
        result = self.cluster.run(PRB, "--deadline", "3")
        self.assertNotEqual(result.returncode, EX_OK, self.output(result))

    def test_r6_acceptance_never_depends_on_docker_logs(self) -> None:
        """R6: a successful log read is not evidence of recovery. Neither the
        source nor an actual run may touch docker logs."""
        self.assertNotIn("docker logs", code_lines(PRB), "R6: no docker logs invocation")
        self.cluster.disable_respawn()
        self.cluster.run(PRB, "--deadline", "2")
        self.assertNotIn("logs", docker_verbs(self.cluster.calls()))

    def test_r6_a_printed_count_is_not_acceptance(self) -> None:
        """R6: the selector's own count is never the thing that decides. With no
        replacement the probe fails even though a worker pid is still printed."""
        self.cluster.disable_respawn()
        result = self.cluster.run(PRB, "--deadline", "2")
        self.assertEqual(result.returncode, EX_DEADLINE, self.output(result))
        self.assertIn("outcome=FAILED", self.output(result))

    def test_r1_end_to_end_probe_does_not_count_its_own_source_in_the_tree(self) -> None:
        """R1 end to end: the probe's own source sits in the tree and the probe
        still selects exactly one process -- the real worker."""
        self.cluster.plant_probe_self()
        result = self.cluster.run(PRB, "--deadline", "10")
        out = self.output(result)
        self.assertEqual(result.returncode, EX_OK, out)
        match = re.search(r"selected=1 target_pid=(\d+) target_starttime=(\d+)", out)
        self.assertIsNotNone(match, out)
        self.assertEqual(
            match.group(1), str(WORKER_PID), "R1: the probe targeted itself, not the worker"
        )
        summary = re.search(r"new_pid=(\d+) new_starttime=(\d+)", out)
        self.assertIsNotNone(summary, out)
        self.assertNotEqual(
            summary.group(2), match.group(2), "R2: the replacement start time must be new"
        )

    def test_r3_probe_never_leaves_the_worker_scope(self) -> None:
        """R3/scope: no systemctl, no wsl, and only inspect/exec on the one
        container -- read from the stub's call log, i.e. from what really ran."""
        self.cluster.run(PRB, "--deadline", "5")
        verbs = docker_verbs(self.cluster.calls())
        for forbidden in ("restart", "stop", "start", "rm", "kill", "logs"):
            self.assertNotIn(forbidden, verbs, f"R5: the probe must not run `docker {forbidden}`")
        self.assertEqual(verbs, {"inspect", "exec"})
        for call in self.cluster.calls():
            self.assertNotIn("FORBIDDEN", call, f"scope violation: {call}")
            if call.startswith("exec"):
                self.assertEqual(call.split()[1], CONTAINER_NAME, call)

    def test_r5_usage_errors_exit_two(self) -> None:
        """R5: the documented usage code is real."""
        self.assertEqual(self.cluster.run(PRB, "--deadline", "0").returncode, EX_USAGE)
        self.assertEqual(self.cluster.run(PRB, "--deadline", "abc").returncode, EX_USAGE)
        self.assertEqual(self.cluster.run(PRB, "--nope").returncode, EX_USAGE)

    def test_r1_probe_is_not_hardwired_to_one_node(self) -> None:
        """The container name and the worker signature are parameters."""
        self.cluster.plant_probe_self()
        result = self.cluster.run(
            PRB,
            "--container", CONTAINER_NAME,
            "--subcommand", "worker verify",
            "--node-flag", "--node-id",
            "--deadline", "10",
        )
        self.assertEqual(result.returncode, EX_OK, self.output(result))


# ===========================================================================
# L1 / L2 / L3 / L5 -- lifecycle-probe.sh
# ===========================================================================
class LifecycleProbeTests(_ClusterCase):
    def test_l3_both_cases_reported_and_exit_zero_when_both_recover(self) -> None:
        """L3/L5: CASE A and CASE B are reported separately and each is earned."""
        result = self.cluster.run(
            LP, "--deadline-a", "5", "--deadline-b", "5", "--poll-interval", "1"
        )
        out = self.output(result)
        self.assertEqual(result.returncode, 0, out)
        self.assertIn("CASE A: PASS", out)
        self.assertIn("CASE B: PASS", out)
        self.assertIn("case_a=PASS case_b=PASS", out)

    def test_l2_failed_worker_lookup_is_a_failure_not_a_warning(self) -> None:
        """L2: the old script only printed the ps output, which never existed."""
        self.cluster.remove_worker()
        self.cluster.disable_respawn()
        result = self.cluster.run(
            LP, "--deadline-a", "3", "--deadline-b", "3", "--poll-interval", "1"
        )
        out = self.output(result)
        self.assertNotEqual(result.returncode, 0, out)
        self.assertIn("CASE B: FAIL", out)
        self.assertNotIn("WARNING", out)
        self.assertIn("no intended worker process found", out)

    def test_l3_case_b_failure_exits_non_zero_while_case_a_still_passes(self) -> None:
        """L3/L5: one failing case must fail the run even when the other passes."""
        self.cluster.disable_respawn()
        result = self.cluster.run(
            LP, "--deadline-a", "5", "--deadline-b", "3", "--poll-interval", "1"
        )
        out = self.output(result)
        self.assertIn("CASE A: PASS", out)
        self.assertIn("CASE B: FAIL", out)
        self.assertEqual(result.returncode, 1, out)
        self.assertIn("case_a=PASS case_b=FAIL", out)

    def test_l3_docker_restart_failure_is_reported_as_a_failure(self) -> None:
        """L3: a docker command failure is surfaced, never swallowed."""
        self.cluster.docker_env["FAKE_DOCKER_MODE"] = "fail-restart"
        result = self.cluster.run(
            LP, "--deadline-a", "3", "--deadline-b", "3", "--poll-interval", "1"
        )
        out = self.output(result)
        self.assertIn("CASE A: FAIL", out)
        self.assertEqual(result.returncode, 3, out)

    def test_l5_case_a_requires_startedat_to_advance(self) -> None:
        """L5: a `docker restart` that changed nothing is not a recovery."""
        self.cluster.touch("restart_keep_startedat")
        result = self.cluster.run(
            LP, "--deadline-a", "5", "--deadline-b", "3", "--poll-interval", "1"
        )
        out = self.output(result)
        self.assertIn("CASE A: FAIL", out)
        self.assertIn("StartedAt did not advance", out)
        self.assertNotEqual(result.returncode, 0, out)

    def test_l5_case_b_requires_restartcount_to_be_unchanged(self) -> None:
        """L5: process recovery must not be rescued by a container restart."""
        self.cluster.touch("bump_restarts_on_respawn")
        result = self.cluster.run(
            LP, "--deadline-a", "5", "--deadline-b", "5", "--poll-interval", "1"
        )
        out = self.output(result)
        self.assertIn("CASE B: FAIL", out)
        self.assertIn("RestartCount changed", out)
        self.assertNotEqual(result.returncode, 0, out)

    def test_l3_lifecycle_never_leaves_the_worker_scope(self) -> None:
        """L1/scope: no systemctl, no wsl, and only the one container."""
        self.cluster.run(LP, "--deadline-a", "3", "--deadline-b", "3", "--poll-interval", "1")
        for call in self.cluster.calls():
            self.assertNotIn("FORBIDDEN", call, f"scope violation: {call}")
            if call.startswith("exec"):
                self.assertEqual(call.split()[1], CONTAINER_NAME, call)

    def test_l1_lifecycle_uses_proc_and_never_ps_or_pkill(self) -> None:
        """L1: the image has no ps and no pkill, so the old process test was a
        silent no-op. The new one reads /proc through the shared selector.

        Asserted against the SHELL CODE and against the old script's own
        mechanism (`sh -c "ps ..."`), so the header that quotes the old command
        for documentation cannot satisfy or break the assertion.
        """
        text = LP.read_text(encoding="utf-8")
        code = code_lines(LP)
        self.assertIsNone(
            re.search(r"(?m)^\s*(ps|pkill)\s", code),
            "L1: no ps/pkill invocation may remain",
        )
        self.assertNotIn("sh -c", code, "L1: the old script's sh -c ps|pkill path must be gone")
        self.assertIn("PRB_HELPER_SOURCE", text)
        self.assertIn("process-recovery-probe.sh", text)
        # And it really does read /proc: the selector's proc root is the only
        # source of process information.
        self.assertIn('"--proc-root"', helper_source())


# ===========================================================================
# OWNER STATE -- the supervisor's owner-intent resolution contract
# ===========================================================================
# worker-supervisor.sh is the one worker script whose owner-state handling was
# changed without a regression test: the four defects below (absent conflated
# with damaged, a parser that disagreed with its own header, a non-atomic
# update, and an untestable interruption) were each real, and each was fixed in
# place. A fix with no test is a comment, not a fix, so they are pinned here.
#
# These tests run the REAL script against the isolated cluster, but only in the
# two owner-side modes that cannot start, stop or inspect anything
# (--resolve-intent, --set-intent) plus bounded resident attempts whose
# container-runtime start is guaranteed to fail against the stub. Every launch
# names the synthetic container and a scratch state dir inside the temp root,
# exactly like the probe tests, so none of this can reach the live worker.
SUP = REPO / "deploy" / "worker" / "worker-supervisor.sh"

# Exit codes the script's own header documents. Hard-coded here on purpose: a
# test that read them out of the script could not detect a change to them.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NO_DOCKERD = 10
EXIT_BAD_INTENT = 12

DAMAGED_SENTINELS = {
    "empty": b"",
    "whitespace_only": b"   \n\t\n",
    "garbage": b"definitely-not-an-intent\n",
    "truncated_prefix": b"sto",
    "truncated_run_no_newline": b"ru",
    "interior_space": b"s top\n",
    "trailing_token": b"stop please\n",
}


class SupervisorOwnerStateTests(_ClusterCase):
    """Owner-intent resolution, pinned as a regression suite."""

    def sentinel(self) -> Path:
        return self.cluster.scratch_state / "supervisor-intent"

    def write_sentinel(self, payload: bytes, mode: int = 0o644) -> Path:
        path = self.sentinel()
        path.write_bytes(payload)
        path.chmod(mode)
        return path

    def run_supervisor(self, *args: str, extra_env: dict[str, str] | None = None,
                       timeout: int = 60) -> subprocess.CompletedProcess[str]:
        """Launch the real supervisor, isolated and bounded.

        ``--state-dir`` is ALWAYS the cluster's scratch dir, so a missing or
        mistyped flag cannot fall through to the production ``/var/lib/worker``.
        The dockerd start attempts are pinned to one, so a `run` intent fails
        fast against the refusing systemctl stub instead of looping for two
        minutes.
        """
        return self.cluster.run(
            SUP,
            "--state-dir", str(self.cluster.scratch_state),
            "--dockerd-attempts", "1",
            "--dockerd-sleep", "1",
            *args,
            extra_env=extra_env,
            timeout=timeout,
        )

    def resolved(self) -> tuple[int, str]:
        result = self.run_supervisor("--resolve-intent")
        return result.returncode, self.output(result).splitlines()[0].strip() if self.output(result).splitlines() else ""

    # --- the resolution rule -------------------------------------------------

    def test_sup_r1_absent_sentinel_is_the_only_run_default(self) -> None:
        """No sentinel is a genuine first start and means run. Nothing else does."""
        self.assertFalse(self.sentinel().exists())
        code, token = self.resolved()
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(token, "run")

    def test_sup_r2_damaged_sentinel_fails_closed_with_exit_12(self) -> None:
        """Present-but-unusable owner state is a FAULT, never permission to run.

        This is the defect that mattered most: the old code resolved an empty,
        whitespace-only and unreadable file to `run`, so a damaged file silently
        granted permission and the supervisor then went on to fail elsewhere
        with a completely misleading reason.
        """
        for name, payload in DAMAGED_SENTINELS.items():
            with self.subTest(sentinel=name):
                self.sentinel().unlink(missing_ok=True)
                self.write_sentinel(payload)
                code, token = self.resolved()
                self.assertEqual(code, EXIT_BAD_INTENT,
                                 f"sentinel {name!r} resolved to {token!r} with exit {code}; "
                                 f"owner state must fail closed with {EXIT_BAD_INTENT}")
                self.assertTrue(token.startswith("bad:"),
                                f"sentinel {name!r} must be reported as damaged, got {token!r}")

    def test_sup_r3_a_sentinel_that_is_not_a_readable_regular_file_is_damaged(self) -> None:
        """A directory or a dangling symlink in the sentinel's place is damage."""
        path = self.sentinel()
        path.mkdir(parents=True, exist_ok=True)
        code, token = self.resolved()
        self.assertEqual(code, EXIT_BAD_INTENT)
        self.assertTrue(token.startswith("bad:"), token)
        path.rmdir()

        link = self.sentinel()
        link.symlink_to(self.cluster.scratch_state / "does-not-exist")
        code, token = self.resolved()
        self.assertEqual(code, EXIT_BAD_INTENT)
        self.assertTrue(token.startswith("bad:"), token)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0,
                     "root bypasses file permissions, so mode 000 is not unreadable")
    def test_sup_r4_unreadable_sentinel_fails_closed(self) -> None:
        """An unreadable file is damaged, not a first start.

        The specific REASON is asserted, not just the exit code. There are two
        independent ways an unreadable sentinel is caught -- the explicit -r
        test, and the failure of the read that follows -- and either alone is
        enough to fail closed. Asserting only the code would let one of them
        be deleted without the suite noticing, so the branch under test is
        named here.
        """
        path = self.write_sentinel(b"run\n", mode=0o000)
        self.addCleanup(path.chmod, 0o644)
        result = self.run_supervisor("--resolve-intent")
        self.assertEqual(result.returncode, EXIT_BAD_INTENT, self.output(result))
        self.assertIn("bad:unreadable", self.output(result))
        self.assertIn("not readable by this uid", self.output(result))

    def test_sup_r5_first_non_blank_line_is_the_value(self) -> None:
        """The parser now does what the header always said it did.

        The header documented "first non-blank line" while the code read
        `head -n 1`, so a sentinel written with a leading blank line resolved to
        run -- the opposite of the owner's instruction, on the one input a
        hand-written sentinel is most likely to have.
        """
        for payload, expected in ((b"\nstop\n", "stop"), (b"  STOP \r\n", "stop"),
                                  (b"\n\n\t pause \n", "pause"), (b"remove\n", "remove"),
                                  (b"run", "run")):
            with self.subTest(payload=payload):
                self.sentinel().unlink(missing_ok=True)
                self.write_sentinel(payload)
                code, token = self.resolved()
                self.assertEqual(code, EXIT_OK, self.output(self.run_supervisor("--resolve-intent")))
                self.assertEqual(token, expected)

    def test_sup_r6_normalisation_strips_edges_but_not_interior_characters(self) -> None:
        """`  STOP \\r` is stop; `s top` is garbage.

        The old normaliser used `tr -d`, which deleted interior spaces as well
        as the surrounding ones, so a corrupted value could be repaired into a
        valid instruction. Normalisation is now edge-only.
        """
        self.sentinel().unlink(missing_ok=True)
        self.write_sentinel(b"  STOP \r\n")
        self.assertEqual(self.resolved(), (EXIT_OK, "stop"))
        self.sentinel().unlink(missing_ok=True)
        self.write_sentinel(b"s top\n")
        code, token = self.resolved()
        self.assertEqual(code, EXIT_BAD_INTENT)
        self.assertTrue(token.startswith("bad:"), token)

    # --- atomic update, and the interruption it has to survive ---------------

    def test_sup_r7_set_intent_round_trips_and_clear_restores_the_run_default(self) -> None:
        """The documented owner command works, and `clear` is a true clear."""
        for value in ("run", "stop", "pause", "remove"):
            with self.subTest(intent=value):
                result = self.run_supervisor("--set-intent", value)
                self.assertEqual(result.returncode, EXIT_OK, self.output(result))
                self.assertEqual(self.resolved(), (EXIT_OK, value))
        result = self.run_supervisor("--set-intent", "clear")
        self.assertEqual(result.returncode, EXIT_OK, self.output(result))
        self.assertFalse(self.sentinel().exists(),
                         "clear must remove the sentinel, leaving the run default")
        self.assertEqual(self.resolved(), (EXIT_OK, "run"))

    def test_sup_r8_set_intent_writes_a_single_complete_line(self) -> None:
        """A reader that opens the sentinel at any instant sees a whole value.

        The update is a temp file in the same directory plus a rename, so there
        is no window in which the sentinel is truncated. A zero-byte or
        multi-value sentinel would mean a reader could still observe damage.
        """
        self.run_supervisor("--set-intent", "stop")
        raw = self.sentinel().read_bytes()
        self.assertTrue(raw.endswith(b"\n"), raw)
        self.assertEqual(len(raw.splitlines()), 1, raw)
        self.assertEqual(raw.decode().strip(), "stop")

    def test_sup_r9_set_intent_rejects_an_unknown_value_without_writing(self) -> None:
        """A typo'd owner command fails loudly and leaves no sentinel behind."""
        result = self.run_supervisor("--set-intent", "stopp")
        self.assertEqual(result.returncode, EXIT_USAGE, self.output(result))
        self.assertFalse(self.sentinel().exists())

    def test_sup_r10_an_interrupted_write_is_executable_and_fails_closed(self) -> None:
        """The torn-write seam is runnable, and it fails closed.

        Previously "fails closed on a partial value" was prose: there was no way
        to interrupt a write and therefore no way to test the claim. The exact
        -match rule is also what makes a genuinely truncated on-disk value
        (`sto`) unrecognised rather than silently honoured.
        """
        self.write_sentinel(b"run\n")
        result = self.run_supervisor(extra_env={"WORKER_SIMULATE_TORN_INTENT": "1"})
        self.assertEqual(result.returncode, EXIT_BAD_INTENT, self.output(result))
        self.assertIn("FAULT", self.output(result))
        self.assertIn("bad:torn", self.output(result))
        health = (self.cluster.scratch_state / "supervisor-health").read_text()
        self.assertIn("fatal-bad-intent-torn-write", health, health)
        # And it started nothing: the seam never authorises work.
        self.assertNotIn("FORBIDDEN systemctl", (self.cluster.state / "calls.log").read_text())

    # --- what the resolution actually gates ----------------------------------

    def test_sup_r11_damage_never_reaches_the_supervision_path(self) -> None:
        """A damaged sentinel stops the run BEFORE any container-runtime work.

        The refusal is observable in two independent ways: the exit code is the
        intent fault, and the refusing runtime was never invoked at all.
        """
        self.write_sentinel(b"")
        before = (self.cluster.state / "calls.log").read_text()
        result = self.run_supervisor()
        self.assertEqual(result.returncode, EXIT_BAD_INTENT, self.output(result))
        after = (self.cluster.state / "calls.log").read_text()
        self.assertEqual(after, before,
                         "a damaged sentinel must be refused before the container "
                         f"runtime is touched, but calls were logged:\n{after}")

    def test_sup_r12_an_absent_sentinel_reaches_the_supervision_path(self) -> None:
        """The other half of the pair: `run` really does proceed.

        Without this, a supervisor that refused everything would pass the whole
        suite above. `systemctl` is the refusing stub here, so the run is
        bounded and fails with the dockerd code -- which is itself the proof
        that intent resolution was satisfied and execution continued.
        """
        code, token = self.resolved()
        self.assertEqual((code, token), (EXIT_OK, "run"))
        result = self.run_supervisor()
        self.assertEqual(result.returncode, EXIT_NO_DOCKERD, self.output(result))
        self.assertIn("FORBIDDEN systemctl", (self.cluster.state / "calls.log").read_text())

    def test_sup_r13_damage_is_a_fault_and_stop_is_an_intended_shutdown(self) -> None:
        """The two are not the same event and must not share a code or wording.

        Collapsing them is how a damaged file gets reported as an owner
        shutdown: an operator reads "stopped" and assumes they asked for it.
        """
        self.sentinel().unlink(missing_ok=True)
        self.write_sentinel(b"stop\n")
        stop = self.run_supervisor()
        self.assertIn(stop.returncode, (EXIT_OK, 13),
                      f"stop must not be a fault: {stop.returncode}\n{self.output(stop)}")
        self.assertNotIn("FAULT", self.output(stop))

        self.sentinel().unlink(missing_ok=True)
        self.write_sentinel(b"nonsense\n")
        damaged = self.run_supervisor()
        self.assertEqual(damaged.returncode, EXIT_BAD_INTENT, self.output(damaged))
        self.assertIn("FAULT", self.output(damaged))

    def test_sup_r14_supervisor_exit_is_not_a_cluster_pause_or_drain(self) -> None:
        """The script must not present its own exit as cluster-side control.

        Exiting is a host-local event. The cluster's view of a node -- pause,
        drain, revoke, removal -- is owned by the control plane, and claiming
        otherwise here is how a reader concludes a drill happened when it did
        not. The boundary is asserted in the script's own text.
        """
        source = SUP.read_text(encoding="utf-8")
        self.assertIn("not a cluster pause", source)
        for forbidden in ("drains the node", "revokes the node", "pauses the cluster"):
            self.assertNotIn(forbidden, source)

    def test_sup_r15_no_launch_ever_touches_a_production_state_dir(self) -> None:
        """Every launch in this class is pinned to the scratch dir.

        The supervisor's own default is /var/lib/worker, so a test that forgot
        --state-dir would write real owner state on a developer machine. This
        asserts the pinning rather than trusting each call site.
        """
        self.assertTrue(str(self.cluster.scratch_state).startswith(str(self.cluster.root)))
        for production in PRODUCTION_STATE_DIRS:
            self.assertNotEqual(str(self.cluster.scratch_state), production)
        self.run_supervisor("--resolve-intent")
        self.assertFalse(Path("/var/lib/worker").exists(),
                         "a supervisor run created the production state dir")


# ===========================================================================
# R5 / R6 / R7 -- source-level contracts of the two scripts
# ===========================================================================
class ScriptContractTests(unittest.TestCase):
    def test_bash_n_passes_on_both_scripts(self) -> None:
        for script in (PRB, LP):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_r5_exit_code_contract_is_documented_in_the_probe(self) -> None:
        source = PRB.read_text(encoding="utf-8")
        self.assertIn("EXIT CODE CONTRACT", source)
        for code in ("0", "2", "3", "4", "5", "6", "7"):
            self.assertRegex(
                source,
                rf"(?m)^\s*#\s*{code}\s{{2,}}\S",
                f"exit code {code} must be documented in the header",
            )

    def test_r5_default_deadline_is_sixty_seconds_and_not_a_fixed_sleep(self) -> None:
        """R5: the old script slept a fixed 25s and had no deadline at all."""
        source = PRB.read_text(encoding="utf-8")
        self.assertRegex(source, r"(?m)^DEADLINE=60$")
        self.assertIsNone(
            re.search(r"(?m)^\s*sleep\s+25\s*$", source),
            "R4/R5: the fixed 25s sleep must be gone",
        )
        self.assertIn("--deadline", source)

    def test_r5_probe_never_calls_systemctl_or_wsl(self) -> None:
        for script in (PRB, LP):
            with self.subTest(script=script.name):
                code = code_lines(script)
                self.assertNotIn("systemctl", code)
                self.assertNotIn("wsl", code)
                self.assertNotRegex(code, r"(?m)^\s*(systemctl|wsl)\b", "must not invoke it")

    def test_r7_probe_marker_is_unique_and_present_in_the_payload(self) -> None:
        """R7: the self-exclusion must be driven by a marker that really is in
        the helper's own source, otherwise it could not fire."""
        source = helper_source()
        match = re.search(r'DEFAULT_MARKER = "([^"]+)"', source)
        self.assertIsNotNone(match, "the payload must define DEFAULT_MARKER")
        marker = match.group(1)

        # Distinctive, and present in the payload -> therefore in the helper's own
        # cmdline, which is what the exclusion keys on.
        self.assertGreaterEqual(len(marker), 16, "the marker must be distinctive")
        self.assertIn(marker, source)

        # The exclusion actually consumes the constant.
        self.assertRegex(
            source, r"found = \[args\.marker, DEFAULT_MARKER\]",
            "the active marker list must always include DEFAULT_MARKER",
        )

        # And the marker cannot collide with the worker it is meant to find.
        self.assertNotIn(marker, joined(WORKER_ARGV))
        self.assertNotIn(marker, WORKER_C_SCRIPT)

    def test_r1_probe_default_signature_matches_the_real_entrypoint_command(self) -> None:
        """R1: the default signature must be the command entrypoint.sh really
        runs, otherwise the probe would look for the wrong process."""
        entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
        self.assertIn(
            f"python3 -c '{WORKER_C_SCRIPT}' worker verify",
            entrypoint,
            "the worker command line this probe targets must exist in entrypoint.sh",
        )
        source = PRB.read_text(encoding="utf-8")
        self.assertIn('IMPORT_STMT="{}"'.format("from qpipe.cli import main"), source)
        self.assertIn('ENTRY_CALL="main()"', source)
        self.assertIn('SUBCOMMAND="worker verify"', source)
        self.assertIn('NODE_FLAG="--node-id"', source)

    def test_r7_payload_and_scripts_are_consistent(self) -> None:
        """R7: both probes must apply one rule, so they cannot drift apart."""
        self.assertIn("process-recovery-probe.sh", LP.read_text(encoding="utf-8"))
        payload = helper_source()
        for token in ("stat", "starttime", "self-pid", "self-pgid", "self-marker-in-c-script"):
            self.assertIn(token, payload, f"the payload must implement {token}")

    def test_r5_both_probes_are_strict_about_their_own_mode(self) -> None:
        for script in (PRB, LP):
            with self.subTest(script=script.name):
                self.assertIn("set -u", script.read_text(encoding="utf-8"))


class RespawnPgidHelperTests(unittest.TestCase):
    """Regression: `enable_respawn(pgrp=...)` used to raise NameError.

    The helper read an undefined `pgid` instead of its own `pgrp` parameter.
    The conditional short-circuited when `pgrp` was None, and the only
    in-suite caller omitted it, so the whole suite stayed green while the
    documented parameter was a guaranteed crash. These tests pass the
    argument, which is the only way the defect is observable.
    """

    def _helper(self) -> tuple[FakeCluster, Path]:
        inst = FakeCluster.__new__(FakeCluster)
        root = Path(tempfile.mkdtemp())
        inst.state = root
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        return inst, root

    def test_default_pgid_is_the_pid_when_pgrp_is_omitted(self) -> None:
        inst, root = self._helper()
        inst.enable_respawn(pid=4242, age=2.0)
        self.assertEqual((root / "respawn_pgid").read_text(), "4242")

    def test_supplied_pgrp_is_written_and_does_not_raise(self) -> None:
        inst, root = self._helper()
        inst.enable_respawn(pid=4242, age=2.0, pgrp=99)
        self.assertEqual((root / "respawn_pgid").read_text(), "99")

    def test_an_explicit_zero_pgid_is_not_silently_replaced_by_the_pid(self) -> None:
        """0 is a real value here, not a falsy placeholder."""
        inst, root = self._helper()
        inst.enable_respawn(pid=4242, age=2.0, pgrp=0)
        self.assertEqual((root / "respawn_pgid").read_text(), "0")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
