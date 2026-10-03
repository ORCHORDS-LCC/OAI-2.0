"""Regression suite for the fleet-worker lifecycle probes (master #241, issue #256).

Pure stdlib, no network, no docker, no ssh, no live host. Run it with:

    python3 -m unittest tests.test_worker_lifecycle -v

How the scripts are exercised
-----------------------------
The two probes are treated as TEXT (source-level contracts) and as PROGRAMS
(their real behaviour, driven against INJECTED FAKE dependencies). A temporary
directory provides:

  * a stub ``docker`` that serves a synthetic ``/proc`` tree, canned
    ``inspect`` state, a simulated ``restart`` and a simulated entrypoint
    respawn loop. It runs the probe's own embedded python payload with the real
    interpreter, so the shipped matching logic executes for real.
  * stub ``systemctl`` and ``wsl`` that log the call and fail. They exist so
    that a probe reaching outside the worker scope fails loudly instead of
    quietly reaching a real host.

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
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

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

CONTAINER_NAME = "oai2-worker"
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
        raise AssertionError("could not extract PRB_HELPER_SOURCE from %s" % PRB)
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
    return "%d (python3) %s\n" % (pid, " ".join(rest))


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
    raise AssertionError("no PRB_SELECT %s= in output:\n%s" % (name, output))


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
CONTAINER = os.environ.get("FAKE_CONTAINER", "oai2-worker")

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


class FakeCluster:
    """A temporary tree with stub executables and a synthetic /proc."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = root / "state"
        self.proc = self.state / "proc"
        self.bin = root / "bin"
        self.state.mkdir(parents=True)
        self.proc.mkdir(parents=True)
        self.bin.mkdir(parents=True)
        self.docker_env: dict[str, str] = {}

        # The real /proc/stat carries `btime`, which is how a process start time
        # (stat field 22) is turned into an age. Without it, age is unprovable
        # and the probe must refuse the process.
        self.proc.joinpath("stat").write_text(
            "cpu  1 2 3 4\nintr 1\nbtime %d\nprocesses 1\n" % BTIME
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
        for tool in ("systemctl", "wsl"):
            self._install(tool, FORBIDDEN_STUB.replace("@@TOOL@@", tool))

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

    def enable_respawn(self, pid: int, age: float = 1.0, pgrp: int | None = None) -> None:
        self.state.joinpath("respawn").touch()
        self.state.joinpath("respawn_pid").write_text(str(pid))
        self.state.joinpath("respawn_pgid").write_text(str(pgid if pgrp is not None else pid))
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
        return [l for l in self.state.joinpath("calls.log").read_text().splitlines() if l]

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PATH"] = "%s%s%s" % (self.bin, os.pathsep, env.get("PATH", ""))
        env["FAKE_DOCKER_STATE"] = str(self.state)
        env["FAKE_CONTAINER"] = CONTAINER_NAME
        env.pop("OAI2_PRB_PROC_ROOT", None)
        env.pop("OAI2_PRB_SELF_PID", None)
        env.pop("OAI2_PRB_SELF_PGID", None)
        env.pop("OAI2_PRB_SIMULATE_KILL", None)
        env.update(self.docker_env)
        return env

    def run(self, script: Path, *args: str, extra_env: dict[str, str] | None = None,
            timeout: int = 120) -> subprocess.CompletedProcess[str]:
        env = self.env()
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(script), *args],
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
            len(matches), 1, "R1: the probe must not count itself. output:\n%s" % self.output(result)
        )
        self.assertEqual(matches[0]["pid"], str(WORKER_PID))
        self.assertEqual(select_field(self.output(result), "selected"), "1")

        rejected = {r["pid"]: r["reason"] for r in rejects}
        self.assertIn(str(SELF_PID), rejected, "the probe's own process must be listed as rejected")
        self.assertTrue(
            rejected[str(SELF_PID)].startswith("self-"),
            "R1: the probe's own process must be rejected by a self exclusion, got %r"
            % rejected[str(SELF_PID)],
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
            ["python3", "-c", "import os  # %s\nprint('worker verify')\n" % marker],
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
            "R1: the marker exclusion must fire before the argv anchor, got %r" % reasons,
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
                    "R1: %s -- matched=%d\n%s" % (why, len(matches), self.output(result)),
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
            "R3: no intended worker must be a failure. output:\n%s" % self.output(result),
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
            result.returncode, EX_DEADLINE, "R4: no replacement must fail. output:\n%s" % self.output(result)
        )
        self.assertIn("no replacement within", self.output(result))
        self.assertLess(elapsed, 30, "R5: the wait must be bounded, took %.1fs" % elapsed)

    def test_r5_exceeding_the_deadline_is_a_failure_with_a_shell_deadline_of_one(self) -> None:
        """R5: even a one-second deadline must be enforced and reported."""
        self.cluster.disable_respawn()
        started = time.monotonic()
        result = self.cluster.run(PRB, "--deadline", "1", "--poll-interval", "1")
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, EX_DEADLINE, self.output(result))
        self.assertLess(elapsed, 20, "a 1s deadline must not take %.1fs" % elapsed)

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
            self.assertIn("[PASS] %s" % criterion, out)
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
            self.assertNotIn(forbidden, verbs, "R5: the probe must not run `docker %s`" % forbidden)
        self.assertEqual(verbs, {"inspect", "exec"})
        for call in self.cluster.calls():
            self.assertNotIn("FORBIDDEN", call, "scope violation: %s" % call)
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
            self.assertNotIn("FORBIDDEN", call, "scope violation: %s" % call)
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
                r"(?m)^\s*#\s*%s\s{2,}\S" % code,
                "exit code %s must be documented in the header" % code,
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
            "python3 -c '%s' worker verify" % WORKER_C_SCRIPT,
            entrypoint,
            "the worker command line this probe targets must exist in entrypoint.sh",
        )
        source = PRB.read_text(encoding="utf-8")
        self.assertIn('IMPORT_STMT="%s"' % "from qpipe.cli import main", source)
        self.assertIn('ENTRY_CALL="main()"', source)
        self.assertIn('SUBCOMMAND="worker verify"', source)
        self.assertIn('NODE_FLAG="--node-id"', source)

    def test_r7_payload_and_scripts_are_consistent(self) -> None:
        """R7: both probes must apply one rule, so they cannot drift apart."""
        self.assertIn("process-recovery-probe.sh", LP.read_text(encoding="utf-8"))
        payload = helper_source()
        for token in ("stat", "starttime", "self-pid", "self-pgid", "self-marker-in-c-script"):
            self.assertIn(token, payload, "the payload must implement %s" % token)

    def test_r5_both_probes_are_strict_about_their_own_mode(self) -> None:
        for script in (PRB, LP):
            with self.subTest(script=script.name):
                self.assertIn("set -u", script.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
