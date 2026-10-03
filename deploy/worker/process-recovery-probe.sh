#!/bin/bash
# =============================================================================
# process-recovery-probe.sh -- proof that the fleet worker PROCESS self-heals
# =============================================================================
#
# Context (master #241, issue #256). The fleet worker runs as a container built
# from a slim python image. That image has python3 but NO ps, NO pkill and NO
# procps, so every process operation must read /proc from inside the container
# with python3. deploy/worker/entrypoint.sh starts the worker as exactly:
#
#   python3 -c 'from qpipe.cli import main; main()' worker verify --url ... --node-id ...
#
# Defects this probe fixes (observed live, not hypothetical):
#
#   R1  SELF-COUNTING. The old probe matched the substrings "worker" AND "verify"
#       anywhere in /proc/<pid>/cmdline. This probe's own `python3 -c` source
#       contains both words literally, so the probe matched ITSELF. A live run
#       against a single real worker printed "count: 2". Any acceptance that
#       trusted that count was wrong.
#   R2  NO IDENTITY. It never recorded the pid / start time of the process it
#       was about to kill, so a recycled pid could be mistaken for the original
#       and a printed number could be mistaken for a replacement.
#   R3  FALSE PASS ON A NO-OP KILL. It printed "killed pids: NONE" and carried
#       on; the script still exited 0.
#   R4  NO REPLACEMENT REQUIREMENT. A fixed sleep(25) then "print whatever is
#       seen", with no assertion.
#   R5  NO DEADLINE AND NO EXIT STATUS. It always exited 0.
#   R6  WRONG EVIDENCE. A successful `docker logs` read plus a printed count
#       were treated as acceptance. Reading logs is not evidence of recovery.
#   R7  NO REGRESSION TEST. Nothing proved the probe could not count itself.
#
# -----------------------------------------------------------------------------
# THE MATCHING RULE (see classify() in the embedded helper below)
# -----------------------------------------------------------------------------
# A process is the intended worker iff ALL of these hold:
#   1. argv[0] basename is python / python3 / python3.N  (it is an interpreter)
#   2. argv[1] == "-c"
#   3. the WHOLE of argv[2], after whitespace normalisation, equals
#        "from qpipe.cli import main; main()"
#      Anchoring the entire script -- not a substring of it -- is what makes the
#      rule incapable of matching a long script that merely mentions the words
#      "worker" and "verify" (which is exactly the R1 failure).
#   4. argv[3:5] == ("worker", "verify")
#   5. "--node-id" occurs in argv[5:]        (it is the node worker, not another
#                                            qpipe subcommand)
#   6. /proc/<pid>/stat is readable, so identity is provable (R2)
#
# AND none of these absolute exclusions apply:
#   a. pid  == this probe helper's own pid
#   b. pgrp == this probe helper's own process group (stat field 5)
#   c. the argv blob contains this probe's own marker string
#   d. argv[1] == "-c" and that -c script contains this probe's own marker
#      (layer (d) is redundant with (c) by design: defence in depth, and it is
#       the layer a regression test can pin independently)
#
# The marker is the literal DEFAULT_MARKER inside the helper source, so the
# helper's own cmdline always carries it, even when --marker is overridden.
# Exclusions (a) and (c)/(d) are independent of (b), so a runtime that happens
# to hand `docker exec` a shared process group still cannot make this probe
# count itself; that case degrades to the safe failure "no worker found"
# (exit 4), never to a false pass. Use --ignore-pgid only if you have actually
# observed that shared-group behaviour.
#
# -----------------------------------------------------------------------------
# EXIT CODE CONTRACT
# -----------------------------------------------------------------------------
#   0  EVERY criterion PASSed. The intended worker was identified by pid AND
#      start time, terminated, and a replacement carrying the same command
#      signature with a new pid AND a new start time appeared before the
#      deadline. This is the only success value.
#   2  usage / bad parameter
#   3  docker command failure (inspect or exec failed, container absent or not
#      running) -- the helper's exit codes are the same numbers, so a helper
#      failure is never confused with a docker failure: the helper always
#      prints a "PRB_RESULT rc=<n>" line, and its absence means transport loss.
#   4  no intended worker process found (nothing was killed)
#   5  termination failure (target vanished before the signal, the signal
#      failed, or the target was still present afterwards)
#   6  no replacement within the bounded deadline (exceeding it is a FAILURE)
#   7  ambiguity / unverifiable identity (more than one intended worker, or the
#      kill target no longer matched the recorded identity)
#
# Acceptance is NEVER a printed count and NEVER `docker logs` output (R6).
# Every step below emits exactly one explicit [PASS] or [FAIL] line and is
# backed by a real check. This script does not call `docker logs` at all.
#
# -----------------------------------------------------------------------------
# SCOPE
# -----------------------------------------------------------------------------
# Worker-scoped only. It may exec and inspect the single container named by
# --container. It never restarts containers, never touches the docker service,
# never calls systemctl or wsl, and never touches a coordinator, a gateway or
# any other container.
#
# -----------------------------------------------------------------------------
# TEST SEAM (used only by tests/test_worker_lifecycle.py)
# -----------------------------------------------------------------------------
#   OAI2_PRB_PROC_ROOT      point the helper at a synthetic /proc tree
#   OAI2_PRB_SELF_PID       the helper's own pid, as seen in that tree
#   OAI2_PRB_SELF_PGID      the helper's own process group, as seen in that tree
#   OAI2_PRB_SIMULATE_KILL  remove the target from the tree instead of
#                           signalling it -- honoured ONLY when the proc root
#                           is not /proc, so a live run can never be diverted
# =============================================================================
set -u

# --- defaults: every one of them is overridable, none is hardwired to a node --
CONTAINER="oai2-worker"
DEADLINE=60
POLL_INTERVAL=1
MAX_AGE=120
MARKER="OAI2_PRB_SELF_HELPER_7f3c1a9d"
IMPORT_STMT="from qpipe.cli import main"
ENTRY_CALL="main()"
SUBCOMMAND="worker verify"
NODE_FLAG="--node-id"
ALLOW_PID_REUSE=0
IGNORE_PGID=0
SELFTEST_PROC_ROOT=""

# --- the selector payload ----------------------------------------------------
# Kept in a variable so this file is also usable as a library: lifecycle-probe.sh
# sources it and reuses exactly the same matching/exclusion rule, so the two
# probes cannot drift apart.
PRB_HELPER_SOURCE=$(cat <<'PRB_HELPER_EOF'
"""Selector for the OAI-2.0 fleet worker process (master #241, issue #256).

Runs inside the slim worker image, which has no ps and no procps, so all
process discovery reads /proc directly.

Defect map:
  R1 self-counting  -- the old rule was `"worker" in cmd and "verify" in cmd`,
                       which matches this helper's own `python3 -c` source and
                       therefore matched the probe itself. The rule below is
                       anchored on the worker's argv shape and excludes self.
  R2 identity       -- every selected process carries pid AND start time, read
                       from field 22 of /proc/<pid>/stat, so a recycled pid can
                       never pass as the original.
  R3 no-op kill     -- a kill that matched nothing returns rc=4, never success.
  R4 replacement    -- mode `wait` requires a new pid and a new start time.
  R5 deadline       -- mode `wait` bounds itself and returns rc=6 when it is
                       exceeded; there is no unbounded path.
  R6 evidence       -- this helper only ever emits real measurements. It never
                       reads docker logs and never reports a count as proof.
  R7 regression     -- classify() is the single testable seam: it is exercised
                       directly against a synthetic /proc tree.

Exit codes are shared with the calling shell: 0 ok, 2 bad arguments,
3 transport/proc-root failure, 4 not found, 5 termination failure,
6 deadline exceeded, 7 ambiguity.
"""

import argparse
import os
import re
import shutil
import signal
import sys
import time

# R1: this literal is the probe's own self-identity marker. It is present in the
# source of this helper, and therefore in the cmdline of this very process.
DEFAULT_MARKER = "OAI2_PRB_SELF_HELPER_7f3c1a9d"
DEFAULT_IMPORT = "from qpipe.cli import main"
DEFAULT_ENTRY = "main()"
DEFAULT_SUBCOMMAND = "worker verify"
DEFAULT_NODE_FLAG = "--node-id"

PYTHON_ARGV0 = re.compile(r"^(python|python3(\.[0-9]+)?)$")

# /proc/<pid>/stat field 3 ("state") is the first token after the closing ")".
FIRST_FIELD_AFTER_COMM = 3
PGRP_INDEX = 5 - FIRST_FIELD_AFTER_COMM        # field 5  -- process group
START_TIME_INDEX = 22 - FIRST_FIELD_AFTER_COMM  # field 22 -- start time

RC_OK = 0
RC_USAGE = 2
RC_TRANSPORT = 3
RC_NOT_FOUND = 4
RC_KILL = 5
RC_DEADLINE = 6
RC_AMBIGUOUS = 7

LIVE_PROC = "/proc"


def slug(text):
    """One shell-safe token, so the shell can read it back out of PRB_RESULT."""
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(text)).strip("-") or "none"


def emit_result(rc, detail=""):
    print("PRB_DETAIL %s" % detail)
    print("PRB_RESULT rc=%d detail=%s" % (rc, slug(detail)))
    sys.stdout.flush()
    return rc


def read_bytes(path):
    try:
        handle = open(path, "rb")
    except OSError:
        return None
    try:
        return handle.read()
    except OSError:
        return None
    finally:
        handle.close()


def read_argv(proc_root, pid):
    """NUL-separated argv, as the kernel sees it. Never space-joined."""
    raw = read_bytes(os.path.join(proc_root, str(pid), "cmdline"))
    if not raw:
        return None
    parts = [chunk.decode("utf-8", "replace") for chunk in raw.split(b"\0")]
    while parts and parts[-1] == "":
        parts.pop()
    return parts or None


def parse_stat(proc_root, pid):
    """(pid, pgrp, starttime) or None. comm may contain spaces, so split on the
    LAST ')' rather than on whitespace."""
    raw = read_bytes(os.path.join(proc_root, str(pid), "stat"))
    if not raw:
        return None
    text = raw.decode("utf-8", "replace")
    left = text.find("(")
    right = text.rfind(")")
    if left < 0 or right <= left:
        return None
    head = text[:text.find(" ")].strip()
    if not head.isdigit():
        return None
    fields = text[right + 1:].split()
    if len(fields) <= START_TIME_INDEX:
        return None
    try:
        return int(head), int(fields[PGRP_INDEX]), int(fields[START_TIME_INDEX])
    except (ValueError, IndexError):
        return None


def boot_time(proc_root):
    raw = read_bytes(os.path.join(proc_root, "stat"))
    if not raw:
        return None
    for line in raw.decode("utf-8", "replace").splitlines():
        if line.startswith("btime "):
            try:
                return int(line.split()[1])
            except (ValueError, IndexError):
                return None
    return None


def clock_ticks():
    try:
        ticks = int(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, ValueError, OSError):
        ticks = 100
    return ticks if ticks > 0 else 100


def age_seconds(proc_root, starttime):
    """Seconds since this process started, or None if it cannot be proven."""
    btime = boot_time(proc_root)
    if btime is None:
        return None
    return time.time() - (btime + float(starttime) / float(clock_ticks()))


def normalise_script(text):
    """Collapse whitespace and normalise ' ; ' so a trivially reformatted worker
    script still matches, without loosening the anchor to a substring."""
    text = re.sub(r"\s+", " ", str(text).strip())
    text = re.sub(r"\s*;\s*", "; ", text)
    return text.rstrip("; ")


def resolve_self(args):
    pid = args.self_pid
    if pid is None:
        env = os.environ.get("OAI2_PRB_SELF_PID", "")
        pid = int(env) if env.isdigit() else os.getpid()
    pgid = args.self_pgid
    if pgid is None:
        env = os.environ.get("OAI2_PRB_SELF_PGID", "")
        pgid = int(env) if env.isdigit() else os.getpgid(0)
    return pid, pgid


def proc_root(args):
    return args.proc_root or os.environ.get("OAI2_PRB_PROC_ROOT") or LIVE_PROC


def markers(args):
    """Always include DEFAULT_MARKER: it is baked into this source, so it is in
    this process's own cmdline even when --marker overrides the active one."""
    found = [args.marker, DEFAULT_MARKER]
    return [m for m in found if m]


def classify(pid, argv, statinfo, args, self_pid, self_pgid, marker_list):
    """R1: the one and only rule for "is this the intended worker?".

    Returns (is_match, reason). Exclusions are evaluated first and are absolute;
    the positive signature is anchored on the whole -c script.
    """
    # (R2) an unreadable stat means identity cannot be proven, so it cannot be
    # an acceptable match. A claim that cannot be proven is not a pass.
    if statinfo is None:
        return False, "no-stat-identity-unprovable"
    _, pgrp, starttime = statinfo
    if starttime <= 0:
        return False, "no-stat-identity-unprovable"

    # --- absolute self exclusions ---
    if pid == self_pid:
        return False, "self-pid"
    if not args.ignore_pgid and self_pgid > 0 and pgrp == self_pgid:
        return False, "self-pgid"
    blob = "\n".join(argv)
    for marker in marker_list:
        if marker in blob:
            return False, "self-marker-in-argv"
    if len(argv) >= 3 and argv[1] == "-c":
        for marker in marker_list:
            if marker in argv[2]:
                return False, "self-marker-in-c-script"

    # --- anchored positive signature ---
    if not PYTHON_ARGV0.match(os.path.basename(argv[0])):
        return False, "argv0-not-python"
    if len(argv) < 3 or argv[1] != "-c":
        return False, "not-a-dash-c-invocation"
    expected = normalise_script(args.import_stmt + "; " + args.entry_call)
    if normalise_script(argv[2]) != expected:
        return False, "c-script-not-anchored-to-worker-import"
    words = args.subcommand.split()
    if list(argv[3:3 + len(words)]) != list(words):
        return False, "subcommand-mismatch"
    tail = argv[3 + len(words):]
    if args.node_flag and args.node_flag not in tail:
        return False, "node-flag-missing"
    return True, "match"


def scan(args, self_pid, self_pgid):
    root = proc_root(args)
    try:
        names = sorted(os.listdir(root))
    except OSError as exc:
        return None, "cannot list %s: %s" % (root, exc)
    marker_list = markers(args)
    records = []
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        argv = read_argv(root, pid)
        if argv is None:
            continue
        statinfo = parse_stat(root, pid)
        matched, reason = classify(
            pid, argv, statinfo, args, self_pid, self_pgid, marker_list
        )
        record = {
            "pid": pid,
            "pgrp": statinfo[1] if statinfo else -1,
            "starttime": statinfo[2] if statinfo else -1,
            "match": matched,
            "reason": reason,
            "age": age_seconds(root, statinfo[2]) if (matched and statinfo) else None,
        }
        records.append(record)
    return records, ""


def print_records(records):
    for record in records:
        if record["match"]:
            print(
                "MATCH pid=%d starttime=%d pgrp=%d age=%.1f"
                % (
                    record["pid"],
                    record["starttime"],
                    record["pgrp"],
                    record["age"] if record["age"] is not None else -1.0,
                )
            )
        else:
            print("REJECT pid=%d reason=%s" % (record["pid"], slug(record["reason"])))


def select_summary(records):
    matched = [r for r in records if r["match"]]
    target = matched[0] if len(matched) == 1 else None
    print(
        "PRB_SELECT selected=%d target_pid=%d target_starttime=%d scanned=%d"
        % (
            len(matched),
            target["pid"] if target else -1,
            target["starttime"] if target else -1,
            len(records),
        )
    )


def find(records, pid):
    for record in records:
        if record["pid"] == pid:
            return record
    return None


def send_signal(args, pid):
    """Live mode signals. Test mode removes the entry from the synthetic tree.
    The simulated branch requires a non-/proc root, so it can never fire on a
    real host."""
    root = proc_root(args)
    if root != LIVE_PROC and os.environ.get("OAI2_PRB_SIMULATE_KILL") == "1":
        shutil.rmtree(os.path.join(root, str(pid)), ignore_errors=True)
        return "simulated"
    os.kill(pid, signal.SIGKILL)
    return "signalled"


def mode_identity(args):
    self_pid, self_pgid = resolve_self(args)
    root = proc_root(args)
    print("PRB_IDENTITY self_pid=%d self_pgid=%d proc_root=%s" % (self_pid, self_pgid, root))
    print("PRB_IDENTITY expected_c_script=%s" % normalise_script(args.import_stmt + "; " + args.entry_call))
    print("PRB_IDENTITY subcommand=[%s] node_flag=[%s]" % (args.subcommand, args.node_flag))
    btime = boot_time(root)
    print("PRB_IDENTITY boot_time=%s clock_ticks=%d" % (btime if btime is not None else -1, clock_ticks()))
    return emit_result(RC_OK, "identity resolved")


def mode_select(args):
    self_pid, self_pgid = resolve_self(args)
    records, error = scan(args, self_pid, self_pgid)
    if records is None:
        return emit_result(RC_TRANSPORT, error)
    print_records(records)
    select_summary(records)
    return emit_result(RC_OK, "scan complete")


def mode_kill(args):
    """R3: terminating nothing is a failure, never a silent success."""
    if args.target_pid is None or args.target_starttime is None:
        return emit_result(RC_USAGE, "kill requires --target-pid and --target-starttime")
    self_pid, self_pgid = resolve_self(args)
    records, error = scan(args, self_pid, self_pgid)
    if records is None:
        return emit_result(RC_TRANSPORT, error)
    matched = [r for r in records if r["match"]]
    if not matched:
        print_records(records)
        return emit_result(RC_NOT_FOUND, "no intended worker process present")
    record = find(records, args.target_pid)
    if record is None:
        return emit_result(
            RC_KILL, "recorded pid %d is no longer present" % args.target_pid
        )
    if not record["match"]:
        return emit_result(
            RC_AMBIGUOUS,
            "pid %d no longer matches the worker signature (%s)" % (args.target_pid, record["reason"]),
        )
    if record["starttime"] != args.target_starttime:
        return emit_result(
            RC_AMBIGUOUS,
            "pid %d start time changed %d -> %d; refusing to signal an unverified process"
            % (args.target_pid, args.target_starttime, record["starttime"]),
        )
    try:
        how = send_signal(args, args.target_pid)
    except OSError as exc:
        return emit_result(RC_KILL, "signal to pid %d failed: %s" % (args.target_pid, exc))
    print("PRB_KILL pid=%d starttime=%d how=%s" % (args.target_pid, record["starttime"], how))
    # R3: prove the process is actually gone rather than assuming the signal worked.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if parse_stat(proc_root(args), args.target_pid) is None:
            return emit_result(RC_OK, "terminated pid %d" % args.target_pid)
        time.sleep(0.1)
    return emit_result(
        RC_KILL, "pid %d still present after SIGKILL" % args.target_pid
    )


def replacement_of(record, baseline_pid, baseline_starttime, args):
    """R2/R4: a replacement is a process with the SAME command signature whose
    identity tuple is new, whose start time is later, and whose age is
    consistent with a fresh process."""
    if record["pid"] == baseline_pid and record["starttime"] == baseline_starttime:
        return False, "same-identity-as-killed-process"
    if record["starttime"] <= baseline_starttime:
        return False, "start-time-not-newer"
    if record["age"] is None:
        return False, "age-unprovable"
    if record["age"] < -1.0:
        return False, "age-in-the-future"
    if record["age"] > args.max_age:
        return False, "age-too-old"
    if record["pid"] == baseline_pid and not args.allow_pid_reuse:
        return False, "pid-reused-and-strict-new-pid-required"
    return True, "replacement"


def mode_wait(args):
    """R4/R5: a bounded wait that REQUIRES a replacement and FAILS on timeout."""
    if args.baseline_pid is None or args.baseline_starttime is None:
        return emit_result(RC_USAGE, "wait requires --baseline-pid and --baseline-starttime")
    self_pid, self_pgid = resolve_self(args)
    deadline = time.time() + args.deadline
    attempts = 0
    last_reason = "no scan performed"
    while True:
        attempts += 1
        records, error = scan(args, self_pid, self_pgid)
        if records is None:
            return emit_result(RC_TRANSPORT, error)
        survivor = find(records, args.baseline_pid)
        if survivor is not None and survivor["starttime"] == args.baseline_starttime:
            return emit_result(
                RC_KILL,
                "killed pid %d is still alive with the original start time"
                % args.baseline_pid,
            )
        candidates = []
        reasons = []
        for record in records:
            if not record["match"]:
                continue
            ok, reason = replacement_of(
                record, args.baseline_pid, args.baseline_starttime, args
            )
            if ok:
                candidates.append(record)
            else:
                reasons.append("%d:%s" % (record["pid"], slug(reason)))
        if len(candidates) == 1:
            winner = candidates[0]
            print(
                "PRB_REPLACEMENT pid=%d starttime=%d age=%.1f attempts=%d"
                % (winner["pid"], winner["starttime"], winner["age"], attempts)
            )
            return emit_result(
                RC_OK,
                "replacement pid %d starttime %d after %d attempt(s)"
                % (winner["pid"], winner["starttime"], attempts),
            )
        if len(candidates) > 1:
            print_records(records)
            return emit_result(
                RC_AMBIGUOUS,
                "%d replacement candidates: %s"
                % (len(candidates), ",".join(str(c["pid"]) for c in candidates)),
            )
        last_reason = ",".join(reasons) or "no worker process present at all"
        if time.time() >= deadline:
            print_records(records)
            return emit_result(
                RC_DEADLINE,
                "no replacement within %gs (%s)" % (args.deadline, last_reason),
            )
        time.sleep(args.poll_interval)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="process-recovery-probe-helper", add_help=True
    )
    parser.add_argument("mode", choices=["identity", "select", "kill", "wait"])
    parser.add_argument("--proc-root", default=None)
    parser.add_argument("--self-pid", type=int, default=None)
    parser.add_argument("--self-pgid", type=int, default=None)
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    parser.add_argument("--import-stmt", default=DEFAULT_IMPORT)
    parser.add_argument("--entry-call", default=DEFAULT_ENTRY)
    parser.add_argument("--subcommand", default=DEFAULT_SUBCOMMAND)
    parser.add_argument("--node-flag", default=DEFAULT_NODE_FLAG)
    parser.add_argument("--ignore-pgid", action="store_true")
    parser.add_argument("--target-pid", type=int, default=None)
    parser.add_argument("--target-starttime", type=int, default=None)
    parser.add_argument("--baseline-pid", type=int, default=None)
    parser.add_argument("--baseline-starttime", type=int, default=None)
    parser.add_argument("--deadline", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-age", type=float, default=120.0)
    parser.add_argument("--allow-pid-reuse", action="store_true")
    return parser


def main(argv):
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return emit_result(RC_USAGE, "bad helper arguments")
    if args.deadline <= 0 or args.poll_interval <= 0 or args.max_age <= 0:
        return emit_result(RC_USAGE, "deadline, poll-interval and max-age must be positive")
    if args.mode == "identity":
        return mode_identity(args)
    if args.mode == "select":
        return mode_select(args)
    if args.mode == "kill":
        return mode_kill(args)
    return mode_wait(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
PRB_HELPER_EOF
)

# Library mode: lifecycle-probe.sh sources this file to reuse PRB_HELPER_SOURCE
# so both probes apply the identical matching/exclusion rule.
if [ "${BASH_SOURCE[0]:-$0}" != "$0" ]; then
  return 0 2>/dev/null || exit 0
fi

# --- output helpers ----------------------------------------------------------
HELPER_OUT=""
HELPER_RC=0

log() { printf '[%s] %s\n' "$(date -u '+%H:%M:%S')" "$*"; }
cr_pass() { printf '  [PASS] %s\n' "$*"; }
cr_fail() { printf '  [FAIL] %s\n' "$*"; }

# die <code> <message>: the only way this script stops. <code> is from the
# contract in the header. Nothing here can be reached with a success code.
die() {
  local code="$1"
  shift
  log "ABORT: $*"
  cr_fail "$* (exit=$code)"
  local slug
  slug=$(printf '%s' "$*" | tr -c 'A-Za-z0-9_.=-' '-' )
  printf 'PRB_SUMMARY outcome=FAILED exit=%d reason=%s\n' "$code" "$slug"
  exit "$code"
}

usage() {
  cat <<'USAGE'
usage: process-recovery-probe.sh [options]
  --container NAME        worker container to inspect (default: oai2-worker)
  --deadline SECONDS      bounded recovery deadline (default: 60).
                          Exceeding it is a FAILURE (exit 6).
  --poll-interval SECONDS replacement poll interval (default: 1)
  --max-age SECONDS       oldest acceptable replacement (default: 120)
  --import-stmt TEXT      worker -c import statement (default: from qpipe.cli import main)
  --entry-call TEXT       worker -c entry call (default: main())
  --subcommand TEXT       worker subcommand words (default: "worker verify")
  --node-flag TEXT        flag proving the node worker (default: --node-id)
  --allow-pid-reuse       accept a replacement that reuses the original pid
                          (a strictly newer start time is still mandatory)
  --ignore-pgid           drop the self-process-group exclusion. Only for a
                          runtime that demonstrably shares a process group with
                          `docker exec`; self-pid and marker exclusions remain.
  --selftest-proc-root D  run only the identity criterion against a synthetic
                          /proc tree. No docker, no signal, no restart. Prints
                          one MATCH/REJECT record per process, for regression
                          testing (R1/R7).
  -h | --help             this text

exit codes: 0 proven recovery, 2 usage, 3 docker failure, 4 no worker found,
5 termination failure, 6 deadline exceeded, 7 ambiguity.
USAGE
}

# --- argument parsing --------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --container) CONTAINER="${2:-}"; shift 2 ;;
    --deadline) DEADLINE="${2:-}"; shift 2 ;;
    --poll-interval) POLL_INTERVAL="${2:-}"; shift 2 ;;
    --max-age) MAX_AGE="${2:-}"; shift 2 ;;
    --import-stmt) IMPORT_STMT="${2:-}"; shift 2 ;;
    --entry-call) ENTRY_CALL="${2:-}"; shift 2 ;;
    --subcommand) SUBCOMMAND="${2:-}"; shift 2 ;;
    --node-flag) NODE_FLAG="${2:-}"; shift 2 ;;
    --marker) MARKER="${2:-}"; shift 2 ;;
    --allow-pid-reuse) ALLOW_PID_REUSE=1; shift ;;
    --ignore-pgid) IGNORE_PGID=1; shift ;;
    --selftest-proc-root) SELFTEST_PROC_ROOT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die 2 "unknown argument: $1" ;;
  esac
done

is_number() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

[ -n "$CONTAINER" ] || die 2 "--container must not be empty"
is_number "$DEADLINE" || die 2 "--deadline must be a positive integer, got '$DEADLINE'"
[ "$DEADLINE" -ge 1 ] || die 2 "--deadline must be >= 1, got '$DEADLINE'"
is_number "$POLL_INTERVAL" || die 2 "--poll-interval must be a positive integer"
[ "$POLL_INTERVAL" -ge 1 ] || die 2 "--poll-interval must be >= 1"
is_number "$MAX_AGE" || die 2 "--max-age must be a positive integer"
[ "$MAX_AGE" -ge 1 ] || die 2 "--max-age must be >= 1"
[ -n "$IMPORT_STMT" ] || die 2 "--import-stmt must not be empty"
[ -n "$ENTRY_CALL" ] || die 2 "--entry-call must not be empty"
[ -n "$SUBCOMMAND" ] || die 2 "--subcommand must not be empty"
[ -n "$NODE_FLAG" ] || die 2 "--node-flag must not be empty"
[ -n "$MARKER" ] || die 2 "--marker must not be empty"

# --- selector plumbing -------------------------------------------------------
# how = local | docker
# R6: the helper's exit code is read from its PRB_RESULT line, never inferred
# from a count and never from log output. A missing PRB_RESULT line means the
# transport (docker exec) lost the result, which is a docker failure, not a
# probe verdict.
run_helper() {
  local how="$1" mode="$2"
  shift 2
  if [ "$how" = "local" ]; then
    python3 -c "$PRB_HELPER_SOURCE" "$mode" \
      --proc-root "$SELFTEST_PROC_ROOT" \
      --marker "$MARKER" \
      --import-stmt "$IMPORT_STMT" \
      --entry-call "$ENTRY_CALL" \
      --subcommand "$SUBCOMMAND" \
      --node-flag="$NODE_FLAG" \
      "$@" 2>&1
    return $?
  fi
  docker exec "$CONTAINER" python3 -c "$PRB_HELPER_SOURCE" "$mode" \
    --marker "$MARKER" \
    --import-stmt "$IMPORT_STMT" \
    --entry-call "$ENTRY_CALL" \
    --subcommand "$SUBCOMMAND" \
    --node-flag="$NODE_FLAG" \
    "$@" 2>&1
}

helper() {
  local how="$1" mode="$2"
  shift 2
  local out rc
  out=$(run_helper "$how" "$mode" "$@")
  rc=$?
  HELPER_OUT="$out"
  printf '%s\n' "$out"
  case "$out" in
    *PRB_RESULT\ rc=*) HELPER_RC=$(field_of "$out" rc) ;;
    *) HELPER_RC=3 ;;   # transport lost the result: docker failure
  esac
  case "$HELPER_RC" in
    ''|*[!0-9]*) HELPER_RC=3 ;;
  esac
  return 0
}

field_of() {
  printf '%s\n' "$1" | sed -n "s/.*[[:space:]]$2=\([^[:space:]]*\).*/\1/p" | head -1
}

# Scoped to the helper's PRB_REPLACEMENT line, so a MATCH record elsewhere in
# the output can never be mistaken for the replacement.
replacement_field_of() {
  printf '%s\n' "$1" \
    | grep '^PRB_REPLACEMENT ' \
    | sed -n "s/.*[[:space:]]$2=\([^[:space:]]*\).*/\1/p" \
    | head -1
}

# --- self test mode: no docker, no signal, no restart (R1/R7) ---------------
if [ -n "$SELFTEST_PROC_ROOT" ]; then
  [ -d "$SELFTEST_PROC_ROOT" ] || die 2 "--selftest-proc-root is not a directory: $SELFTEST_PROC_ROOT"
  log "=== selector self test against synthetic $SELFTEST_PROC_ROOT (no docker, no signals) ==="
  helper local identity
  [ "$HELPER_RC" -eq 0 ] || die "$HELPER_RC" "selector identity check failed"
  helper local select
  if [ "$HELPER_RC" -ne 0 ]; then
    die "$HELPER_RC" "selector scan failed against $SELFTEST_PROC_ROOT"
  fi
  log "=== selector self test complete: selected=$(field_of "$HELPER_OUT" selected) ==="
  exit 0
fi

# =============================================================================
# main
# =============================================================================
log "=== worker PROCESS recovery proof (master #241 requirement B, issue #256) ==="
log "container=$CONTAINER deadline=${DEADLINE}s poll=${POLL_INTERVAL}s max_age=${MAX_AGE}s"
log "signature: python3 -c '$IMPORT_STMT; $ENTRY_CALL' $SUBCOMMAND $NODE_FLAG <value>"

# --- criterion 0: the worker container is reachable and running --------------
# R3/R5: a docker failure is a failure, stated as one.
if ! INSPECT_OUT=$(docker inspect "$CONTAINER" --format '{{.State.Running}}' 2>&1); then
  die 3 "docker inspect failed for '$CONTAINER': $INSPECT_OUT"
fi
RUNNING=$(printf '%s' "$INSPECT_OUT" | tr -d '[:space:]')
if [ "$RUNNING" != "true" ]; then
  die 3 "container '$CONTAINER' is not running (State.Running=$RUNNING)"
fi
cr_pass "criterion container-running: '$CONTAINER' is running"

# --- criterion 1: identify the intended worker, by pid AND start time -------
# R1: the anchored signature plus the absolute self exclusions. R2: the start
# time is captured here, BEFORE anything is terminated.
helper docker select
if [ "$HELPER_RC" -ne 0 ]; then
  die "$HELPER_RC" "worker scan failed: $(field_of "$HELPER_OUT" detail)"
fi
SELECTED=$(field_of "$HELPER_OUT" selected)
TARGET_PID=$(field_of "$HELPER_OUT" target_pid)
TARGET_STARTTIME=$(field_of "$HELPER_OUT" target_starttime)
case "$SELECTED" in
  ''|*[!0-9]*) die 3 "docker exec returned no usable selection: $HELPER_OUT" ;;
esac
if [ "$SELECTED" -eq 0 ]; then
  # R3: nothing to terminate is a failure, not a shrug. The old script printed
  # "killed pids: NONE" here and exited 0.
  die 4 "no intended worker process matched the signature in '$CONTAINER'; refusing to signal anything"
fi
if [ "$SELECTED" -ne 1 ]; then
  # R1/R7: more than one match means the exclusion rule regressed, or the node
  # really is running duplicates. Either way: refuse to guess.
  die 7 "ambiguous: $SELECTED processes match the worker signature in '$CONTAINER'; refusing to choose"
fi
cr_pass "criterion identify: worker pid=$TARGET_PID starttime=$TARGET_STARTTIME (probe excluded, identity recorded)"

# --- criterion 2: terminate exactly that process ----------------------------
helper docker kill --target-pid "$TARGET_PID" --target-starttime "$TARGET_STARTTIME"
case "$HELPER_RC" in
  0) cr_pass "criterion terminate: pid=$TARGET_PID (starttime=$TARGET_STARTTIME) terminated" ;;
  *) die "$HELPER_RC" "termination failed: $(field_of "$HELPER_OUT" detail)" ;;
esac

# --- criterion 3: a replacement must appear before the deadline -------------
# R4/R5: bounded, and exceeding the deadline is a FAILURE (exit 6), not a
# "looks fine" print.
REUSE_FLAG=""
[ "$ALLOW_PID_REUSE" -eq 1 ] && REUSE_FLAG="--allow-pid-reuse"
# shellcheck disable=SC2086
helper docker wait --baseline-pid "$TARGET_PID" --baseline-starttime "$TARGET_STARTTIME" \
  --deadline "$DEADLINE" --poll-interval "$POLL_INTERVAL" --max-age "$MAX_AGE" $REUSE_FLAG
case "$HELPER_RC" in
  0) : ;;
  *) die "$HELPER_RC" "recovery not proven: $(field_of "$HELPER_OUT" detail)" ;;
esac
NEW_PID=$(replacement_field_of "$HELPER_OUT" pid)
NEW_STARTTIME=$(replacement_field_of "$HELPER_OUT" starttime)
NEW_AGE=$(replacement_field_of "$HELPER_OUT" age)
case "$NEW_PID$NEW_STARTTIME$NEW_AGE" in
  ''|*[!0-9.]*) die 3 "docker exec returned no usable replacement record: $HELPER_OUT" ;;
esac
# The helper already enforced the strict new-pid rule. Repeat it here only when
# the caller did NOT opt in, so --allow-pid-reuse is honoured exactly once.
if [ "$ALLOW_PID_REUSE" -eq 0 ] && [ "$NEW_PID" = "$TARGET_PID" ]; then
  die 6 "replacement reused pid $NEW_PID under the strict rule; pass --allow-pid-reuse if that is expected"
fi
if [ "$NEW_STARTTIME" = "$TARGET_STARTTIME" ]; then
  die 6 "replacement reused the original start time $NEW_STARTTIME; it is not a new process"
fi
cr_pass "criterion replacement: worker pid=$NEW_PID starttime=$NEW_STARTTIME age=${NEW_AGE}s appeared within ${DEADLINE}s"

# --- criterion 4: this was a PROCESS recovery, not a container restart ------
if ! INSPECT_OUT=$(docker inspect "$CONTAINER" --format '{{.State.Running}}' 2>&1); then
  die 3 "docker inspect failed after recovery for '$CONTAINER': $INSPECT_OUT"
fi
RUNNING=$(printf '%s' "$INSPECT_OUT" | tr -d '[:space:]')
if [ "$RUNNING" != "true" ]; then
  die 3 "container '$CONTAINER' is not running after process recovery"
fi
cr_pass "criterion container-still-up: '$CONTAINER' survived the process kill (no container restart involved)"

# --- verdict: every criterion above is a real check, so 0 is earned ----------
printf 'PRB_SUMMARY selected=1 target_pid=%s target_starttime=%s new_pid=%s new_starttime=%s new_age=%s deadline=%s outcome=PROVEN-RECOVERY\n' \
  "$TARGET_PID" "$TARGET_STARTTIME" "$NEW_PID" "$NEW_STARTTIME" "$NEW_AGE" "$DEADLINE"
log "PROVEN: worker pid=$TARGET_PID (start $TARGET_STARTTIME) was replaced by pid=$NEW_PID (start $NEW_STARTTIME) within ${DEADLINE}s"
log "=== RESULT: PASS ==="
exit 0
