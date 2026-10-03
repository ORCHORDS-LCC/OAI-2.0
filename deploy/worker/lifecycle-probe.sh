#!/bin/bash
# =============================================================================
# lifecycle-probe.sh -- fleet worker LIFECYCLE recovery proof
# =============================================================================
#
# Context (master #241, issue #256). The worker container is a slim python
# image: it has python3 but NO ps, NO pkill and NO procps. The old version of
# this script looked for the worker process with
#
#     docker exec <c> sh -c "ps -eo pid,args | grep -E 'qpipe|worker'"
#     docker exec <c> sh -c "pkill -f 'qpipe.cli' || pkill -f 'worker verify'"
#
# Neither command exists in that image, so the lookup failed, the output was
# swallowed by a pipe into `head`, every `docker exec` was followed by `|| true`,
# and the script still exited 0. The process test therefore did nothing at all
# while reporting success.
#
# Defects this probe fixes:
#
#   L1 (R6) A missing process tool made the process test a silent no-op. Process
#            discovery now reads /proc with python3, which is present.
#   L2 (R3) A failed worker lookup was a printed line, not a failure.
#   L3 (R5) There was no exit status: the script always exited 0.
#   L4 (R6) Evidence was `docker logs` output plus greps. Reading logs is not
#            evidence that recovery happened.
#   L5 (R1) Matching was a loose `grep -E 'qpipe|worker'` that also matches this
#            probe's own process. The anchored selector below cannot.
#
# Two cases are reported SEPARATELY, each as an explicit PASS or FAIL, and the
# script exits non-zero if either fails:
#
#   CASE A  container restart recovery
#           The worker container is restarted. The case PASSes only if the
#           container is running afterwards, its StartedAt advanced, and a
#           worker process exists again within the deadline.
#
#   CASE B  worker-process recovery
#           A single worker process is terminated in place while the container
#           keeps running. The case PASSes only if that exact process is gone,
#           a replacement with a NEW start time appears within the deadline,
#           the container is still running, and the container's RestartCount did
#           NOT change -- which is what proves the entrypoint's supervised loop
#           (requirement B) did the work rather than the restart policy.
#
# EXIT CODE CONTRACT
#   0  CASE A PASS and CASE B PASS
#   1  at least one case FAILED
#   2  usage / bad parameter / helper payload unavailable
#   3  docker command failure (inspect, exec or restart failed, container absent
#      or not running). Reported as a FAIL for the affected case AND returned
#      as the process exit code, so a broken docker CLI can never read as PASS.
#
# The process matching/exclusion rule is NOT re-implemented here: this script
# sources the payload from process-recovery-probe.sh so both probes apply the
# identical rule and cannot drift apart.
#
# SCOPE
#   Worker-scoped only. It inspects, execs into and restarts the single
#   container named by --container. It never touches the docker service, never
#   calls systemctl or wsl, and never touches a coordinator, a gateway or any
#   other container.
# =============================================================================
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LIB="$SCRIPT_DIR/process-recovery-probe.sh"
if [ ! -r "$LIB" ]; then
  printf 'lifecycle-probe: cannot read the selector payload %s\n' "$LIB" >&2
  exit 2
fi
# shellcheck source=./process-recovery-probe.sh
. "$LIB"
if [ -z "${PRB_HELPER_SOURCE:-}" ]; then
  printf 'lifecycle-probe: selector payload missing from %s\n' "$LIB" >&2
  exit 2
fi

# --- defaults: nothing is hardwired to one node ------------------------------
CONTAINER="oai2-worker"
DEADLINE_A=120
DEADLINE_B=60
POLL_INTERVAL=1
MAX_AGE=180
MARKER="OAI2_PRB_SELF_HELPER_7f3c1a9d"
IMPORT_STMT="from qpipe.cli import main"
ENTRY_CALL="main()"
SUBCOMMAND="worker verify"
NODE_FLAG="--node-id"

HELPER_OUT=""
HELPER_RC=0
CASE_A="FAIL not-run"
CASE_B="FAIL not-run"
DOCKER_RC=0

log() { printf '[%s] %s\n' "$(date -u '+%H:%M:%S')" "$*"; }
case_line() { printf 'CASE %s: %s\n' "$1" "$2"; }

usage() {
  cat <<'USAGE'
usage: lifecycle-probe.sh [options]
  --container NAME        worker container (default: oai2-worker)
  --deadline-a SECONDS    CASE A (container restart) deadline (default: 120)
  --deadline-b SECONDS    CASE B (process recovery) deadline (default: 60)
  --poll-interval SECONDS poll interval while waiting (default: 1)
  --max-age SECONDS       oldest acceptable replacement process (default: 180)
  --import-stmt TEXT      worker -c import statement (default: from qpipe.cli import main)
  --entry-call TEXT       worker -c entry call (default: main())
  --subcommand TEXT       worker subcommand words (default: "worker verify")
  --node-flag TEXT        flag proving the node worker (default: --node-id)
  -h | --help             this text

exit codes: 0 both cases passed, 1 a case failed, 2 usage, 3 docker failure.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --container) CONTAINER="${2:-}"; shift 2 ;;
    --deadline-a) DEADLINE_A="${2:-}"; shift 2 ;;
    --deadline-b) DEADLINE_B="${2:-}"; shift 2 ;;
    --poll-interval) POLL_INTERVAL="${2:-}"; shift 2 ;;
    --max-age) MAX_AGE="${2:-}"; shift 2 ;;
    --import-stmt) IMPORT_STMT="${2:-}"; shift 2 ;;
    --entry-call) ENTRY_CALL="${2:-}"; shift 2 ;;
    --subcommand) SUBCOMMAND="${2:-}"; shift 2 ;;
    --node-flag) NODE_FLAG="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; printf 'lifecycle-probe: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

is_number() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

for pair in "container:$CONTAINER" "deadline-a:$DEADLINE_A" "deadline-b:$DEADLINE_B" \
            "poll-interval:$POLL_INTERVAL" "max-age:$MAX_AGE" "import-stmt:$IMPORT_STMT" \
            "entry-call:$ENTRY_CALL" "subcommand:$SUBCOMMAND" "node-flag:$NODE_FLAG"; do
  if [ -z "${pair#*:}" ]; then
    printf 'lifecycle-probe: --%s must not be empty\n' "${pair%%:*}" >&2
    exit 2
  fi
done
for num in "$DEADLINE_A" "$DEADLINE_B" "$POLL_INTERVAL" "$MAX_AGE"; do
  if ! is_number "$num" || [ "$num" -lt 1 ]; then
    printf 'lifecycle-probe: deadlines, poll-interval and max-age must be positive integers, got %s\n' "$num" >&2
    exit 2
  fi
done

# --- the /proc-backed selector, shared with process-recovery-probe.sh --------
# L1: no ps, no pkill, no procps -- this image has python3 and /proc only.
run_helper() {
  local mode="$1"
  shift
  docker exec "$CONTAINER" python3 -c "$PRB_HELPER_SOURCE" "$mode" \
    --marker "$MARKER" \
    --import-stmt "$IMPORT_STMT" \
    --entry-call "$ENTRY_CALL" \
    --subcommand "$SUBCOMMAND" \
    --node-flag="$NODE_FLAG" \
    "$@" 2>&1
}

helper() {
  local mode="$1"
  shift
  local out
  out=$(run_helper "$mode" "$@")
  HELPER_OUT="$out"
  printf '%s\n' "$out"
  case "$out" in
    *PRB_RESULT\ rc=*) HELPER_RC=$(field_of "$out" rc) ;;
    *) HELPER_RC=3 ;;   # transport lost the result: that is a docker failure
  esac
  case "$HELPER_RC" in
    ''|*[!0-9]*) HELPER_RC=3 ;;
  esac
  return 0
}

field_of() {
  printf '%s\n' "$1" | sed -n "s/.*[[:space:]]$2=\([^[:space:]]*\).*/\1/p" | head -1
}

replacement_field_of() {
  printf '%s\n' "$1" \
    | grep '^PRB_REPLACEMENT ' \
    | sed -n "s/.*[[:space:]]$2=\([^[:space:]]*\).*/\1/p" \
    | head -1
}

detail_of() { field_of "$HELPER_OUT" detail; }

# inspect <format> -> value on stdout, non-zero on failure.
inspect() {
  local out
  if ! out=$(docker inspect "$CONTAINER" --format "$1" 2>&1); then
    DOCKER_RC=3
    printf 'docker inspect failed: %s\n' "$out" >&2
    return 1
  fi
  printf '%s' "$out" | tr -d '[:space:]'
}

# require_exactly_one_worker <label>
# L2: a failed or ambiguous lookup is a FAILURE, never a warning line.
require_exactly_one_worker() {
  local label="$1"
  helper select
  if [ "$HELPER_RC" -ne 0 ]; then
    case "$HELPER_RC" in
      3) DOCKER_RC=3 ;;
    esac
    printf '  %s: lookup failed (%s): %s\n' "$label" "$HELPER_RC" "$(detail_of)"
    return 1
  fi
  local selected
  selected=$(field_of "$HELPER_OUT" selected)
  case "$selected" in
    ''|*[!0-9]*) printf '  %s: unusable selection: %s\n' "$label" "$HELPER_OUT"; return 1 ;;
  esac
  if [ "$selected" -eq 0 ]; then
    printf '  %s: no intended worker process found in %s\n' "$label" "$CONTAINER"
    return 1
  fi
  if [ "$selected" -ne 1 ]; then
    printf '  %s: ambiguous, %s processes match the worker signature\n' "$label" "$selected"
    return 1
  fi
  WORKER_PID=$(field_of "$HELPER_OUT" target_pid)
  WORKER_STARTTIME=$(field_of "$HELPER_OUT" target_starttime)
  printf '  %s: worker pid=%s starttime=%s\n' "$label" "$WORKER_PID" "$WORKER_STARTTIME"
  return 0
}

WORKER_PID=""
WORKER_STARTTIME=""

# =============================================================================
# preflight
# =============================================================================
log "=== worker LIFECYCLE recovery proof (master #241, issue #256) ==="
log "container=$CONTAINER deadline-a=${DEADLINE_A}s deadline-b=${DEADLINE_B}s"
log "process discovery: /proc via python3 (this image has no ps and no pkill)"

if ! BEFORE_RUNNING=$(inspect '{{.State.Running}}'); then
  printf 'lifecycle-probe: docker is unusable; neither case can be judged\n' >&2
  printf 'CASE A: FAIL docker-inspect-failed\n'
  printf 'CASE B: FAIL docker-inspect-failed\n'
  exit 3
fi
if [ "$BEFORE_RUNNING" != "true" ]; then
  printf 'lifecycle-probe: container %s is not running (State.Running=%s)\n' "$CONTAINER" "$BEFORE_RUNNING" >&2
  printf 'CASE A: FAIL container-not-running\n'
  printf 'CASE B: FAIL container-not-running\n'
  exit 3
fi

# =============================================================================
# CASE A -- container restart recovery
# =============================================================================
log "=== CASE A: restart the CONTAINER and require the worker to return ==="
A_FAIL=""
if ! require_exactly_one_worker "case-a before"; then
  A_FAIL="no provable worker process before the restart"
elif ! BEFORE_STARTED=$(inspect '{{.State.StartedAt}}'); then
  A_FAIL="docker inspect failed"
else
  if docker restart "$CONTAINER" >/dev/null 2>&1; then
    log "  restart issued; waiting up to ${DEADLINE_A}s for the worker to re-register"
    # A container restart repopulates the PID namespace, so the worker may
    # legitimately land on the same pid. Identity here is the start time.
    helper wait --baseline-pid "$WORKER_PID" --baseline-starttime "$WORKER_STARTTIME" \
      --deadline "$DEADLINE_A" --poll-interval "$POLL_INTERVAL" --max-age "$MAX_AGE" \
      --allow-pid-reuse
    if [ "$HELPER_RC" -ne 0 ]; then
      A_FAIL="worker did not return within ${DEADLINE_A}s ($(detail_of))"
    else
      A_NEW_PID=$(replacement_field_of "$HELPER_OUT" pid)
      A_NEW_STARTTIME=$(replacement_field_of "$HELPER_OUT" starttime)
      A_AGE=$(replacement_field_of "$HELPER_OUT" age)
      if [ "$A_NEW_STARTTIME" = "$WORKER_STARTTIME" ]; then
        A_FAIL="post-restart worker has the pre-restart start time ($A_NEW_STARTTIME); identity was not refreshed"
      else
        A_AFTER=$(inspect '{{.State.Running}}') || A_FAIL="docker inspect failed after restart"
        if [ -z "$A_FAIL" ]; then
          A_AFTER_STARTED=$(inspect '{{.State.StartedAt}}') || A_FAIL="docker inspect failed after restart"
        fi
        if [ -z "$A_FAIL" ] && [ "$A_AFTER" != "true" ]; then
          A_FAIL="container is not running after the restart (State.Running=$A_AFTER)"
        fi
        if [ -z "$A_FAIL" ] && [ "$A_AFTER_STARTED" = "$BEFORE_STARTED" ]; then
          A_FAIL="StartedAt did not advance ($BEFORE_STARTED); the container did not actually restart"
        fi
        if [ -z "$A_FAIL" ]; then
          printf '  worker after restart: pid=%s starttime=%s age=%ss (was pid=%s starttime=%s)\n' \
            "$A_NEW_PID" "$A_NEW_STARTTIME" "$A_AGE" "$WORKER_PID" "$WORKER_STARTTIME"
        fi
      fi
    fi
  else
    DOCKER_RC=3
    A_FAIL="docker restart failed"
  fi
fi
if [ -n "$A_FAIL" ]; then
  CASE_A="FAIL $A_FAIL"
  printf '  reason: %s\n' "$A_FAIL"
else
  CASE_A="PASS"
fi
case_line A "$CASE_A"

# =============================================================================
# CASE B -- worker-process recovery (the container must NOT restart)
# =============================================================================
log "=== CASE B: kill the worker PROCESS in place; the entrypoint must respawn it ==="
B_FAIL=""
if [ "$DOCKER_RC" -ne 0 ]; then
  B_FAIL="skipped: docker was already failing"
elif ! require_exactly_one_worker "case-b before"; then
  B_FAIL="no provable worker process to terminate"
else
  B_PID="$WORKER_PID"
  B_STARTTIME="$WORKER_STARTTIME"
  B_RESTARTS_BEFORE=$(inspect '{{.RestartCount}}') || B_RESTARTS_BEFORE="unreadable"
  helper kill --target-pid "$B_PID" --target-starttime "$B_STARTTIME"
  if [ "$HELPER_RC" -ne 0 ]; then
    case "$HELPER_RC" in
      3) DOCKER_RC=3 ;;
    esac
    B_FAIL="termination failed ($(detail_of))"
  else
    printf '  terminated worker pid=%s starttime=%s\n' "$B_PID" "$B_STARTTIME"
    helper wait --baseline-pid "$B_PID" --baseline-starttime "$B_STARTTIME" \
      --deadline "$DEADLINE_B" --poll-interval "$POLL_INTERVAL" --max-age "$MAX_AGE" \
      --allow-pid-reuse
    if [ "$HELPER_RC" -ne 0 ]; then
      B_FAIL="no replacement within ${DEADLINE_B}s ($(detail_of))"
    else
      B_NEW_PID=$(replacement_field_of "$HELPER_OUT" pid)
      B_NEW_STARTTIME=$(replacement_field_of "$HELPER_OUT" starttime)
      B_AGE=$(replacement_field_of "$HELPER_OUT" age)
      if [ "$B_NEW_STARTTIME" = "$B_STARTTIME" ]; then
        B_FAIL="replacement reuses the pre-kill start time ($B_NEW_STARTTIME); it is not a new process"
      else
        B_RUNNING=$(inspect '{{.State.Running}}') || B_FAIL="docker inspect failed after the process kill"
        if [ -z "$B_FAIL" ] && [ "$B_RUNNING" != "true" ]; then
          B_FAIL="container stopped during a process-level recovery (State.Running=$B_RUNNING)"
        fi
        if [ -z "$B_FAIL" ]; then
          B_RESTARTS_AFTER=$(inspect '{{.RestartCount}}') || B_RESTARTS_AFTER="unreadable"
          # A process kill must NOT have been rescued by the restart policy.
          if [ "$B_RESTARTS_AFTER" != "$B_RESTARTS_BEFORE" ]; then
            B_FAIL="RestartCount changed $B_RESTARTS_BEFORE -> $B_RESTARTS_AFTER; the container restarted, so this is not process-level recovery"
          else
            printf '  worker after respawn: pid=%s starttime=%s age=%ss (was pid=%s starttime=%s, RestartCount stayed %s)\n' \
              "$B_NEW_PID" "$B_NEW_STARTTIME" "$B_AGE" "$B_PID" "$B_STARTTIME" "$B_RESTARTS_AFTER"
          fi
        fi
      fi
    fi
  fi
fi
if [ -n "$B_FAIL" ]; then
  CASE_B="FAIL $B_FAIL"
  printf '  reason: %s\n' "$B_FAIL"
else
  CASE_B="PASS"
fi
case_line B "$CASE_B"

# =============================================================================
# verdict -- L3: the exit status is earned, and 0 requires BOTH cases to pass
# =============================================================================
printf 'LIFECYCLE_SUMMARY case_a=%s case_b=%s\n' \
  "$(printf '%s' "$CASE_A" | cut -d' ' -f1)" "$(printf '%s' "$CASE_B" | cut -d' ' -f1)"

if [ "$DOCKER_RC" -ne 0 ]; then
  log "=== RESULT: FAIL (docker command failure) ==="
  exit 3
fi
if [ "${CASE_A%% *}" = "PASS" ] && [ "${CASE_B%% *}" = "PASS" ]; then
  log "=== RESULT: PASS (CASE A and CASE B both recovered) ==="
  exit 0
fi
log "=== RESULT: FAIL (CASE A: $CASE_A | CASE B: $CASE_B) ==="
exit 1
