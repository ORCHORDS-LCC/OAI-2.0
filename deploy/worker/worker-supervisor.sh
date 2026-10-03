#!/bin/bash
# ===========================================================================
# worker-supervisor.sh - resident container-runtime supervisor for the
# OAI-2.0 fleet worker container (node p50). Master #241, issue #251.
#
# WHAT THIS PROCESS IS
#   The single lifecycle owner for requirement A of master #241:
#
#     A. host / container runtime start  -> THIS script + the scheduled task
#     B. worker-process restart          -> the container entrypoint's loop
#     C. application-level reconnect     -> the worker client's registration
#                                           and heartbeat against the pipeline
#
#   It is invoked ATTACHED and never returns in normal operation. Holding the
#   WSL2 session is what keeps the VM, dockerd and therefore the worker alive:
#   WSL2 tears the VM down once no session is attached, and a container restart
#   policy cannot help, because that policy lives in dockerd and dockerd is
#   what died.
#
#   It invokes a real FILE, never `bash -c` with a multi-line script. An earlier
#   launcher passed the script body to `bash -c` as a PowerShell here-string;
#   PowerShell split it on newlines, bash received only the first line, exited
#   immediately and the node left the fleet within a minute of every launch.
#
# SCOPE (never violated)
#   This script acts on exactly ONE container: the worker container (default
#   oai2-worker). It never starts, stops or reconfigures a coordinator, gateway,
#   model server or any other pipeline service. The only system unit it touches
#   is the container-runtime unit, and only ever to check or start it - never
#   enable, disable, mask or reconfigure it. It never touches WSL
#   configuration. It never holds or logs a credential.
#
# EXIT CODE CONTRACT (also mirrored by start-worker.ps1, which passes these
# through unchanged, and by install-worker-task.ps1's task settings)
#
#     0   INTENDED owner shutdown. The owner asked for stop/pause/removal, or
#         sent SIGTERM/SIGINT, or ran --help. Not a failure: the scheduled task
#         records SUCCESS and, because Task Scheduler only auto-restarts a
#         task that FAILED, the task is not bounced back up. This is what makes
#         "the owner stopped the worker" indistinguishable from "the worker is
#         broken" impossible.
#     2   Invalid invocation or unusable environment: unknown option, missing
#         option value, non-numeric value where a number is required, or a
#         state directory that cannot be created/written.
#    10   dockerd could not be made active inside the bounded startup window
#         (--dockerd-attempts x --dockerd-sleep seconds). Startup only.
#    11   The worker container is absent, or cannot be started, inside the
#         missing-container grace window. Recreating it needs the image and the
#         node's runtime secrets, which this process deliberately does not have,
#         so retrying forever would be a hot loop that can never succeed.
#    12   The owner-intent file exists but holds an unrecognised value. Fail
#         CLOSED: nothing is started or restarted.
#    13   Owner intent was read and understood, but could not be carried out
#         (for example intent=stop while dockerd is unreachable). Reported as a
#         failure so it is retried and surfaced, instead of claiming a clean
#         stop that did not happen.
#
#   In STEADY state the supervisor does not exit for infrastructure failures. It
#   retries dockerd and the container with bounded exponential backoff, logs a
#   loud ERROR on every failed attempt, escalates once past the failure cap and
#   records everything in the health file. It deliberately does NOT exit there,
#   because exiting releases the WSL2 session and takes dockerd down with it -
#   turning a recoverable runtime fault into an unrecoverable node outage.
#
# OWNER-INTENT SENTINEL (durable stop / pause / removal)
#   Path:   /var/lib/worker/supervisor-intent   (dir overridable with
#           --state-dir / WORKER_STATE_DIR)
#   Values: first non-blank line only; surrounding spaces, CR and letter case
#           are ignored. `run` is the default and may be written explicitly.
#
#     run      supervise normally: keep dockerd up, keep the container running
#     stop     the owner wants the worker down. The supervisor stops the
#              container once, then exits 0. Exits 13 if that stop failed.
#     pause    the owner has already paused it, or wants the supervisor to stay
#              out of the way. The supervisor touches nothing and exits 0.
#     remove   the owner removed the container, or does not want it recreated.
#              The supervisor never creates a container, so it logs the
#              container's real presence if it can observe it and exits 0. It
#              does not delete anything.
#
#   Examples (run as root inside WSL):
#     printf 'stop\n'   > /var/lib/worker/supervisor-intent
#     printf 'pause\n'  > /var/lib/worker/supervisor-intent
#     printf 'remove\n' > /var/lib/worker/supervisor-intent
#     printf 'run\n'    > /var/lib/worker/supervisor-intent   # or: rm the file
#
#   Clearing the file, or writing `run`, restores automatic startup. The
#   sentinel is durable: it lives on disk, so it still says `stop` after the
#   supervisor, the task or the whole VM has been restarted. An unreadable or
#   empty file is treated as `run`. An UNRECOGNISED value fails closed with
#   exit 12 and starts nothing.
#
# STATE FILES (all under --state-dir, default /var/lib/worker)
#   supervisor-intent   owner sentinel, values above
#   supervisor-health   heartbeat, rewritten every loop; inspect with
#                       `cat /var/lib/worker/supervisor-health`. Keys:
#                       updated_utc, supervisor_pid, state, container,
#                       container_running, container_status, dockerd_active,
#                       docker_ready, consecutive_failures, last_action,
#                       last_error, last_exit_code, last_exit_reason.
#   supervisor.log      the same log lines as stdout, rotated once at 4 MiB
#                       (one generation, supervisor.log.1) - this process is
#                       resident for weeks, an unbounded log is a real defect.
#
# DEFECTS THIS REWRITE FIXES (issue #251; all observed live, not hypothetical)
#   D1  The old script ran a bounded `for _ in $(seq 1 60)` dockerd start loop
#       and then, when dockerd was still down, logged "continuing to retry".
#       That was false: the loop had ended, and the following infinite loop
#       only ever looked at the container, so a dead dockerd was never restarted
#       and never logged again. dockerd recovery is now a first-class, bounded,
#       observable path inside the main loop (see ensure_dockerd).
#   D2  The old script restarted the worker even after the owner had stopped or
#       removed it. Owner intent is now a durable sentinel honoured at the top
#       of every loop iteration AND inside every wait: all sleeps go through
#       sleep_responsive, which re-reads the sentinel in 5 s slices, so a durable
#       stop takes effect within ~5 s even in the middle of a 120 s backoff.
#   D3  Owner shutdown and a crash were indistinguishable. They are now
#       separated by exit code (0 vs 10/11/12/13) AND by log text.
#   D4  `docker start ... || true` and `2>/dev/null` discarded the only evidence
#       of a broken runtime. Real docker/systemd error text is now captured and
#       logged, and the health file carries the last error for an owner who is
#       not watching the log.
#   D5  `docker start` retried forever at a fixed 30 s cadence with no escalation
#       and no cap. Retries now use bounded exponential backoff with a ceiling,
#       and a loud one-shot ESCALATION past --max-failures.
#   D6  There was no exit contract at all. See the table above; it is also what
#       lets start-worker.ps1 report a real failure to Task Scheduler instead of
#       always claiming SUCCESS.
#
# USAGE
#   wsl.exe -d Ubuntu-24.04 -u root -e bash /mnt/c/Users/P50/q-pipe/worker-supervisor.sh
#
#   --container NAME           worker container (default: oai2-worker)
#   --state-dir PATH           state directory (default: /var/lib/worker)
#   --interval SECONDS         steady-state poll interval (default: 30)
#   --dockerd-attempts N       start attempts per recovery pass (default: 60)
#   --dockerd-sleep SECONDS    delay between those attempts (default: 2)
#   --backoff-max SECONDS      retry backoff ceiling (default: 120)
#   --max-failures N           consecutive failures before escalation (default: 5)
#   --missing-attempts N       grace before exit 11 (default: 3)
#   --docker-unit NAME         container-runtime unit (default: docker)
#   -h, --help                 this text
# ===========================================================================
set -u

# ------------------------------------------------------------------ defaults
DEF_CONTAINER='oai2-worker'
DEF_STATE_DIR='/var/lib/worker'
DEF_INTERVAL='30'
DEF_DOCKERD_ATTEMPTS='60'
DEF_DOCKERD_SLEEP='2'
DEF_BACKOFF_MAX='120'
DEF_MAX_FAILURES='5'
DEF_MISSING_ATTEMPTS='3'
DEF_DOCKER_UNIT='docker'
SLEEP_SLICE='5'            # owner intent is honoured within this many seconds
DOCKER_READY_ATTEMPTS='15'
LOG_MAX_BYTES='4194304'   # 4 MiB, then one rotation

container="${DEF_CONTAINER}"
state_dir="${WORKER_STATE_DIR:-$DEF_STATE_DIR}"
interval="${WORKER_POLL_INTERVAL:-$DEF_INTERVAL}"
dockerd_attempts="${WORKER_DOCKERD_ATTEMPTS:-$DEF_DOCKERD_ATTEMPTS}"
dockerd_sleep="${WORKER_DOCKERD_SLEEP:-$DEF_DOCKERD_SLEEP}"
backoff_max="${WORKER_BACKOFF_MAX:-$DEF_BACKOFF_MAX}"
max_failures="${WORKER_MAX_FAILURES:-$DEF_MAX_FAILURES}"
missing_attempts="${WORKER_MISSING_ATTEMPTS:-$DEF_MISSING_ATTEMPTS}"
docker_unit="${WORKER_DOCKER_UNIT:-$DEF_DOCKER_UNIT}"

EXIT_OK=0
EXIT_USAGE=2
EXIT_NO_DOCKERD=10
EXIT_NO_CONTAINER=11
EXIT_BAD_INTENT=12
EXIT_INTENT_UNMET=13

# ------------------------------------------------------------------- logging
append_log() {
  [ -n "$LOG_FILE" ] || return 0
  if [ -f "$LOG_FILE" ]; then
    bytes=$(wc -c <"$LOG_FILE" 2>/dev/null | tr -cd '0-9')
    [ -n "$bytes" ] || bytes=0
    if [ "$bytes" -gt "$LOG_MAX_BYTES" ]; then
      mv -f "$LOG_FILE" "$LOG_FILE.1" 2>/dev/null || rm -f "$LOG_FILE" 2>/dev/null || :
    fi
  fi
  printf '%s\n' "$1" >>"$LOG_FILE" 2>/dev/null || :
}

log() {
  level=$1
  shift
  line=$(printf '[%s] %-5s %s' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$level" "$*")
  printf '%s\n' "$line"
  append_log "$line"
}

# Collapse docker/systemd multi-line output into one greppable log line.
one_line() { printf '%s' "$*" | tr '\n' ' ' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

# --------------------------------------------------------------- health file
HEALTH_STATE='starting'
HEALTH_CONTAINER_RUNNING='unknown'
HEALTH_CONTAINER_STATUS='unknown'
HEALTH_DOCKERD_ACTIVE='unknown'
HEALTH_DOCKER_READY='unknown'
HEALTH_FAILURES='0'
HEALTH_LAST_ACTION='none'
HEALTH_LAST_ERROR=''
HEALTH_EXIT_CODE=''
HEALTH_EXIT_REASON=''

write_health() {
  [ -n "$HEALTH_FILE" ] || return 0
  tmp="${HEALTH_FILE}.tmp.$$"
  {
    printf 'updated_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'supervisor_pid=%s\n' "$$"
    printf 'state=%s\n' "$HEALTH_STATE"
    printf 'container=%s\n' "$container"
    printf 'container_running=%s\n' "$HEALTH_CONTAINER_RUNNING"
    printf 'container_status=%s\n' "$HEALTH_CONTAINER_STATUS"
    printf 'dockerd_active=%s\n' "$HEALTH_DOCKERD_ACTIVE"
    printf 'docker_ready=%s\n' "$HEALTH_DOCKER_READY"
    printf 'consecutive_failures=%s\n' "$HEALTH_FAILURES"
    printf 'last_action=%s\n' "$HEALTH_LAST_ACTION"
    printf 'last_error=%s\n' "$HEALTH_LAST_ERROR"
    printf 'last_exit_code=%s\n' "$HEALTH_EXIT_CODE"
    printf 'last_exit_reason=%s\n' "$HEALTH_EXIT_REASON"
  } >"$tmp" 2>/dev/null || { rm -f "$tmp" 2>/dev/null || :; return 0; }
  mv -f "$tmp" "$HEALTH_FILE" 2>/dev/null || rm -f "$tmp" 2>/dev/null || :
  return 0
}

# Heartbeat: refresh the health file at most once per 15s. Called from every
# wait and every recovery iteration, so the file stays fresh precisely while
# things are going wrong - a health file that only updates on success is
# useless to an owner trying to work out why the node is down.
LAST_HEALTH_WRITE=0
heartbeat() {
  now=$(date +%s)
  if [ "$now" -ge $(( LAST_HEALTH_WRITE + 15 )) ]; then
    LAST_HEALTH_WRITE=$now
    write_health
  fi
  return 0
}

# Single place where this process returns. $1 exit code, $2 reason,
# $3 health state. Log wording follows the exit code, never the other way
# round: exit 0 always means an intended owner shutdown.
finish() {
  code=$1
  reason=$2
  HEALTH_STATE=$3
  HEALTH_EXIT_CODE=$code
  HEALTH_EXIT_REASON=$(one_line "$reason")
  write_health
  case "$code" in
    0) log INFO "supervisor exiting 0 (INTENDED owner shutdown): ${reason}" ;;
    2) log ERROR "supervisor exiting 2 (invalid invocation or unusable environment): ${reason}" ;;
    *) log ERROR "supervisor exiting ${code} (FAILURE, not an owner shutdown): ${reason}" ;;
  esac
  exit "$code"
}

on_signal() {
  trap '' TERM INT
  finish "$EXIT_OK" "received $1; owner asked for shutdown" "stopped-signal"
}

# ------------------------------------------------------------ argument parsing
usage() {
  cat <<EOF
Usage: worker-supervisor.sh [options]

  --container NAME        worker container (default: ${DEF_CONTAINER})
  --state-dir PATH        state directory (default: ${DEF_STATE_DIR})
  --interval SECONDS      steady-state poll interval (default: ${DEF_INTERVAL})
  --dockerd-attempts N    start attempts per recovery pass (default: ${DEF_DOCKERD_ATTEMPTS})
  --dockerd-sleep SECONDS delay between attempts (default: ${DEF_DOCKERD_SLEEP})
  --backoff-max SECONDS   retry backoff ceiling (default: ${DEF_BACKOFF_MAX})
  --max-failures N        consecutive failures before escalation (default: ${DEF_MAX_FAILURES})
  --missing-attempts N    grace before exit ${EXIT_NO_CONTAINER} (default: ${DEF_MISSING_ATTEMPTS})
  --docker-unit NAME      container-runtime unit (default: ${DEF_DOCKER_UNIT})
  -h, --help              this text

Exit codes: 0 intended owner shutdown, 2 invalid invocation/environment,
${EXIT_NO_DOCKERD} dockerd not active in the startup window, ${EXIT_NO_CONTAINER} worker
container absent/unstartable, ${EXIT_BAD_INTENT} unreadable owner intent,
${EXIT_INTENT_UNMET} owner intent could not be carried out.
EOF
}

usage_error() {
  printf 'error: %s\n\n' "$1" >&2
  usage >&2
  exit "$EXIT_USAGE"
}

is_positive_int() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
  esac
  [ "$1" -ge 1 ] 2>/dev/null
}

need_value() {
  # $1 = option as written, $2 = number of remaining argv entries
  if [ "$2" -lt 2 ]; then
    usage_error "$1 requires a value"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -c|--container)        need_value "$1" "$#"; container=$2; shift 2 ;;
    --container=*)         container=${1#*=}; shift ;;
    -s|--state-dir)        need_value "$1" "$#"; state_dir=$2; shift 2 ;;
    --state-dir=*)         state_dir=${1#*=}; shift ;;
    -i|--interval)         need_value "$1" "$#"; interval=$2; shift 2 ;;
    --interval=*)          interval=${1#*=}; shift ;;
    --dockerd-attempts)    need_value "$1" "$#"; dockerd_attempts=$2; shift 2 ;;
    --dockerd-attempts=*)  dockerd_attempts=${1#*=}; shift ;;
    --dockerd-sleep)       need_value "$1" "$#"; dockerd_sleep=$2; shift 2 ;;
    --dockerd-sleep=*)     dockerd_sleep=${1#*=}; shift ;;
    --backoff-max)         need_value "$1" "$#"; backoff_max=$2; shift 2 ;;
    --backoff-max=*)       backoff_max=${1#*=}; shift ;;
    --max-failures)        need_value "$1" "$#"; max_failures=$2; shift 2 ;;
    --max-failures=*)      max_failures=${1#*=}; shift ;;
    --missing-attempts)    need_value "$1" "$#"; missing_attempts=$2; shift 2 ;;
    --missing-attempts=*)  missing_attempts=${1#*=}; shift ;;
    --docker-unit)         need_value "$1" "$#"; docker_unit=$2; shift 2 ;;
    --docker-unit=*)       docker_unit=${1#*=}; shift ;;
    -h|--help)             usage; exit "$EXIT_OK" ;;
    --)                    shift; break ;;
    -*)                    usage_error "unknown option: $1" ;;
    *)                     usage_error "unexpected argument: $1" ;;
  esac
done

for pair in \
  "interval:$interval" \
  "dockerd-attempts:$dockerd_attempts" \
  "dockerd-sleep:$dockerd_sleep" \
  "backoff-max:$backoff_max" \
  "max-failures:$max_failures" \
  "missing-attempts:$missing_attempts"
do
  name=${pair%%:*}
  value=${pair#*:}
  is_positive_int "$value" || usage_error "--${name} must be a positive integer, got '${value}'"
done
case "$container" in
  '') usage_error "--container must not be empty" ;;
esac
case "$state_dir" in
  '') usage_error "--state-dir must not be empty" ;;
esac
case "$docker_unit" in
  '') usage_error "--docker-unit must not be empty" ;;
esac

INTENT_FILE="$state_dir/supervisor-intent"
HEALTH_FILE="$state_dir/supervisor-health"
LOG_FILE="$state_dir/supervisor.log"

# A supervisor that cannot read owner intent cannot be durably stopped by its
# owner, so an unusable state directory is fatal rather than a warning.
if ! mkdir -p "$state_dir" 2>/dev/null || [ ! -w "$state_dir" ]; then
  printf 'error: state directory %s is not usable (cannot create or write); owner intent would be unreadable\n' "$state_dir" >&2
  exit "$EXIT_USAGE"
fi

trap 'on_signal SIGTERM' TERM
trap 'on_signal SIGINT' INT

# --------------------------------------------------------------- docker state
have_systemctl() { command -v systemctl >/dev/null 2>&1; }

dockerd_active() {
  if have_systemctl; then
    systemctl is-active --quiet "$docker_unit"
  else
    # No systemd in this distro: the daemon is up iff the socket answers.
    docker info >/dev/null 2>&1
  fi
}

docker_ready() { docker info >/dev/null 2>&1; }

# $1 = number of consecutive failures (>=1). Doubles from the poll interval
# and stops at the ceiling, so a permanently broken runtime cannot hot-loop.
backoff_delay() {
  n=$1
  d=$interval
  while [ "$n" -gt 1 ] && [ "$d" -lt "$backoff_max" ]; do
    d=$(( d * 2 ))
    n=$(( n - 1 ))
  done
  [ "$d" -gt "$backoff_max" ] && d=$backoff_max
  printf '%s' "$d"
}

CONTAINER_ERR=''
CONTAINER_START_ERR=''
CONTAINER_STOP_ERR=''
DOCKERD_ERR=''

container_state() {
  # Sets HEALTH_CONTAINER_RUNNING / HEALTH_CONTAINER_STATUS, returns 1 when
  # the container cannot be inspected at all (absent, or daemon unreachable).
  CONTAINER_ERR=''
  out=$(docker container inspect --format '{{.State.Running}}|{{.State.Status}}' "$container" 2>&1)
  rc=$?
  if [ "$rc" -ne 0 ]; then
    HEALTH_CONTAINER_RUNNING='unknown'
    HEALTH_CONTAINER_STATUS='absent-or-unreachable'
    CONTAINER_ERR=$(one_line "$out")
    return 1
  fi
  HEALTH_CONTAINER_RUNNING=${out%%|*}
  HEALTH_CONTAINER_STATUS=${out#*|}
  return 0
}

container_start() {
  CONTAINER_START_ERR=''
  out=$(docker start "$container" 2>&1)
  rc=$?
  [ "$rc" -eq 0 ] && return 0
  CONTAINER_START_ERR=$(one_line "$out")
  return 1
}

container_stop() {
  CONTAINER_STOP_ERR=''
  out=$(docker stop "$container" 2>&1)
  rc=$?
  [ "$rc" -eq 0 ] && return 0
  CONTAINER_STOP_ERR=$(one_line "$out")
  return 1
}

# ------------------------------------------------------------- owner intent
# Returns the normalised intent on stdout: run|stop|pause|remove, or
# invalid:<verbatim>, or run when the file is absent/empty/unreadable.
read_intent() {
  if [ ! -f "$INTENT_FILE" ]; then
    printf 'run'
    return 0
  fi
  raw=$(tr -d '\r' <"$INTENT_FILE" 2>/dev/null | head -n 1 | tr -d ' \t' | tr 'A-Z' 'a-z')
  case "$raw" in
    ''|run)        printf 'run' ;;
    stop)          printf 'stop' ;;
    pause)         printf 'pause' ;;
    remove)        printf 'remove' ;;
    *)             printf 'invalid:%s' "${raw:-<empty-but-not-run>}" ;;
  esac
}

# Called at the top of every loop iteration and every sleep slice, so an owner
# stop takes effect within SLEEP_SLICE seconds even while the supervisor is
# inside a backoff. Returns 0 only for `run`; every other value returns through
# one of the documented exit codes.
apply_intent() {
  intent=$(read_intent)
  case "$intent" in
    run)
      return 0
      ;;
    stop)
      # The owner wants it down. Stop it once, then get out of the way.
      if container_state; then
        if [ "$HEALTH_CONTAINER_RUNNING" = 'true' ]; then
          if container_stop; then
            finish "$EXIT_OK" "owner intent=stop; container ${container} stopped" "stopped-intent-stop"
          fi
          finish "$EXIT_INTENT_UNMET" \
            "owner intent=stop but stopping ${container} failed: ${CONTAINER_STOP_ERR:-no error text}" \
            "stop-unmet"
        fi
        finish "$EXIT_OK" "owner intent=stop; container ${container} already ${HEALTH_CONTAINER_STATUS}" "stopped-intent-stop"
      fi
      # Inspect failed. An absent container already satisfies "stop"; only an
      # unreachable daemon leaves the intent unfulfilled.
      if docker_ready; then
        finish "$EXIT_OK" "owner intent=stop; container ${container} is absent, so there is nothing to stop" "stopped-intent-stop"
      fi
      finish "$EXIT_INTENT_UNMET" \
        "owner intent=stop but the docker daemon does not answer, so the stop could not be confirmed: ${CONTAINER_ERR:-no error text}" \
        "stop-unmet"
      ;;
    pause)
      finish "$EXIT_OK" "owner intent=pause; container left untouched, nothing restarted" "stopped-intent-pause"
      ;;
    remove)
      if container_state; then
        finish "$EXIT_OK" \
          "owner intent=remove; not recreating it (container ${container} is still ${HEALTH_CONTAINER_STATUS})" \
          "stopped-intent-remove"
      fi
      if docker_ready; then
        finish "$EXIT_OK" "owner intent=remove; container ${container} is absent; not recreating it" "stopped-intent-remove"
      fi
      finish "$EXIT_OK" \
        "owner intent=remove; not recreating it (${container} presence not verifiable: ${CONTAINER_ERR:-no error text})" \
        "stopped-intent-remove"
      ;;
    invalid:*)
      finish "$EXIT_BAD_INTENT" \
        "owner intent file ${INTENT_FILE} holds an unrecognised value '${intent#invalid:}'; allowed: run, stop, pause, remove. Nothing was started or restarted." \
        "fatal-bad-intent"
      ;;
  esac
  return 0
}

# ------------------------------------------------------------------- recovery
STARTUP_PHASE='true'
ESCALATED=''

reset_failures() {
  if [ "$HEALTH_FAILURES" != '0' ] || [ -n "$ESCALATED" ]; then
    log INFO "recovery succeeded after ${HEALTH_FAILURES} consecutive failure(s)"
  fi
  HEALTH_FAILURES='0'
  ESCALATED=''
  HEALTH_LAST_ERROR=''
}

escalate_once() {
  # $1 = topic. One loud line per outage, not one per poll.
  [ "$ESCALATED" = "$1" ] && return 0
  ESCALATED=$1
  log ERROR "ESCALATION: ${HEALTH_LAST_ERROR:-unknown error}; ${HEALTH_FAILURES} consecutive failures exceeded --max-failures=${max_failures}. Holding WSL2 open and retrying every ${backoff_max}s. If this persists, inspect ${HEALTH_FILE} and fix the runtime by hand."
}

# Sleeps in slices so owner intent is honoured promptly instead of up to a full
# poll interval or a full backoff later, and refreshes the health heartbeat
# while it waits. EVERY wait in this script goes through here, including
# backoff waits: a raw `sleep 120` would leave an owner-requested stop
# unrecognised for two minutes, which is the D2 defect in a different costume.
sleep_responsive() {
  total=$1
  slept=0
  while [ "$slept" -lt "$total" ]; do
    apply_intent
    heartbeat
    step=$SLEEP_SLICE
    remaining=$(( total - slept ))
    [ "$step" -gt "$remaining" ] && step=$remaining
    sleep "$step"
    slept=$(( slept + step ))
  done
}

wait_docker_ready() {
  n=0
  while :; do
    docker_ready && return 0
    n=$(( n + 1 ))
    [ "$n" -ge "$DOCKER_READY_ATTEMPTS" ] && return 1
    sleep_responsive 2
  done
}

# Recovers the container RUNTIME, not just the container. Returns 0 once
# dockerd is active and the daemon answers. Exits ${EXIT_NO_DOCKERD} if it
# cannot manage that inside the bounded startup window; in steady state it
# keeps retrying forever instead of exiting (see the header: exiting releases
# the WSL2 session and takes dockerd down with it).
ensure_dockerd() {
  if dockerd_active && docker_ready; then
    HEALTH_DOCKERD_ACTIVE='true'
    HEALTH_DOCKER_READY='true'
    reset_failures
    return 0
  fi

  if dockerd_active; then
    HEALTH_DOCKERD_ACTIVE='true'
    HEALTH_DOCKER_READY='false'
    log ERROR "dockerd unit ${docker_unit} is active but 'docker info' does not answer; treating the runtime as down"
  else
    HEALTH_DOCKERD_ACTIVE='false'
    HEALTH_DOCKER_READY='false'
    log ERROR "dockerd is not active; entering bounded recovery (${dockerd_attempts} attempts x ${dockerd_sleep}s)"
  fi
  HEALTH_STATE='recovering-dockerd'
  write_health

  attempt=1
  while :; do
    apply_intent
    heartbeat
    if dockerd_active; then
      HEALTH_DOCKERD_ACTIVE='true'
      if wait_docker_ready; then
        HEALTH_DOCKER_READY='true'
        HEALTH_LAST_ACTION="start dockerd ${docker_unit}"
        log INFO "dockerd active and answering (attempt ${attempt}); container runtime is back"
        reset_failures
        return 0
      fi
      HEALTH_DOCKER_READY='false'
      DOCKERD_ERR="unit ${docker_unit} is active but the docker daemon still does not answer 'docker info'"
    else
      HEALTH_DOCKERD_ACTIVE='false'
      DOCKERD_ERR='unit not active'
    fi

    HEALTH_FAILURES=$(( HEALTH_FAILURES + 1 ))
    HEALTH_LAST_ACTION="start dockerd ${docker_unit} (attempt ${attempt}/${dockerd_attempts})"
    if have_systemctl; then
      if start_out=$(systemctl start "$docker_unit" 2>&1); then
        log ERROR "attempt ${attempt}/${dockerd_attempts}: 'systemctl start ${docker_unit}' returned success but ${DOCKERD_ERR}"
      else
        DOCKERD_ERR=$(one_line "$start_out")
        log ERROR "attempt ${attempt}/${dockerd_attempts}: 'systemctl start ${docker_unit}' failed: ${DOCKERD_ERR:-no error text}"
      fi
    else
      DOCKERD_ERR="systemctl is unavailable in this distro, so ${docker_unit} cannot be started from here"
      log ERROR "attempt ${attempt}/${dockerd_attempts}: ${DOCKERD_ERR}"
    fi
    HEALTH_LAST_ERROR=$(one_line "$DOCKERD_ERR")
    write_health

    if [ "$attempt" -ge "$dockerd_attempts" ]; then
      if [ "$STARTUP_PHASE" = 'true' ]; then
        finish "$EXIT_NO_DOCKERD" \
          "dockerd not active after ${attempt} attempt(s) (bounded ${dockerd_attempts}x${dockerd_sleep}s window); last error: ${DOCKERD_ERR:-none}" \
          "fatal-no-dockerd"
      fi
      escalate_once dockerd
      delay=$backoff_max
      attempt=1
    else
      delay=$(backoff_delay "$attempt")
      attempt=$(( attempt + 1 ))
    fi
    HEALTH_LAST_ACTION="waiting ${delay}s before re-checking dockerd"
    log ERROR "retrying dockerd in ${delay}s (consecutive failures: ${HEALTH_FAILURES})"
    sleep_responsive "$delay"
  done
}

# Keeps the ONE worker container up. Exits ${EXIT_NO_CONTAINER} when it is
# absent or unstartable inside the grace window.
ensure_container() {
  HEALTH_STATE='recovering-container'
  missing=0

  while :; do
    apply_intent
    heartbeat

    if container_state; then
      missing=0
      HEALTH_LAST_ERROR=''
      if [ "$HEALTH_CONTAINER_RUNNING" = 'true' ]; then
        HEALTH_STATE='running'
        HEALTH_LAST_ACTION='none (container already running)'
        reset_failures
        [ "$STARTUP_PHASE" = 'true' ] && STARTUP_PHASE='false'
        write_health
        return 0
      fi
      log WARN "container ${container} is ${HEALTH_CONTAINER_STATUS}; starting it"
      if container_start; then
        log INFO "container ${container} start requested"
        HEALTH_LAST_ACTION="docker start ${container}"
        sleep_responsive 2
        if container_state && [ "$HEALTH_CONTAINER_RUNNING" = 'true' ]; then
          HEALTH_STATE='running'
          log INFO "container ${container} is running"
          reset_failures
          [ "$STARTUP_PHASE" = 'true' ] && STARTUP_PHASE='false'
          write_health
          return 0
        fi
        log WARN "'docker start ${container}' returned success but the container is ${HEALTH_CONTAINER_STATUS:-unknown}; re-checking"
      else
        HEALTH_FAILURES=$(( HEALTH_FAILURES + 1 ))
        HEALTH_LAST_ERROR="docker start ${container}: ${CONTAINER_START_ERR:-no error text}"
        log ERROR "docker start ${container} failed: ${CONTAINER_START_ERR:-no error text}"
        write_health
      fi
    else
      missing=$(( missing + 1 ))
      HEALTH_FAILURES=$(( HEALTH_FAILURES + 1 ))
      HEALTH_LAST_ERROR="container ${container} absent: ${CONTAINER_ERR:-no error text}"
      if [ "$missing" -ge "$missing_attempts" ]; then
        reason="container ${container} is absent after ${missing} inspect attempt(s): ${CONTAINER_ERR:-no error text}"
        finish "$EXIT_NO_CONTAINER" \
          "${reason}. It will not be recreated here: that needs the worker image and the node's runtime secrets. Set ${INTENT_FILE} to 'stop' or 'remove', or redeploy the container and let the task retry." \
          "fatal-no-container"
      fi
      log ERROR "attempt ${missing}/${missing_attempts}: container ${container} cannot be inspected: ${CONTAINER_ERR:-no error text}"
      write_health
    fi

    [ "$HEALTH_FAILURES" -ge "$max_failures" ] && escalate_once container
    delay=$(backoff_delay "$HEALTH_FAILURES")
    HEALTH_LAST_ACTION="retrying container in ${delay}s"
    log ERROR "retrying container ${container} in ${delay}s (consecutive failures: ${HEALTH_FAILURES})"
    sleep_responsive "$delay"
  done
}

# ---------------------------------------------------------------------- main
log INFO "supervisor starting: container=${container} state_dir=${state_dir} interval=${interval}s unit=${docker_unit} pid=$$"
log INFO "ownership: this process owns requirement A only (host/container runtime). Worker-process restart belongs to the container entrypoint; application reconnect belongs to the worker client."
log INFO "exit contract: 0 intended owner shutdown | 2 invalid invocation | 10 dockerd unavailable | 11 worker container absent/unstartable | 12 unreadable owner intent | 13 owner intent unmet"

intent_now=$(read_intent)
log INFO "owner intent (${INTENT_FILE}): ${intent_now}"
apply_intent

if ! have_systemctl; then
  log WARN "systemctl not found in this distro; dockerd is probed with 'docker info' and cannot be started from here"
fi

ensure_dockerd
log INFO "dockerd is active and answering"

ensure_container
log INFO "worker container ${container} is up; entering the steady-state supervision loop (this process now runs until owner intent, a signal, or a fatal condition)"

# Attached and resident from here. No exit on infrastructure failure: see the
# header. Every return path goes through finish() and the documented codes.
while :; do
  sleep_responsive "$interval"
  ensure_dockerd
  ensure_container
done
