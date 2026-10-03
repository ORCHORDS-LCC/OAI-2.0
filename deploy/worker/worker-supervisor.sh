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
#    12   The owner-intent file EXISTS but is unusable, so owner state is
#         damaged: not a readable regular file, unreadable, empty (no non-blank
#         line), an unrecognised value, or a half-written one. Fail CLOSED:
#         nothing is started, stopped or restarted. This is a FAULT and is
#         deliberately distinct from a requested `stop`, which is exit 0. Only
#         an ABSENT sentinel defaults to run. See the resolution rule below.
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
# OWNER-INTENT SENTINEL (durable owner intent; the ONLY thing this script reads
# as permission to act)
#   Path:   /var/lib/worker/supervisor-intent   (dir overridable with
#           --state-dir / WORKER_STATE_DIR)
#
#   INTENT RESOLUTION RULE - this text is the specification, and read_intent()
#   implements it. Numbered so a test can name the rule it is checking.
#     1. ABSENT is the only case that means `run`. If the path does not exist -
#        a genuine first start - the intent is `run`. This is the ONLY default
#        in this script. `run` may also be written explicitly.
#     2. PRESENT-BUT-UNUSABLE fails closed. A path that exists but is not a
#        readable regular file (a directory, a socket, a dangling symlink) is
#        UNREADABLE -> exit 12.
#     3. PRESENT-BUT-EMPTY fails closed. Zero bytes, or no non-blank line
#        anywhere in the file, is a DAMAGED file, not permission -> exit 12.
#     4. Otherwise the FIRST NON-BLANK LINE is the value. Blank lines above it
#        are skipped. That line is normalised: CR removed, leading and
#        trailing spaces/tabs removed, ASCII letters lowercased.
#     5. The normalised line must then equal run, stop, pause or remove
#        EXACTLY. Any other token is UNRECOGNISED -> exit 12, including a
#        truncated prefix such as `sto`, which is what an interrupted write
#        leaves behind on disk. Interior spaces are NOT removed: `s top` is
#        unrecognised, not `stop`.
#
#   So a damaged owner-state file NEVER resolves to `run`. Absent and damaged
#   are different states with different outcomes, and only absence is benign.
#   A `stop` is a requested shutdown (exit 0); a damaged file is a FAULT (exit
#   12). The two are reported differently in the log and in the health file.
#
#   Values:
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
#   Clearing the sentinel, or writing `run`, restores automatic startup. The
#   sentinel is durable: it lives on disk, so it still says `stop` after the
#   supervisor, the task or the whole VM has been restarted. UNREADABLE, EMPTY
#   and UNRECOGNISED all fail closed with exit 12 and start nothing.
#
# ATOMIC UPDATE (recommended owner procedure) - the exact commands
#   The write matters as much as the read: the sentinel is durable owner state.
#   A plain `printf 'stop\n' > <file>` truncates FIRST, so a reader arriving in
#   that window sees a zero-byte or half-written file. Under rule 1-3 that now
#   fails closed (exit 12) - safe, but still an outage.
#
#   Instead write a temp file in the SAME directory and rename it over the
#   sentinel. rename(2) is atomic within a filesystem, so a concurrent reader
#   sees either the whole old value or the whole new one, never a mixture.
#   Same directory is not a style choice: a rename across filesystems is not
#   atomic, and /var/lib/worker and /tmp usually are not the same one.
#   write_intent_atomically() below does exactly this, and is exposed to the
#   owner so there is one supported command rather than a recipe to re-derive:
#
#     bash worker-supervisor.sh --set-intent stop     # run|stop|pause|remove
#     bash worker-supervisor.sh --set-intent pause
#     bash worker-supervisor.sh --set-intent remove
#     bash worker-supervisor.sh --set-intent run      # explicit
#     bash worker-supervisor.sh --set-intent clear    # remove the sentinel
#     bash worker-supervisor.sh --resolve-intent      # what would be honoured
#
#   Equivalent by hand, if an owner prefers not to use the flag:
#     printf 'stop\n' > /var/lib/worker/.supervisor-intent.tmp \
#       && mv -f /var/lib/worker/.supervisor-intent.tmp \
#               /var/lib/worker/supervisor-intent
#
#   The supervisor's own read is resilient to a concurrent rename by
#   construction: it opens the path ONCE, so the descriptor it reads through
#   names one inode for the whole read. A rename swaps the directory entry, not
#   that descriptor, so a read can return the entire old value or the entire
#   new value and never a blend. It does not read the file twice and stitch
#   results together, and it re-checks the path type after opening.
#   What this does NOT cover: a machine-level crash (power loss) during the
#   write. There is no fsync, so durability across that is out of scope.
#
# TEST SEAM: simulating an interrupted write
#   WORKER_SIMULATE_TORN_INTENT=1 makes read_intent() behave as if an
#   interrupted, half-completed owner write had been observed. It is TEST-ONLY
#   and off unless that variable is exactly 1, and with it on the supervisor
#   fails closed (exit 12, health state fatal-bad-intent-torn-write) exactly
#   as a genuinely half-written file does. It exists so the failure path is
#   executable in CI rather than asserted in prose. It never authorises
#   anything: with the seam on, no container is started.
#
# WHAT `pause` HERE DOES NOT DO - read this before reaching for cluster controls
#   This script can do exactly ONE kind of thing: stop itself from managing
#   anything. That is "suspend supervisor management" and nothing more. Every
#   value in the sentinel above is a statement about THIS process.
#
#   "Pause worker admissions" is a DIFFERENT control: the worker stays up and
#   keeps its registration and heartbeat, but must not be handed new jobs. This
#   script does not implement it, cannot implement it from here, and does not
#   hold the credential or endpoint to implement it. Admission is a
#   control-plane decision - see oai2/runtime/admission.py (AdmissionPolicy,
#   AdmissionQueue, AdmissionDecision) together with the pipeline's own
#   routing. It is owned there, not here. `pause` must not be described, in an
#   incident note or a runbook, as a pause of worker admissions: it is not
#   one, and the difference is the whole point of writing it down.
#
#   Plainly, for the end-to-end controls this script does NOT provide. A
#   supervisor exit is NOT a cluster pause, drain, revoke or removal:
#     - It releases THIS process and the WSL2 session that holds dockerd up. On
#       WSL2 the runtime goes down with the session; it does not "settle".
#     - It does not stop the worker from reconnecting or re-registering. That
#       is requirement C and belongs to the worker client (see WHAT THIS
#       PROCESS IS).
#     - It does not hold, cancel or fail a job. Queued and running jobs belong
#       to the pipeline.
#     - It does not cordon, drain or evict node p50, and it deletes nothing.
#     - It does not report node health upstream. It writes one local health
#       file and nothing else.
#     - It does not stop the scheduled task from launching it again; that is
#       the task's own configuration.
#   Status of those controls, from this script's point of view, is BLOCKED /
#   NOT RUN: not attempted here, no evidence either way. Nothing in this file
#   may be cited as evidence that they work, and none of them are covered by
#   the exit-code contract above - that contract describes THIS process only.
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
# DEFECTS FIXED BY THE PREVIOUS REWRITE (issue #251; all observed live, not
# hypothetical). Listed for history; the D numbers below are that revision's and
# are unrelated to the D1-D4 owner-state list that follows.
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
# OWNER-STATE DEFECTS FIXED IN THIS REVISION (the owner-intent sentinel; these
# D1-D4 are a different list from the historical one above)
#   D1  Absent was conflated with damaged. The old read_intent() did
#       `[ ! -f ] -> run` and then folded an unreadable read into the same
#       `''|run` case as a real `run`, so an owner-state file that was empty,
#       zero bytes after a failed write, or unreadable all resolved to "run":
#       permission manufactured out of a broken file. Absence is a legitimate
#       first-start state; damage is not. Resolution now distinguishes the two,
#       and only absence defaults to run (rules 1-3 above). Unreadable, empty
#       and unrecognised each fail closed with exit 12 and their own health
#       state, so a fault is never logged as a request to stop.
#   D2  The parser and this header disagreed. The header promised "first
#       non-blank line"; the code used `head -n 1`, i.e. the first LINE. A file
#       whose first line was blank and whose second line was `stop` resolved to
#       `run` - the exact opposite of the documented rule and of the owner's
#       intent, and it did so by starting to supervise. One rule is now
#       documented (rule 4) and implemented by the same code path that the tests
#       and --resolve-intent exercise. Normalisation also stopped deleting
#       INTERIOR whitespace, so `s top` is no longer accepted as `stop`.
#   D3  There was no atomic update path, only a truncating `>` recipe, so a
#       reader could observe a half-written file. write_intent_atomically()
#       (temp file in the same directory, then rename) is now implemented and
#       exposed as --set-intent, and the exact owner commands are documented
#       above. The supervisor's read is single-open, so a concurrent rename
#       yields a whole value and never a torn one.
#   D4  An interruption mid-write could not be tested at all, so "fails closed
#       on a partial value" was prose rather than a check. The exact-match rule
#       in rule 5 means a truncated value such as `sto` is unrecognised and
#       fails closed, and WORKER_SIMULATE_TORN_INTENT=1 makes the in-flight
#       case executable in CI. Both paths are exercised, not asserted.
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
#   --set-intent VALUE         write the owner sentinel atomically and exit
#                              (run|stop|pause|remove|clear); never supervises
#   --resolve-intent           print the intent that would be honoured and exit
#                              (0 if run/stop/pause/remove, 12 if unusable)
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

# Owner-side modes. Empty means "supervise", which is the whole point of
# the script; these two exit immediately and never touch a container.
set_intent=''
resolve_intent=''

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
  --set-intent VALUE      write the owner sentinel atomically and exit; never
                          supervises (run|stop|pause|remove|clear)
  --resolve-intent        print the intent that would be honoured and exit
  -h, --help              this text

Exit codes: 0 intended owner shutdown, 2 invalid invocation/environment,
${EXIT_NO_DOCKERD} dockerd not active in the startup window, ${EXIT_NO_CONTAINER} worker
container absent/unstartable, ${EXIT_BAD_INTENT} owner intent file present but
unreadable, empty or unrecognised, ${EXIT_INTENT_UNMET} owner intent could not be
carried out.

Owner intent: only an ABSENT sentinel defaults to run. A present-but-damaged
sentinel fails closed with ${EXIT_BAD_INTENT} and starts nothing. See the header
of this script for the full resolution rule, the atomic write commands
(--set-intent) and for what this script does NOT do (it is not a cluster pause,
drain, revoke or removal).
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
    --set-intent)          need_value "$1" "$#"; set_intent=$2; shift 2 ;;
    --set-intent=*)        set_intent=${1#*=}; shift ;;
    --resolve-intent)      resolve_intent=1; shift ;;
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
# Make an arbitrary owner-supplied byte sequence safe to put in one log line
# and in the key=value health file: printable only, single line, bounded. The
# old code interpolated the raw value straight into both.
sanitise() {
  printf '%s' "$1" | tr -cd '[:print:]' | tr -s ' \t' ' ' | cut -c1-80
}

# The single implementation of the rule documented in the header. Sets two
# globals rather than printing, so the explanation survives into the log and
# the health file instead of being lost in a subshell:
#   INTENT     run | stop | pause | remove | bad:<kind>:<detail>
#   INTENT_WHY one short line saying how that was arrived at
# Only `run` is a benign default, and only for an absent sentinel.
INTENT=''
INTENT_WHY=''

read_intent() {
  INTENT=''
  INTENT_WHY=''

  # Test seam first, so it wins over whatever is actually on disk. Documented
  # in the header: it can only ever make the supervisor fail closed.
  if [ "${WORKER_SIMULATE_TORN_INTENT:-0}" = '1' ]; then
    INTENT_WHY='test seam WORKER_SIMULATE_TORN_INTENT=1: simulated interrupted owner write'
    INTENT='bad:torn:an owner write was interrupted part way through (simulated), so the value on disk may be half of a real intent'
    return 0
  fi

  # Rule 1: genuine absence is the ONLY default. A dangling symlink is not
  # absence - the owner pointed at something that is not there - so it is
  # caught by the -L test below instead of being read as "first start".
  if [ ! -e "$INTENT_FILE" ] && [ ! -L "$INTENT_FILE" ]; then
    INTENT_WHY='sentinel absent (first start), the one documented default'
    INTENT='run'
    return 0
  fi

  # Rule 2: present, but not a readable regular file.
  if [ ! -f "$INTENT_FILE" ]; then
    INTENT_WHY='sentinel is present but is not a regular file'
    if [ -L "$INTENT_FILE" ]; then
      INTENT="bad:unreadable:${INTENT_FILE} is a symlink whose target does not exist (dangling); that is a damaged owner-state file, not a first start"
    elif [ -d "$INTENT_FILE" ]; then
      INTENT="bad:unreadable:${INTENT_FILE} is a directory, not an intent file"
    else
      INTENT="bad:unreadable:${INTENT_FILE} exists but is not a regular file"
    fi
    return 0
  fi
  if [ ! -r "$INTENT_FILE" ]; then
    INTENT_WHY='sentinel is present but not readable by this uid'
    INTENT="bad:unreadable:${INTENT_FILE} is not readable by uid $(id -u 2>/dev/null || printf '?')"
    return 0
  fi

  # One open, one inode. A concurrent rename swaps the directory entry, not
  # this descriptor, so the value read here is always a WHOLE file: either the
  # complete old intent or the complete new one. There is no second read to
  # stitch against the first, which is where a torn value would come from.
  if ! raw=$(cat -- "$INTENT_FILE" 2>/dev/null); then
    INTENT_WHY='sentinel is present but could not be read'
    INTENT="bad:unreadable:${INTENT_FILE} could not be opened or read"
    return 0
  fi

  # Rule 4: the FIRST NON-BLANK LINE, not the first line. Blank lines above it
  # are skipped rather than being taken as the value, so a leading empty line
  # can no longer turn a `stop` into a `run`. `grep -m1 -v` prints nothing (and
  # exits 1) when there is no non-blank line at all; that is rule 3 below, and
  # it is deliberately not an error here.
  raw=$(printf '%s' "$raw" | tr -d '\r' | grep -m1 -v '^[[:space:]]*$' 2>/dev/null)

  # Rule 4 normalisation: CR already gone; strip SURROUNDING whitespace only
  # (the old `tr -d ' \t'` deleted interior spaces too, so `s top` was
  # accepted as `stop`); then ASCII lowercase.
  raw=$(printf '%s' "$raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | tr 'A-Z' 'a-z')

  case "$raw" in
    '')
      # Rule 3: the file is there but says nothing. Damaged, not permission.
      INTENT_WHY='sentinel is present but holds no non-blank line'
      INTENT="bad:empty:${INTENT_FILE} is present but empty (zero bytes, or nothing but blank lines); a damaged owner-state file is never treated as permission to run"
      ;;
    run|stop|pause|remove)
      INTENT_WHY="sentinel present, first non-blank line normalised to '${raw}'"
      INTENT=$raw
      ;;
    *)
      # Rule 5: exact match only. A truncated prefix such as `sto` - exactly
      # what an interrupted write leaves on disk - lands here and fails closed.
      INTENT_WHY="sentinel present, first non-blank line normalised to an unrecognised token"
      INTENT="bad:unrecognised:${INTENT_FILE} holds the unrecognised value '$(sanitise "$raw")'; allowed values are exactly run, stop, pause, remove"
      ;;
  esac
  return 0
}

# Write owner intent atomically: a temp file in the SAME directory, then a
# rename over the sentinel. rename(2) is atomic within a filesystem, so a
# reader sees either the whole old value or the whole new one. Same directory
# is required for that, not stylistic - a cross-filesystem rename is not atomic.
# $1 = run|stop|pause|remove, or 'clear' to remove the sentinel.
# Returns 0 on success, 1 on failure. Never starts or stops anything itself.
write_intent_atomically() {
  wia_value=$1
  case "$wia_value" in
    clear)
      rm -f "$INTENT_FILE" 2>/dev/null || return 1
      return 0
      ;;
    run|stop|pause|remove) ;;
    *) return 1 ;;
  esac
  mkdir -p "$state_dir" 2>/dev/null || return 1
  [ -w "$state_dir" ] || return 1
  wia_tmp="$state_dir/.supervisor-intent.tmp.$$"
  if ! printf '%s\n' "$wia_value" >"$wia_tmp" 2>/dev/null; then
    rm -f "$wia_tmp" 2>/dev/null || :
    return 1
  fi
  chmod 0644 "$wia_tmp" 2>/dev/null || :
  # The rename is the commit point. A temp file left behind by a killed run is
  # harmless: the supervisor only ever reads supervisor-intent.
  mv -f "$wia_tmp" "$INTENT_FILE" 2>/dev/null || {
    rm -f "$wia_tmp" 2>/dev/null || :
    return 1
  }
  return 0
}

# ------------------------------------------------- owner-side modes (no docker)
# These two run the SAME read_intent / write_intent_atomically code the
# supervisor uses, so what the owner is told is exactly what the supervisor
# will do. They are placed after those definitions and before the traps, the
# log and any docker call: neither mode can start, stop or inspect a container.
if [ -n "$set_intent" ]; then
  case "$set_intent" in
    run|stop|pause|remove|clear) ;;
    *) usage_error "--set-intent must be one of run, stop, pause, remove, clear; got '${set_intent}'" ;;
  esac
  if ! write_intent_atomically "$set_intent"; then
    printf 'error: could not write owner intent %s atomically (state dir %s must be writable)\n' "$INTENT_FILE" "$state_dir" >&2
    exit "$EXIT_USAGE"
  fi
  if [ "$set_intent" = 'clear' ]; then
    printf 'owner intent cleared: %s removed; an absent sentinel is the one case that means run\n' "$INTENT_FILE"
  else
    read_intent
    printf 'owner intent set: %s now resolves to %s (%s)\n' "$INTENT_FILE" "$INTENT" "$INTENT_WHY"
  fi
  exit "$EXIT_OK"
fi

if [ -n "$resolve_intent" ]; then
  read_intent
  printf '%s\n' "$INTENT"
  printf '(%s)\n' "$INTENT_WHY"
  case "$INTENT" in
    bad:*) exit "$EXIT_BAD_INTENT" ;;
    *)     exit "$EXIT_OK" ;;
  esac
fi

trap 'on_signal SIGTERM' TERM
trap 'on_signal SIGINT' INT

# Called at the top of every loop iteration and every sleep slice, so an owner
# stop takes effect within SLEEP_SLICE seconds even while the supervisor is
# inside a backoff. Returns 0 only for `run`; every other value returns through
# one of the documented exit codes.
apply_intent() {
  read_intent
  intent=$INTENT
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
    bad:*)
      # A FAULT in owner state, not a request to stop. Exit 12, never 0, and a
      # health state distinct from every intentional shutdown above, so the log
      # and the health file say which of the two happened. Nothing is started,
      # stopped or restarted on this path: it is reached before any docker call.
      bad_rest=${intent#bad:}
      bad_kind=${bad_rest%%:*}
      bad_detail=${bad_rest#*:}
      case "$bad_kind" in
        torn)       bad_state='fatal-bad-intent-torn-write' ;;
        unreadable) bad_state='fatal-bad-intent-unreadable' ;;
        empty)      bad_state='fatal-bad-intent-empty' ;;
        *)          bad_state='fatal-bad-intent-unrecognised' ;;
      esac
      finish "$EXIT_BAD_INTENT" \
        "owner intent FAULT, not a shutdown: ${bad_detail} (${INTENT_WHY:-no further detail}). An absent sentinel is the only thing that means run; a present but damaged one never authorises anything. Nothing was started, stopped or restarted. Repair or remove the sentinel, then let the task retry. Note this is NOT a cluster pause, drain, revoke or removal - see the header." \
        "$bad_state"
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
log INFO "exit contract: 0 intended owner shutdown | 2 invalid invocation | 10 dockerd unavailable | 11 worker container absent/unstartable | 12 owner intent present but unreadable/empty/unrecognised | 13 owner intent unmet"
log INFO "scope limit: this script only suspends ITS OWN supervision. It is not a cluster pause, drain, revoke or removal; worker admissions are owned by the control plane (oai2/runtime/admission.py)."

read_intent
log INFO "owner intent (${INTENT_FILE}): ${INTENT} -- ${INTENT_WHY}"
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
