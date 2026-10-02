#!/bin/sh
# OAI-2.0 fleet worker entrypoint (worker-only image).
#
# Responsibilities are deliberately split (master #241, #251, #256):
#   A. host / container runtime start   -> this container being started
#   B. worker-process restart           -> this script's supervised loop
#   C. application-level reconnect      -> the worker's own outbound registration
#                                            and heartbeat against the existing pipeline
#
# A container restart policy alone is NOT network reconnection, and a heartbeat is
# NOT job success: READY is only real after an assigned lease has executed.
#
# Secrets are never baked in. They arrive as environment variables from the
# owner's secret store at run time:
#   QPIPE_CLUSTER_URL     existing pipeline control-plane address (required)
#   QPIPE_CLUSTER_TOKEN   scoped node credential (required)
#   QPIPE_NODE_ID         scoped node identity (required, one per node)
set -u

: "${QPIPE_CLUSTER_URL:?QPIPE_CLUSTER_URL is required}"
: "${QPIPE_CLUSTER_TOKEN:?QPIPE_CLUSTER_TOKEN is required}"
: "${QPIPE_NODE_ID:?QPIPE_NODE_ID is required}"

WORK_ROOT="${QPIPE_WORK_ROOT:-/var/lib/worker/root}"
PROFILES="${QPIPE_PROFILES:-/var/lib/worker/profiles/profiles.json}"
HEARTBEAT="${QPIPE_HEARTBEAT:-10}"
POLL="${QPIPE_POLL:-2}"
LEASE_SECONDS="${QPIPE_LEASE_SECONDS:-600}"
MAX_BACKOFF="${QPIPE_MAX_BACKOFF:-60}"

mkdir -p "$WORK_ROOT" 2>/dev/null || true

log() { printf '%s [worker] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }

log "node=${QPIPE_NODE_ID} cluster=${QPIPE_CLUSTER_URL} work_root=${WORK_ROOT} lease=${LEASE_SECONDS}s"
log "image carries the worker only; coordinator/gateway/lease authority stay in the existing pipeline"

backoff="$POLL"
while true; do
    started=$(date +%s)
    # The worker client runs as a child of this supervisor. We never start, and
    # never substitute, any coordinator or gateway.
    python3 -c 'from qpipe.cli import main; main()' worker verify \
        --url "$QPIPE_CLUSTER_URL" \
        --node-id "$QPIPE_NODE_ID" \
        --commands-file "$PROFILES" \
        --work-root "$WORK_ROOT" \
        --heartbeat "$HEARTBEAT" \
        --poll "$POLL" \
        --lease-seconds "$LEASE_SECONDS"
    code=$?
    ran=$(( $(date +%s) - started ))

    # A crash or a lost pipeline must not become a hot restart loop.
    if [ "$ran" -gt 300 ]; then
        backoff="$POLL"
    else
        backoff=$(( backoff * 2 ))
        if [ "$backoff" -gt "$MAX_BACKOFF" ]; then
            backoff="$MAX_BACKOFF"
        fi
    fi
    log "worker exited code=${code} after ${ran}s; reconnecting in ${backoff}s"
    sleep "$backoff"
done
