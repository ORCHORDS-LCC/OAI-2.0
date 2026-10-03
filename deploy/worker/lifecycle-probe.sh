#!/bin/bash
# Worker-scoped lifecycle recovery probe (master #241 requirements A/B/C).
# Scoped strictly to the worker container: it never touches the host OS, never
# reboots, and never touches a coordinator, gateway or pipeline service.
set -u
log() { printf '[%s] %s\n' "$(date -u '+%H:%M:%S')" "$*"; }

log "=== BEFORE ==="
docker inspect oai2-worker --format 'running={{.State.Running}} started={{.State.StartedAt}} restarts={{.RestartCount}}'
echo "worker process in container:"
docker exec oai2-worker sh -c "ps -eo pid,args | grep -E 'qpipe|worker' | grep -v grep" 2>&1 | head -5

log "=== TEST B: kill the worker PROCESS inside the container (entrypoint must respawn it) ==="
# Kill only the worker python process; the container and entrypoint stay up.
docker exec oai2-worker sh -c "pkill -f 'qpipe.cli' || pkill -f 'worker verify'" 2>&1 || true
sleep 20
echo "worker process after kill:"
docker exec oai2-worker sh -c "ps -eo pid,args | grep -E 'qpipe|worker' | grep -v grep" 2>&1 | head -5
docker inspect oai2-worker --format 'container_still_running={{.State.Running}} restarts={{.RestartCount}}'

log "=== TEST A: restart the CONTAINER (docker restart policy / entrypoint must recover) ==="
docker restart oai2-worker >/dev/null 2>&1
log "restart issued; waiting for the worker to re-register"
sleep 35
docker inspect oai2-worker --format 'after_restart running={{.State.Running}} started={{.State.StartedAt}} restarts={{.RestartCount}} exit={{.State.ExitCode}}'
echo "worker process after container restart:"
docker exec oai2-worker sh -c "ps -eo pid,args | grep -E 'qpipe|worker' | grep -v grep" 2>&1 | head -5
echo "--- last worker log lines (proves reconnect/registration) ---"
docker logs --tail 8 oai2-worker 2>&1
