#!/bin/bash
# Resident container-runtime supervisor for the OAI-2.0 fleet worker (node=p50).
#
# This script must be invoked as an ATTACHED, never-returning process:
#
#     wsl.exe -d Ubuntu-24.04 -u root -e bash /mnt/c/Users/P50/q-pipe/worker-supervisor.sh
#
# WSL2 tears the VM down once it has no attached session, and dockerd dies with
# the VM, so a detached launch is not enough. An earlier version of the launcher
# passed a multi-line script to `bash -c`; PowerShell split it on newlines, bash
# received only the first line and exited immediately, and the node dropped out
# of the fleet within a minute of every launch. Invoking a real file with a
# single argument avoids that entirely.
#
# Ownership split for master #241:
#   A. host / container runtime start  -> this script (attached, resident)
#   B. worker process restart          -> the container entrypoint's loop
#   C. application reconnect           -> the worker client registration + heartbeat
set -u

log() { printf '[%s] %s\n' "$(date -u '+%H:%M:%S')" "$*"; }

log "supervisor starting: ensuring dockerd"
for _ in $(seq 1 60); do
  if systemctl is-active --quiet docker; then
    break
  fi
  systemctl start docker >/dev/null 2>&1 || true
  sleep 2
done

if systemctl is-active --quiet docker; then
  log "dockerd active"
else
  log "WARNING: dockerd not active; continuing to retry"
fi

if docker start oai2-worker >/dev/null 2>&1; then
  log "container start requested"
else
  log "container start returned non-zero (may already be running)"
fi
docker ps --filter name=oai2-worker --format '{{.Names}} | {{.Status}}' || true

# Stay attached forever. While this process holds the WSL2 session, the VM
# stays up, dockerd stays up, and the container keeps its restart policy.
# If the container ever exits, bring it back so the worker re-registers with the
# existing pipeline. This is scoped strictly to the worker: it never starts,
# stops or reconfigures a coordinator, gateway or any pipeline service.
while true; do
  if ! docker ps --format '{{.Names}}' | grep -qx 'oai2-worker'; then
    log "container not running; starting"
    docker start oai2-worker >/dev/null 2>&1 || true
  fi
  sleep 30
done
