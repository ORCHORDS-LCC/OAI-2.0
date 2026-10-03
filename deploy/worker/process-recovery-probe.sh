#!/bin/bash
# Worker-process recovery probe without procps (the image has no ps/pkill).
# Scoped strictly to the worker container.
set -u
log() { printf '[%s] %s\n' "$(date -u '+%H:%M:%S')" "$*"; }

log "=== worker processes via /proc ==="
docker exec oai2-worker python3 -c '
import os
for p in sorted(os.listdir("/proc")):
    if not p.isdigit():
        continue
    try:
        cmd = open(f"/proc/{p}/cmdline","rb").read().replace(b"\0",b" ").decode(errors="replace").strip()
    except Exception:
        continue
    if "qpipe" in cmd or "worker verify" in cmd:
        print(f"  pid={p}  {cmd[:110]}")
' 2>&1

log "=== kill the worker PROCESS only (container + entrypoint stay up) ==="
docker exec oai2-worker python3 -c '
import os, signal
killed=[]
for p in sorted(os.listdir("/proc")):
    if not p.isdigit():
        continue
    try:
        cmd = open(f"/proc/{p}/cmdline","rb").read().replace(b"\0",b" ").decode(errors="replace")
    except Exception:
        continue
    if int(p) == os.getpid():
        continue
    if "worker" in cmd and "verify" in cmd:
        os.kill(int(p), signal.SIGKILL); killed.append(p)
print("  killed pids:", killed or "NONE")
' 2>&1

log "waiting 25s for the entrypoint supervised loop to respawn it"
sleep 25

log "=== worker processes after kill (respawn proves requirement B) ==="
docker exec oai2-worker python3 -c '
import os
found=0
for p in sorted(os.listdir("/proc")):
    if not p.isdigit():
        continue
    try:
        cmd = open(f"/proc/{p}/cmdline","rb").read().replace(b"\0",b" ").decode(errors="replace").strip()
    except Exception:
        continue
    if "worker" in cmd and "verify" in cmd:
        found+=1
        print(f"  pid={p}  {cmd[:110]}")
print("  count:", found)
' 2>&1
docker inspect oai2-worker --format 'container_running={{.State.Running}} restarts={{.RestartCount}}'
log "=== recent log ==="
docker logs --tail 4 oai2-worker 2>&1
