# Host/container-runtime startup owner (master #241 requirement A).
#
# WSL2 and dockerd do NOT start on Windows boot by default, so the container's
# --restart policy alone is not enough: something must bring the runtime up AND
# keep it up. This task is the single lifecycle owner for "host -> container
# runtime".
#   A. host / container runtime start   -> this task + worker-supervisor.sh
#   B. worker process restart           -> the container entrypoint's own loop
#   C. application-level reconnect      -> the worker's registration + heartbeat
#
# Two earlier versions of this launcher failed, both in the same way - the node
# dropped out of the fleet within about a minute of every launch:
#
#   1. It started dockerd, printed container status, and exited. WSL2 then had
#      no attached session, shut the VM down, and took dockerd with it. A
#      container restart policy cannot help, because the policy lives in
#      dockerd, and dockerd is what died.
#   2. It kept going but passed the supervision loop to `bash -c` as a
#      multi-line PowerShell here-string. PowerShell splits that on newlines, so
#      bash received only the first line and exited 0 immediately.
#
# The fix is a single attached `wsl.exe ... bash <file>` invocation. Passing a
# real file as one argument removes the quoting problem, and the process never
# returns, which is what keeps the WSL2 VM - and therefore dockerd and the
# worker - alive.
$ErrorActionPreference = 'Continue'

$wsl = 'C:\Windows\System32\wsl.exe'
$supervisor = '/mnt/c/Users/P50/q-pipe/worker-supervisor.sh'

Write-Output ("[{0}] starting resident container-runtime supervisor for node={1}" -f (Get-Date -Format 'HH:mm:ss'), $env:OAI2_NODE)

# Attached and blocking. Returns only if the supervisor itself dies.
& $wsl -d Ubuntu-24.04 -u root -e bash $supervisor
