# ===========================================================================
# start-worker.ps1 - host/container-runtime startup owner (master #241
# requirement A). Windows side of the OAI-2.0 fleet worker (node p50).
# Installed and run by the OAI2WorkerRuntime scheduled task; see
# install-worker-task.ps1, which registers that task reproducibly.
#
# WHAT THIS PROCESS IS
#   WSL2 and dockerd do NOT start on Windows boot by default, so the
#   container's --restart policy alone is not enough: something must bring the
#   runtime up AND keep it up. This launcher plus worker-supervisor.sh is the
#   single lifecycle owner for that, and for nothing else:
#
#     A. host / container runtime start  -> this task + worker-supervisor.sh
#     B. worker-process restart          -> the container entrypoint's loop
#     C. application-level reconnect     -> the worker client's registration
#                                           and heartbeat against the pipeline
#
#   It hands the supervisor to WSL as ONE attached, blocking invocation. That
#   attachment is the point: WSL2 tears the VM down once no session is
#   attached, and a container restart policy cannot help, because the policy
#   lives in dockerd and dockerd is what died with the VM.
#
#   It invokes a real FILE path, never `bash -c` with a multi-line script. An
#   earlier launcher passed the script body to `bash -c` as a PowerShell
#   here-string; PowerShell split it on newlines, bash received only the first
#   line, exited immediately, and the node dropped out of the fleet within a
#   minute of every launch.
#
# SCOPE
#   Starts nothing except the one WSL distribution that runs the worker
#   supervisor. It never touches WSL configuration, never starts, stops or
#   reconfigures a coordinator, gateway, model server or any other pipeline
#   service, and never passes a credential to anything.
#
# DEFECTS THIS REWRITE FIXES (issue #251; all observed live)
#   D7  The old launcher ran wsl/bash and then fell off the end, so PowerShell
#       exited 0 no matter what happened. Task Scheduler therefore recorded
#       SUCCESS for a failed WSL launch, a missing supervisor file and a bash
#       error alike - a dead worker looked like a healthy task. This launcher
#       checks $LASTEXITCODE and exits non-zero, so a real failure is recorded
#       as a real failure and the task's RestartCount/RestartInterval applies.
#   D8  An intended owner shutdown and an unexpected supervisor exit were
#       indistinguishable, both being 0. The supervisor now has a documented
#       exit contract and this launcher passes those codes through, so exit 0
#       means exactly "the owner asked for this" and every other code means
#       "something broke". Task Scheduler only auto-restarts a task that
#       FAILED, so this is also what stops an owner-requested stop from being
#       immediately undone by the task's own restart policy.
#   D9  C:\Users\P50\q-pipe\worker-supervisor.sh and the task identity were
#       hardcoded, which blocked any other supported account or install path.
#       Every one of them is now a parameter with the deployed value as the
#       default. There is deliberately no -Password parameter and no secret of
#       any kind in this file; S4U registration (see install-worker-task.ps1)
#       stores no credential, and WSL cannot run as SYSTEM anyway
#       (Wsl/WSL_E_LOCAL_SYSTEM_NOT_SUPPORTED).
#
# EXIT CODES (Task Scheduler records these; non-zero = task failure)
#     0   intended owner shutdown (supervisor exit 0)
#     2   invalid invocation of this launcher
#    10   pass-through: dockerd unavailable in the bounded startup window
#    11   pass-through: worker container absent or unstartable
#    12   pass-through: owner-intent file holds an unrecognised value
#    13   pass-through: owner intent could not be carried out
#    20   wsl.exe not found at the configured path
#    21   the WSL distribution could not be entered, or bash / the supervisor
#         file was not found inside it (wsl exit 127)
#    22   the supervisor file was not found on the Windows side
#    23   wsl.exe returned no exit code at all (unexpected; treated as failure)
#
#   Any other non-zero code from the supervisor is passed through unchanged,
#   including 137 if the supervisor process was killed outright.
#
# USAGE
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File start-worker.ps1
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File start-worker.ps1 `
#       -ScriptDirectory 'C:\Users\P50\q-pipe' -ContainerName oai2-worker
#
#   Output note: Task Scheduler does not keep this script's stdout. The durable
#   operator-facing artifacts are, inside WSL,
#   /var/lib/worker/supervisor-health (heartbeat) and /var/lib/worker/supervisor.log.
# ===========================================================================
[CmdletBinding()]
param(
    # WSL side.
    [string] $WslDistro   = 'Ubuntu-24.04',
    [string] $WslUser     = 'root',
    [string] $WslPath     = "$env:SystemRoot\System32\wsl.exe",

    # Where the supervisor file lives ON WINDOWS. Converted to a WSL path
    # unless -SupervisorPath is given explicitly.
    [string] $ScriptDirectory = 'C:\Users\P50\q-pipe',
    [string] $ScriptName      = 'worker-supervisor.sh',
    [string] $SupervisorPath  = '',

    # The ONE container this fleet node runs. Passed straight through to the
    # supervisor's --container option.
    [string] $ContainerName = 'oai2-worker',

    # Purely for the operator-facing log line. Never a secret.
    [string] $Node = $env:OAI2_NODE
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Continue'

function Write-Log {
    param(
        [Parameter(Mandatory = $true)][string] $Level,
        [Parameter(Mandatory = $true)][string] $Message
    )
    Write-Output ('[{0}] {1,-5} {2}' -f (Get-Date -Format 'HH:mm:ss'), $Level, $Message)
}

function Stop-With {
    param(
        [Parameter(Mandatory = $true)][int]    $Code,
        [Parameter(Mandatory = $true)][string] $Message
    )
    if ($Code -eq 0) { Write-Log 'INFO' $Message } else { Write-Log 'ERROR' $Message }
    Write-Log 'INFO' ("launcher exiting {0}" -f $Code)
    exit $Code
}

# Drive-letter paths map into WSL as /mnt/<drive>/... . Done here rather than
# by shelling out to wslpath so the mapping cannot depend on wsl's argument
# quoting, and so a failure is a clear message instead of a wrong path.
function ConvertTo-WslPath {
    param([Parameter(Mandatory = $true)][string] $WindowsPath)

    if ($WindowsPath -notmatch '^([A-Za-z]):[\\/](.*)$') {
        throw "Cannot derive a WSL path from '$WindowsPath'. Pass -SupervisorPath explicitly."
    }
    $drive = $Matches[1].ToLowerInvariant()
    $rest  = ($Matches[2] -replace '\\', '/')
    return ('/mnt/{0}/{1}' -f $drive, $rest)
}

# ------------------------------------------------------------- validate input
if ([string]::IsNullOrWhiteSpace($ContainerName)) {
    Stop-With -Code 2 -Message '-ContainerName must not be empty.'
}
if ([string]::IsNullOrWhiteSpace($WslDistro)) {
    Stop-With -Code 2 -Message '-WslDistro must not be empty.'
}
if ([string]::IsNullOrWhiteSpace($WslUser)) {
    Stop-With -Code 2 -Message '-WslUser must not be empty.'
}

if ([string]::IsNullOrWhiteSpace($SupervisorPath)) {
    if ([string]::IsNullOrWhiteSpace($ScriptDirectory)) {
        Stop-With -Code 2 -Message '-ScriptDirectory must not be empty.'
    }
    if ([string]::IsNullOrWhiteSpace($ScriptName)) {
        Stop-With -Code 2 -Message '-ScriptName must not be empty.'
    }
    $windowsSupervisor = Join-Path -Path $ScriptDirectory -ChildPath $ScriptName
} else {
    $windowsSupervisor = $SupervisorPath
}

try {
    $resolvedSupervisor = ConvertTo-WslPath -WindowsPath $windowsSupervisor
} catch {
    Stop-With -Code 2 -Message $_.Exception.Message
}

# -------------------------------------------------------------- pre-flight
Write-Log 'INFO' ("starting resident container-runtime supervisor: node={0} distro={1} user={2} container={3}" -f `
    $(if ($Node) { $Node } else { '<unset>' }), $WslDistro, $WslUser, $ContainerName)
Write-Log 'INFO' ("supervisor file: {0}  ->  {1}" -f $windowsSupervisor, $resolvedSupervisor)

# D7: a missing supervisor file must fail the task, not silently succeed.
if (-not (Test-Path -LiteralPath $windowsSupervisor -PathType Leaf)) {
    Stop-With -Code 22 -Message "supervisor file not found: $windowsSupervisor"
}
# The supervisor needs to be executable by bash; it only has to be readable by
# bash, but a CRLF file would break it, so check the first line's shebang too.
$firstLine = (Get-Content -LiteralPath $windowsSupervisor -TotalCount 1 -ErrorAction SilentlyContinue)
if ($firstLine -notmatch '^#!') {
    Stop-With -Code 22 -Message "supervisor file does not start with a shebang: $windowsSupervisor"
}

if (-not (Test-Path -LiteralPath $WslPath -PathType Leaf)) {
    Stop-With -Code 20 -Message "wsl.exe not found at '$WslPath'. WSL2 is required; the Windows container runtime cannot substitute for it."
}

# Enter the distribution once, before the real attached run, so a broken
# distribution is reported as such instead of surfacing as an opaque exit code.
try {
    & $WslPath -d $WslDistro -u $WslUser -e true
    $probeCode = $LASTEXITCODE
} catch {
    Stop-With -Code 21 -Message ("could not execute wsl.exe: {0}" -f $_.Exception.Message)
}
if ($null -eq $probeCode) {
    Stop-With -Code 21 -Message "wsl.exe -d $WslDistro -u $WslUser -e true returned no exit code."
}
if ($probeCode -ne 0) {
    Stop-With -Code 21 -Message ("cannot enter WSL distribution '{0}' as user '{1}' (wsl exit {2}). Check that the distribution is installed and this account may run it." -f $WslDistro, $WslUser, $probeCode)
}

# ------------------------------------------------------------- attached run
# Blocks for the lifetime of the supervisor. Returns only when the supervisor
# itself returns, through its documented exit codes.
Write-Log 'INFO' 'entering the attached supervisor run; this task stays running until the owner stops the worker or the supervisor returns'
try {
    & $WslPath -d $WslDistro -u $WslUser -e bash $resolvedSupervisor --container $ContainerName
    $code = $LASTEXITCODE
} catch {
    Stop-With -Code 23 -Message ("wsl.exe failed while running the supervisor: {0}" -f $_.Exception.Message)
}

# D8: the supervisor's code IS the task's result. Never flatten it to 0.
if ($null -eq $code) {
    Stop-With -Code 23 -Message 'wsl.exe returned no exit code after the supervisor run; treating it as an unexpected failure.'
}
$code = [int] $code

switch ($code) {
    0 {
        Stop-With -Code 0 -Message 'supervisor reported an intended owner shutdown; nothing was restarted.'
    }
    10 {
        Stop-With -Code 10 -Message 'dockerd was not active inside the bounded startup window (supervisor exit 10). The task will be retried by its own restart policy.'
    }
    11 {
        Stop-With -Code 11 -Message 'the worker container is absent or could not be started (supervisor exit 11).'
    }
    12 {
        Stop-With -Code 12 -Message 'the owner-intent file holds an unrecognised value (supervisor exit 12); nothing was started.'
    }
    13 {
        Stop-With -Code 13 -Message 'owner intent was read but could not be carried out (supervisor exit 13).'
    }
    127 {
        Stop-With -Code 21 -Message 'bash or the supervisor file was not found inside the WSL distribution (wsl exit 127).'
    }
    default {
        if ($code -lt 0 -or $code -gt 255) {
            Stop-With -Code 23 -Message ("supervisor returned unmappable code {0}; reported as an unexpected failure." -f $code)
        }
        Stop-With -Code $code -Message ("supervisor exited {0}; propagated as a task failure." -f $code)
    }
}
