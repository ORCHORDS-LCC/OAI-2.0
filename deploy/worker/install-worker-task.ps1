# ===========================================================================
# install-worker-task.ps1 - reproducibly install/update the OAI2WorkerRuntime
# scheduled task (master #241 requirement A, issue #251).
#
# WHY THIS FILE EXISTS
#   The working task configuration on the P50 was an undocumented manual host
#   change. Nothing in the repository described it, so it could not be
#   reproduced, reviewed or diffed - and its settings were exactly the settings
#   that decide whether a dead worker is reported as a dead worker. This script
#   makes the working configuration a reviewable artifact instead: the whole
#   configuration is encoded here, and applying it twice produces the same
#   registered task.
#
# WHAT IT REGISTERS
#   Principal      the P50 account, LogonType S4U, RunLevel Highest
#   Trigger        AtStartup
#   Action         powershell.exe -NoProfile -ExecutionPolicy Bypass
#                  -WindowStyle Hidden -File <start-worker.ps1>
#   Settings       ExecutionTimeLimit  PT0S (unlimited)
#                  MultipleInstances   IgnoreNew
#                  RestartCount        3
#                  RestartInterval     5 minutes
#                  StartWhenAvailable  true
#
# WHY S4U AND NEVER SYSTEM
#   WSL cannot run as SYSTEM: it fails with Wsl/WSL_E_LOCAL_SYSTEM_NOT_SUPPORTED.
#   S4U ("do not store password") runs the task as the interactive user without
#   keeping a credential, and is verified to work for WSL. Consequently this
#   script has NO -Password parameter, never prompts for one, and never echoes,
#   logs or stores any secret. If S4U registration is not possible in the target
#   environment, this script FAILS LOUDLY - it does not fall back to SYSTEM,
#   because a SYSTEM task would fail at every WSL launch and would look like a
#   broken worker rather than a broken registration.
#
# EXIT CODES
#     0  success (a successful apply, -DryRun or -ShowCurrent)
#     2  invalid invocation (bad parameter, or launcher not deployed)
#     3  the scheduled-task registration failed
#     4  the task was registered but does not match what was requested
#     5  -ShowCurrent: the task is not registered on this host
#     6  elevation is required to register an AtStartup task
#
# MODES
#     (no flags)          apply/update the task, then verify what was registered
#     -DryRun             print the configuration that would be applied, change
#                         nothing
#     -ShowCurrent        print what is registered now, change nothing
#     -ShowCurrent -DryRun  both reports, still change nothing
#
# USAGE
#   # preview only - prints the configuration, changes nothing
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File install-worker-task.ps1 -DryRun
#
#   # compare what is registered now against what this script would apply
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File install-worker-task.ps1 -ShowCurrent -DryRun
#
#   # apply (elevated)
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File install-worker-task.ps1
#
#   # another supported account / install path
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File install-worker-task.ps1 `
#       -TaskName OAI2WorkerRuntime -Account '.\P51' -ScriptDirectory 'C:\Users\P51\q-pipe'
#
# SCOPE
#   Registers exactly one scheduled task and starts nothing. It never touches
#   WSL configuration, never touches a coordinator, gateway, model server or any
#   other pipeline service, and holds no credential.
# ===========================================================================
[CmdletBinding()]
param(
    [string] $TaskName       = 'OAI2WorkerRuntime',
    [string] $Account        = '.\P50',
    [string] $ScriptDirectory = 'C:\Users\P50\q-pipe',
    [string] $LauncherName   = 'start-worker.ps1',

    [int]    $RestartCount           = 3,
    [int]    $RestartIntervalMinutes = 5,
    [string] $PowerShellPath = 'powershell.exe',

    # Preview: print what would be applied, apply nothing.
    [switch] $DryRun,
    # Print the currently registered configuration for comparison.
    [switch] $ShowCurrent
)

# No Set-StrictMode here on purpose: the MSFT_Task* objects expose different
# property sets across Windows versions, and property access is checked
# explicitly further down instead of by the parser.

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
    exit $Code
}

function Test-Administrator {
    try {
        $identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    } catch {
        return $false
    }
}

function Test-HasProperty {
    param($Object, [string] $Name)
    if ($null -eq $Object) { return $false }
    return ($Object.PSObject.Properties.Name -contains $Name)
}

function Get-CurrentTask {
    param([string] $Name)
    return (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue)
}

function Get-IntervalMinutes {
    # Get-ScheduledTask reports RestartInterval as an ISO-8601 duration
    # ('PT5M'), or as a TimeSpan depending on version. Convert both so the
    # verification below compares real minutes; $null means "unrecognised",
    # which is reported as a warning rather than a false failure.
    param($Value)
    if ($Value -is [TimeSpan]) { return [int] [Math]::Round($Value.TotalMinutes) }
    $m = [regex]::Match([string] $Value, '^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$')
    if (-not $m.Success) { return $null }
    $h = 0; $min = 0; $sec = 0
    if ($m.Groups[1].Success) { $h   = [int] $m.Groups[1].Value }
    if ($m.Groups[2].Success) { $min = [int] $m.Groups[2].Value }
    if ($m.Groups[3].Success) { $sec = [int] $m.Groups[3].Value }
    return (($h * 60) + $min)
}

function Format-ActionArgument {
    param(
        [string] $PowerShell,
        [string] $Launcher
    )
    # The launcher path is quoted so an install directory with a space in it
    # still yields a valid action. This is the ONLY place the action string is
    # built, so the registered task and every printed preview agree.
    return ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $Launcher)
}

# Display only. This function deliberately returns nothing: its output IS the
# report, and a returned value would be indistinguishable from those lines for
# the caller. Presence is decided by the caller via Get-CurrentTask.
function Show-CurrentTask {
    param(
        [string] $Name,
        $Task
    )
    Write-Output ''
    Write-Output "=== CURRENTLY REGISTERED: $Name ==="
    Write-Output ('  State            : {0}' -f $Task.State)
    if ($Task.Principal) {
        Write-Output ('  Principal        : {0} (LogonType={1}, RunLevel={2})' -f `
            $Task.Principal.UserId, $Task.Principal.LogonType, $Task.Principal.RunLevel)
    }
    foreach ($a in @($Task.Actions)) {
        Write-Output ('  Action Execute   : {0}' -f $a.Execute)
        Write-Output ('  Action Arguments : {0}' -f $a.Arguments)
    }
    foreach ($t in @($Task.Triggers)) {
        Write-Output ('  Trigger          : {0}' -f $t.CimClass.CimClassName)
    }
    if ($Task.Settings) {
        Write-Output ('  ExecutionLimit   : {0}' -f $Task.Settings.ExecutionTimeLimit)
        Write-Output ('  MultipleInstances: {0}' -f $Task.Settings.MultipleInstances)
        Write-Output ('  StartWhenAvailable: {0}' -f $Task.Settings.StartWhenAvailable)
        if (Test-HasProperty $Task.Settings 'RestartCount') {
            Write-Output ('  RestartCount     : {0}' -f $Task.Settings.RestartCount)
        }
        if (Test-HasProperty $Task.Settings 'RestartInterval') {
            Write-Output ('  RestartInterval  : {0}' -f $Task.Settings.RestartInterval)
        }
    }
    Write-Output ''
}

# ------------------------------------------------------------ validate input
if ([string]::IsNullOrWhiteSpace($TaskName)) {
    Stop-With -Code 2 -Message '-TaskName must not be empty.'
}
if ([string]::IsNullOrWhiteSpace($Account)) {
    Stop-With -Code 2 -Message '-Account must not be empty.'
}
if ([string]::IsNullOrWhiteSpace($ScriptDirectory)) {
    Stop-With -Code 2 -Message '-ScriptDirectory must not be empty.'
}
if ([string]::IsNullOrWhiteSpace($LauncherName)) {
    Stop-With -Code 2 -Message '-LauncherName must not be empty.'
}
if ($RestartCount -lt 0) {
    Stop-With -Code 2 -Message "-RestartCount must be >= 0, got $RestartCount."
}
if ($RestartIntervalMinutes -lt 0) {
    Stop-With -Code 2 -Message "-RestartIntervalMinutes must be >= 0, got $RestartIntervalMinutes."
}
if ([string]::IsNullOrWhiteSpace($PowerShellPath)) {
    Stop-With -Code 2 -Message '-PowerShellPath must not be empty.'
}

$launcher = Join-Path -Path $ScriptDirectory -ChildPath $LauncherName
$actionArguments = Format-ActionArgument -PowerShell $PowerShellPath -Launcher $launcher

Write-Log 'INFO' ("task={0} account={1} launcher={2}" -f $TaskName, $Account, $launcher)

if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    if ($DryRun) {
        Write-Log 'WARN' ("launcher not found at '{0}'; the preview below is still valid, but the task would fail until the launcher is deployed there." -f $launcher)
    } else {
        Stop-With -Code 2 -Message "launcher not found at '$launcher'. Deploy start-worker.ps1 there first, or pass -ScriptDirectory."
    }
}

# ------------------------------------------------------------- current state
# -ShowCurrent is strictly read-only: it reports and stops. To change the task,
# run without it. This keeps "just look at it" from ever mutating a host.
if ($ShowCurrent) {
    $currentTask = Get-CurrentTask -Name $TaskName
    if ($null -eq $currentTask) {
        Write-Log 'WARN' ("task '{0}' is not registered on this host" -f $TaskName)
    } else {
        Show-CurrentTask -Name $TaskName -Task $currentTask
    }
    if (-not $DryRun) {
        Write-Log 'INFO' ("-ShowCurrent is read-only; task '{0}' was not modified. Re-run without -ShowCurrent to apply." -f $TaskName)
        if ($null -eq $currentTask) { exit 5 }
        exit 0
    }
}

# ------------------------------------------------------------- build the task
$action = New-ScheduledTaskAction -Execute $PowerShellPath -Argument $actionArguments
$trigger = New-ScheduledTaskTrigger -AtStartup
# S4U = "do not store password". No credential is supplied, requested or stored.
$principal = New-ScheduledTaskPrincipal -UserId $Account -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount $RestartCount `
    -RestartInterval (New-TimeSpan -Minutes $RestartIntervalMinutes) `
    -StartWhenAvailable

Write-Output ''
Write-Output '=== CONFIGURATION THIS SCRIPT APPLIES ==='
Write-Output ('  Task             : {0}' -f $TaskName)
Write-Output ('  Action           : {0} {1}' -f $PowerShellPath, $actionArguments)
Write-Output ('  Trigger          : AtStartup')
Write-Output ('  Principal        : {0} (LogonType=S4U, RunLevel=Highest)' -f $Account)
Write-Output ('  ExecutionLimit   : PT0S (unlimited)')
Write-Output ('  MultipleInstances: IgnoreNew')
Write-Output ('  RestartCount     : {0}' -f $RestartCount)
Write-Output ('  RestartInterval  : {0} minute(s)' -f $RestartIntervalMinutes)
Write-Output ('  StartWhenAvailable: true')
Write-Output ('  Password stored  : none (S4U)')
Write-Output ''

if ($DryRun) {
    Write-Log 'INFO' "DRY RUN: nothing was applied. Task '$TaskName' is unchanged."
    exit 0
}

# ------------------------------------------------------------------- apply
if (-not (Test-Administrator)) {
    Stop-With -Code 6 -Message ("registering an AtStartup task with RunLevel=Highest requires an elevated shell. Re-run from an elevated PowerShell. No partial change was made.")
}

Write-Log 'INFO' ("registering/updating task '{0}' (this is idempotent: -Force replaces it with exactly the configuration above)" -f $TaskName)
try {
    Register-ScheduledTask `
        -TaskName    $TaskName `
        -Action      $action `
        -Trigger     $trigger `
        -Principal   $principal `
        -Settings    $settings `
        -Description 'OAI-2.0 fleet worker container-runtime supervisor (master #241 requirement A). Managed by deploy/worker/install-worker-task.ps1.' `
        -Force `
        -ErrorAction Stop | Out-Null
} catch {
    # Deliberately NO fallback to SYSTEM: WSL cannot run as SYSTEM
    # (Wsl/WSL_E_LOCAL_SYSTEM_NOT_SUPPORTED), so a SYSTEM task would fail at
    # every launch while looking like a broken worker.
    Stop-With -Code 3 -Message ("Register-ScheduledTask failed: {0}`n  S4U registration is mandatory here; a SYSTEM fallback was NOT attempted because WSL cannot run as SYSTEM (Wsl/WSL_E_LOCAL_SYSTEM_NOT_SUPPORTED)." -f $_.Exception.Message)
}

# ------------------------------------------------------------------ verify
$registered = Get-CurrentTask -Name $TaskName
if ($null -eq $registered) {
    Stop-With -Code 3 -Message "registration reported success but task '$TaskName' cannot be read back."
}

$problems = New-Object System.Collections.ArrayList

if ([string] $registered.Principal.LogonType -ne 'S4U') {
    [void] $problems.Add(("LogonType is '{0}', expected 'S4U'. A SYSTEM or interactive task must not be used here: WSL cannot run as SYSTEM." -f $registered.Principal.LogonType))
} else {
    # Windows may normalise the account (P50 vs .\P50 vs HOST\P50). Warn, do
    # not fail: the LogonType above is the property that actually matters.
    $wanted  = ($Account -replace '^[.\\]+', '')
    $actual  = ($registered.Principal.UserId -replace '^[.\\]+', '' -replace '^[^\\]+\\', '')
    if ($wanted -ne $actual -and $wanted -notlike "*$actual") {
        [void] $problems.Add(("principal account is '{0}' but '{1}' was requested (compare -ShowCurrent output before trusting this task)." -f $registered.Principal.UserId, $Account))
    }
}

if ([string] $registered.Settings.MultipleInstances -ne 'IgnoreNew') {
    [void] $problems.Add(("MultipleInstances is '{0}', expected 'IgnoreNew'." -f $registered.Settings.MultipleInstances))
}

$limit = [string] $registered.Settings.ExecutionTimeLimit
if ($limit -ne 'PT0S' -and $limit -ne '00:00:00') {
    [void] $problems.Add(("ExecutionTimeLimit is '{0}', expected PT0S (unlimited)." -f $limit))
}

if (Test-HasProperty $registered.Settings 'RestartCount') {
    if ([int] $registered.Settings.RestartCount -ne $RestartCount) {
        [void] $problems.Add(("RestartCount is '{0}', expected {1}." -f $registered.Settings.RestartCount, $RestartCount))
    }
}
if (Test-HasProperty $registered.Settings 'RestartInterval') {
    $ri        = $registered.Settings.RestartInterval
    $riMinutes = Get-IntervalMinutes -Value $ri
    if ($null -eq $riMinutes) {
        Write-Log 'WARN' ("could not interpret the registered RestartInterval '{0}'; expected {1} minute(s). Verify by hand." -f $ri, $RestartIntervalMinutes)
    } elseif ($riMinutes -ne $RestartIntervalMinutes) {
        [void] $problems.Add(("RestartInterval is '{0}' (parsed as {1} minute(s)), expected {2}." -f $ri, $riMinutes, $RestartIntervalMinutes))
    }
}
if (-not ([bool] $registered.Settings.StartWhenAvailable)) {
    [void] $problems.Add('StartWhenAvailable is false, expected true.')
}
$registeredActions = @($registered.Actions)
if ($registeredActions.Count -ne 1 -or $registeredActions[0].Execute -ne $PowerShellPath) {
    [void] $problems.Add("the registered action does not match '$PowerShellPath $actionArguments'.")
} elseif ($registeredActions[0].Arguments -ne $actionArguments) {
    [void] $problems.Add(("action arguments are '{0}', expected '{1}'." -f $registeredActions[0].Arguments, $actionArguments))
}

if ($problems.Count -gt 0) {
    foreach ($p in $problems) { Write-Log 'ERROR' $p }
    Stop-With -Code 4 -Message ("task '{0}' is registered but does NOT match the requested configuration. Re-run with -ShowCurrent -DryRun to compare." -f $TaskName)
}

Write-Log 'INFO' ("task '{0}' registered and verified against the requested configuration." -f $TaskName)
Write-Log 'INFO' ("re-run this script at any time: it is idempotent, so a second run yields the same registered configuration.")
Write-Log 'INFO' "start it now with: Start-ScheduledTask -TaskName '$TaskName'"
Write-Log 'INFO' "operator artifacts after it runs (inside WSL): /var/lib/worker/supervisor-health, /var/lib/worker/supervisor.log"
exit 0
