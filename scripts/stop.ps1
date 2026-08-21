<#
.SYNOPSIS
    Gracefully shuts down the app: the speech server and the llama.cpp chat
    sidecar, releasing their VRAM.

.DESCRIPTION
    Reads the PID file the launcher wrote and stops only processes that are
    still the ones it started (matched by name), speech server first so no new
    chat turns reach a dying sidecar. Reports the VRAM that came back.
#>
[CmdletBinding()]
param(
    [string]$Preset = "windows-cuda-release",
    [int]$Device = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pidFile = Join-Path $repoRoot "build\$Preset\app.pids"

function Get-FreeVram {
    try {
        return [int](& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$Device 2>$null |
            Select-Object -First 1)
    } catch { return -1 }
}

$before = Get-FreeVram

$targets = @()
if (Test-Path -LiteralPath $pidFile) {
    foreach ($line in Get-Content -LiteralPath $pidFile) {
        if ($line -match '^(tts|llm)=(\d+)$' -and [int]$Matches[2] -ne 0) {
            $targets += [pscustomobject]@{ Kind = $Matches[1]; ProcessId = [int]$Matches[2] }
        }
    }
} else {
    Write-Host "No PID file at $pidFile; matching by process name instead."
    foreach ($name in @("audiocpp_server", "llama-server")) {
        foreach ($process in @(Get-Process -Name $name -ErrorAction SilentlyContinue)) {
            $targets += [pscustomobject]@{ Kind = $name; ProcessId = $process.Id }
        }
    }
}

# Speech server first: it fronts the sidecar, so nothing new is routed to a
# dying LLM.
$order = @("tts", "audiocpp_server", "llm", "llama-server")
$targets = $targets | Sort-Object { $order.IndexOf($_.Kind) }

$stopped = 0
foreach ($target in $targets) {
    $process = Get-Process -Id $target.ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) { continue }
    if ($process.ProcessName -notin @("audiocpp_server", "llama-server")) {
        Write-Warning "PID $($target.ProcessId) is now $($process.ProcessName); skipping."
        continue
    }
    Write-Host "Stopping $($process.ProcessName) (PID $($process.Id))..."
    # Both servers exit promptly on termination and hold no state that needs a
    # flush: the character store writes synchronously on save.
    Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
    if (-not $process.WaitForExit(10000)) {
        Write-Warning "$($process.ProcessName) did not exit in 10 s; forcing."
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        $process.WaitForExit(5000) | Out-Null
    }
    $stopped += 1
}

if (Test-Path -LiteralPath $pidFile) {
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

if ($stopped -eq 0) {
    Write-Host "Nothing was running."
} else {
    # VRAM release lags process exit by a moment.
    Start-Sleep -Seconds 2
    $after = Get-FreeVram
    if ($before -ge 0 -and $after -ge 0) {
        Write-Host "Stopped $stopped process(es); $($after - $before) MiB of VRAM released ($after MiB now free)."
    } else {
        Write-Host "Stopped $stopped process(es)."
    }
}
