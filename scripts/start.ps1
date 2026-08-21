[CmdletBinding()]
param(
    [string]$Preset = "windows-cuda-release",
    [string]$ModelPath = "",
    [int]$Port = 18080,
    [int]$Device = 0,
    [int]$Threads = 12,
    [ValidateRange(0, 10000)]
    [int]$CudaKeepaliveMs = 1,
    [ValidateRange(1, 1000)]
    [int]$CudaKeepaliveWorkMs = 20,
    [string]$LogFile = "",
    [switch]$DisableGgmlCudaGraphCapture,
    [string[]]$WarmVoices = @(
        "demo_2_man",
        "demo_3_woman",
        "demo_4_woman",
        # Keep the first quick-start voice's CUDA graph active when startup
        # finishes; the WebUI prewarms other voices when they are selected.
        "demo_1_man"
    ),
    [switch]$SkipWarmup,
    # The llama.cpp chat sidecar. The default model is Peach-9B-8k-Roleplay
    # Q8_0; -SkipLlm starts speech-only.
    [string]$LlmModelPath = "",
    [int]$LlmPort = 18081,
    [int]$LlmGpuLayers = 99,
    [int]$LlmContext = 8192,
    [int]$LlmThreads = 6,
    [switch]$SkipLlm,
    [switch]$Restart,
    [switch]$OpenBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ListeningProcessId {
    param([Parameter(Mandatory = $true)][int]$LocalPort)

    $connection = Get-NetTCPConnection -State Listen -LocalPort $LocalPort -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $connection) {
        return [int]$connection.OwningProcess
    }

    foreach ($line in (& netstat.exe -ano -p tcp)) {
        if ($line -match "^\s*TCP\s+\S+:$LocalPort\s+\S+\s+LISTENING\s+(\d+)\s*$") {
            return [int]$Matches[1]
        }
    }
    return 0
}

function Test-ServerHealth {
    param([Parameter(Mandatory = $true)][string]$BaseUrl)
    try {
        $health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 2
        return $health.status -eq "ok"
    } catch {
        return $false
    }
}

function Invoke-StreamingWarmup {
    param(
        [Parameter(Mandatory = $true)][System.Net.Http.HttpClient]$Client,
        [Parameter(Mandatory = $true)][string]$Endpoint,
        [Parameter(Mandatory = $true)][string]$Voice
    )

    $payload = [ordered]@{
        model = "qwen3-tts"
        input = "The system is warmed and ready for immediate natural speech."
        voice = $Voice
        response_format = "pcm"
        stream_format = "audio"
        stream = $true
        chunk_frames = 1
        decoder_context_frames = 25
        stream_accumulate = $false
        max_tokens = 128
        seed = 1234
        do_sample = $true
        top_k = 50
        top_p = 1.0
        temperature = 0.9
        subtalker_do_sample = $true
        subtalker_top_k = 50
        subtalker_top_p = 1.0
        subtalker_temperature = 0.9
    }
    $request = [System.Net.Http.HttpRequestMessage]::new(
        [System.Net.Http.HttpMethod]::Post,
        $Endpoint)
    # The native server closes a completed streaming response. Explicitly avoid
    # reusing that socket for the next voice warmup on Windows HttpClient.
    $request.Headers.ConnectionClose = $true
    $request.Content = [System.Net.Http.StringContent]::new(
        ($payload | ConvertTo-Json -Compress),
        [System.Text.Encoding]::UTF8,
        "application/json")
    $response = $null
    $stream = $null
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $firstPcmMs = $null
    $totalBytes = 0L
    try {
        $response = $Client.SendAsync(
            $request,
            [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) {
            $errorBody = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            throw "Warmup for $Voice failed with HTTP $([int]$response.StatusCode): $errorBody"
        }
        $stream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $buffer = [byte[]]::new(65536)
        while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            if ($null -eq $firstPcmMs) {
                $firstPcmMs = $watch.Elapsed.TotalMilliseconds
            }
            $totalBytes += $read
        }
        $watch.Stop()
        if ($totalBytes -le 0) {
            throw "Warmup for $Voice returned no PCM audio."
        }
        [pscustomobject]@{
            Voice = $Voice
            FirstPcmMs = [math]::Round([double]$firstPcmMs, 2)
            TotalMs = [math]::Round($watch.Elapsed.TotalMilliseconds, 2)
            AudioSeconds = [math]::Round($totalBytes / 48000.0, 3)
        }
    } finally {
        if ($null -ne $stream) { $stream.Dispose() }
        if ($null -ne $response) { $response.Dispose() }
        $request.Dispose()
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildRoot = Join-Path $repoRoot "build\$Preset"
$serverExe = Join-Path $buildRoot "bin\audiocpp_server.exe"
$voiceDir = Join-Path $repoRoot "webui\native\demo_voices"
$configPath = Join-Path $buildRoot "qwen3_tts_server.json"
$stdoutPath = Join-Path $buildRoot "qwen3_tts_server.stdout.log"
$stderrPath = Join-Path $buildRoot "qwen3_tts_server.stderr.log"
$baseUrl = "http://127.0.0.1:$Port"

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $repoRoot "models\Qwen3-TTS-12Hz-1.7B-Base-GGUF\qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf"
}
$ModelPath = [System.IO.Path]::GetFullPath($ModelPath)

if (-not (Test-Path -LiteralPath $serverExe -PathType Leaf)) {
    throw "CUDA server not found: $serverExe"
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Qwen3-TTS 1.7B Q8 model not found: $ModelPath"
}
if (-not (Test-Path -LiteralPath $voiceDir -PathType Container)) {
    throw "Demo voice directory not found: $voiceDir"
}
if ($Threads -le 0) { throw "Threads must be positive." }
if ($Port -le 0 -or $Port -gt 65535) { throw "Port must be in 1..65535." }

# Register the bundled library as model voice presets. The server loads preset
# WAVs once at startup, avoiding filesystem reads and WAV parsing on the
# latency-critical request path while preserving the exact reference samples.
$voicePresets = [ordered]@{}
$promptTextPath = Join-Path $voiceDir "prompt_text"
if (Test-Path -LiteralPath $promptTextPath -PathType Leaf) {
    foreach ($line in Get-Content -LiteralPath $promptTextPath -Encoding UTF8) {
        $separator = $line.IndexOf('|')
        if ($separator -le 0) { continue }
        $voiceName = $line.Substring(0, $separator).Trim()
        $transcript = $line.Substring($separator + 1)
        $voiceWav = Join-Path $voiceDir "$voiceName.wav"
        if ($voiceName -and (Test-Path -LiteralPath $voiceWav -PathType Leaf)) {
            $voicePresets[$voiceName] = [ordered]@{
                voice_ref = ([System.IO.Path]::GetFullPath($voiceWav) -replace "\\", "/")
                reference_text = $transcript
            }
        }
    }
}

$effectiveConfig = [ordered]@{
    host = "127.0.0.1"
    port = $Port
    backend = "cuda"
    device = $Device
    threads = $Threads
    lazy_load = $false
    ui = $true
    ui_management = $false
    models = @(
        [ordered]@{
            id = "qwen3-tts"
            family = "qwen3_tts"
            path = ($ModelPath -replace "\\", "/")
            task = "tts"
            mode = "streaming"
            lazy = $false
            voice_presets = $voicePresets
            session_options = [ordered]@{
                "qwen3_tts.mem_saver" = "false"
                # Standard attention: measured on this 3090, flash_attention
                # gave no warm speedup and nearly tripled the cost of a prefill
                # graph capture, so the parity default is also the fast one.
                "qwen3_tts.perf_mode" = "off"
                # Chat speaks one sentence at a time, so text lengths vary far
                # more than studio use did. Slots cover the bundled voices plus
                # the active character, and enough prompt-shape graphs that a
                # conversation stops paying capture spikes after its first
                # exchanges.
                "qwen3_tts.voice_prompt_cache_slots" = "6"
                "qwen3_tts.prefill_graph_cache_slots" = "16"
            }
            default_request_options = [ordered]@{
                chunk_frames = "1"
                decoder_context_frames = "25"
                stream_accumulate = "false"
            }
        }
    )
}
$configJson = $effectiveConfig | ConvertTo-Json -Depth 8
[System.IO.Directory]::CreateDirectory($buildRoot) | Out-Null
[System.IO.File]::WriteAllText(
    $configPath,
    $configJson,
    [System.Text.UTF8Encoding]::new($false))

# ---- VRAM preflight ---------------------------------------------------------
# Budget on a 24 GB card: Qwen3-TTS resident ~6.5 GB, Peach-2.0 Q8 weights
# ~9.4 GB + 8k KV ~1 GB, CUDA runtime overhead ~1.5 GB. Anything already
# serving on our ports is reused, so its memory does not count against a
# fresh launch.
$RequiredVramMiB = 19000
try {
    $freeVram = [int](& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$Device 2>$null |
        Select-Object -First 1)
} catch { $freeVram = -1 }
$ttsRunning = (Get-ListeningProcessId -LocalPort $Port) -ne 0
$llmRunning = (Get-ListeningProcessId -LocalPort $LlmPort) -ne 0
if ($freeVram -ge 0 -and -not ($ttsRunning -and $llmRunning) -and -not $Restart) {
    $needed = $RequiredVramMiB
    if ($ttsRunning) { $needed -= 6500 }
    if ($llmRunning -or $SkipLlm) { $needed -= 11500 }
    if ($freeVram -lt $needed) {
        throw ("Not enough free VRAM: {0} MiB free on device {1}, about {2} MiB needed. " -f $freeVram, $Device, $needed) +
            "Close other GPU workloads (or pass -SkipLlm for speech-only) and try again."
    }
    Write-Host "VRAM preflight: $freeVram MiB free on device $Device (need ~$needed MiB)."
}

# ---- llama.cpp chat sidecar -------------------------------------------------
# Started before the speech server so the 9.4 GB weight load overlaps the TTS
# warmup instead of following it.
$llmExe = Join-Path $repoRoot "build\llama-cpp\llama-server.exe"
if ([string]::IsNullOrWhiteSpace($LlmModelPath)) {
    $LlmModelPath = Join-Path $repoRoot "models\Peach-2.0-9B-8k-Roleplay-GGUF\Peach-2.0-9B-8k-Roleplay.Q8_0.gguf"
}
if (-not [System.IO.Path]::IsPathRooted($LlmModelPath)) {
    $LlmModelPath = Join-Path $repoRoot $LlmModelPath
}
$LlmModelPath = [System.IO.Path]::GetFullPath($LlmModelPath)
$llmAvailable = $false
if (-not $SkipLlm) {
    if (-not (Test-Path -LiteralPath $llmExe -PathType Leaf)) {
        Write-Warning "llama-server not found at $llmExe; starting speech-only."
    } elseif (-not (Test-Path -LiteralPath $LlmModelPath -PathType Leaf)) {
        Write-Warning "LLM model not found at $LlmModelPath; starting speech-only."
    } else {
        $llmAvailable = $true
    }
}

$llmPid = 0
if ($llmAvailable) {
    $llmPid = Get-ListeningProcessId -LocalPort $LlmPort
    if ($llmPid -ne 0 -and $null -eq (Get-Process -Id $llmPid -ErrorAction SilentlyContinue)) {
        $llmPid = 0
    }
    if ($llmPid -ne 0 -and $Restart) {
        $llmProcess = Get-Process -Id $llmPid -ErrorAction Stop
        if ($llmProcess.ProcessName -ne "llama-server") {
            throw "Port $LlmPort belongs to PID $llmPid ($($llmProcess.ProcessName)); it was not stopped."
        }
        Write-Host "Stopping the existing llama-server on port $LlmPort..."
        Stop-Process -Id $llmPid
        $llmProcess.WaitForExit(10000) | Out-Null
        $llmPid = 0
    }
    if ($llmPid -eq 0) {
        Write-Host "Starting llama-server (Peach) on port $LlmPort..."
        # -ngl 99: every layer on the GPU. --jinja: apply the chat template
        # embedded in the GGUF. Flash attention defaults to auto (on for CUDA).
        # One parallel slot keeps the whole KV budget on this conversation,
        # which is what makes cache_prompt reuse effective turn over turn.
        $llmStart = @{
            FilePath = $llmExe
            ArgumentList = @(
                "-m", $LlmModelPath,
                "--host", "127.0.0.1",
                "--port", [string]$LlmPort,
                "-ngl", [string]$LlmGpuLayers,
                "-c", [string]$LlmContext,
                "--jinja",
                "--no-webui",
                "--cache-reuse", "256",
                "--parallel", "1",
                "-t", [string]$LlmThreads)
            WorkingDirectory = (Split-Path -Parent $llmExe)
            PassThru = $true
            WindowStyle = "Hidden"
            RedirectStandardOutput = (Join-Path $buildRoot "llama_server.stdout.log")
            RedirectStandardError = (Join-Path $buildRoot "llama_server.stderr.log")
        }
        $llmProcess = Start-Process @llmStart
        $llmPid = $llmProcess.Id
        try { $llmProcess.PriorityClass = "High" } catch {
            Write-Warning "Could not raise the llama-server priority: $($_.Exception.Message)"
        }
    } else {
        Write-Host "Reusing the llama-server already listening on port $LlmPort."
    }
}
# -----------------------------------------------------------------------------

$listenerPid = Get-ListeningProcessId -LocalPort $Port
if ($listenerPid -ne 0 -and $Restart) {
    $listener = Get-Process -Id $listenerPid -ErrorAction Stop
    $expectedExe = [System.IO.Path]::GetFullPath($serverExe)
    if ($listener.ProcessName -ne "audiocpp_server" -or
        [System.IO.Path]::GetFullPath($listener.Path) -ne $expectedExe) {
        throw "Port $Port belongs to PID $listenerPid ($($listener.ProcessName)); it was not stopped."
    }
    Write-Host "Stopping the existing audio.cpp server on port $Port..."
    Stop-Process -Id $listenerPid
    $listener.WaitForExit(10000) | Out-Null
    $listenerPid = 0
}

$serverProcess = $null
if ($listenerPid -eq 0) {
    Write-Host "Starting the resident Qwen3-TTS 1.7B CUDA server..."
    # Some managed launch environments inject both Path and PATH. Windows
    # PowerShell 5.1's Start-Process rejects those case-insensitive duplicates.
    $processEnvironment = [System.Environment]::GetEnvironmentVariables()
    $pathKeys = @($processEnvironment.Keys | Where-Object { $_ -ieq "PATH" })
    if ($pathKeys.Count -gt 1) {
        $pathValue = $pathKeys |
            ForEach-Object { [string]$processEnvironment[$_] } |
            Sort-Object Length -Descending |
            Select-Object -First 1
        foreach ($pathKey in $pathKeys) {
            [System.Environment]::SetEnvironmentVariable(
                [string]$pathKey,
                $null,
                [System.EnvironmentVariableTarget]::Process)
        }
        [System.Environment]::SetEnvironmentVariable(
            "Path",
            $pathValue,
            [System.EnvironmentVariableTarget]::Process)
    }
    $startArguments = @{
        FilePath = $serverExe
        ArgumentList = @(
            "--config", $configPath,
            "--voice-dir", $voiceDir,
            "--character-dir", (Join-Path $repoRoot "character"),
            "--llm-port", [string]$(if ($llmAvailable) { $LlmPort } else { 0 }),
            "--cuda-keepalive-ms", [string]$CudaKeepaliveMs,
            "--cuda-keepalive-work-ms", [string]$CudaKeepaliveWorkMs)
        WorkingDirectory = $repoRoot
        PassThru = $true
        WindowStyle = "Hidden"
        RedirectStandardOutput = $stdoutPath
        RedirectStandardError = $stderrPath
    }
    if (-not [string]::IsNullOrWhiteSpace($LogFile)) {
        $resolvedLogFile = [System.IO.Path]::GetFullPath($LogFile)
        $logParent = Split-Path -Parent $resolvedLogFile
        if ($logParent) {
            [System.IO.Directory]::CreateDirectory($logParent) | Out-Null
        }
        $startArguments.ArgumentList += @("--log-file", $resolvedLogFile)
    }
    # Keep ggml CUDA graph capture enabled for the normal low-latency path. The
    # switch is a diagnostic fallback for driver/toolkit problems; reusable
    # model graph objects and CUDA kernels remain active even when it is used.
    $previousCudaGraphDisable = [System.Environment]::GetEnvironmentVariable(
        "GGML_CUDA_DISABLE_GRAPHS",
        [System.EnvironmentVariableTarget]::Process)
    try {
        if ($DisableGgmlCudaGraphCapture) {
            [System.Environment]::SetEnvironmentVariable(
                "GGML_CUDA_DISABLE_GRAPHS",
                "1",
                [System.EnvironmentVariableTarget]::Process)
        }
        $serverProcess = Start-Process @startArguments
    } finally {
        [System.Environment]::SetEnvironmentVariable(
            "GGML_CUDA_DISABLE_GRAPHS",
            $previousCudaGraphDisable,
            [System.EnvironmentVariableTarget]::Process)
    }
    $listenerPid = $serverProcess.Id
} else {
    Write-Host "Reusing the server already listening on $baseUrl."
    $serverProcess = Get-Process -Id $listenerPid -ErrorAction SilentlyContinue
}

if ($null -ne $serverProcess) {
    try { $serverProcess.PriorityClass = "High" } catch {
        Write-Warning "Could not raise the server process priority: $($_.Exception.Message)"
    }
}

$ready = $false
for ($attempt = 0; $attempt -lt 600; $attempt++) {
    if (Test-ServerHealth -BaseUrl $baseUrl) {
        $ready = $true
        break
    }
    if ($null -ne $serverProcess -and $serverProcess.HasExited) {
        throw "The server exited during startup. See $stderrPath"
    }
    Start-Sleep -Milliseconds 250
}
if (-not $ready) {
    throw "The server did not become healthy within 150 seconds. See $stderrPath"
}

$models = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 10
$loaded = @(@($models.data) | Where-Object {
    $_.id -eq "qwen3-tts" -and $_.loaded -eq $true -and
    ([System.IO.Path]::GetFullPath([string]$_.path) -eq $ModelPath)
})
if ($loaded.Count -ne 1) {
    throw "The server on $baseUrl is not serving the requested resident Qwen3-TTS 1.7B model."
}

if ($llmAvailable) {
    $llmReady = $false
    $llmDeadline = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $llmDeadline) {
        try {
            $llmHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$LlmPort/health" -TimeoutSec 2
            if ($llmHealth.status -eq "ok") { $llmReady = $true; break }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    if (-not $llmReady) {
        Write-Warning "llama-server did not become healthy; chat will report unavailable. See $(Join-Path $buildRoot 'llama_server.stderr.log')"
        $llmAvailable = $false
    } else {
        Write-Host "llama-server is ready (Peach loaded)."
    }
}

$warmupResults = @()
if (-not $SkipWarmup -and $WarmVoices.Count -gt 0) {
    Add-Type -AssemblyName System.Net.Http
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds(90)
    try {
        foreach ($voice in $WarmVoices) {
            Write-Host "Warming voice $voice..."
            $warmupResults += Invoke-StreamingWarmup `
                -Client $client `
                -Endpoint "$baseUrl/v1/audio/speech" `
                -Voice $voice
        }
    } finally {
        $client.Dispose()
    }
    $warmupResults | Format-Table -AutoSize

    # The active character's voice is the one chat and MCP actually use; a
    # request that names no voice resolves to it. Warm it last so its CUDA
    # graphs and voice prompt stay the freshest entries in the caches.
    Write-Host "Warming the active character voice..."
    try {
        $characterWarm = [ordered]@{
            model = "qwen3-tts"
            input = "The system is warmed and ready for immediate natural speech."
            response_format = "pcm"
            stream_format = "audio"
            stream = $true
            chunk_frames = 1
            decoder_context_frames = 25
            stream_accumulate = $false
            max_tokens = 128
            seed = 1234
        }
        $characterResponse = Invoke-WebRequest -Uri "$baseUrl/v1/audio/speech" -Method Post `
            -ContentType "application/json" -Body ($characterWarm | ConvertTo-Json -Compress) `
            -TimeoutSec 120 -UseBasicParsing
        Write-Host ("Character voice warmed ({0} bytes of PCM)." -f $characterResponse.RawContentLength)
    } catch {
        Write-Warning "Character voice warmup failed: $($_.Exception.Message)"
    }
}

$pidFile = Join-Path $buildRoot "app.pids"
@("tts=$listenerPid", "llm=$(if ($llmAvailable) { $llmPid } else { 0 })") |
    Set-Content -LiteralPath $pidFile -Encoding ascii

$llmSummary = if ($llmAvailable) { "chat LLM on port $LlmPort" } else { "chat disabled" }
Write-Host "Ready at $baseUrl/ - speech + $llmSummary (PID $listenerPid, CUDA device $Device, threads $Threads)."
if ($OpenBrowser) {
    Start-Process "$baseUrl/"
}

[pscustomobject]@{
    Url = "$baseUrl/"
    ProcessId = $listenerPid
    Model = $ModelPath
    Threads = $Threads
    WarmedVoices = @($warmupResults).Count
    Config = $configPath
}
