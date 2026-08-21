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
    # The llama.cpp chat sidecar. The speech server owns the process and can
    # switch between the registered models from Settings; -LlmModelPath
    # overrides where Peach 2.0 lives, -SkipLlm starts speech-only.
    [string]$LlmModelPath = "",
    [int]$LlmPort = 18081,
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
    # The 1.7B is the voice-quality pick (the 0.6B measured faster but lost
    # too much of the clone); pass the 0.6B path explicitly to trade voice
    # fidelity for ~250 ms per reply.
    $ModelPath = Join-Path $repoRoot "models\Qwen3-TTS-12Hz-1.7B-Base-GGUF\qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf"
}
$ModelPath = [System.IO.Path]::GetFullPath($ModelPath)

if (-not (Test-Path -LiteralPath $serverExe -PathType Leaf)) {
    throw "CUDA server not found: $serverExe"
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Qwen3-TTS model not found: $ModelPath"
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

# ---- llama.cpp chat sidecar registry ----------------------------------------
# The speech server spawns and owns llama-server (kill-on-close job object), so
# the launcher only decides which models are on the menu. Settings can switch
# between every entry whose GGUF is on disk.
$llmExe = Join-Path $repoRoot "build\llama-cpp\llama-server.exe"
if ([string]::IsNullOrWhiteSpace($LlmModelPath)) {
    $LlmModelPath = Join-Path $repoRoot "models\Peach-2.0-9B-8k-Roleplay-GGUF\Peach-2.0-9B-8k-Roleplay.Q8_0.gguf"
}
if (-not [System.IO.Path]::IsPathRooted($LlmModelPath)) {
    $LlmModelPath = Join-Path $repoRoot $LlmModelPath
}
$LlmModelPath = [System.IO.Path]::GetFullPath($LlmModelPath)
$gemmaModelPath = Join-Path $repoRoot "models\gemma-4-26B-A4B-qat-heretic-GGUF\gemma-4-26B-A4B-it-qat-heretic-UD-Q4_K_XL.gguf"

$llmModels = @()
if (Test-Path -LiteralPath $LlmModelPath -PathType Leaf) {
    $llmModels += , [ordered]@{
        id = "peach"
        name = "Peach 2.0"
        path = ($LlmModelPath -replace "\\", "/")
    }
}
if (Test-Path -LiteralPath $gemmaModelPath -PathType Leaf) {
    # All 16.8 GB of Q4_K_M cannot sit beside the resident TTS model on a
    # 24 GB card, but most of it can: only the first 9 layers' MoE experts
    # stay in system RAM, and a 1024-token micro-batch keeps the split
    # prefill efficient (measured on this 3090: prompt 590 t/s and decode
    # 47 t/s versus 129/17 with everything on the CPU, at 14.5 GB beside
    # the trimmed TTS caches). Gemma 4 is a reasoning model, and a voice
    # conversation cannot wait through a hidden think phase -- a zero
    # reasoning budget makes it answer directly.
    $llmModels += , [ordered]@{
        id = "gemma"
        name = "Gemma 4 26B heretic"
        path = ($gemmaModelPath -replace "\\", "/")
        # Two expert layers deeper when the small TTS model frees the VRAM.
        extra_args = @("--n-cpu-moe", $(if ($ModelPath -match '0\.6b') { "5" } else { "7" }),
            "-ub", "1024", "--reasoning-budget", "0")
    }
}
$vanillaGemmaPath = Join-Path $repoRoot "models\gemma-4-26B-A4B-it-qat-GGUF\gemma-4-26B_q4_0-it.gguf"
if (Test-Path -LiteralPath $vanillaGemmaPath -PathType Leaf) {
    # Google's own QAT release, alignment intact: the "vanilla" alternative
    # for anyone who wants stock Gemma behavior. Slightly larger than the
    # heretic UD quant, so it keeps one more expert layer on the CPU.
    $llmModels += , [ordered]@{
        id = "gemma-vanilla"
        name = "Gemma 4 26B (vanilla)"
        path = ($vanillaGemmaPath -replace "\\", "/")
        extra_args = @("--n-cpu-moe", $(if ($ModelPath -match '0\.6b') { "7" } else { "10" }),
            "-ub", "1024", "--reasoning-budget", "0")
    }
}
$orcaModelPath = Join-Path $repoRoot "models\Qwen3.8-27B-OrcaRouter-GGUF\Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf"
if (Test-Path -LiteralPath $orcaModelPath -PathType Leaf) {
    # Dense 27B like the Huihui build: sharp functionally (its memory and
    # language mirroring judged 10/10) but it cannot sit fully beside the TTS
    # model, so whole layers go to the CPU and decode is slow. Its matched
    # MTP draft head measured flat here (33% acceptance, heavy head, hybrid
    # attention rollback), so speculation stays off.
    $llmModels += , [ordered]@{
        id = "orca-qwen"
        name = "Qwen3.8 27B OrcaRouter (slow)"
        path = ($orcaModelPath -replace "\\", "/")
        extra_args = @("-ngl", "42", "-ub", "1024",
            "--chat-template-kwargs", '{"enable_thinking":false}')
    }
}
$huihuiModelPath = Join-Path $repoRoot "models\Huihui-Qwen3.8-27B-abliterated-GGUF\Huihui-Qwen3.8-27B-abliterated.Q4_K_M.gguf"
if (Test-Path -LiteralPath $huihuiModelPath -PathType Leaf) {
    # A DENSE 27B: there is no expert-offload trick, so fitting beside the
    # TTS model means putting whole layers on the CPU, and measured decode is
    # ~4 t/s -- slower than the speech itself. Registered because it is asked
    # for, but not suited to live voice at this quant; a ~Q3 quant that fits
    # fully on the GPU would run ~35 t/s. Its template ignores
    # --reasoning-budget, so thinking is disabled through the template kwarg.
    $llmModels += , [ordered]@{
        id = "huihui-qwen"
        name = "Qwen3.8 27B heretic (slow)"
        path = ($huihuiModelPath -replace "\\", "/")
        extra_args = @("-ngl", "46", "-ub", "1024",
            "--chat-template-kwargs", '{"enable_thinking":false}')
    }
}
$llmAvailable = $false
if (-not $SkipLlm) {
    if (-not (Test-Path -LiteralPath $llmExe -PathType Leaf)) {
        Write-Warning "llama-server not found at $llmExe; starting speech-only."
    } elseif ($llmModels.Count -eq 0) {
        Write-Warning "No chat model files found under models\; starting speech-only."
    } else {
        $llmAvailable = $true
    }
}
# ------------------------------------------------------------------------------

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
                # more than studio use did. Three voice slots cover the active
                # character plus a couple of demo auditions -- chat only ever
                # uses one voice, and the reclaimed VRAM buys two more Gemma
                # expert layers on the GPU. Twelve prompt-shape graphs still
                # stop a conversation paying capture spikes after its first
                # exchanges.
                "qwen3_tts.voice_prompt_cache_slots" = "3"
                "qwen3_tts.prefill_graph_cache_slots" = "12"
            }
            default_request_options = [ordered]@{
                chunk_frames = "1"
                decoder_context_frames = "25"
                stream_accumulate = "false"
            }
        }
    )
}
if ($llmAvailable) {
    $effectiveConfig["llm_host"] = "127.0.0.1"
    $effectiveConfig["llm_port"] = $LlmPort
    $effectiveConfig["llm_server_exe"] = (([System.IO.Path]::GetFullPath($llmExe)) -replace "\\", "/")
    $effectiveConfig["llm_default"] = [string]$llmModels[0].id
    $effectiveConfig["llm_log_dir"] = ($buildRoot -replace "\\", "/")
    $effectiveConfig["llm_models"] = $llmModels
}
$configJson = $effectiveConfig | ConvertTo-Json -Depth 8
[System.IO.Directory]::CreateDirectory($buildRoot) | Out-Null
[System.IO.File]::WriteAllText(
    $configPath,
    $configJson,
    [System.Text.UTF8Encoding]::new($false))

# ---- VRAM preflight ---------------------------------------------------------
# Budget on a 24 GB card: Qwen3-TTS resident ~6.5 GB, the chat model up to
# ~13.3 GB (Gemma 26B with 10 expert layers on the CPU; Peach Q8 is ~11.5 GB),
# CUDA runtime overhead ~1.5 GB. Anything already serving on our ports is
# reused, so its memory does not count against a fresh launch.
$RequiredVramMiB = 21000
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

# The speech server spawns its own llama-server child; anything already bound
# to the LLM port when a fresh server is about to start is a leftover (an old
# launcher-owned sidecar, or the job object still tearing down) and would block
# the child's bind.
if ($llmAvailable -and $listenerPid -eq 0) {
    $stalePid = Get-ListeningProcessId -LocalPort $LlmPort
    if ($stalePid -ne 0) {
        $stale = Get-Process -Id $stalePid -ErrorAction SilentlyContinue
        if ($null -ne $stale -and $stale.ProcessName -eq "llama-server") {
            Write-Host "Stopping a leftover llama-server on port $LlmPort..."
            Stop-Process -Id $stalePid
            $stale.WaitForExit(10000) | Out-Null
        } elseif ($null -ne $stale) {
            throw "Port $LlmPort belongs to PID $stalePid ($($stale.ProcessName)); it was not stopped."
        }
    }
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
    # The 26B MoE takes a while to read from disk and stage into system RAM;
    # give the child real time before declaring chat unavailable.
    $llmDeadline = (Get-Date).AddSeconds(300)
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
        $llmName = "chat model"
        try {
            $llmSettings = Invoke-RestMethod -Uri "$baseUrl/v1/llm-settings" -TimeoutSec 5
            $active = @($llmSettings.models) | Where-Object { $_.id -eq $llmSettings.model }
            if (@($active).Count -ge 1) { $llmName = @($active)[0].name }
        } catch {}
        Write-Host "llama-server is ready ($llmName loaded)."
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

# The sidecar is the server's child, so its PID is discovered from the port it
# serves; stop.ps1 uses the file as a hint and name-sweeps regardless.
$llmPid = if ($llmAvailable) { Get-ListeningProcessId -LocalPort $LlmPort } else { 0 }
$pidFile = Join-Path $buildRoot "app.pids"
@("tts=$listenerPid", "llm=$llmPid") |
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
