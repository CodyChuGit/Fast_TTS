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
    # Demo-voice warms are opt-in now: with a single voice-prompt cache slot,
    # warming demos would evict the character the pipeline actually speaks.
    [string[]]$WarmVoices = @(),
    [switch]$SkipWarmup,
    # The llama.cpp chat sidecar. The speech server owns the process and can
    # switch between the registered models from Settings; -LlmModelPath
    # overrides where Peach 2.0 lives, -SkipLlm starts speech-only.
    [string]$LlmModelPath = "",
    [int]$LlmPort = 18081,
    [switch]$SkipLlm,
    # Pure-LLM benchmarking: the TTS registers lazily (never loaded, no VRAM),
    # the ASR is skipped, and Gemma keeps every expert layer on the GPU. The
    # LLM tab then measures the model with nothing else on the card.
    [switch]$LlmOnly,
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
    # The ear-approved default: the 1.7B talker at q4_k with the speech codec
    # kept at f16 (a homemade audiocpp_gguf requant). Same voice as the stock
    # q8_0 package to the ear, and the freed VRAM funds two more Gemma expert
    # layers on the GPU. The q8_0_v2 package stays on disk as the fallback.
    $ModelPath = Join-Path $repoRoot "models\Qwen3-TTS-12Hz-1.7B-Base-GGUF\qwen3-tts-12hz-1.7b-base-q4_k-mix.gguf"
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
    # 24 GB card, but most of it can: only the first few layers' MoE experts
    # stay in system RAM. Flash attention + q8_0 KV (llm_manager base args)
    # freed ~300 MB, spent here on two more GPU expert layers, and a
    # 2048-token micro-batch speeds the split bulk prefill (grid-measured on
    # this 3090 beside the Q4-mix TTS: prefill 1060 -> 1749 t/s, decode
    # 53.8 -> ~60 t/s, 22.9 GB total). Gemma 4 is a reasoning model, and a
    # voice conversation cannot wait through a hidden think phase -- a zero
    # reasoning budget makes it answer directly.
    $llmModels += , [ordered]@{
        id = "gemma"
        name = "Gemma 4 26B heretic"
        path = ($gemmaModelPath -replace "\\", "/")
        # Deeper on the GPU when the small TTS model frees the VRAM; the
        # larger-TTS branch stays conservative (one layer from KV savings).
        # When the streaming-STT model is registered, THREE expert layers go
        # back to the CPU: at two, total VRAM sat at 96% and a TTS prefill
        # graph capture wedged the model mid-allocation. Headroom is a
        # correctness requirement here, not a tuning preference.
        extra_args = @($(if ($LlmOnly) {
                # The whole card belongs to the LLM: all experts on the GPU.
                @("--n-cpu-moe", "0", "-ub", "2048")
            } elseif ($ModelPath -match '0\.6b|q4_k-mix') {
                @("--n-cpu-moe", $(if (Test-Path -LiteralPath (Join-Path $repoRoot "models\Qwen3-ASR-0.6B-GGUF\qwen3-asr-0.6b-q4_k.gguf") -PathType Leaf) { "5" } else { "3" }), "-ub", "2048")
            } else {
                @("--n-cpu-moe", "6", "-ub", "1024")
            }) + @("--reasoning-budget", "0"))
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
        extra_args = @("--n-cpu-moe", $(if ($ModelPath -match '0\.6b|q4_k-mix') { "6" } else { "9" }),
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
    if ($LlmOnly) {
        # With the card to itself the dense 27B fits fully on the GPU for the
        # first time -- this is the model the voice-mode budget could never
        # afford. Three flavors race from the Settings switch; measured
        # (2026-08-23, 3090, story turns): plain 18.4 t/s decode, ngram-mod
        # 19.9 t/s (free +8%), MTP 18.0 at its best tuning (n-max 4) and 11
        # untuned -- the Q8 head's draft cost cancels its 33-40% acceptance,
        # so ngram is the one speculation worth switching on.
        $orcaCommon = @("-ngl", "99", "-ub", "2048",
            "--chat-template-kwargs", '{"enable_thinking":false}')
        $llmModels += , [ordered]@{
            id = "orca-qwen"
            name = "Qwen3.8 27B OrcaRouter"
            path = ($orcaModelPath -replace "\\", "/")
            extra_args = $orcaCommon
        }
        $orcaMtpPath = Join-Path $repoRoot "models\Qwen3.8-27B-OrcaRouter-GGUF\Qwen3.8-27B-Uncensored-OrcaRouter-MTP-Q8_0.gguf"
        if (Test-Path -LiteralPath $orcaMtpPath -PathType Leaf) {
            $llmModels += , [ordered]@{
                id = "orca-qwen-mtp"
                name = "Qwen3.8 27B OrcaRouter (MTP)"
                path = ($orcaModelPath -replace "\\", "/")
                extra_args = $orcaCommon + @("--spec-type", "draft-mtp",
                    "-md", ($orcaMtpPath -replace "\\", "/"), "-ngld", "99",
                    "-ctkd", "q8_0", "-ctvd", "q8_0",
                    "--spec-draft-n-max", "4", "--spec-draft-n-min", "1",
                    "--spec-draft-p-min", "0.5")
            }
        }
        $llmModels += , [ordered]@{
            id = "orca-qwen-ngram"
            name = "Qwen3.8 27B OrcaRouter (ngram)"
            path = ($orcaModelPath -replace "\\", "/")
            extra_args = $orcaCommon + @("--spec-type", "ngram-mod")
        }
    } else {
        $llmModels += , [ordered]@{
            id = "orca-qwen"
            name = "Qwen3.8 27B OrcaRouter (slow)"
            path = ($orcaModelPath -replace "\\", "/")
            extra_args = @("-ngl", "42", "-ub", "1024",
                "--chat-template-kwargs", '{"enable_thinking":false}')
        }
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
        extra_args = @($(if ($LlmOnly) { @("-ngl", "99", "-ub", "2048") }
            else { @("-ngl", "46", "-ub", "1024") }) +
            @("--chat-template-kwargs", '{"enable_thinking":false}'))
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

# Streaming STT (the Voice tab): registered whenever the converted ASR
# weights exist. Lazy, so boot stays fast and the ~1.5 GB VRAM is only spent
# once voice is actually used; after the first use the model stays warm.
$asrModelPath = Join-Path $repoRoot "models\Qwen3-ASR-0.6B-GGUF\qwen3-asr-0.6b-q4_k.gguf"
$voiceAvailable = (Test-Path -LiteralPath $asrModelPath -PathType Leaf) -and -not $LlmOnly

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
            lazy = ([bool]$LlmOnly)
            voice_presets = $voicePresets
            session_options = [ordered]@{
                "qwen3_tts.mem_saver" = "false"
                # Standard attention: measured on this 3090, flash_attention
                # gave no warm speedup and nearly tripled the cost of a prefill
                # graph capture, so the parity default is also the fast one.
                "qwen3_tts.perf_mode" = "off"
                # One voice slot: chat is the product and it speaks one
                # character. Prefill graphs are the hidden VRAM monster --
                # measured ~470 MB EACH -- so five slots caps them at ~2.3 GB;
                # an unusual sentence length outside the five hottest buckets
                # pays a one-time recapture instead of the whole machine
                # paging (measured: graph creep to 23.5+ GB collapses llama
                # decode to single digits via unified-memory fallback).
                "qwen3_tts.voice_prompt_cache_slots" = "1"
                "qwen3_tts.prefill_graph_cache_slots" = "5"
            }
            default_request_options = [ordered]@{
                chunk_frames = "1"
                decoder_context_frames = "25"
                stream_accumulate = "false"
            }
        }
    )
}
if ($voiceAvailable) {
    $effectiveConfig["models"] = @($effectiveConfig["models"]) + , [ordered]@{
        id = "qwen3-asr"
        family = "qwen3_asr"
        path = ($asrModelPath -replace "\\", "/")
        task = "asr"
        mode = "offline"
        # Eager: the ~2.6 GB loads at boot so the first spoken turn never
        # pays a ten-second model load.
        lazy = $false
    }
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
if (-not $SkipWarmup -and -not $LlmOnly) {
    if ($WarmVoices.Count -gt 0) {
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
    }

    # Warm the ACTIVE CHARACTER across the sentence-length buckets chat
    # actually produces: each length captures its ~470 MB prefill graph at
    # boot, so no conversation ever pays a capture spike, and the single
    # voice-prompt slot ends up holding the character.
    $characterWarmSentences = @(
        "Okay, sounds good.",
        "That sounds like a really lovely way to spend the evening.",
        "Honestly, I think we should try the new place first and then decide if we still want dessert afterwards.",
        "Mm, let me think about it for a moment, because there are a few different ways we could plan this and I want to pick the one that feels the most fun for both of us tonight."
    )
    foreach ($sentence in $characterWarmSentences) {
        try {
            $characterWarm = [ordered]@{
                model = "qwen3-tts"
                input = $sentence
                response_format = "pcm"
                stream_format = "audio"
                stream = $true
                chunk_frames = 1
                decoder_context_frames = 25
                stream_accumulate = $false
                seed = 777
            }
            $characterResponse = Invoke-WebRequest -Uri "$baseUrl/v1/audio/speech" -Method Post `
                -ContentType "application/json" -Body ($characterWarm | ConvertTo-Json -Compress) `
                -TimeoutSec 180 -UseBasicParsing
            Write-Host ("Character warm ({0} chars -> {1} bytes)." -f $sentence.Length, $characterResponse.RawContentLength)
        } catch {
            Write-Warning "Character warmup failed: $($_.Exception.Message)"
        }
    }
}

# (A background page-cache warm of the whole model menu was tried here and
# removed: streaming ~70 GB of GGUFs competes with the CPU-resident experts
# for memory bandwidth and showed up as multi-second first-token spikes
# mid-chat. Model switches simply pay their disk read.)

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
