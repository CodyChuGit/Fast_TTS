# qwen3-tts-streamer

Native, low-latency Qwen3-TTS streaming built as a small, upstream-friendly
extension to [`audio.cpp`](https://github.com/0xShug0/audio.cpp). The talker,
code predictor, GGUF loaders, voice encoders, speech decoder, CPU/CUDA backends,
CLI, and HTTP server remain the audio.cpp implementations. No PyTorch or Python
runtime is used for inference.

## What is genuinely streaming

The talker performs prompt prefill once, preserves its KV cache and RNG, and
then invokes a callback whenever one *complete* temporal codec frame (all
codebooks) is available. The callback runs before the talker computes the next
frame. After `chunk_frames` frames, the speech decoder receives the previous
left context plus the new frames; samples belonging to the old context are
discarded and only new PCM is emitted.

```text
text / voice prompt -> talker prefill + KV cache
                    -> complete codec frame(s)
                    -> 25-frame decoder context + new frames
                    -> trim old-context samples
                    -> 24 kHz mono float32 PCM event
```

The default is two 12.5 Hz frames, or about 160 ms of speech per decode. The
session logs `qwen3_tts.streaming.first_pcm_before_generation_end=1` when the
request proves that PCM was emitted before generation completed.

Features:

- true incremental generation; no completed-WAV slicing
- Qwen3-TTS Base voice cloning, including ICL and speaker-embedding-only modes
- CustomVoice named speakers and instructions
- VoiceDesign natural-language instructions
- Q8_0, BF16, and native GGUF packages already supported by audio.cpp
- CPU and CUDA through the existing ggml backend
- HTTP SSE or raw chunked s16le PCM at `POST /v1/audio/speech`
- bounded memory by default and synchronous backpressure
- prompt/reference caching, cancellation, long-text segmentation, TTFA/RTF metrics

## Build

Linux CPU:

```bash
cmake -S . -B build \
  -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS=qwen3_tts \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

NVIDIA CUDA:

```bash
cmake -S . -B build \
  -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS=qwen3_tts \
  -DENGINE_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Add `-DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_EXAMPLES=ON
-DENGINE_BUILD_WARMBENCH=ON` to build the unit test, live player, and latency
benchmark.

## Model packages

Use the existing audio.cpp model catalog; no new conversion is required:

```bash
python3 tools/model_manager_v2.py install qwen3_tts
```

Specific installable ids include `qwen3_tts_1_7b_base_q8_0`,
`qwen3_tts_1_7b_base_bf16`, `qwen3_tts_1_7b_customvoice_q8_0`, and
`qwen3_tts_1_7b_voicedesign_q8_0`. A standalone GGUF file or its extracted
directory can be passed to `--model`.

## CLI

Base voice clone, streaming events with a completed WAV explicitly requested:

```bash
build/bin/audiocpp_cli \
  --task tts --mode streaming --family qwen3_tts \
  --model models/Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf \
  --backend cuda \
  --text "You should hear the first chunk while this sentence is still being generated." \
  --voice-ref reference.wav --reference-text "The exact reference transcript." \
  --chunk-frames 2 --decoder-context-frames 25 \
  --stream-accumulate true --out output.wav --metrics
```

Use `--text-file long.txt` for long input. Without `--stream-accumulate true`,
PCM chunks are released after the event consumer handles them and the complete
waveform is not retained in RAM.

Speaker-embedding-only clone:

```bash
build/bin/audiocpp_cli --task tts --mode streaming --family qwen3_tts \
  --model /path/to/base.gguf --backend cuda --text "Hello" \
  --voice-ref reference.wav --speaker-embedding-only true --chunk-frames 2
```

CustomVoice:

```bash
build/bin/audiocpp_cli --task tts --mode streaming --family qwen3_tts \
  --model /path/to/customvoice.gguf --backend cuda --text "Hello" \
  --speaker Vivian --instruct "Warm, restrained, and conversational." --chunk-frames 2
```

VoiceDesign:

```bash
build/bin/audiocpp_cli --task vdes --mode streaming --family qwen3_tts \
  --model /path/to/voicedesign.gguf --backend cuda --text "Hello" \
  --instruct "A low, calm adult narrator with a slight smile." --chunk-frames 2
```

For unmistakable live playback on Linux (float PCM is piped to `aplay` as each
event arrives):

```bash
build/bin/qwen3_tts_live --model /path/to/base.gguf \
  --text "Playback starts before synthesis finishes." \
  --reference-audio reference.wav --reference-text "The exact transcript." \
  --backend cuda --chunk-frames 2
```

Pass `--player-command "..."` for another raw-f32 player.

## HTTP streaming

`server.json`:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "models": [
    {
      "id": "qwen3-tts",
      "family": "qwen3_tts",
      "path": "/absolute/path/to/base.gguf",
      "task": "tts",
      "mode": "streaming"
    }
  ]
}
```

```bash
build/bin/audiocpp_server --config server.json
```

Raw chunked s16le PCM (24 kHz, mono):

```bash
curl --no-buffer http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3-tts",
    "input":"Hello from true Qwen streaming.",
    "voice_ref":"/absolute/path/to/reference.wav",
    "reference_text":"The exact reference transcript.",
    "response_format":"pcm",
    "stream_format":"audio",
    "stream":true,
    "chunk_frames":2
  }' > speech.s16le
```

Set `"stream_format":"sse"` to receive base64 `speech.audio.delta` events.
The producer invokes the network sink synchronously, so a slow socket applies
backpressure rather than growing an unbounded queue. A write failure/client
disconnect unwinds generation immediately and leaves the loaded model reusable.

The embedded WebUI uses the raw PCM transport when its selected Qwen3-TTS model
is configured with `mode=streaming`. It schedules each received 24 kHz mono PCM
chunk through the browser audio context immediately, then wraps the accumulated
PCM in a WAV only after generation so the result can still be replayed or saved.
For configured quick-start voices, selecting a voice and pausing briefly while
editing text sends a two-frame warmup. This keeps the selected CUDA graph and
exact text prefill hot before Generate is pressed and prepares a deterministic
160 ms PCM prefix; the real request still uses the configured model, full
reference, sampling options, and decoder context unchanged.

## Latency tuning

| `chunk_frames` | Tradeoff |
|---:|---|
| 1 | lowest algorithmic latency, highest decoder overhead |
| 2 | recommended real-time default, about 160 ms of speech |
| 4 | balanced throughput and update rate |
| 8 | throughput-oriented |

`decoder_context_frames=25` matches the established offline decoder's left
context. Lower values are accepted but can reduce boundary quality.

### Quality-preserving low-latency launch on Windows

After building the `windows-cuda-release` preset and installing the 1.7B Base
Q8 model, run:

```powershell
scripts\start_qwen3_tts_low_latency.ps1 -Restart
```

The launcher keeps the 1.7B model resident, loads all bundled voice WAVs and
UTF-8 transcripts once at startup, caches converted host projection weights,
uses all configured CPU threads for bit-identical prompt projection, retains
four voice prompts and four prompt-shape graphs, uses the full 25-frame
speech-decoder context, and warms the CUDA streaming path for all four embedded
demo voices. It finishes with `demo_1_man` active; choosing another WebUI voice
prewarms that voice in the background. For a fixed seed, the WebUI also prepares
and retains the exact first two codec frames for the current text. Generate can
schedule that verified 160 ms PCM prefix immediately while the identical full
request catches up, then byte-check and remove the duplicate prefix from live
playback. The saved WAV still comes entirely from the full server response. It
defaults to one codec frame per PCM event and raw PCM transport; these change
delivery latency without changing the model, reference conditioning, sampling
settings, or emitted waveform. The companion `.cmd` file starts or reuses the
server and opens the WebUI.

On WDDM, the launcher also enables a low-priority CUDA heartbeat on one SM
(`20 ms` work / `1 ms` rest by default). This prevents the RTX 3090 from
dropping to its P8 idle clock between requests, which otherwise adds a large
wake-up penalty to the first CUDA pass. It does not touch model tensors or the
generation stream, but it intentionally increases idle GPU power. Use
`-CudaKeepaliveMs 0` to disable it, or tune `-CudaKeepaliveWorkMs` and
`-CudaKeepaliveMs` for a different latency/power tradeoff.

For another resident deployment, set matching session options in `server.json`:

```json
{
  "qwen3_tts.mem_saver": "false",
  "qwen3_tts.voice_prompt_cache_slots": "4",
  "qwen3_tts.prefill_graph_cache_slots": "4"
}
```

## Tests and benchmark

```bash
ctest --test-dir build -R qwen3_tts_stream_decoder_test --output-on-failure
build/bin/qwen3_tts_stream_latency --model /path/to/base.gguf \
  --reference-audio reference.wav --reference-text "The exact transcript." \
  --backend cuda
```

The deterministic unit test covers chunk sizes 1/2/4/8, flush behavior,
reference trimming, and duplicate/dropped boundary samples. The model benchmark
reports only measurements obtained on the current machine: model load time,
TTFA, codec frames/s, decoder ms/chunk, audio seconds/s, RTF, peak RSS where
available, and the first-PCM-before-generation-end assertion.

## Current limitations

- The decoder uses bounded 25-frame re-decoding rather than a native causal
  convolution-state cache. This is correct and continuous but spends more work
  per chunk than a future stateful decoder could.
- Streaming transport is HTTP SSE or raw chunked PCM; WebSocket transport has
  not been added because audio.cpp's current speech server already provides
  backpressure and disconnect cancellation through HTTP streaming.
- Streaming WAV headers are not emitted; use raw PCM for low latency or request
  accumulated output for a completed WAV.
- CUDA talker/decoder overlap is intentionally not enabled; both execute in the
  proven synchronous order.
