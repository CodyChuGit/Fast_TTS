# Super Fast Voice

A **real-time voice pipeline** for one machine and one job: *talk with a
character, out loud, with the lowest latency a single RTX 3090 can give.*

Speak — your words appear while you say them. Pause — she answers in her own
cloned voice, usually because her reply was already being generated from the
stable half of your sentence *while you were still talking*. Speak over her —
she stops. One server, three resident engines, no cloud, no Python in the
hot path, no framework between the microphone and the GPU.

```
 microphone ──► Silero VAD ──► Qwen3-ASR (streaming re-decode)
                                   │
                     stable prefix │ tentative tail
                                   ▼
                     speculative LLM generation  ◄── prompt cache, coalesced warms
                       (Gemma-4-26B, llama.cpp)
                                   │  sentences, streamed
                                   ▼
                     Qwen3-TTS 1.7B (cloned character voice)
                                   │  PCM, streamed per codec frame
                                   ▼
                               speakers ──► barge-in cuts everything above
```

## Measured latencies (RTX 3090, warm, end to end)

| Interaction | Result |
|---|---|
| Type, pause 2–3 s, send | her voice in **~150 ms** (reply pre-generated + pre-synthesized) |
| Type and send immediately | **~750 ms** to her voice |
| Speak, stop talking | final transcript in **170–300 ms**, reply follows |
| Words on screen while speaking | stable prefix updates every **~300 ms** |
| Plain TTS (speak endpoint) | first PCM in **~140 ms** |
| Whole stack resident | 23.2 GB VRAM, all three engines warm |

These are honest numbers from the probes in `scripts/` (`real_user_sim.py`
emulates typing users; `voice_probe.py` streams real-time-paced speech), not
best-case singles.

## What makes it fast

Every stage overlaps the next, and everything that can be guessed is guessed:

- **Speculative replies** — while you type (or speak), finished-looking
  drafts trigger full reply generation server-side; the send attaches to the
  in-flight buffer. Keys are punctuation-tolerant, so `"…tonight"` speculated
  matches `"…tonight?"` sent. Send-shaped drafts also pre-synthesize the
  first sentence of her *audio*.
- **Streaming STT with a stable/tentative split** — the growing utterance is
  re-decoded continuously; tokens that survive consecutive hypotheses commit
  to a stable prefix (the thing worth speculating on), the rest stays
  visibly tentative. Semantic endpointing holds the turn open when you stop
  on "and…" and commits fast when you sound finished; a quiet-tail decode
  usually has the final transcript computed *before* the VAD fires.
- **A TTS engine tuned to its floor** — per-voice prompt caching, GPU-side
  KV state transfer (no host round trips), pinned generation seed so the
  character always sounds like herself. Sentence-by-sentence synthesis rides
  ahead of playback.
- **An LLM sidecar that never blocks the send** — prompt-cache prewarms are
  coalesced (newest wins, one in flight, generations take priority),
  speculation streams are serialized, and zombie requests that lose races
  are dropped before they reach the GPU. Flash attention + q8 KV bought the
  VRAM back for more resident experts.
- **Character hygiene** — replies are scrubbed before the client, the TTS,
  or the next prompt sees them; per-turn steering rides inside the newest
  message so the prompt cache stays hot.

## Models

| Role | Model | Footprint |
|---|---|---|
| Reply generation | Gemma-4-26B-A4B (QAT, Q4_K_XL) via managed [llama.cpp](https://github.com/ggml-org/llama.cpp) sidecar | ~13 GB (MoE experts split CPU/GPU) |
| Character voice | Qwen3-TTS 1.7B, 12.5 Hz codec, Q4 mix | ~2 GB |
| Ears | Qwen3-ASR 0.6B, Q8_0 + bundled Silero VAD | ~1.5 GB (lazy) |

The character is defined by a reference recording + persona; every saved
character keeps its voice and can be re-activated in one click.

## Run

```powershell
scripts/start.ps1        # builds nothing; launches TTS server + llama sidecar
# UI at http://127.0.0.1:8080  — Speak · Live · Voice · Chat · Settings
```

Building from source (CUDA):

```powershell
cmake --preset windows-cuda-release
cmake --build build/windows-cuda-release --target audiocpp_server
cd webui/native && npm install && npm run build   # embedded on next build
```

## Surface

| Endpoint | What |
|---|---|
| `POST /v1/chat/speak` | chat turn → SSE of tokens + sentences + PCM audio (`prewarm`/`speculate` flags drive the latency machinery) |
| `POST /v1/voice/live` | chunked 16 kHz PCM in → SSE transcript events out, one connection |
| `POST /v1/voice/sessions` (+`/audio`, `/events`, `/stop`) | the same events for browsers |
| `POST /v1/audio/speech` | plain TTS in the character voice |
| `/mcp` | MCP (streamable HTTP): agents call `speak` and get WAV back |
| `/v1/character`, `/v1/characters` | the character library |
| `/v1/llm-settings` | master prompt + sampling, persisted |

Voice tuning rides on the query string:
`min_silence_ms`, `endpoint_hold_ms`, `endpoint_hold_incomplete_ms`,
`partial_interval_ms`, `stable_hypothesis_count`, `language`, …

## Tools

- `scripts/real_user_sim.py` — five typing personas with real client timing;
  the regression gate for anything touching the send path
- `scripts/voice_probe.py` — streams a spoken utterance at real-time pace,
  prints the event timeline and stop→final latency
- `scripts/latency_matrix.py` — A/B harness across the latency strategies
- `scripts/filler_qa.py` — multi-seed synthesis lottery that curates any
  pre-rendered clip library by waveform scoring

## Heritage

The synthesis/recognition engine grew out of the excellent
[audio.cpp](https://github.com/0xShug0/audio.cpp) architecture (ggml-based,
many model families); replies run through [llama.cpp](https://github.com/ggml-org/llama.cpp).
This repository has since become its own thing: a single-purpose, measured,
end-to-end **real-time conversation pipeline** — the engines are the organs,
the latency architecture is the animal.
