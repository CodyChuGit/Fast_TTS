# Super Fast TTS

A standalone character-voice server for one machine and one job: **talk to a
character, out loud, with the lowest net latency a single RTX 3090 can give**.
Two models run resident side by side —

- **[Peach-2.0-9B-8k-Roleplay](https://huggingface.co/ClosedCharacter/Peach-2.0-9B-8k-Roleplay)**
  (Q8_0, via a managed [llama.cpp](https://github.com/ggml-org/llama.cpp)
  sidecar) writes the character's replies, and
- **Qwen3-TTS 1.7B** (streaming CUDA inference, built on the
  [audio.cpp](https://github.com/0xShug0/audio.cpp) engine) speaks them in the
  character's voice

— and the pipeline between them streams: tokens flow out as they are generated,
each completed sentence is synthesized while the model writes the next one, and
audio starts before the reply is half-written.

Measured on an RTX 3090, warm, end to end from sending a message:

| | first token | first audio |
|---|---:|---:|
| fresh conversation | 69 ms | **689 ms** |
| later turns | 49 ms | **708 ms** |

Plain text-to-speech (no LLM) reaches first audio in ~300–420 ms.

## Run

Build once, then:

```powershell
scripts\start.ps1 -Restart      # launch both servers, warm everything
scripts\stop.ps1                # graceful shutdown, releases the VRAM
```

`start.ps1` preflights free VRAM (~19 GB needed for both models), starts the
llama.cpp sidecar so the 9.4 GB weight load overlaps the TTS warmup, raises
both processes to high priority, warms every bundled voice **and the active
character's voice**, records both PIDs for `stop.ps1`, and opens
**http://127.0.0.1:18080/**. `-SkipLlm` starts speech-only.

## The character

One character is active at a time and it is **server state**: name, voice, and
persona, persisted in `character/` across restarts.

- **Voice** — a bundled demo voice, or cloned from 5–15 s of clean speech plus
  its transcript (Settings warns when a sample is too long or short, plays back
  what the clone is conditioned on, and has a Test button).
- **Persona** — who the character is; it becomes Peach's system prompt, so the
  same character both *is* and *sounds like* itself.
- **Library** — every save is a preset; any saved character activates in one
  click and the whole server switches: the web page, MCP callers, everything.

## Surfaces

| Endpoint | Purpose |
|---|---|
| `GET /` | The app: Speak, Chat, Settings |
| `POST /v1/chat/speak` | One SSE stream: LLM tokens + interleaved character audio |
| `POST /v1/audio/speech` | OpenAI-compatible TTS; no `voice` field → the character speaks |
| `POST /mcp` | Model Context Protocol (streamable HTTP) with a `speak` tool |
| `GET/POST /v1/character`, `/v1/characters*` | The character and its library |
| `GET /health` | Liveness, backend, LLM sidecar state |

## How the latency is made

- The TTS model stays resident with live CUDA graphs, a keepalive against GPU
  downclocking, cached voice prompts, and a 25-frame sliding decoder that emits
  PCM per codec frame — engine details in
  [README_QWEN3_TTS_STREAMING.md](README_QWEN3_TTS_STREAMING.md).
- llama.cpp runs fully on-GPU (`-ngl 99`, flash attention) with prompt caching
  (`cache_prompt` + `--cache-reuse 256`), so a turn's prefill covers only what
  is new — later-turn first tokens land in ~50 ms.
- The sentence segmenter cuts the **first** segment early at a clause boundary
  once ~60 characters have streamed — first audio does not wait for the first
  period — and every later sentence is synthesized while the previous one
  plays. Roleplay `*actions*` display in chat but are stripped from speech.
- Replies default to Peach's card sampling (temperature 0.3, top_p 0.5,
  repetition penalty 1.1) with a voice-sized 160-token ceiling.

## Repository

This began as a fork of [audio.cpp](https://github.com/0xShug0/audio.cpp) and
keeps its engine (`src/`, `include/`, ggml CUDA backends) plus the Qwen3-TTS
streaming work; the general-purpose studio, model manager, and multi-model
surface have been removed. The build compiles exactly one model family:

```powershell
cmake -S . -B build/windows-cuda-release -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=qwen3_tts
cmake --build build/windows-cuda-release --target audiocpp_server
```

llama.cpp binaries live in `build/llama-cpp/` (prebuilt CUDA release) and the
Peach GGUF in `models/Peach-2.0-9B-8k-Roleplay-GGUF/`.

Tests: `ctest --test-dir build/windows-cuda-release -R "server_"` covers the
MCP protocol, the character store and library, the sentence segmenter, and the
LLM stream parsers — none need a model. `cd webui/native && npm run check &&
npm run test:stream` covers the frontend.
