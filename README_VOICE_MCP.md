# Voice MCP

This app is an **MCP server for audio**. It does one thing: turn text into speech
in the voice of a single character — **F** by default — using the resident,
CUDA-warmed Qwen3-TTS streaming model. Agent platforms such as Open WebUI
connect over the Model Context Protocol and call the same voice the web page
uses.

The web UI is a standalone front door for that one function: a page to make F
speak, a settings page to replace F with another character, and the MCP
connection details. It is no longer a general audio.cpp studio.

## Run

```powershell
scripts\start_qwen3_tts_low_latency.ps1 -Restart
```

Then open **http://127.0.0.1:18080/**. The launcher keeps the 1.7B model
resident, warms the bundled voices, and enables the CUDA keepalive, so a warm
request reaches first audio in well under a second.

## The character

The character is **server state**, persisted in `character/` next to the repo:
the name and voice chosen in Settings apply to the web page, to every MCP call,
and to plain `/v1/audio/speech` requests that name no voice — and they survive
restarts.

- **Default**: `F`, voiced by the bundled `demo_3_woman` preset.
- **Replace with a bundled voice**: Settings → Bundled voice. These are warmed
  at startup, so they keep the lowest latency.
- **Replace with a custom character**: Settings → Custom recording. Upload or
  record 5–15 seconds of clean speech plus its transcript; the voice is cloned
  from it. The first request after saving pays a one-time prompt warmup.

The same operations over HTTP:

```bash
curl http://127.0.0.1:18080/v1/character
curl -X POST http://127.0.0.1:18080/v1/character \
  -H 'Content-Type: application/json' -d '{"name":"F","preset":"demo_3_woman"}'
curl -X POST http://127.0.0.1:18080/v1/character \
  -F name=Nova -F "transcript=Exactly what the sample says." -F file=@voice.wav
```

## MCP

Endpoint: **`http://127.0.0.1:18080/mcp`** — Model Context Protocol over
streamable HTTP (POST one JSON-RPC message per request; responses are plain
JSON, which the transport permits and every client accepts). The server is
stateless: no session ids, `GET`/`DELETE` answer 405.

One tool is exposed:

| Tool | Arguments | Returns |
|---|---|---|
| `speak` | `text` (required), `seed` (optional) | `audio` content (base64 WAV, 24 kHz mono) plus a text summary |

The tool description and server instructions embed the current character's
name, so a connected agent knows whose voice it is invoking. Tool failures come
back as `isError` tool results the agent can read; malformed calls get JSON-RPC
errors. Text is capped at 4000 characters per call — the instructions tell
agents to speak one utterance at a time.

```bash
curl -X POST http://127.0.0.1:18080/mcp -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0","id":1,"method":"tools/call",
  "params":{"name":"speak","arguments":{"text":"Hello from F."}}}'
```

## Open WebUI

Two ways to wire it in, usable together:

1. **As an MCP tool server** — Admin Settings → External Tools → add
   `http://<host>:18080/mcp` as a streamable-HTTP MCP server. Models can then
   call `speak` and attach F's audio to their replies. On an Open WebUI that
   only accepts OpenAPI tool servers, bridge with
   `uvx mcpo --port 8600 --server-type streamable-http -- http://<host>:18080/mcp`
   and add `http://localhost:8600` instead.
2. **As the TTS engine** — Admin Settings → Audio → Text-to-Speech → OpenAI,
   API base `http://<host>:18080/v1`, any API key, any voice name. Every
   read-aloud then uses the character's voice through the OpenAI-compatible
   `/v1/audio/speech` endpoint this server already serves.

If Open WebUI runs in Docker, start this server with `--host 0.0.0.0` (or use
`host.docker.internal`) so the container can reach it.

### On streaming

This server streams: PCM leaves before generation finishes, and the web page
plays it live. But an MCP `tools/call` returns one complete clip, and Open
WebUI's read-aloud sends the finished message text — so today the audible
result starts after the text is done. For spoken replies that begin while the
model is still writing, the **client** needs to stream text into this server —
sentence-sized `speak` calls (or `/v1/audio/speech` requests) issued as the LLM
emits them, played back in order. That is an Open WebUI-side change; this
server is already fast enough for it (a warm sentence reaches first audio in
~0.6–0.8 s and generates about 2× real time).

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | The web page: speak as the character, settings, MCP connection info |
| `POST /mcp` | MCP streamable HTTP (initialize, tools/list, tools/call speak) |
| `GET /v1/character` | The active character |
| `POST /v1/character` | Replace the character (JSON preset or multipart custom voice) |
| `POST /v1/audio/speech` | OpenAI-compatible speech; no `voice` field → the character speaks |
| `GET /health`, `GET /v1/models` | Liveness and model state |

Engine internals — the incremental talker, KV cache, sliding decoder, and the
latency tuning the launcher applies — are documented in
[README_QWEN3_TTS_STREAMING.md](README_QWEN3_TTS_STREAMING.md); none of that
changed.

## Tests

```bash
ctest --test-dir build/windows-cuda-release -R "server_mcp_test|server_character_test"
cd webui/native && npm run check && npm run test:stream
```

The MCP test covers the protocol handshake, tool listing, audio results, and
error classification against a stubbed speech engine; the character test covers
the store's persistence and fallback rules. Neither needs a model.
