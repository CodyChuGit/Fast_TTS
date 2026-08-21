# Voice MCP WebUI

The embedded web page is the front door for this app's single function: making
the configured character — **F** by default — speak. It is compiled from a small
SvelteKit app in `native/` and embedded into `audiocpp_server`, so the running
binary needs no frontend files.

Views:

- **Speak** — a text box, live streaming playback while generation runs, the
  first-audio latency for each request, a WAV download of the last utterance,
  and the MCP connection panel with Open WebUI instructions.
- **Settings** — replace the character: rename it, pick one of the bundled demo
  voices, or clone a custom voice from an uploaded or microphone-recorded
  sample plus its transcript. The character is server state and applies to MCP
  callers too; see [README_VOICE_MCP.md](../README_VOICE_MCP.md).

## Frontend development

Node.js is needed only to modify and rebuild the frontend:

```bash
cd webui/native
npm ci
npm run check
npm run test:stream
npm run build        # writes native/dist, embedded on the next server build
```

The page keeps one interactive audio context for the whole session and feeds it
16-bit PCM chunks as they arrive from `/v1/audio/speech`, so playback starts on
the first decoded frames rather than after the clip completes.
