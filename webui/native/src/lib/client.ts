// The whole client surface of this app: health, the character, and speaking.
// This UI is a front door to one function, so the client is deliberately small.

export interface ServerHealth {
  status: string;
  backend: string;
  models: number;
  // True when a llama.cpp chat sidecar is attached.
  llm: boolean;
}

export interface Character {
  name: string;
  source: 'preset' | 'custom';
  preset?: string;
  transcript?: string;
  // Who the character is; the chat LLM's system prompt.
  persona?: string;
  available_presets: string[];
}

async function errorFrom(response: Response): Promise<Error> {
  let message = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    message = body?.error?.message || body?.message || message;
  } catch {
    // Not JSON; the status line is the best we have.
  }
  return new Error(message);
}

async function json<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && typeof init.body === 'string' && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) throw await errorFrom(response);
  return response.json() as Promise<T>;
}

export function health(): Promise<ServerHealth> {
  return json<ServerHealth>('/health');
}

export function character(): Promise<Character> {
  return json<Character>('/v1/character');
}

export function setCharacterPreset(name: string, preset: string, persona: string): Promise<Character> {
  return json<Character>('/v1/character', {
    method: 'POST',
    body: JSON.stringify({ name, preset, persona })
  });
}

// The saved-character library. Every save lands in it, and any entry can be
// re-activated in one click; the active entry is the voice the whole app uses.
export interface SavedCharacterEntry {
  id: string;
  name: string;
  source: 'preset' | 'custom';
  preset?: string;
  active: boolean;
}

export interface CharacterLibrary {
  active_id: string;
  characters: SavedCharacterEntry[];
}

export function listCharacters(): Promise<CharacterLibrary> {
  return json<CharacterLibrary>('/v1/characters');
}

export function activateCharacter(id: string): Promise<Character> {
  return json<Character>('/v1/characters/activate', {
    method: 'POST',
    body: JSON.stringify({ id })
  });
}

export function deleteCharacter(id: string): Promise<CharacterLibrary> {
  return json<CharacterLibrary>('/v1/characters/delete', {
    method: 'POST',
    body: JSON.stringify({ id })
  });
}

// Edits the character while keeping its saved voice -- a rename (and, for a
// custom character, a transcript touch-up) never demands the recording again.
export function updateCharacter(name: string, transcript?: string, persona?: string): Promise<Character> {
  const body: Record<string, string> = { name };
  if (transcript) body.transcript = transcript;
  if (persona !== undefined) body.persona = persona;
  return json<Character>('/v1/character', {
    method: 'POST',
    body: JSON.stringify(body)
  });
}

export function setCharacterCustom(
  name: string,
  transcript: string,
  voice: Blob,
  persona: string
): Promise<Character> {
  const form = new FormData();
  form.append('name', name);
  form.append('transcript', transcript);
  form.append('persona', persona);
  form.append('file', voice, 'voice.wav');
  return json<Character>('/v1/character', { method: 'POST', body: form });
}

export interface SpeakStats {
  firstPcmMs: number | null;
  wallMs: number;
  audioSeconds: number;
}

// Streams PCM from the speech endpoint. No voice is named, so the server's
// character voice applies -- the same voice an MCP caller gets.
export async function speakStream(
  text: string,
  seed: number | null,
  onPcmChunk: (chunk: Uint8Array) => void,
  signal?: AbortSignal
): Promise<{ chunks: Uint8Array[]; stats: SpeakStats }> {
  const started = performance.now();
  const body: Record<string, unknown> = {
    input: text,
    response_format: 'pcm',
    stream_format: 'audio',
    stream: true,
    stream_accumulate: false,
    chunk_frames: 1,
    decoder_context_frames: 25
  };
  if (seed !== null) body.seed = seed;

  const response = await fetch('/v1/audio/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal
  });
  if (!response.ok) throw await errorFrom(response);
  if (!response.body) throw new Error('The speech response has no readable body.');

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let firstPcmMs: number | null = null;
  let totalBytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      if (firstPcmMs === null) firstPcmMs = performance.now() - started;
      const chunk = value.slice();
      chunks.push(chunk);
      totalBytes += chunk.byteLength;
      onPcmChunk(chunk);
    }
  } finally {
    reader.releaseLock();
  }
  if (!totalBytes) throw new Error('The speech request produced no audio.');
  return {
    chunks,
    stats: {
      firstPcmMs,
      wallMs: performance.now() - started,
      // 24 kHz mono 16-bit PCM.
      audioSeconds: totalBytes / 48000
    }
  };
}

// Where an MCP client connects. Derived from the page location so the shown
// URL is correct however the server is bound.
export function mcpEndpoint(origin: string): string {
  return `${origin.replace(/\/+$/, '')}/mcp`;
}
export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatStats {
  first_token_ms: number;
  first_audio_ms: number;
  wall_ms: number;
  audio_seconds: number;
}

export type ChatEvent =
  | { type: 'token'; text: string }
  | { type: 'sentence'; text: string }
  | { type: 'audio'; audio: string }
  | { type: 'error'; message: string }
  | { type: 'done'; stats: ChatStats }
  | { type: 'start' };

export function base64ToBytes(data: string): Uint8Array {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

// Streams a chat turn: the reply arrives as token events while completed
// sentences come back as interleaved PCM audio events, so the character talks
// while still "typing". One SSE connection carries everything.
export async function chatSpeak(
  messages: ChatMessage[],
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch('/v1/chat/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
    signal
  });
  if (!response.ok) throw await errorFrom(response);
  if (!response.body) throw new Error('The chat response has no readable body.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === '[DONE]') continue;
          try {
            onEvent(JSON.parse(payload) as ChatEvent);
          } catch {
            // A malformed frame is dropped; the stream continues.
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
// The roleplay engine: the master prompt every character plays through, and
// the sampling that shapes delivery. {name} and {persona} in the master prompt
// are filled from the active character.
export interface LlmModelOption {
  id: string;
  name: string;
  // False while the model's GGUF has not been downloaded yet.
  installed: boolean;
}

export interface LlmSettings {
  master_prompt: string;
  temperature: number;
  top_p: number;
  repeat_penalty: number;
  max_tokens: number;
  // Grow replies over the conversation toward the max_tokens ceiling.
  length_ramp: boolean;
  // The chat model running now (a registry id), and everything on the menu.
  // Empty model + empty models means the server does not manage the sidecar.
  model: string;
  models: LlmModelOption[];
}

export function getLlmSettings(): Promise<LlmSettings> {
  return json<LlmSettings>('/v1/llm-settings');
}

export function setLlmSettings(settings: Partial<LlmSettings>): Promise<LlmSettings> {
  return json<LlmSettings>('/v1/llm-settings', {
    method: 'POST',
    body: JSON.stringify(settings)
  });
}

export function resetLlmSettings(): Promise<LlmSettings> {
  return json<LlmSettings>('/v1/llm-settings', {
    method: 'POST',
    body: JSON.stringify({ reset: true })
  });
}
