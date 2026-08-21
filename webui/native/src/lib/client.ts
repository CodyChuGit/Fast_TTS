// The whole client surface of this app: health, the character, and speaking.
// This UI is a front door to one function, so the client is deliberately small.

export interface ServerHealth {
  status: string;
  backend: string;
  models: number;
}

export interface Character {
  name: string;
  source: 'preset' | 'custom';
  preset?: string;
  transcript?: string;
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

export function setCharacterPreset(name: string, preset: string): Promise<Character> {
  return json<Character>('/v1/character', {
    method: 'POST',
    body: JSON.stringify({ name, preset })
  });
}

export function setCharacterCustom(name: string, transcript: string, voice: Blob): Promise<Character> {
  const form = new FormData();
  form.append('name', name);
  form.append('transcript', transcript);
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
