import type { LoadedModel, ServerHealth } from './types';
import { encodePcm16BytesWav } from './audio';

async function errorFrom(response: Response): Promise<Error> {
  let message = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    message = body?.error?.message || body?.message || message;
  } catch {
    const text = await response.text();
    if (text) message = text;
  }
  return new Error(message);
}

export async function jsonRequest<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal
): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(path, { ...init, headers, signal });
  if (!response.ok) throw await errorFrom(response);
  return response.json() as Promise<T>;
}

export async function health(): Promise<ServerHealth> {
  return jsonRequest<ServerHealth>('/health');
}

export async function models(): Promise<LoadedModel[]> {
  const response = await jsonRequest<{ data: LoadedModel[] }>('/v1/models');
  return response.data;
}

export async function loadModel(body: Record<string, unknown>): Promise<void> {
  await jsonRequest('/v1/models/load', { method: 'POST', body: JSON.stringify(body) });
}

export async function unloadModel(id: string): Promise<void> {
  await jsonRequest('/v1/models/unload', {
    method: 'POST',
    body: JSON.stringify({ id })
  });
}

export async function pathStatus(path: string): Promise<{ exists: boolean; directory: boolean; file: boolean }> {
  return jsonRequest('/v1/ui/path-status', {
    method: 'POST',
    body: JSON.stringify({ path })
  });
}

export interface ModelInstallJob {
  id: string;
  state: 'idle' | 'queued' | 'running' | 'cancelling' | 'cancelled' | 'cleaned' | 'complete' | 'failed';
  message: string;
  exit_code: number;
  downloaded_bytes: number;
  total_bytes: number;
  progress_percent: number;
  started_at_ms: number;
  finished_at_ms: number;
}

export async function installModelPackage(body: { id: string; overwrite?: boolean }): Promise<ModelInstallJob> {
  return jsonRequest('/v1/ui/models/install', {
    method: 'POST',
    body: JSON.stringify(body)
  });
}

export async function stopModelInstall(id: string): Promise<ModelInstallJob> {
  return jsonRequest('/v1/ui/models/install/stop', {
    method: 'POST',
    body: JSON.stringify({ id })
  });
}

export async function cleanPartialModelInstall(id: string): Promise<{ id: string; cleaned: boolean; message: string }> {
  return jsonRequest('/v1/ui/models/clean-partial', {
    method: 'POST',
    body: JSON.stringify({ id })
  });
}

export async function deleteModelPackage(id: string): Promise<{ id: string; removed: boolean; message: string }> {
  return jsonRequest('/v1/ui/models/delete', {
    method: 'POST',
    body: JSON.stringify({ id })
  });
}

export async function modelInstallJobs(): Promise<ModelInstallJob[]> {
  const response = await jsonRequest<{ data: ModelInstallJob[] }>('/v1/ui/models/install-status');
  return response.data;
}

export interface ModelPackageSize {
  id: string;
  size_bytes: number | null;
  state: 'pending' | 'ok' | 'gated' | 'unknown' | 'error';
  message: string;
  installed: boolean;
  version_state: 'not_installed' | 'unknown' | 'up_to_date' | 'update_available';
  local_revision: string;
  remote_revision: string;
}

export interface ModelPackageSizesResponse {
  state: 'idle' | 'running' | 'complete' | 'failed';
  message: string;
  data: ModelPackageSize[];
}

export async function modelPackageSizes(): Promise<ModelPackageSizesResponse> {
  return jsonRequest<ModelPackageSizesResponse>('/v1/ui/models/package-sizes');
}

export interface ModelsRootResponse {
  models_root: string;
  default_models_root: string;
  is_default: boolean;
}

export async function getModelsRoot(): Promise<ModelsRootResponse> {
  return jsonRequest<ModelsRootResponse>('/v1/ui/models-root');
}

export async function setModelsRoot(path = ''): Promise<ModelsRootResponse> {
  return jsonRequest<ModelsRootResponse>('/v1/ui/models-root', {
    method: 'POST',
    body: JSON.stringify({ path })
  });
}

export interface DirectoryBrowserResponse {
  current: string;
  parent: string;
  roots: string[];
  directories: Array<{ name: string; path: string }>;
}

export async function browseDirectories(path = ''): Promise<DirectoryBrowserResponse> {
  return jsonRequest<DirectoryBrowserResponse>('/v1/ui/browse-directories', {
    method: 'POST',
    body: JSON.stringify({ path })
  });
}

export async function availableVoices(model = ''): Promise<string[]> {
  const query = model ? `?model=${encodeURIComponent(model)}` : '';
  const response = await jsonRequest<{ voices: string[] }>(`/v1/audio/voices${query}`);
  return response.voices;
}

export async function uploadWav(blob: Blob, signal?: AbortSignal): Promise<string> {
  const response = await jsonRequest<{ path: string }>('/v1/ui/upload', {
    method: 'POST',
    headers: {
      'Content-Type': 'audio/wav'
    },
    body: blob
  }, signal);
  return response.path;
}

export interface SpeechResponse {
  blob: Blob;
  wallMs: string | null;
  rtf: string | null;
  firstPcmMs?: string | null;
}

export async function speech(
  body: Record<string, unknown>,
  signal?: AbortSignal
): Promise<SpeechResponse> {
  const response = await fetch('/v1/audio/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal
  });
  if (!response.ok) throw await errorFrom(response);
  return {
    blob: await response.blob(),
    wallMs: response.headers.get('X-AudioCPP-Wall-Ms'),
    rtf: response.headers.get('X-AudioCPP-RTF')
  };
}

export interface SpeechStreamOptions {
  sampleRate: number;
  channels: number;
  onPcmChunk?: (chunk: Uint8Array, first: boolean) => void;
}

export async function speechStream(
  body: Record<string, unknown>,
  options: SpeechStreamOptions,
  signal?: AbortSignal
): Promise<SpeechResponse> {
  const started = performance.now();
  const response = await fetch('/v1/audio/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...body,
      response_format: 'pcm',
      stream_format: 'audio',
      stream: true,
      stream_accumulate: false
    }),
    signal
  });
  if (!response.ok) throw await errorFrom(response);
  if (!response.body) throw new Error('Streaming speech response has no readable body.');

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let first = true;
  let firstPcmMs: number | null = null;
  let totalBytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      const chunk = value.slice();
      if (first) firstPcmMs = performance.now() - started;
      chunks.push(chunk);
      totalBytes += chunk.byteLength;
      options.onPcmChunk?.(chunk, first);
      first = false;
    }
  } finally {
    reader.releaseLock();
  }

  if (!totalBytes) throw new Error('Streaming speech response produced no PCM audio.');
  const wallMs = performance.now() - started;
  const audioMs = totalBytes / (options.sampleRate * options.channels * 2) * 1000;
  return {
    blob: encodePcm16BytesWav(chunks, options.sampleRate, options.channels),
    wallMs: wallMs.toFixed(3),
    rtf: audioMs > 0 ? (wallMs / audioMs).toFixed(6) : null,
    firstPcmMs: firstPcmMs?.toFixed(3) || null
  };
}

export async function warmSpeechVoice(
  body: Record<string, unknown>,
  signal?: AbortSignal
): Promise<Uint8Array> {
  const response = await fetch('/v1/audio/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...body,
      response_format: 'pcm',
      stream_format: 'audio',
      stream: true,
      stream_accumulate: false,
      chunk_frames: 1,
      decoder_context_frames: 25,
      // Three tokens produce two complete 80 ms codec frames. Besides warming
      // the exact prompt, the UI can reuse this deterministic 160 ms prefix
      // while the identical foreground request catches up.
      max_tokens: 3
    }),
    signal
  });
  if (!response.ok) throw await errorFrom(response);
  return new Uint8Array(await response.arrayBuffer());
}

export async function transcription(body: Record<string, unknown>, signal?: AbortSignal) {
  return jsonRequest<Record<string, unknown>>('/v1/audio/transcriptions', {
    method: 'POST',
    body: JSON.stringify(body)
  }, signal);
}

export async function runTask(body: Record<string, unknown>, signal?: AbortSignal) {
  return jsonRequest<Record<string, unknown>>('/v1/tasks/run', {
    method: 'POST',
    body: JSON.stringify(body)
  }, signal);
}

export function base64AudioUrl(data: string): string {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
}
