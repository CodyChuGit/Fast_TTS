function writeAscii(view: DataView, offset: number, text: string) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

// Qwen3-TTS emits one 80 ms codec frame at a time in the lowest-latency mode.
// A 64 ms lead is the smallest practical safety margin that remained continuous
// across the warmed RTX 3090 voice matrix (one codec frame is 80 ms). If
// playback ever drains, the same lead is restored before resuming so late
// chunks do not click.
const INITIAL_PLAYBACK_LEAD_SECONDS = 0.064;
const CONTINUATION_LEAD_SECONDS = 0.01;
// Startup jitter buffer: hold this much audio (three 80 ms codec frames)
// before the first sample plays, or start anyway once the deadline passes.
// Warm generation outpaces playback 2-3x, so a reserve established once never
// shrinks; it exists purely to ride out the uneven arrival of the first
// chunks.
const DEFAULT_START_BUFFER_SECONDS = 0.24;
const DEFAULT_START_MAX_WAIT_MS = 350;
// Ceiling for the re-arm lead after repeated mid-stream underruns.
const MAX_UNDERRUN_LEAD_SECONDS = 0.32;

// Keep one interactive output context alive for the page lifetime. Creating,
// resuming, and closing a context around every request adds work directly on
// the click-to-audio path and can force the browser to reopen the device.
let sharedPlaybackContext: AudioContext | null = null;

export async function primePcm16Playback(): Promise<AudioContext> {
  if (!sharedPlaybackContext || sharedPlaybackContext.state === 'closed') {
    sharedPlaybackContext = new AudioContext({ latencyHint: 'interactive' });
  }
  if (sharedPlaybackContext.state !== 'running') await sharedPlaybackContext.resume();
  return sharedPlaybackContext;
}

export async function closePcm16Playback(): Promise<void> {
  const context = sharedPlaybackContext;
  sharedPlaybackContext = null;
  if (context && context.state !== 'closed') await context.close();
}

function pcm16WavHeader(dataBytes: number, sampleRate: number, channels: number): ArrayBuffer {
  const bytes = new ArrayBuffer(44);
  const view = new DataView(bytes);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataBytes, true);
  return bytes;
}

export function encodePcm16BytesWav(
  chunks: Uint8Array[],
  sampleRate: number,
  channels: number
): Blob {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new Error('PCM stream sample rate must be a positive integer.');
  }
  if (!Number.isInteger(channels) || channels <= 0) {
    throw new Error('PCM stream channel count must be a positive integer.');
  }
  const frameBytes = channels * 2;
  const receivedBytes = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const dataBytes = receivedBytes - (receivedBytes % frameBytes);
  const parts: BlobPart[] = [pcm16WavHeader(dataBytes, sampleRate, channels)];
  let remaining = dataBytes;
  for (const chunk of chunks) {
    if (remaining <= 0) break;
    const length = Math.min(remaining, chunk.byteLength);
    const copy = new Uint8Array(length);
    copy.set(chunk.subarray(0, length));
    parts.push(copy.buffer as ArrayBuffer);
    remaining -= length;
  }
  return new Blob(parts, { type: 'audio/wav' });
}

export class Pcm16StreamPlayer {
  private readonly sampleRate: number;
  private readonly channels: number;
  private readonly startBufferSeconds: number;
  private readonly startMaxWaitMs: number;
  private context: AudioContext | null = null;
  private nextStartTime = 0;
  private carry = new Uint8Array(0);
  private sources = new Set<AudioBufferSourceNode>();
  private drainResolver: (() => void) | null = null;
  private stopped = false;
  private initialPlaybackLeadSeconds = INITIAL_PLAYBACK_LEAD_SECONDS;
  private hasScheduledAudio = false;
  private begun = false;
  private pending: AudioBuffer[] = [];
  private pendingSeconds = 0;
  private startTimer: ReturnType<typeof setTimeout> | null = null;
  private underruns = 0;

  constructor(
    sampleRate: number,
    channels = 1,
    options: { startBufferSeconds?: number; startMaxWaitMs?: number } = {}
  ) {
    this.sampleRate = sampleRate;
    this.channels = channels;
    this.startBufferSeconds = options.startBufferSeconds ?? DEFAULT_START_BUFFER_SECONDS;
    this.startMaxWaitMs = options.startMaxWaitMs ?? DEFAULT_START_MAX_WAIT_MS;
  }

  async start(): Promise<void> {
    if (this.context) return;
    this.context = await primePcm16Playback();
    this.nextStartTime = this.context.currentTime;
  }

  setInitialPlaybackLeadSeconds(seconds: number): void {
    if (!Number.isFinite(seconds) || seconds < 0) {
      throw new Error('Initial PCM playback lead must be a non-negative finite number.');
    }
    if (this.begun || this.sources.size) {
      throw new Error('Initial PCM playback lead must be set before the first audio chunk.');
    }
    this.initialPlaybackLeadSeconds = seconds;
  }

  push(chunk: Uint8Array): void {
    if (!chunk.byteLength || this.stopped) return;
    if (!this.context) throw new Error('PCM stream player has not been started.');

    let bytes = chunk;
    if (this.carry.byteLength) {
      bytes = new Uint8Array(this.carry.byteLength + chunk.byteLength);
      bytes.set(this.carry, 0);
      bytes.set(chunk, this.carry.byteLength);
    }

    const frameBytes = this.channels * 2;
    const completeBytes = bytes.byteLength - (bytes.byteLength % frameBytes);
    this.carry = completeBytes < bytes.byteLength ? bytes.slice(completeBytes) : new Uint8Array(0);
    if (!completeBytes) return;

    const frames = completeBytes / frameBytes;
    const audio = this.context.createBuffer(this.channels, frames, this.sampleRate);
    const view = new DataView(bytes.buffer, bytes.byteOffset, completeBytes);
    for (let channel = 0; channel < this.channels; channel += 1) {
      const samples = audio.getChannelData(channel);
      for (let frame = 0; frame < frames; frame += 1) {
        samples[frame] = view.getInt16((frame * this.channels + channel) * 2, true) / 32768;
      }
    }

    // Startup jitter buffer. Generation runs faster than real time, but the
    // FIRST chunks arrive unevenly (prefill tail, graph capture, event
    // batching); starting on the very first 80 ms frame therefore stutters.
    // Hold playback until a few frames are queued -- once that reserve exists,
    // the faster-than-real-time producer only ever grows it. The deadline
    // bounds added latency when a clip is shorter than the reserve.
    if (!this.begun) {
      this.pending.push(audio);
      this.pendingSeconds += frames / this.sampleRate;
      if (this.pendingSeconds >= this.startBufferSeconds) {
        this.beginPlayback();
      } else if (this.startTimer === null) {
        this.startTimer = setTimeout(() => {
          this.startTimer = null;
          if (!this.begun && !this.stopped) this.beginPlayback();
        }, this.startMaxWaitMs);
      }
      return;
    }
    this.scheduleBuffer(audio);
  }

  private beginPlayback(): void {
    if (this.begun || !this.context) return;
    this.begun = true;
    if (this.startTimer !== null) {
      clearTimeout(this.startTimer);
      this.startTimer = null;
    }
    this.nextStartTime = this.context.currentTime + this.initialPlaybackLeadSeconds;
    const queued = this.pending;
    this.pending = [];
    this.pendingSeconds = 0;
    for (const buffer of queued) {
      this.scheduleBuffer(buffer);
    }
  }

  private scheduleBuffer(audio: AudioBuffer): void {
    if (!this.context) return;
    const source = this.context.createBufferSource();
    source.buffer = audio;
    source.connect(this.context.destination);
    source.onended = () => {
      this.sources.delete(source);
      if (!this.sources.size && this.drainResolver) {
        const resolve = this.drainResolver;
        this.drainResolver = null;
        resolve();
      }
    };
    this.sources.add(source);

    const floorLead = this.hasScheduledAudio ? CONTINUATION_LEAD_SECONDS : 0;
    let startAt = Math.max(this.nextStartTime, this.context.currentTime + floorLead);
    if (this.hasScheduledAudio && this.nextStartTime <= this.context.currentTime) {
      // A real underrun: the stream drained mid-playback. Re-arm with a lead
      // that grows on every repeat, so persistent jitter converges on a buffer
      // deep enough to absorb it instead of clicking at every chunk.
      this.underruns += 1;
      const lead = Math.min(
        INITIAL_PLAYBACK_LEAD_SECONDS * (1 + this.underruns),
        MAX_UNDERRUN_LEAD_SECONDS
      );
      startAt = this.context.currentTime + lead;
    }
    source.start(startAt);
    this.nextStartTime = startAt + audio.length / this.sampleRate;
    this.hasScheduledAudio = true;
  }

  async finish(): Promise<void> {
    if (!this.context || this.stopped) return;
    this.carry = new Uint8Array(0);
    // A clip shorter than the startup reserve never crossed the threshold;
    // play what was gathered.
    if (!this.begun && this.pending.length) {
      this.beginPlayback();
    }
    if (this.startTimer !== null) {
      clearTimeout(this.startTimer);
      this.startTimer = null;
    }
    if (this.sources.size) {
      await new Promise<void>((resolve) => {
        this.drainResolver = resolve;
      });
    }
    this.releaseContext();
  }

  async stop(): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    this.carry = new Uint8Array(0);
    this.pending = [];
    this.pendingSeconds = 0;
    if (this.startTimer !== null) {
      clearTimeout(this.startTimer);
      this.startTimer = null;
    }
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        // The source may already have ended between cancellation and cleanup.
      }
    }
    this.sources.clear();
    if (this.drainResolver) {
      const resolve = this.drainResolver;
      this.drainResolver = null;
      resolve();
    }
    this.releaseContext();
  }

  private releaseContext(): void {
    this.context = null;
    this.stopped = true;
  }
}

// Removes an already-played deterministic prefix from the live response while
// checking every byte. Network reads may split the prefix at arbitrary points.
export class VerifiedPcmPrefixSkipper {
  private offset = 0;
  private readonly prefix: Uint8Array;

  constructor(prefix: Uint8Array) {
    if (!prefix.byteLength) throw new Error('PCM playback prefix must not be empty.');
    this.prefix = prefix;
  }

  consume(chunk: Uint8Array): Uint8Array {
    if (!chunk.byteLength || this.complete) return chunk;
    const count = Math.min(chunk.byteLength, this.prefix.byteLength - this.offset);
    for (let index = 0; index < count; index += 1) {
      if (chunk[index] !== this.prefix[this.offset + index]) {
        throw new Error('Prepared PCM prefix differs from the live deterministic stream.');
      }
    }
    this.offset += count;
    return chunk.subarray(count);
  }

  get complete(): boolean {
    return this.offset === this.prefix.byteLength;
  }

  get consumedBytes(): number {
    return this.offset;
  }
}

export function encodePcm16Wav(buffer: AudioBuffer): Blob {
  const channels = buffer.numberOfChannels;
  const frames = buffer.length;
  const bytes = new ArrayBuffer(44 + frames * channels * 2);
  const view = new DataView(bytes);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + frames * channels * 2, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, frames * channels * 2, true);

  const samples = Array.from({ length: channels }, (_, channel) => buffer.getChannelData(channel));
  let offset = 44;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, samples[channel][frame]));
      view.setInt16(offset, sample < 0 ? sample * 32768 : sample * 32767, true);
      offset += 2;
    }
  }
  return new Blob([bytes], { type: 'audio/wav' });
}

export async function concatenateAudioBlobs(blobs: Blob[]): Promise<Blob> {
  if (!blobs.length) throw new Error('No audio chunks were generated.');
  if (blobs.length === 1) return blobs[0];

  const context = new AudioContext();
  try {
    const decoded = await Promise.all(blobs.map(async (blob) =>
      context.decodeAudioData(await blob.arrayBuffer())));
    const sampleRate = decoded[0].sampleRate;
    const channels = decoded[0].numberOfChannels;
    for (const chunk of decoded) {
      if (chunk.sampleRate !== sampleRate || chunk.numberOfChannels !== channels) {
        throw new Error('Generated chunks use different audio formats and cannot be joined.');
      }
    }
    const totalFrames = decoded.reduce((sum, chunk) => sum + chunk.length, 0);
    const merged = context.createBuffer(channels, totalFrames, sampleRate);
    let offset = 0;
    for (const chunk of decoded) {
      for (let channel = 0; channel < channels; channel += 1) {
        merged.copyToChannel(chunk.getChannelData(channel), channel, offset);
      }
      offset += chunk.length;
    }
    return encodePcm16Wav(merged);
  } finally {
    await context.close();
  }
}

export async function browserDecodeToWav(file: File, targetSampleRate?: number): Promise<Blob> {
  if (!targetSampleRate && (file.type === 'audio/wav' || file.name.toLowerCase().endsWith('.wav'))) {
    return file;
  }
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    if (!targetSampleRate || decoded.sampleRate === targetSampleRate) {
      return encodePcm16Wav(decoded);
    }
    const frames = Math.max(1, Math.ceil(decoded.duration * targetSampleRate));
    const offline = new OfflineAudioContext(decoded.numberOfChannels, frames, targetSampleRate);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    return encodePcm16Wav(await offline.startRendering());
  } finally {
    await context.close();
  }
}
