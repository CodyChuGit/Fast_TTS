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
// After a mid-stream underrun the stream re-arms by accumulating REAL audio,
// not by waiting. With a producer below real time (the LLM decoding beside
// the TTS at the start of a chat reply), a fixed wait gains almost nothing
// and drains again at once, turning the first seconds into a chain of gaps.
// The reserve target grows with every repeat; the deadline bounds the pause
// when the producer trickles or the stream is about to end.
const REBUFFER_STEP_SECONDS = 0.16;
const MAX_REBUFFER_SECONDS = 0.48;
const REBUFFER_MAX_WAIT_MS = 800;

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
  private startBufferSeconds: number;
  private startMaxWaitMs: number;
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
  private rebuffering = false;
  private rebufferTargetSeconds = 0;
  private rebufferStepSeconds = REBUFFER_STEP_SECONDS;
  private maxRebufferSeconds = MAX_REBUFFER_SECONDS;
  private rebufferMaxWaitMs = REBUFFER_MAX_WAIT_MS;
  // Caption sync: each scheduled buffer records where it starts on the
  // context clock and where it falls in the pushed PCM stream, so the UI can
  // ask which stream second is at the speakers right now.
  private segments: Array<{ ctxStart: number; streamStart: number; seconds: number }> = [];
  private scheduledSeconds = 0;
  private receivedSeconds = 0;
  private lastPlayedSeconds = 0;

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

  // Deepen the startup jitter reserve before any audio has been pushed --
  // used when the producer announces it will pace slowly (a slow-decoding
  // LLM feeding the TTS), where one deeper hold beats many small underruns.
  // Paced mode also raises the rebuffer profile: a slow producer's long
  // replies accumulate a thin-margin deficit, and the default 480 ms
  // ceiling (tuned for fast producers) re-drains within seconds.
  setStartBuffer(seconds: number, maxWaitMs: number): void {
    if (!Number.isFinite(seconds) || seconds < 0) {
      throw new Error('PCM start reserve must be a non-negative finite number.');
    }
    if (this.begun || this.sources.size) {
      throw new Error('PCM start reserve must be set before the first audio chunk.');
    }
    this.startBufferSeconds = seconds;
    this.startMaxWaitMs = maxWaitMs;
    this.rebufferStepSeconds = 0.35;
    this.maxRebufferSeconds = 1.2;
    this.rebufferMaxWaitMs = 1500;
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
    this.receivedSeconds += frames / this.sampleRate;
    const audio = this.context.createBuffer(this.channels, frames, this.sampleRate);
    const view = new DataView(bytes.buffer, bytes.byteOffset, completeBytes);
    for (let channel = 0; channel < this.channels; channel += 1) {
      const samples = audio.getChannelData(channel);
      for (let frame = 0; frame < frames; frame += 1) {
        samples[frame] = view.getInt16((frame * this.channels + channel) * 2, true) / 32768;
      }
    }

    // A real underrun: the scheduled tail already played out before this
    // chunk arrived. Restarting on the lone chunk would drain again while
    // the producer is slow, so fall back to accumulating a reserve instead.
    if (
      this.begun &&
      !this.rebuffering &&
      this.hasScheduledAudio &&
      this.nextStartTime <= this.context.currentTime
    ) {
      this.underruns += 1;
      this.rebuffering = true;
      this.rebufferTargetSeconds = Math.min(
        this.rebufferStepSeconds * this.underruns,
        this.maxRebufferSeconds
      );
    }

    // Jitter buffer, both at startup and after an underrun. Warm generation
    // runs faster than real time, but the FIRST chunks arrive unevenly
    // (prefill tail, graph capture, the LLM decoding on the same GPU);
    // starting on the very first 80 ms frame therefore stutters. Hold
    // playback until a reserve of audio is queued -- once it exists, a
    // faster-than-real-time producer only ever grows it. The deadline bounds
    // added latency when a clip is shorter than the reserve.
    if (!this.begun || this.rebuffering) {
      this.pending.push(audio);
      this.pendingSeconds += frames / this.sampleRate;
      const targetSeconds = this.begun ? this.rebufferTargetSeconds : this.startBufferSeconds;
      const maxWaitMs = this.begun ? this.rebufferMaxWaitMs : this.startMaxWaitMs;
      if (this.pendingSeconds >= targetSeconds) {
        this.flushPending();
      } else if (this.startTimer === null) {
        this.startTimer = setTimeout(() => {
          this.startTimer = null;
          if (!this.stopped) this.flushPending();
        }, maxWaitMs);
      }
      return;
    }
    this.scheduleBuffer(audio);
  }

  private flushPending(): void {
    if (!this.context) return;
    if (this.startTimer !== null) {
      clearTimeout(this.startTimer);
      this.startTimer = null;
    }
    if (!this.begun) {
      this.begun = true;
      this.nextStartTime = this.context.currentTime + this.initialPlaybackLeadSeconds;
    } else if (this.rebuffering) {
      this.rebuffering = false;
      if (this.nextStartTime <= this.context.currentTime) {
        this.nextStartTime = this.context.currentTime + CONTINUATION_LEAD_SECONDS;
      }
    }
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
    const startAt = Math.max(this.nextStartTime, this.context.currentTime + floorLead);
    source.start(startAt);
    const seconds = audio.length / this.sampleRate;
    this.segments.push({ ctxStart: startAt, streamStart: this.scheduledSeconds, seconds });
    if (this.segments.length > 512) this.segments.splice(0, 256);
    this.scheduledSeconds += seconds;
    this.nextStartTime = startAt + seconds;
    this.hasScheduledAudio = true;
  }

  // The stream position (seconds of pushed PCM) currently at the speakers.
  // Between scheduled buffers (an underrun gap) the position holds at the end
  // of the last played buffer; after the context is released it stays frozen.
  playedSeconds(): number {
    const context = this.context;
    if (!context) return this.lastPlayedSeconds;
    const now = context.currentTime;
    for (let index = this.segments.length - 1; index >= 0; index -= 1) {
      const segment = this.segments[index];
      if (now >= segment.ctxStart) {
        this.lastPlayedSeconds = Math.min(
          segment.streamStart + (now - segment.ctxStart),
          segment.streamStart + segment.seconds
        );
        break;
      }
    }
    return this.lastPlayedSeconds;
  }

  // Seconds of complete PCM received so far, including audio still queued in
  // the jitter buffer -- the caption layer's generation frontier.
  bufferedSeconds(): number {
    return this.receivedSeconds;
  }

  async finish(): Promise<void> {
    if (!this.context || this.stopped) return;
    this.carry = new Uint8Array(0);
    // A clip shorter than the startup reserve never crossed the threshold,
    // and an underrun near the end may hold a final tail; play what was
    // gathered either way.
    if (this.pending.length) {
      this.flushPending();
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
    this.rebuffering = false;
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
    if (this.context) this.playedSeconds();
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
