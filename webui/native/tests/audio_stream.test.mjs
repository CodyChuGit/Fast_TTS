import assert from 'node:assert/strict';
import { after, afterEach, test } from 'node:test';

import {
  closePcm16Playback,
  encodePcm16BytesWav,
  Pcm16StreamPlayer,
  primePcm16Playback,
  VerifiedPcmPrefixSkipper
} from '../src/lib/audio.ts';

test('encodePcm16BytesWav preserves split PCM and drops an incomplete frame', async () => {
  const wav = encodePcm16BytesWav([
    new Uint8Array([0x01]),
    new Uint8Array([0x02, 0x03, 0x04, 0xff])
  ], 24000, 1);
  const bytes = new Uint8Array(await wav.arrayBuffer());
  const view = new DataView(bytes.buffer);

  assert.equal(wav.type, 'audio/wav');
  assert.equal(wav.size, 48);
  assert.equal(new TextDecoder().decode(bytes.subarray(0, 4)), 'RIFF');
  assert.equal(new TextDecoder().decode(bytes.subarray(8, 12)), 'WAVE');
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 24000);
  assert.equal(view.getUint32(40, true), 4);
  assert.deepEqual([...bytes.subarray(44)], [0x01, 0x02, 0x03, 0x04]);
});

class FakeAudioBuffer {
  constructor(channels, frames, sampleRate) {
    this.numberOfChannels = channels;
    this.length = frames;
    this.sampleRate = sampleRate;
    this.data = Array.from({ length: channels }, () => new Float32Array(frames));
  }

  getChannelData(channel) {
    return this.data[channel];
  }
}

class FakeAudioBufferSourceNode {
  constructor(context) {
    this.context = context;
    this.buffer = null;
    this.onended = null;
    this.stopped = false;
    this.startAt = null;
  }

  connect() {}

  start(startAt) {
    this.startAt = startAt;
    queueMicrotask(() => this.onended?.());
  }

  stop() {
    this.stopped = true;
    queueMicrotask(() => this.onended?.());
  }
}

class FakeAudioContext {
  static latest = null;

  constructor() {
    FakeAudioContext.latest = this;
    this.currentTime = 1;
    this.destination = {};
    this.state = 'suspended';
    this.buffers = [];
    this.sources = [];
  }

  async resume() {
    this.state = 'running';
  }

  createBuffer(channels, frames, sampleRate) {
    const buffer = new FakeAudioBuffer(channels, frames, sampleRate);
    this.buffers.push(buffer);
    return buffer;
  }

  createBufferSource() {
    const source = new FakeAudioBufferSourceNode(this);
    this.sources.push(source);
    return source;
  }

  async close() {
    this.state = 'closed';
  }
}

const originalAudioContext = globalThis.AudioContext;
globalThis.AudioContext = FakeAudioContext;
afterEach(async () => {
  await closePcm16Playback();
});
after(() => {
  if (originalAudioContext === undefined) delete globalThis.AudioContext;
  else globalThis.AudioContext = originalAudioContext;
});

test('Pcm16StreamPlayer decodes PCM split across network reads and drains playback', async () => {
  const player = new Pcm16StreamPlayer(24000, 1);
  await player.start();
  player.push(new Uint8Array([0x00]));
  player.push(new Uint8Array([0x00, 0xff, 0x7f, 0x7b]));
  await player.finish();

  const context = FakeAudioContext.latest;
  assert.equal(context.state, 'running');
  assert.equal(context.buffers.length, 1);
  assert.equal(context.buffers[0].length, 2);
  assert.equal(context.buffers[0].sampleRate, 24000);
  assert.equal(context.buffers[0].data[0][0], 0);
  assert.ok(Math.abs(context.buffers[0].data[0][1] - 32767 / 32768) < 1e-6);
  assert.equal(context.sources.length, 1);
  assert.ok(context.sources[0].startAt >= 1.064);
  assert.ok(context.sources[0].startAt < 1.065);
});

test('Pcm16StreamPlayer schedules later chunks contiguously', async () => {
  const player = new Pcm16StreamPlayer(24000, 1);
  await player.start();
  player.push(new Uint8Array([0x00, 0x00, 0x01, 0x00]));
  player.push(new Uint8Array([0x02, 0x00, 0x03, 0x00]));
  await player.finish();

  const context = FakeAudioContext.latest;
  assert.equal(context.sources.length, 2);
  assert.equal(
    context.sources[1].startAt,
    context.sources[0].startAt + 2 / 24000
  );
});

test('Pcm16StreamPlayer can schedule a prepared prefix without artificial lead', async () => {
  const player = new Pcm16StreamPlayer(24000, 1);
  await player.start();
  player.setInitialPlaybackLeadSeconds(0);
  player.push(new Uint8Array([0x00, 0x00]));
  await player.finish();

  const context = FakeAudioContext.latest;
  assert.equal(context.sources[0].startAt, 1);
});

test('Pcm16StreamPlayer restores its safety lead if prepared playback drains', async () => {
  const player = new Pcm16StreamPlayer(24000, 1);
  await player.start();
  player.setInitialPlaybackLeadSeconds(0);
  player.push(new Uint8Array([0x00, 0x00]));
  const context = FakeAudioContext.latest;
  context.currentTime = 2;
  player.push(new Uint8Array([0x00, 0x00]));
  await player.finish();

  assert.ok(context.sources[1].startAt >= 2.064);
  assert.ok(context.sources[1].startAt < 2.065);
});

test('PCM stream players reuse a primed interactive AudioContext', async () => {
  const primed = await primePcm16Playback();
  const first = new Pcm16StreamPlayer(24000, 1);
  await first.start();
  first.push(new Uint8Array([0x00, 0x00]));
  await first.finish();

  const second = new Pcm16StreamPlayer(24000, 1);
  await second.start();
  second.push(new Uint8Array([0x00, 0x00]));
  await second.finish();

  assert.equal(FakeAudioContext.latest, primed);
  assert.equal(primed.sources.length, 2);
});

test('VerifiedPcmPrefixSkipper verifies and removes a prefix split across reads', () => {
  const skipper = new VerifiedPcmPrefixSkipper(new Uint8Array([1, 2, 3, 4]));
  assert.deepEqual([...skipper.consume(new Uint8Array([1]))], []);
  assert.deepEqual([...skipper.consume(new Uint8Array([2, 3, 4, 5, 6]))], [5, 6]);
  assert.equal(skipper.consumedBytes, 4);
  assert.equal(skipper.complete, true);
});

test('VerifiedPcmPrefixSkipper rejects a non-identical live prefix', () => {
  const skipper = new VerifiedPcmPrefixSkipper(new Uint8Array([1, 2]));
  assert.throws(
    () => skipper.consume(new Uint8Array([1, 9])),
    /differs from the live deterministic stream/
  );
});
