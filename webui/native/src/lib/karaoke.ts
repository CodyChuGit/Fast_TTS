// Word-level caption timing for streamed TTS. The engine exposes no forced
// alignment, so timing is proportional: each word carries a weight from its
// speakable characters plus a pause weight for trailing punctuation, and a
// word is revealed when the playback clock crosses its share of the clip.
// While the clip is still being generated the duration is a rate estimate;
// the moment the true duration is known the remaining words re-lay onto it,
// so any estimate error is confined to pacing drift mid-clip. The measured
// rate of every finished clip becomes the prior for the next one, which
// converges on the active character's actual speaking speed after one use.

export interface CaptionWord {
  text: string;
  weight: number;
}

const CJK = /[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff66-\uff9f]/;
const SPEAKABLE = /[\p{L}\p{N}]/gu;
// `SPEAKABLE` is global for match-counting; `.test()` needs a stateless copy.
const SPEAKABLE_ONE = /[\p{L}\p{N}]/u;
// Sentence-final punctuation buys the longest pause, clause punctuation a
// shorter one. Closing quotes/brackets after the mark still count.
const FULL_STOP = /[.!?…。！？]["')\]”’]*$/;
const HALF_STOP = /[,;:、，；：—-]["')\]”’]*$/;

const RATE_STORAGE_KEY = 'karaoke.secondsPerWeight';
const DEFAULT_SECONDS_PER_WEIGHT = 0.062;
const MIN_SECONDS_PER_WEIGHT = 0.03;
const MAX_SECONDS_PER_WEIGHT = 0.14;

function clampRate(rate: number): number {
  return Math.min(MAX_SECONDS_PER_WEIGHT, Math.max(MIN_SECONDS_PER_WEIGHT, rate));
}

function loadRate(): number {
  try {
    const stored = Number.parseFloat(localStorage.getItem(RATE_STORAGE_KEY) ?? '');
    if (Number.isFinite(stored)) return clampRate(stored);
  } catch {
    // Storage can be unavailable; the default prior is fine.
  }
  return DEFAULT_SECONDS_PER_WEIGHT;
}

function saveRate(measured: number): void {
  try {
    // Blend rather than overwrite so one odd clip does not swing the prior.
    const blended = clampRate(0.5 * loadRate() + 0.5 * measured);
    localStorage.setItem(RATE_STORAGE_KEY, blended.toFixed(5));
  } catch {
    // Nothing to do; the next clip just keeps the current prior.
  }
}

function wordWeight(token: string): number {
  const speakable = token.match(SPEAKABLE)?.length ?? 0;
  let weight = Math.max(1, speakable);
  if (FULL_STOP.test(token)) weight += 2.8;
  else if (HALF_STOP.test(token)) weight += 1.1;
  return weight;
}

// CJK text has no spaces; each character is close to one syllable, so it is
// split into per-character caption units. Mixed tokens split at the script
// boundary and Latin runs inside them stay whole.
function splitCjk(token: string): string[] {
  const parts: string[] = [];
  let latin = '';
  for (const char of token) {
    if (CJK.test(char)) {
      if (latin) {
        parts.push(latin);
        latin = '';
      }
      parts.push(char);
    } else {
      latin += char;
    }
  }
  if (latin) parts.push(latin);
  // Trailing punctuation folds into the previous unit so pauses land there.
  const merged: string[] = [];
  for (const part of parts) {
    if (merged.length && !SPEAKABLE_ONE.test(part) && !CJK.test(part)) {
      merged[merged.length - 1] += part;
    } else {
      merged.push(part);
    }
  }
  return merged;
}

export function tokenizeCaption(text: string): CaptionWord[] {
  const words: CaptionWord[] = [];
  for (const raw of text.split(/\s+/)) {
    if (!raw) continue;
    const pieces = CJK.test(raw) ? splitCjk(raw) : [raw];
    for (const piece of pieces) {
      words.push({ text: piece, weight: wordWeight(piece) });
    }
  }
  return words;
}

export class CaptionTimeline {
  readonly words: CaptionWord[];
  private readonly startWeights: number[];
  private readonly totalWeight: number;
  private finalSeconds: number | null = null;
  private revealed = 0;

  constructor(text: string) {
    this.words = tokenizeCaption(text);
    this.startWeights = [];
    let cumulative = 0;
    for (const word of this.words) {
      this.startWeights.push(cumulative);
      cumulative += word.weight;
    }
    this.totalWeight = cumulative;
  }

  // The true clip length, once known. Also teaches the rate prior.
  finish(totalSeconds: number): void {
    if (!Number.isFinite(totalSeconds) || totalSeconds <= 0 || !this.totalWeight) return;
    this.finalSeconds = totalSeconds;
    saveRate(totalSeconds / this.totalWeight);
  }

  // Words whose spoken onset the playback clock has passed. Monotonic: a
  // duration correction never hides a word that was already shown.
  update(playedSeconds: number): number {
    if (!this.words.length) return 0;
    const duration = this.finalSeconds ?? this.totalWeight * loadRate();
    if (duration <= 0 || playedSeconds <= 0) return this.revealed;
    const playedWeight = (playedSeconds / duration) * this.totalWeight;
    let count = this.revealed;
    while (count < this.words.length && this.startWeights[count] <= playedWeight) count += 1;
    if (count > this.revealed) this.revealed = count;
    return this.revealed;
  }

  revealAll(): number {
    this.revealed = this.words.length;
    return this.revealed;
  }

  get length(): number {
    return this.words.length;
  }
}
