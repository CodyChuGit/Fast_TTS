// Syllable-level caption timing for streamed TTS. The engine exposes no
// forced alignment, so timing is proportional: every syllable carries a
// weight from its speakable characters (plus a pause weight for trailing
// punctuation), and a syllable is revealed when the playback clock crosses
// its share of the clip. While the clip is still being generated the duration
// is a rate estimate; the moment the true duration is known the remaining
// syllables re-lay onto it, so any estimate error is confined to pacing drift
// mid-clip. The measured rate of every finished clip becomes the prior for
// the next one, which converges on the active character's speaking speed
// after one use. Syllabification is heuristic (vowel groups with a silent-e
// merge) -- plenty for karaoke pacing, not linguistics.

export interface CaptionSyllable {
  text: string;
  weight: number;
}

export interface CaptionWord {
  text: string;
  syllables: CaptionSyllable[];
  weight: number;
  // Flat index of this word's first syllable in the whole caption -- the
  // reveal counter counts syllables, so rendering needs the offset.
  offset: number;
}

const CJK = /[⺀-鿿가-힯豈-﫿ｦ-ﾟ]/;
const SPEAKABLE = /[\p{L}\p{N}]/gu;
// `SPEAKABLE` is global for match-counting; `.test()` needs a stateless copy.
const SPEAKABLE_ONE = /[\p{L}\p{N}]/u;
const LETTER_RUNS = /[a-zÀ-ɏͰ-ϿЀ-ӿ]+|[^a-zÀ-ɏͰ-ϿЀ-ӿ]+/gi;
const LETTER_ONE = /[a-zÀ-ɏͰ-ϿЀ-ӿ]/i;
// One approximate syllable: leading consonants, a vowel group, and trailing
// consonants when the word ends or another consonant cluster follows.
const VOWEL_GROUP = /[^aeiouy]*[aeiouy]+(?:[^aeiouy]*$|[^aeiouy](?=[^aeiouy]))?/gi;
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

function syllabifyRun(run: string): string[] {
  const pieces = run.match(VOWEL_GROUP);
  if (!pieces || pieces.length === 0) return [run];
  // A final "<consonant>e" chunk is usually a silent e ("ti|me", "ho|me");
  // fold it into the syllable before it.
  if (pieces.length > 1 && /^[^aeiouy]?e$/i.test(pieces[pieces.length - 1])) {
    const tail = pieces.pop() as string;
    pieces[pieces.length - 1] += tail;
  }
  return pieces;
}

// Splits one whitespace-delimited token into syllable display chunks.
// Non-letter runs (digits, punctuation, apostrophes already inside the run)
// ride along with the previous syllable so nothing floats alone.
function syllabify(token: string): string[] {
  const chunks = token.match(LETTER_RUNS) ?? [token];
  const out: string[] = [];
  for (const chunk of chunks) {
    if (LETTER_ONE.test(chunk)) {
      out.push(...syllabifyRun(chunk));
    } else if (out.length) {
      out[out.length - 1] += chunk;
    } else {
      out.push(chunk);
    }
  }
  return out;
}

function syllableWeight(text: string): number {
  return Math.max(1, text.match(SPEAKABLE)?.length ?? 0);
}

function buildWord(token: string, offset: number): CaptionWord {
  const syllables: CaptionSyllable[] = syllabify(token).map((text) => ({
    text,
    weight: syllableWeight(text)
  }));
  // The pause lands on the word's last syllable, where the mark is.
  if (FULL_STOP.test(token)) syllables[syllables.length - 1].weight += 2.8;
  else if (HALF_STOP.test(token)) syllables[syllables.length - 1].weight += 1.1;
  return {
    text: token,
    syllables,
    weight: syllables.reduce((sum, syllable) => sum + syllable.weight, 0),
    offset
  };
}

// CJK text has no spaces; each character is close to one syllable, so every
// character becomes its own word unit. Mixed tokens split at the script
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
  let offset = 0;
  for (const raw of text.split(/\s+/)) {
    if (!raw) continue;
    const pieces = CJK.test(raw) ? splitCjk(raw) : [raw];
    for (const piece of pieces) {
      const word = buildWord(piece, offset);
      offset += word.syllables.length;
      words.push(word);
    }
  }
  return words;
}

interface UnitLayout {
  startWeights: number[];
  totalWeight: number;
}

function layoutUnits(words: CaptionWord[]): UnitLayout {
  const startWeights: number[] = [];
  let cumulative = 0;
  for (const word of words) {
    for (const syllable of word.syllables) {
      startWeights.push(cumulative);
      cumulative += syllable.weight;
    }
  }
  return { startWeights, totalWeight: cumulative };
}

export class CaptionTimeline {
  readonly words: CaptionWord[];
  private readonly layout: UnitLayout;
  private finalSeconds: number | null = null;
  private revealed = 0;

  constructor(text: string) {
    this.words = tokenizeCaption(text);
    this.layout = layoutUnits(this.words);
  }

  // The true clip length, once known. Also teaches the rate prior.
  finish(totalSeconds: number): void {
    if (!Number.isFinite(totalSeconds) || totalSeconds <= 0 || !this.layout.totalWeight) return;
    this.finalSeconds = totalSeconds;
    saveRate(totalSeconds / this.layout.totalWeight);
  }

  // Syllables whose spoken onset the playback clock has passed. Monotonic: a
  // duration correction never hides a syllable that was already shown.
  update(playedSeconds: number): number {
    const units = this.layout.startWeights.length;
    if (!units) return 0;
    const duration = this.finalSeconds ?? this.layout.totalWeight * loadRate();
    if (duration <= 0 || playedSeconds <= 0) return this.revealed;
    const playedWeight = (playedSeconds / duration) * this.layout.totalWeight;
    let count = this.revealed;
    while (count < units && this.layout.startWeights[count] <= playedWeight) count += 1;
    if (count > this.revealed) this.revealed = count;
    return this.revealed;
  }

  revealAll(): number {
    this.revealed = this.layout.startWeights.length;
    return this.revealed;
  }
}

// Exact caption timing for the chat stream. The server emits each sentence's
// text BEFORE that sentence's audio chunks, so every received PCM second is
// attributed to a known sentence -- sentence boundaries need no estimation at
// all. Only the syllable pacing inside one sentence is proportional.

interface CaptionSentence {
  words: CaptionWord[];
  layout: UnitLayout;
  start: number;   // stream seconds where this sentence's audio begins
  seconds: number; // audio attributed to it so far
  revealed: number;
}

export interface SentenceCaption {
  index: number;
  words: CaptionWord[];
  visible: number;
}

export class SentenceCaptionTrack {
  private sentences: CaptionSentence[] = [];
  private pushedSeconds = 0;

  addSentence(text: string): void {
    const words = tokenizeCaption(text);
    this.sentences.push({
      words,
      layout: layoutUnits(words),
      start: this.pushedSeconds,
      seconds: 0,
      revealed: 0
    });
  }

  addAudioSeconds(seconds: number): void {
    if (!Number.isFinite(seconds) || seconds <= 0) return;
    // Audio before any sentence event should not happen; keep the stream
    // accounting correct anyway with an implicit empty sentence.
    if (!this.sentences.length) this.addSentence('');
    this.sentences[this.sentences.length - 1].seconds += seconds;
    this.pushedSeconds += seconds;
  }

  // The sentence whose audio is at the speakers, with its reveal count in
  // syllables. A sentence still receiving audio paces against its provisional
  // length; reveal is monotonic per sentence so growth never hides anything.
  captionAt(playedSeconds: number): SentenceCaption | null {
    let index = -1;
    for (let i = 0; i < this.sentences.length; i += 1) {
      if (this.sentences[i].start <= playedSeconds) index = i;
      else break;
    }
    if (index < 0) return null;
    const sentence = this.sentences[index];
    const units = sentence.layout.startWeights.length;
    if (playedSeconds >= sentence.start + sentence.seconds && index < this.sentences.length - 1) {
      sentence.revealed = units;
    } else if (sentence.seconds > 0 && sentence.layout.totalWeight > 0) {
      const playedWeight =
        ((playedSeconds - sentence.start) / sentence.seconds) * sentence.layout.totalWeight;
      let count = sentence.revealed;
      while (count < units && sentence.layout.startWeights[count] <= playedWeight) count += 1;
      if (count > sentence.revealed) sentence.revealed = count;
    }
    return { index, words: sentence.words, visible: sentence.revealed };
  }
}
