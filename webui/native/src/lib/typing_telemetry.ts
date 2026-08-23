// Typing-episode recorder: the raw material for the send predictor.
//
// An episode spans from the first keystroke into an empty input to the moment
// the draft is sent (outcome "sent") or abandoned (cleared / navigated away,
// outcome "abandoned"). Every input event is recorded RAW -- timestamp, draft
// length, what changed -- rather than as derived features, so the offline
// feature pipeline can evolve without re-instrumenting the client or
// invalidating previously collected data.
//
// Draft text is recorded only in snapshots (first 200 chars) at edit events;
// this is a local single-user app and the corpus feeds a local predictor.

interface TypingEvent {
  // ms since the episode began (performance.now() deltas, sub-ms precision).
  t: number;
  // Draft length after this event.
  len: number;
  // Net character delta (+1 typing, negative for deletions/cuts, large + for paste).
  d: number;
  // Classification of the newest trailing character after the event:
  // w=word char, s=space, t=terminal punct (.!?。！？), c=comma-ish pause
  // punct, o=other, e=empty.
  k: string;
}

export interface TypingEpisode {
  started_unix_ms: number;
  outcome: 'sent' | 'abandoned';
  // Total wall time from first keystroke to outcome, ms.
  duration_ms: number;
  // The draft at outcome time (what was actually sent, or what was thrown away).
  final_text: string;
  // Snapshots of the draft at coarse checkpoints (every 16 events), letting
  // offline analysis reconstruct linguistic state without per-key text logs.
  snapshots: Array<{ t: number; text: string }>;
  events: TypingEvent[];
}

function classify(text: string): string {
  if (!text.length) return 'e';
  const ch = text[text.length - 1];
  if ('.!?。！？'.includes(ch)) return 't';
  if (',;:，；：、—-'.includes(ch)) return 'c';
  if (ch === ' ' || ch === '\n' || ch === '\t') return 's';
  if (/[\p{L}\p{N}]/u.test(ch)) return 'w';
  return 'o';
}

export class TypingRecorder {
  private events: TypingEvent[] = [];
  private snapshots: Array<{ t: number; text: string }> = [];
  private episodeStart = 0;
  private startedUnixMs = 0;
  private lastLen = 0;
  private active = false;

  // Call on every input event with the current draft text.
  record(text: string): void {
    const now = performance.now();
    if (!this.active) {
      if (!text.length) return;
      this.active = true;
      this.episodeStart = now;
      this.startedUnixMs = Date.now();
      this.lastLen = 0;
      this.events = [];
      this.snapshots = [];
    }
    const t = now - this.episodeStart;
    this.events.push({
      t: Math.round(t * 10) / 10,
      len: text.length,
      d: text.length - this.lastLen,
      k: classify(text)
    });
    this.lastLen = text.length;
    if (this.events.length % 16 === 1) {
      this.snapshots.push({ t: Math.round(t), text: text.slice(0, 200) });
    }
    // A dead-man bound: a pathological episode (bot, stuck key) stops
    // growing instead of ballooning the payload.
    if (this.events.length > 4000) this.active = false;
  }

  // The draft was sent. Flush the episode with its final text.
  sent(finalText: string): void {
    this.finish('sent', finalText);
  }

  // The draft was discarded without sending (cleared, page hidden).
  abandoned(finalText: string): void {
    if (this.active && finalText.length) this.finish('abandoned', finalText);
    this.active = false;
  }

  private finish(outcome: 'sent' | 'abandoned', finalText: string): void {
    if (!this.active || !this.events.length) {
      this.active = false;
      return;
    }
    this.active = false;
    const episode: TypingEpisode = {
      started_unix_ms: this.startedUnixMs,
      outcome,
      duration_ms: Math.round(performance.now() - this.episodeStart),
      final_text: finalText.slice(0, 500),
      snapshots: this.snapshots,
      events: this.events
    };
    this.events = [];
    this.snapshots = [];
    // keepalive lets the flush survive a page unload race; failures are
    // silently dropped -- telemetry must never break the app.
    void fetch('/v1/telemetry/typing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(episode),
      keepalive: true
    }).catch(() => undefined);
  }
}
