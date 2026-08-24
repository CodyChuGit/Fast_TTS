# Anticipating the end of an utterance before it happens

Research synthesis (2026-08-23) for the speech side of send prediction:
fire LLM speculation while the user is still speaking, not after the
silence.

## What the field knows

- **Cue conjunction beats any single cue** (Gravano & Hirschberg 2011):
  falling/high-rising final intonation, slowing speech rate, dropping
  intensity, lower pitch, longer phrase duration, and textual completion —
  turn-taking probability rises linearly with how many appear together.
- **Text tells you an end is possible; prosody tells you it is actual**
  (Bögels & Torreira 2015/2021): listeners only anticipate turn ends at
  syntactic completion points that also carry phrase-final lengthening and
  a boundary tone.
- **Continuous prediction models work in production**: Voice Activity
  Projection (VAP, Ekstedt & Skantze) outputs P(who speaks) over a 2 s
  horizon at 50 Hz, 76-85% balanced shift/hold accuracy, 15 ms CPU
  inference (open: VAP-Realtime, MaAI). Krisp's 6M-param model: 0.82
  balanced accuracy vs 0.59 for VAD-silence timers; at a fixed 6% false
  positives it detects shifts 0.4 s faster than a heavy post-silence
  classifier.
- **ASR-native forecasting**: decoder end-of-query token probability
  halved endpoint latency at Google (Chang 2019; Bijwadia 2022: median
  −30.8%); AssemblyAI ships it (end_of_turn_confidence, 160 ms silence
  when confident vs 2400 ms fallback). Hypothesis-stability literature:
  instability lives in the last 1-2 words; prefix age is the dominant
  stability predictor (matches our StableTracker design).
- **Filled pauses are the veto**: near-flat F0 + frozen spectral envelope
  sustained >150 ms identifies um/uh/lengthening in real time at 85/92
  recall/precision with two cheap features (Goto 1999). OpenAI semantic
  VAD extends its timeout on trailing "uhhm" for the same reason.
- **Industry speculation policies**: LiveKit fires LLM speculation on new
  transcripts and gates TTS on turn confirmation (our design too); Vapi
  and OpenAI map P(done) to a variable hold time. Nobody publicly fires
  speculation on prosody BEFORE the silence — that is the open edge.

## Plan for this pipeline (in order of effort)

1. **Partial-growth stall trigger** (no new DSP): while VAD still reports
   speech, if ≥2 consecutive 280 ms re-decodes add no new tokens AND the
   hypothesis looks complete (existing shape check), emit an
   `endpoint_likely` SSE event; the client fires speculation on it. Fires
   one re-decode cycle or more before the VAD-silence + hold path.
2. **Adaptive hold from the same signals** (replace the binary 110/350 ms
   rule with a continuous mapping; stretch the hold on filler evidence).
3. **Filler veto**: flat-F0 + frozen-spectrum detector (YIN pitch, 10 ms
   hop) or ASR filler tokens in the tentative tail; suppresses the trigger
   and stretches the hold.
4. **Optional big step**: run VAP/MaAI on CPU (user mic + TTS output as
   the stereo pair) and use p_now/p_future as the trigger score.

Expected: 300-600 ms earlier speculation per spoken turn at ~5-15% extra
speculative calls; false-positive management identical to the typing side
(one in flight, newest wins, cancel on prefix change).

Full source list in the conversation research report (agent run
2026-08-23); companion doc: send-prediction.md (typing side).
