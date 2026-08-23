# Predicting the send before it happens

Research synthesis (2026-08-23) for the send predictor: fire speculative LLM
work while the user is still typing, so the reply is ready at send time.

## The four literatures this assembles

1. **Writing-process keystroke research** — pause duration scales with
   linguistic boundary size (within-word < between-word < sentence <
   paragraph; Leijten & Van Waes 2013). A pause after a sentence-final token
   is a categorically different signal than a mid-word pause. Bursts =
   typing runs between pauses ≥2 s. Per-user baseline inter-key interval
   varies 3-4x across people (136M-keystroke CHI'18 dataset), so pause
   features must be normalized to the user's own rolling IKI.
2. **Turn-taking / endpointing** (the speech version of this exact problem):
   timing-only endpointing is decisively beaten by timing + linguistic
   completeness (TurnGPT, LiveKit turn detector — a 0.5B LM scoring
   P(utterance complete); Pipecat Smart Turn). **Endpoint Anticipation**
   (arXiv 2606.13450) is the template for the whole loop: forecast the
   endpoint at 320-2560 ms horizons, speculatively launch LLM + TTS, verify
   or discard. Result: 1195→690 ms response latency (505 ms saved) at 28.4%
   wasted compute. Its metrics are ours: **Median Realized Anticipation**,
   **Premature Anticipation Rate**, **Expected Redundant Computation**.
3. **Industry prefetch triggering** — Google Instant patent US8706750B2:
   fast typing SUPPRESSES speculation (specific intent, still going); the
   pause is the trigger; fire at language boundaries; separate cheap
   speculation from visible commitment. Chrome Speculation Rules eagerness
   tiers (immediate/eager/moderate/conservative) = cheap action at low
   confidence, expensive action only at high confidence, capped in-flight
   with FIFO cancel, waste-based throttling. instant.page: hover 65 ms =
   P(click) 0.5, worth ~300 ms of lead. Smart Compose: pick a target
   trigger RATE (an FP budget), derive the threshold from it.
4. **Direct prior art** — IBM US10395658B2 (speculative pipeline on partial
   utterances, ~1 s pause = completion, cost-benefit per branch); Theai
   US11960983B1 (LLM prefetch on partial input, serve if discrepancy below
   threshold); PredGen arXiv 2506.15556 (~2x latency cut using input-time
   idle GPU).

## Feature set (evidence-ranked)

1. Current pause / rolling median IKI (the strongest single signal)
2. Draft completeness: terminal punctuation class; no dangling
   conjunction/preposition; (later) P(end) from an LM over the draft
3. Draft length vs the user's own sent-length distribution
4. Burst structure: keys in current burst, time since burst start
5. Recent edits: backspace rate, mid-text editing vs appending
6. IKI slowdown trend over the last 5-10 keys (typing decelerates
   approaching completion)
7. Send-affordance micro-behaviors (hover over send ≈ 300 ms of free lead)

## Architecture decisions for this repo

- **Labels are automatic**: every keystroke gets `sent_within_1500ms`
  retroactively from the episode outcome. The collector
  (`/v1/telemetry/typing` → `data/typing_episodes.jsonl`, commit 97a392e)
  records raw events precisely so features can evolve offline.
- **Model**: logistic regression over ~10 features first (per-keystroke
  cost: microseconds in JS); tiny GBDT only if LR plateaus. Evidence says
  frozen-simple beats fancy at this task size.
- **Trainer**: `scripts/send_predictor.py` — featurizes episodes, trains,
  reports MRA / premature-rate / redundancy, exports weights JSON for the
  client.
- **Firing policy** (to implement client-side once a model exists):
  - Tier 1, cheap+continuous: incremental draft prefill on word boundaries
    (rollback-capable models only — orca pays full passes, so its tiers
    stay conservative).
  - Tier 2, p≈0.5: full-context prefill.
  - Tier 3, p≈0.8: whole-reply speculation (+ early TTS on send-shaped).
  - Schmitt trigger: after a cancel, re-arm only below a lower threshold.
  - One in-flight speculation, newest wins (already the pipeline's law).
  - Threshold calibrated to a chosen trigger rate, then adapted online from
    fired→sent vs fired→kept-typing outcomes (bandit-style, ICASSP'23).

## Expected numbers (grounded in the sources)

0.5-2 s of usable lead per message; speech analog realized a median 640 ms
at 28% redundant compute. Personal calibration should beat that: typed text
is noise-free and one user's rhythms are consistent.
