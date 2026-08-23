"""Train the send predictor from collected typing episodes.

Reads data/typing_episodes.jsonl (the raw corpus the WebUI collector
appends), featurizes every keystroke moment, trains a logistic regression
on the automatic label `sent within 1500 ms`, reports the Endpoint
Anticipation metrics (median realized anticipation, premature-fire rate),
and exports calibrated weights to data/send_predictor_weights.json for the
client-side scorer.

Usage:
    python scripts/send_predictor.py            # train on real episodes
    python scripts/send_predictor.py --selftest # synthetic end-to-end check
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HORIZON_MS = 1500.0
FIRE_THRESHOLD = 0.6
FEATURE_NAMES = [
    "pause_ratio",      # current pause / rolling median inter-key interval
    "terminal_punct",   # draft ends with . ! ? (CJK included)
    "clause_punct",     # draft ends with , ; : etc.
    "trailing_space",   # last char is whitespace (word just completed)
    "len_norm",         # draft length / 60, capped at 2
    "burst_keys",       # keystrokes since the last >=2 s pause, /20 capped
    "log_elapsed",      # log10(ms since episode start)
    "backspace_rate",   # deletions / events over the last 5 s
    "slowdown",         # mean(last 3 IKI) / mean(previous 7 IKI), capped
]


def rolling_median(values: list[float], fallback: float = 240.0) -> float:
    if not values:
        return fallback
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def featurize_episode(episode: dict) -> list[tuple[list[float], float, float]]:
    """Yields (features, ms_until_send_or_inf, sample_time_ms) per moment.

    Moments are every keystroke event plus synthetic checkpoints inside long
    pauses (every 250 ms), because "the user has now been quiet for 900 ms"
    is a different state than "quiet for 100 ms" even with no new key.
    """
    events = episode.get("events") or []
    if len(events) < 4:
        return []
    sent = episode.get("outcome") == "sent"
    send_t = float(episode.get("duration_ms", events[-1]["t"]))
    ikis: list[float] = []
    samples = []
    for i in range(1, len(events)):
        ev, prev = events[i], events[i - 1]
        iki = max(0.0, float(ev["t"]) - float(prev["t"]))
        window = [e for e in events[max(0, i - 40):i + 1]]
        recent_5s = [e for e in window if float(ev["t"]) - float(e["t"]) <= 5000.0]
        backspaces = sum(1 for e in recent_5s if e.get("d", 0) < 0)
        burst_keys = 0
        for j in range(i, 0, -1):
            if float(events[j]["t"]) - float(events[j - 1]["t"]) >= 2000.0:
                break
            burst_keys += 1
        med = rolling_median(ikis[-20:])
        last3 = ikis[-3:]
        prev7 = ikis[-10:-3]
        slowdown = (sum(last3) / len(last3)) / max(1.0, sum(prev7) / len(prev7)) if last3 and prev7 else 1.0
        ikis.append(iki)

        # The moment of this keystroke, plus checkpoints through the pause
        # that FOLLOWS it (until the next event or the outcome).
        gap_end = float(events[i + 1]["t"]) if i + 1 < len(events) else send_t
        checkpoints = [0.0]
        pause_at = 250.0
        while float(ev["t"]) + pause_at < gap_end and pause_at <= 3000.0:
            checkpoints.append(pause_at)
            pause_at += 250.0
        for pause in checkpoints:
            now = float(ev["t"]) + pause
            features = [
                min(pause / max(60.0, med), 20.0),
                1.0 if ev.get("k") == "t" else 0.0,
                1.0 if ev.get("k") == "c" else 0.0,
                1.0 if ev.get("k") == "s" else 0.0,
                min(float(ev.get("len", 0)) / 60.0, 2.0),
                min(burst_keys / 20.0, 2.0),
                math.log10(max(now, 1.0)),
                backspaces / max(1, len(recent_5s)),
                min(slowdown, 4.0),
            ]
            until_send = (send_t - now) if sent else math.inf
            samples.append((features, until_send, now))
    return samples


def sigmoid(z: float) -> float:
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def train(samples: list[tuple[list[float], int]], epochs: int = 60,
          lr: float = 0.25) -> list[float]:
    dim = len(FEATURE_NAMES) + 1  # + bias
    w = [0.0] * dim
    n = len(samples)
    positives = sum(label for _, label in samples)
    pos_weight = max(1.0, (n - positives) / max(1, positives))
    rng = random.Random(7)
    order = list(range(n))
    for epoch in range(epochs):
        rng.shuffle(order)
        step = lr / (1.0 + 0.05 * epoch)
        for idx in order:
            features, label = samples[idx]
            z = w[-1] + sum(wi * xi for wi, xi in zip(w, features))
            p = sigmoid(z)
            g = (p - label) * (pos_weight if label else 1.0)
            for j, xj in enumerate(features):
                w[j] -= step * (g * xj + 1e-4 * w[j])
            w[-1] -= step * g
    return w


def score(w: list[float], features: list[float]) -> float:
    return sigmoid(w[-1] + sum(wi * xi for wi, xi in zip(w, features)))


def evaluate(w: list[float], episodes: list[dict]) -> dict:
    leads, premature_fires, sent_count, fired_abandoned = [], 0, 0, 0
    abandoned = 0
    for ep in episodes:
        samples = featurize_episode(ep)
        if not samples:
            continue
        if ep.get("outcome") == "sent":
            sent_count += 1
            send_t = float(ep.get("duration_ms", 0))
            # Realized anticipation: the earliest moment from which the score
            # stays above threshold until the send.
            fired_at = None
            for features, _until, now in samples:
                if score(w, features) >= FIRE_THRESHOLD:
                    if fired_at is None:
                        fired_at = now
                else:
                    fired_at = None
            if fired_at is not None:
                leads.append(send_t - fired_at)
            # Premature: fired more than 4 s before the send at any point.
            if any(score(w, f) >= FIRE_THRESHOLD and (send_t - now) > 4000.0
                   for f, _u, now in samples):
                premature_fires += 1
        else:
            abandoned += 1
            if any(score(w, f) >= FIRE_THRESHOLD for f, _u, _now in samples):
                fired_abandoned += 1
    leads.sort()
    return {
        "sent_episodes": sent_count,
        "abandoned_episodes": abandoned,
        "hit_rate": len(leads) / max(1, sent_count),
        "median_realized_anticipation_ms": leads[len(leads) // 2] if leads else 0.0,
        "premature_rate": premature_fires / max(1, sent_count),
        "fired_on_abandoned_rate": fired_abandoned / max(1, abandoned),
    }


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    if not path.exists():
        return episodes
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            episodes.append(json.loads(line)["episode"])
        except (json.JSONDecodeError, KeyError):
            continue
    return episodes


def synthesize_corpus(n: int = 240) -> list[dict]:
    """Two personas with structurally different rhythms, for pipeline
    verification only -- a model trained here proves the machinery, not
    real-world accuracy."""
    rng = random.Random(42)
    episodes = []
    for i in range(n):
        events, t, length = [], 0.0, 0
        decisive = rng.random() < 0.6
        words = rng.randint(4, 14) if decisive else rng.randint(8, 30)
        for w_i in range(words):
            for _ in range(rng.randint(2, 7)):
                t += rng.gauss(140, 40) if decisive else rng.gauss(220, 90)
                length += 1
                events.append({"t": round(max(t, 0), 1), "len": length, "d": 1, "k": "w"})
            t += rng.gauss(220, 60)
            length += 1
            events.append({"t": round(t, 1), "len": length, "d": 1, "k": "s"})
            if not decisive and rng.random() < 0.25:
                t += rng.uniform(1200, 4000)  # mid-thought stare
        sent = rng.random() < (0.9 if decisive else 0.65)
        if sent:
            length += 1
            t += rng.gauss(260, 80)
            events.append({"t": round(t, 1), "len": length, "d": 1, "k": "t"})
            t += rng.uniform(350, 1400)  # the pre-send pause
        episodes.append({
            "outcome": "sent" if sent else "abandoned",
            "duration_ms": round(t),
            "final_text": "x" * length,
            "snapshots": [],
            "events": events,
        })
    return episodes


def main() -> int:
    selftest = "--selftest" in sys.argv
    root = Path(__file__).resolve().parent.parent
    if selftest:
        episodes = synthesize_corpus()
        print(f"selftest corpus: {len(episodes)} synthetic episodes")
    else:
        episodes = load_episodes(root / "data" / "typing_episodes.jsonl")
        print(f"loaded {len(episodes)} collected episodes")
        if len(episodes) < 40:
            print("Not enough real episodes yet (need ~40+). Keep chatting -- "
                  "the collector is recording. Run --selftest to verify the "
                  "pipeline meanwhile.")
            return 0

    rng = random.Random(1)
    rng.shuffle(episodes)
    split = int(len(episodes) * 0.8)
    train_eps, test_eps = episodes[:split], episodes[split:]

    samples = []
    for ep in train_eps:
        for features, until_send, _now in featurize_episode(ep):
            samples.append((features, 1 if until_send <= HORIZON_MS else 0))
    positives = sum(l for _, l in samples)
    print(f"training samples: {len(samples)} ({positives} positive)")
    w = train(samples)

    metrics = evaluate(w, test_eps)
    print("held-out metrics:")
    for key, value in metrics.items():
        print(f"  {key:34s} {value:.3f}" if isinstance(value, float) else f"  {key:34s} {value}")

    out = root / "data" / "send_predictor_weights.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "weights": w[:-1],
        "bias": w[-1],
        "fire_threshold": FIRE_THRESHOLD,
        "horizon_ms": HORIZON_MS,
        "trained_on_episodes": len(train_eps),
        "selftest": selftest,
        "metrics": metrics,
    }, indent=1), encoding="utf-8")
    print(f"weights -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
