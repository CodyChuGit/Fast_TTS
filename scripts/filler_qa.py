# Curates the hesitation-filler library by synthesis lottery: each entry is
# rendered with several seeds through the running server's character voice,
# every candidate's waveform is scored (duration band for the text, single
# vocal gesture, natural decay, no clipping), and the winner is written into
# character/fillers/ under the entry's existing cache filename. Restart the
# server afterwards so it loads the curated clips.
#
#   python scripts/filler_qa.py            # score + regenerate flagged entries
#   python scripts/filler_qa.py --all      # regenerate everything
#   python scripts/filler_qa.py --report   # score current clips, change nothing
#
# The entry list must mirror filler.cpp's library() in order; the mapping from
# cache filename to entry is captured in fillers/mapping.json the first time
# this runs (derived from file creation order of the original build).

import argparse, glob, json, os, sys, time, urllib.request

import numpy as np

SERVER = "http://127.0.0.1:18080"
FILLER_DIR = os.path.join(os.path.dirname(__file__), "..", "character", "fillers")
SR = 24000
THR = 400

# (size, chinese, text) -- keep in exact library() order.
LIBRARY = [
    ("S", 0, "Mmm."), ("S", 0, "Uhh..."), ("S", 0, "Ooh."), ("S", 0, "Hmm?"),
    ("S", 0, "Oh!"), ("S", 0, "Right..."),
    ("M", 0, "Ummm, okay so..."), ("M", 0, "Hmm, let me think..."),
    ("M", 0, "Oh, that? Well..."), ("M", 0, "Yeah, yeah, okay..."),
    ("M", 0, "Mmm, good question..."), ("M", 0, "Wait, let me see..."),
    ("L", 0, "Okay okay, hold on... let me actually think about that for a second..."),
    ("L", 0, "Hmm... that's actually such a good question, give me a sec..."),
    ("L", 0, "Ooh, um, okay... so, how do I put this..."),
    ("L", 0, "Mmm, wait wait... okay, I think I know what I want to say..."),
    ("S", 1, "嗯…"), ("S", 1, "哦？"), ("S", 1, "呃…"), ("S", 1, "嗯哼。"), ("S", 1, "哦——"),
    ("M", 1, "嗯…让我想想哦…"), ("M", 1, "哎呀，这个嘛…"), ("M", 1, "哦，那个呀…"),
    ("M", 1, "嗯嗯，好问题…"), ("M", 1, "等一下哦,我想想…"),
    ("L", 1, "哎呀等一下等一下,让我好好想一想这个问题哦…"),
    ("L", 1, "嗯——这个问题还挺有意思的,让我想想看…"),
    ("L", 1, "唔,怎么说呢…让我组织一下语言哈…"),
    ("M", 1, "嗯嗯，这样啊…"),
]

# Preferred spoken-duration band per size: hesitations should be tight.
BANDS = {"S": (0.35, 1.1), "M": (1.0, 2.4), "L": (2.4, 4.4)}
SEEDS = [777, 1234, 4242, 9001, 31337, 55555]


def synthesize(text, seed):
    body = json.dumps({
        "input": text, "seed": seed, "response_format": "pcm",
        "stream_format": "audio", "stream": True, "stream_accumulate": False,
    }).encode()
    req = urllib.request.Request(SERVER + "/v1/audio/speech", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return np.frombuffer(r.read(), dtype=np.int16).astype(np.float32)


def trim(x, attack_ms=30, release_ms=100):
    loud = np.abs(x) > THR
    if not loud.any():
        return x[:0]
    first = int(np.argmax(loud))
    last = len(x) - int(np.argmax(loud[::-1]))
    b0 = max(0, first - attack_ms * SR // 1000)
    b1 = min(len(x), last + release_ms * SR // 1000)
    return x[b0:b1]


def internal_quiet_runs(x, min_ms=200):
    loud = np.abs(x) > THR
    if not loud.any():
        return []
    first = int(np.argmax(loud))
    last = len(x) - int(np.argmax(loud[::-1]))
    q = ~loud[first:last]
    runs, i = [], 0
    while i < len(q):
        if q[i]:
            j = i
            while j < len(q) and q[j]:
                j += 1
            if (j - i) >= min_ms * SR // 1000:
                runs.append((j - i) / SR)
            i = j
        else:
            i += 1
    return runs


def score(x, size):
    """Lower is better. x is a trimmed candidate."""
    if len(x) < SR // 10:
        return 99.0, "empty"
    dur = len(x) / SR
    lo, hi = BANDS[size]
    notes = []
    s = 0.0
    if dur < lo:
        s += (lo - dur) * 2.0
        notes.append(f"short {dur:.2f}s")
    elif dur > hi:
        s += (dur - hi) * 3.0
        notes.append(f"long {dur:.2f}s")
    tail = float(np.sqrt(np.mean(x[-int(0.12 * SR):] ** 2)))
    s += tail / 1500.0
    if tail > 900:
        notes.append(f"hot-tail {tail:.0f}")
    gaps = internal_quiet_runs(x)
    if size == "S" and gaps:
        s += 2.0 * len(gaps)
        notes.append(f"{len(gaps)} gap(s)")
    peak = float(np.max(np.abs(x)))
    if peak > 32200:
        s += 1.0
        notes.append("clipping")
    return s, ", ".join(notes) or "clean"


def load_mapping():
    path = os.path.join(FILLER_DIR, "mapping.json")
    saved = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if len(saved) == len(LIBRARY):
            return saved
    files = sorted(glob.glob(os.path.join(FILLER_DIR, "*.pcm")), key=os.path.getmtime)
    if len(files) != len(LIBRARY):
        sys.exit(f"cannot map: {len(files)} cache files vs {len(LIBRARY)} entries; "
                 "boot the server once so new entries synthesize, then rerun")
    if saved:
        # Entries appended since the saved mapping: the not-yet-mapped files,
        # in creation order, take the not-yet-mapped indexes in order.
        new_files = [os.path.basename(f) for f in files if os.path.basename(f) not in saved]
        new_indexes = sorted(set(range(len(LIBRARY))) - set(saved.values()))
        mapping = dict(saved)
        mapping.update(dict(zip(new_files, new_indexes)))
    else:
        mapping = {os.path.basename(f): i for i, f in enumerate(files)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=1)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="regenerate every entry")
    ap.add_argument("--report", action="store_true", help="score only, change nothing")
    args = ap.parse_args()

    mapping = load_mapping()
    by_index = {v: k for k, v in mapping.items()}
    regenerated = 0
    for i, (size, zh, text) in enumerate(LIBRARY):
        path = os.path.join(FILLER_DIR, by_index[i])
        current = trim(np.frombuffer(open(path, "rb").read(), dtype=np.int16).astype(np.float32))
        cur_score, cur_notes = score(current, size)
        label = text if len(text) <= 30 else text[:27] + "..."
        if args.report or (not args.all and cur_score < 0.8):
            print(f"keep  {label:<32} {len(current)/SR:5.2f}s score {cur_score:4.2f} ({cur_notes})")
            continue
        print(f"regen {label:<32} {len(current)/SR:5.2f}s score {cur_score:4.2f} ({cur_notes})")
        best, best_score, best_notes, best_seed = current, cur_score, cur_notes, None
        for seed in SEEDS:
            try:
                cand = trim(synthesize(text, seed))
            except Exception as e:
                print(f"      seed {seed}: synthesis failed ({e})")
                continue
            c_score, c_notes = score(cand, size)
            marker = " <-- best so far" if c_score < best_score else ""
            print(f"      seed {seed}: {len(cand)/SR:5.2f}s score {c_score:4.2f} ({c_notes}){marker}")
            if c_score < best_score:
                best, best_score, best_notes, best_seed = cand, c_score, c_notes, seed
            time.sleep(0.2)
        if best_seed is not None:
            with open(path, "wb") as f:
                f.write(best.astype(np.int16).tobytes())
            regenerated += 1
            print(f"      -> replaced with seed {best_seed} ({len(best)/SR:.2f}s, {best_notes})")
        else:
            print("      -> original kept (no candidate beat it)")
    print(f"\n{regenerated} clip(s) replaced. Restart the server to load them.")


if __name__ == "__main__":
    main()
