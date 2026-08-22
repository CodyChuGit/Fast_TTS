# Emulates real users against the live server, reproducing the WebUI's exact
# client behavior: typing with word-boundary pauses and occasional mid-draft
# stalls, the 300 ms prewarm debounce, the 180/400/850 ms speculation tiers,
# assistant history stored with the spoken hesitation prefix (as the real
# client stores it), and humans who start their next message shortly after
# the reply finishes streaming. Reports, per send, which path fired and what
# the listener actually experienced.
#
#   python -u scripts/real_user_sim.py

import base64, json, random, re, socket, time, urllib.request

SERVER = ("127.0.0.1", 18080)
BUFFER_S = 0.15  # the browser player's startup buffer

SENT_END = re.compile(r'[.!?…~。！？～][)\]"\'’”』」]*$')
QUESTION_START = re.compile(r'^(what|why|how|when|where|who|which|can|could|would|'
                            r'should|do|does|did|is|are|will)\b', re.I)


def speculate_delay(draft):
    if SENT_END.search(draft):
        return 0.12
    cjk = any('㐀' <= c <= '鿿' for c in draft)
    if not cjk and QUESTION_START.match(draft):
        return 0.30
    if (not cjk and len(draft.split()) <= 3) or (cjk and len(draft) <= 6):
        return 0.50
    return 1.50


def post_async(payload):
    def go():
        try:
            req = urllib.request.Request(f"http://{SERVER[0]}:{SERVER[1]}/v1/chat/speak",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            pass
    import threading
    threading.Thread(target=go, daemon=True).start()


def stream(messages):
    body = json.dumps({"messages": messages}).encode()
    s = socket.create_connection(SERVER)
    s.sendall(b"POST /v1/chat/speak HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
    t0 = time.perf_counter()
    raw, marks = b"", []
    while True:
        c = s.recv(65536)
        if not c:
            break
        raw += c
        marks.append((time.perf_counter() - t0, len(raw)))
    s.close()
    base = raw.find(b"\r\n\r\n") + 4
    data = raw[base:]
    out, spans, p = b"", [], 0
    while True:
        j = data.find(b"\r\n", p)
        if j < 0:
            break
        try:
            n = int(data[p:j], 16)
        except ValueError:
            break
        if n == 0 or len(data) < j + 2 + n:
            break
        out += data[j + 2:j + 2 + n]
        spans.append((len(out), base + j + 2 + n))  # (out END offset, raw END)
        p = j + 2 + n + 2

    def arrival(out_pos):
        raw_pos = next((rp for o_end, rp in spans if o_end >= out_pos), len(raw))
        return next((t for t, ln in marks if ln >= raw_pos), marks[-1][0])

    events, cursor = [], 0
    for frame in out.split(b"\n\n"):
        end = cursor + len(frame)
        cursor = end + 2
        for line in frame.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            pl = line[5:].strip()
            if not pl or pl == b"[DONE]":
                continue
            try:
                events.append((arrival(end), json.loads(pl)))
            except json.JSONDecodeError:
                pass
    return events


class Persona:
    def __init__(self, name, word_pause, stall_prob, stall_ms, pre_enter_ms,
                 post_reply_s, editor=False, chinese=False):
        self.__dict__.update(name=name, word_pause=word_pause, stall_prob=stall_prob,
                             stall_ms=stall_ms, pre_enter_ms=pre_enter_ms,
                             post_reply_s=post_reply_s, editor=editor, chinese=chinese)


PERSONAS = [
    Persona("fast-sender", 0.12, 0.08, 0.7, 0.20, 1.2),
    Persona("average", 0.25, 0.20, 0.9, 0.60, 2.0),
    Persona("hover-reader", 0.25, 0.15, 0.9, 3.20, 1.5),
    Persona("draft-editor", 0.22, 0.15, 0.9, 0.40, 1.8, editor=True),
    Persona("zh-user", 0.30, 0.15, 0.9, 0.50, 1.8, chinese=True),
]

SCRIPTS_EN = [
    ["hey! how was your day today?",
     "nice. what's the best thing you ate this week",
     "haha okay that sounds amazing honestly",
     "do you think we should try making that together sometime?",
     "alright, pick a day and I'm there"],
    ["I finally watched that show everyone keeps talking about",
     "the ending was so confusing though, what did you think it meant?",
     "hmm that's actually a really interesting take",
     "okay now recommend me something happier please!",
     "perfect. adding it to the list right now"],
]
SCRIPTS_ZH = [
    ["今天过得怎么样呀",
     "你最近有没有看什么好看的剧？",
     "哈哈那我周末也去看看",
     "对了,你觉得学做饭难吗",
     "那你教我做一道菜吧！"],
]


def type_message(persona, history, text, rng):
    """Emulates typing `text`, firing prewarm/speculate as the client would.
    Returns seconds spent typing (send happens right after)."""
    words = list(text) if persona.chinese else text.split()
    step = 3 if persona.chinese else 1  # ZH: bursts of ~3 characters
    draft = ""
    last_prewarmed = last_speculated = None
    spent = 0.0

    def draft_messages(d):
        return history + [{"role": "user", "content": d.strip()}]

    def idle(pause, current):
        # Timers fire DURING the pause at their offsets, exactly like the
        # client's setTimeout -- firing them at pause end would (and once
        # did) hide the speculate-vs-send race behind zero head start.
        nonlocal last_prewarmed, last_speculated, spent
        d = current.strip()
        elapsed = 0.0
        if d and pause >= 0.20 and d != last_prewarmed and d != last_speculated:
            time.sleep(0.20 - elapsed)
            elapsed = 0.20
            last_prewarmed = d
            post_async({"messages": draft_messages(d), "prewarm": True})
        if d:
            delay = speculate_delay(d)
            if pause >= delay and d != last_speculated:
                time.sleep(max(0.0, delay - elapsed))
                elapsed = max(elapsed, delay)
                last_speculated = d
                post_async({"messages": draft_messages(d), "speculate": True})
        time.sleep(pause - elapsed)
        spent += pause

    i = 0
    while i < len(words):
        chunk = words[i:i + step]
        draft = (draft + ("" if persona.chinese else " ") + ("".join(chunk) if persona.chinese else " ".join(chunk))).strip()
        i += step
        if i < len(words):
            pause = persona.word_pause * rng.uniform(0.7, 1.4)
            if rng.random() < persona.stall_prob:
                pause = persona.stall_ms * rng.uniform(0.8, 1.3)
            idle(pause, draft)
    if persona.editor and len(words) > 4 and rng.random() < 0.7:
        # Pause long enough to speculate, then rewrite the ending.
        idle(1.2, draft)
        words_kept = words[:-2]
        draft = ("".join(words_kept) if persona.chinese else " ".join(words_kept)).strip()
        tail = "了吧" if persona.chinese else " for real though"
        draft = draft + tail
        idle(0.25, draft)
        text = draft
    idle(persona.pre_enter_ms * rng.uniform(0.8, 1.3), draft)
    return spent, draft.strip()


def run_turn(persona, history, text, rng):
    typing_s, final_text = type_message(persona, history, text, rng)
    messages = history + [{"role": "user", "content": final_text}]
    events = stream(messages)

    reply_tokens, sentences = [], []
    filler_s = n_filler = 0
    first_audio_at = real_audio_at = None
    stats, errors = {}, []
    for at, e in events:
        t = e.get("type")
        if t == "token":
            reply_tokens.append(e["text"])
        elif t == "sentence":
            sentences.append(e["text"])
        elif t == "audio":
            n = len(base64.b64decode(e["audio"]))
            if first_audio_at is None:
                first_audio_at = at
            # A filler emission is token+sentence+one big clip, so at its
            # audio event tokens == sentences; real speech has streamed many
            # tokens past the sentence count by the time its audio starts.
            is_filler = (n >= 12000 and real_audio_at is None and
                         len(reply_tokens) == len(sentences))
            if is_filler:
                filler_s += n / 48000
                n_filler += 1
            elif real_audio_at is None:
                real_audio_at = at
        elif t == "error":
            errors.append(e.get("message", "?"))
        elif t == "done":
            stats = e.get("stats", {})

    reply = "".join(reply_tokens)
    if real_audio_at is None:
        real_audio_at = stats.get("first_audio_ms", 0) / 1000.0
    sound_at = (first_audio_at if first_audio_at is not None else real_audio_at) + BUFFER_S
    if filler_s > 0:
        content_at = max(sound_at + filler_s, real_audio_at)
        seam = real_audio_at - (sound_at + filler_s)
    else:
        content_at = real_audio_at + BUFFER_S
        seam = 0.0
    hit = stats.get("speculative_hit", False)
    tok = stats.get("first_token_ms", -1)
    path = ("spec-settled" if hit and tok <= 60 else
            "spec-early" if hit else "miss")
    history.append({"role": "user", "content": final_text})
    history.append({"role": "assistant", "content": reply})
    return dict(persona=persona.name, path=path, sound_ms=sound_at * 1000,
                content_ms=content_at * 1000, seam_ms=seam * 1000,
                fillers=" + ".join(sentences[:n_filler]) or "-",
                tok_ms=tok, typing_s=typing_s, errors=errors,
                empty=not reply.strip(), audio_s=stats.get("audio_seconds", 0))


def main():
    rng = random.Random(42)
    results = []
    for persona in PERSONAS:
        script = rng.choice(SCRIPTS_ZH) if persona.chinese else rng.choice(SCRIPTS_EN)
        history = []
        print(f"\n=== {persona.name} ===")
        for turn, text in enumerate(script):
            r = run_turn(persona, history, text, rng)
            results.append(r)
            flag = " ERR:" + ";".join(r["errors"]) if r["errors"] else (" EMPTY" if r["empty"] else "")
            print(f"  t{turn} {r['path']:<12} sound {r['sound_ms']:5.0f}ms content {r['content_ms']:5.0f}ms "
                  f"seam {r['seam_ms']:+5.0f}ms tok {r['tok_ms']:5.0f} [{r['fillers']}]{flag}")
            time.sleep(persona.post_reply_s)

    print("\n=== aggregate ===")
    import statistics as st
    by_path = {}
    for r in results:
        by_path.setdefault(r["path"], []).append(r)
    for path, rs in sorted(by_path.items()):
        snd = [r["sound_ms"] for r in rs]
        cnt = [r["content_ms"] for r in rs]
        gaps = [max(0, r["seam_ms"]) for r in rs]
        print(f"{path:<12} n={len(rs):<3} sound p50 {st.median(snd):5.0f} max {max(snd):5.0f} | "
              f"content p50 {st.median(cnt):5.0f} max {max(cnt):5.0f} | "
              f"audible gap p50 {st.median(gaps):4.0f} max {max(gaps):4.0f}")
    n_err = sum(1 for r in results if r["errors"] or r["empty"])
    print(f"reliability: {len(results) - n_err}/{len(results)} clean turns")


if __name__ == "__main__":
    main()
