# Measures the chat pipeline's perceived-latency strategies against each
# other on the live server: legacy short-clause opener (filler:false),
# hesitation fillers with contextual picks, and both speculation paths.
# Reports, per run, when sound starts, when real content starts, and the
# seam between hesitation and real voice (negative = overlap = seamless;
# positive = silence the listener hears).
#
#   python scripts/latency_matrix.py

import json, socket, time, urllib.request

SERVER = ("127.0.0.1", 18080)
BUFFER_S = 0.15  # client-side startup buffer the browser player uses


def post(payload):
    req = urllib.request.Request(f"http://{SERVER[0]}:{SERVER[1]}/v1/chat/speak",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300).read()


def stream(payload):
    """Returns list of (arrival_s, event_dict) parsed from the SSE stream."""
    body = json.dumps(payload).encode()
    s = socket.create_connection(SERVER)
    s.sendall(b"POST /v1/chat/speak HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
    t0 = time.perf_counter()
    raw = b""
    marks = []  # (time, raw_len) so we can assign arrival times to bytes
    while True:
        c = s.recv(65536)
        if not c:
            break
        raw += c
        marks.append((time.perf_counter() - t0, len(raw)))
    s.close()
    data = raw[raw.find(b"\r\n\r\n") + 4:]
    base = raw.find(b"\r\n\r\n") + 4
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
        spans.append((len(out), base + j + 2, n))
        out += data[j + 2:j + 2 + n]
        p = j + 2 + n + 2

    def arrival(raw_pos):
        for t, ln in marks:
            if ln >= raw_pos:
                return t
        return marks[-1][0]

    events = []
    cursor = 0
    for frame in out.split(b"\n\n"):
        end = cursor + len(frame)
        raw_end = None
        for o, rp, n in spans:
            if o <= end <= o + n:
                raw_end = rp + (end - o)
                break
        cursor = end + 2
        for line in frame.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            pl = line[5:].strip()
            if not pl or pl == b"[DONE]":
                continue
            try:
                events.append((arrival(raw_end or len(raw)), json.loads(pl)))
            except json.JSONDecodeError:
                pass
    return events


def measure(messages, filler=True, prewarm=True, speculate_wait=None):
    payload = {"messages": messages}
    if not filler:
        payload["filler"] = False
    if speculate_wait is not None:
        post({"messages": messages, "speculate": True})
        time.sleep(speculate_wait)
    elif prewarm:
        post({"messages": messages, "prewarm": True})
        time.sleep(2.2)
    events = stream(payload)
    import base64 as b64
    # A hesitation clip is one big audio event (>=0.3 s); real TTS streams
    # tiny chunks (~0.08 s). Size separates them cleanly.
    filler_s, n_filler_clips = 0.0, 0
    first_audio_at = real_audio_at = None
    sentences, stats = [], {}
    for at, e in events:
        t = e.get("type")
        if t == "sentence":
            sentences.append(e["text"])
        elif t == "audio":
            n = len(b64.b64decode(e["audio"]))
            if first_audio_at is None:
                first_audio_at = at
            if n >= 12000 and real_audio_at is None:
                filler_s += n / 48000
                n_filler_clips += 1
            elif real_audio_at is None:
                real_audio_at = at
        elif t == "done":
            stats = e.get("stats", {})
    filler_texts = sentences[:n_filler_clips]
    first_real_sentence = sentences[n_filler_clips] if len(sentences) > n_filler_clips else ""
    if real_audio_at is None:
        real_audio_at = stats.get("first_audio_ms", 0) / 1000.0
    sound_at = (first_audio_at if first_audio_at is not None else real_audio_at) + BUFFER_S
    if filler_s > 0:
        filler_end = sound_at + filler_s
        content_plays = max(filler_end, real_audio_at)
        seam = real_audio_at - filler_end
    else:
        filler_end = None
        content_plays = real_audio_at + BUFFER_S
        seam = 0.0
    return dict(sound_ms=sound_at * 1000, content_ms=content_plays * 1000,
                seam_ms=seam * 1000, filler_s=filler_s,
                fillers=" + ".join(filler_texts) or "-",
                opener=first_real_sentence[:52],
                tok_ms=stats.get("first_token_ms", -1))


def show(label, r):
    print(f"{label:<26} sound {r['sound_ms']:6.0f}ms  content {r['content_ms']:6.0f}ms  "
          f"seam {r['seam_ms']:+6.0f}ms  tok {r['tok_ms']:5.0f}  [{r['fillers']}]")
    print(f"{'':<26} opener: {r['opener']!r}")


TOPICS = ["favorite way to spend a sunday", "what makes a good friend",
          "best snack for a movie night", "how to stay motivated",
          "the weirdest dream you remember"]

def turn0(i):
    return [{"role": "user", "content": f"so tell me, {TOPICS[i % len(TOPICS)]}?"}]

def turn1(i):
    return turn0(i) + [
        {"role": "assistant", "content": "Mmm. Honestly, slow mornings and a long walk, nothing fancy."},
        {"role": "user", "content": "that does sound nice. i might steal that idea this weekend"}]


if __name__ == "__main__":
    print("warmup"); measure(turn0(99), prewarm=False)

    print("\n-- legacy (filler off, short clause opener), prewarmed --")
    for i in range(3):
        show(f"legacy turn0 #{i}", measure(turn0(i), filler=False))
    for i in range(2):
        show(f"legacy turn1 #{i}", measure(turn1(i), filler=False))

    print("\n-- fillers (contextual + wait-targeted), prewarmed --")
    for i in range(3):
        show(f"filler turn0 #{i}", measure(turn0(i)))
    for i in range(2):
        show(f"filler turn1 #{i}", measure(turn1(i)))

    print("\n-- classifier fit --")
    show("question EN", measure([{"role": "user", "content": "why is the sky blue at noon but red at sunset"}]))
    show("excited EN", measure([{"role": "user", "content": "omg guess what happened at work today!!"}]))
    show("statement EN", measure([{"role": "user", "content": "i finally finished that book you mentioned last week"}]))
    show("question ZH", measure([{"role": "user", "content": "你觉得一个人旅行安全吗"}]))
    show("statement ZH", measure([{"role": "user", "content": "我今天把房间彻底打扫了一遍"}]))

    print("\n-- speculation paths --")
    show("spec settled (5s hover)", measure(turn1(3), speculate_wait=5.0))
    show("spec early (0.3s)", measure(turn1(4), speculate_wait=0.3))
