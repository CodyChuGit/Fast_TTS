# Streams a spoken utterance into /v1/voice/live at real-time pace (20 ms
# chunks over one chunked-transfer connection, SSE events read concurrently)
# and reports the event sequence plus the latencies that matter: first
# partial, first stable, and end-of-turn -> final transcript.
#
#   python scripts/voice_probe.py [--wav path.wav] [--repeat N]
#
# Without --wav, a test utterance is synthesized through the server's own TTS
# and resampled to 16 kHz, so the probe is self-contained.

import argparse, json, socket, subprocess, sys, threading, time, urllib.request
from pathlib import Path

SERVER = ("127.0.0.1", 18080)
SCRATCH = Path(__file__).resolve().parent.parent / "build" / "voice_probe"


def synthesize_test_wav(text: str, out_path: Path):
    body = json.dumps({
        "input": text, "seed": 777, "response_format": "pcm",
        "stream_format": "audio", "stream": True, "stream_accumulate": False,
    }).encode()
    req = urllib.request.Request(f"http://{SERVER[0]}:{SERVER[1]}/v1/audio/speech",
                                 data=body, headers={"Content-Type": "application/json"})
    pcm = urllib.request.urlopen(req, timeout=180).read()
    raw = out_path.with_suffix(".24k.pcm")
    raw.write_bytes(pcm)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", str(raw),
        "-af", "adelay=400|400,apad=pad_dur=0.8",
        "-ar", "16000", "-ac", "1", "-f", "s16le", str(out_path),
    ], check=True)


def load_pcm16(path: Path) -> bytes:
    if path.suffix.lower() == ".wav":
        out = path.with_suffix(".16k.pcm")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
            "-ar", "16000", "-ac", "1", "-f", "s16le", str(out),
        ], check=True)
        return out.read_bytes()
    return path.read_bytes()


def stream_utterance(pcm: bytes, label: str, realtime: bool = True):
    s = socket.create_connection(SERVER)
    s.sendall(
        b"POST /v1/voice/live?sample_rate=16000&channels=1&sample_format=s16le HTTP/1.1\r\n"
        b"Host: x\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")

    events = []
    t0 = time.perf_counter()

    def reader():
        raw = b""
        try:
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                raw += chunk
                while b"\n\n" in raw:
                    frame, raw = raw.split(b"\n\n", 1)
                    for line in frame.split(b"\n"):
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == b"[DONE]":
                            continue
                        # De-chunk artifacts: frames may carry HTTP chunk size
                        # lines; find the JSON braces.
                        start = payload.find(b"{")
                        if start < 0:
                            continue
                        try:
                            events.append((time.perf_counter() - t0, json.loads(payload[start:])))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    step = 640  # 20 ms of 16 kHz s16le
    for offset in range(0, len(pcm), step):
        block = pcm[offset:offset + step]
        s.sendall(f"{len(block):x}\r\n".encode() + block + b"\r\n")
        if realtime:
            time.sleep(0.02)
    s.sendall(b"0\r\n\r\n")
    thread.join(timeout=30)
    s.close()

    print(f"\n== {label} ({len(pcm) / 32000:.1f}s of audio) ==")
    finals = []
    for at, event in events:
        kind = event.get("type")
        text = event.get("text", "")
        extra = f" {text!r}" if text else ""
        print(f"  {at * 1000:6.0f}ms  {kind}{extra}")
        if kind == "final_transcript":
            finals.append(event)
    for final in finals:
        timings = final.get("timings", {})
        print(f"  -> timings: first_partial {timings.get('first_partial_ms')}ms | "
              f"first_stable {timings.get('first_stable_ms')}ms | "
              f"eot->final {timings.get('eot_to_final_ms')}ms | "
              f"final_decode {timings.get('final_decode_ms')}ms | "
              f"audio {timings.get('audio_ms')}ms")
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--text", default="Can you check the weather in New York for me tomorrow?")
    args = parser.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    if args.wav:
        pcm = load_pcm16(Path(args.wav))
    else:
        target = SCRATCH / "utterance.16k.pcm"
        if not target.exists():
            print("synthesizing test utterance through the server TTS...")
            synthesize_test_wav(args.text, target)
        pcm = target.read_bytes()

    print("warmup pass (loads the lazy ASR model)...")
    stream_utterance(pcm, "warmup", realtime=False)
    for index in range(args.repeat):
        stream_utterance(pcm, f"pass {index + 1}")


if __name__ == "__main__":
    main()
