# Qwen3-TTS streaming latency benchmark

This benchmark runs the same deterministic request with `chunk_frames` 1, 2,
4, and 8. It reports measured TTFA, codec frames/s, decoder time/chunk, audio
seconds generated per wall second, RTF, Linux process peak RSS, and whether the
first PCM event occurred before generation completed. It never contains
hard-coded performance results.

Build with `-DENGINE_BUILD_WARMBENCH=ON`, then run:

```bash
build/bin/qwen3_tts_stream_latency \
  --model models/Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf \
  --reference-audio reference.wav \
  --reference-text "The exact reference transcript." \
  --backend cuda
```
