#pragma once

#include "voice_stability.h"

#include "../streaming/streaming.h"

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace minitts::server::voice {

// ---------------------------------------------------------------------------
// One live voice connection: PCM in, transcript events out.
// ---------------------------------------------------------------------------

struct TurnParams {
    float vad_threshold = 0.5F;
    int min_speech_ms = 100;
    // Acoustic silence before the VAD reports a possible end. The turn does
    // not finalize here -- an adaptive hold follows, and a speculative final
    // decode starts, so resumed speech cancels cheaply.
    int min_silence_ms = 120;
    // Additional hold after the VAD end before the turn commits: short when
    // the hypothesis looks complete, long when it ends in a continuation
    // word ("and", "的", "um") -- semantic endpointing without a second
    // model.
    int endpoint_hold_ms = 110;
    int endpoint_hold_incomplete_ms = 350;
    int max_utterance_ms = 30000;
    int partial_interval_ms = 280;
    // Below this much captured speech, partial decodes are skipped: the ASR
    // hallucinates on sub-second snippets and returns fragments the tracker
    // would just have to disbelieve.
    int min_partial_audio_ms = 700;
    int speech_pad_ms = 240;     // audio retained from before the VAD trigger
    // Loudness gate, in multiples of the measured noise floor. A microphone's
    // hiss is steady and its speech is not: a controller headset mic measured
    // a floor of ~0.035 RMS with real speech at 0.14-0.57, so an utterance
    // whose loudest moment never rises this far above the floor is room tone
    // the VAD mistook for voice, and is dropped without transcription (the
    // ASR hallucinates confident fragments on noise). Zero disables the gate.
    float min_speech_snr = 3.0F;
    // Absolute floor for that gate, so a silent studio mic (noise floor near
    // zero) still requires real signal before an utterance is accepted.
    float min_speech_rms = 0.012F;
    StabilityParams stability;
    std::string language;        // empty = auto-detect
};

// Parses `key=value&...` query text into TurnParams overrides; unknown keys
// are ignored so transport-level parameters can share the query string.
TurnParams turn_params_from_query(const std::string & query);

struct TranscribeResult {
    std::string text;
    std::string language;
};

struct VoiceHooks {
    // Runs the ASR model on one utterance-so-far snapshot (16 kHz mono).
    // Called from the runner's decode worker thread.
    std::function<TranscribeResult(const engine::runtime::AudioBuffer &)> transcribe;
    // Emits one SSE payload (the `data:` JSON). Must be thread-safe: the
    // audio loop and the decode worker both emit.
    std::function<void(const std::string & json)> emit;
    // True once the client is gone; the runner unwinds promptly.
    std::function<bool()> aborted;
};

// Drives one connection: reads 16 kHz mono float PCM off `stream`, gates it
// with streaming Silero VAD, re-decodes the growing utterance on a worker
// thread at `partial_interval_ms`, tracks hypothesis stability, and emits
// typed events. Handles any number of utterances until the stream ends.
//
// Events (all carry "t_ms" on the connection's monotonic clock):
//   speech_started    {utterance_id}
//   partial_transcript{utterance_id, text}          -- tentative tail
//   stable_transcript {utterance_id, text}          -- full committed prefix
//   speech_ended      {utterance_id}
//   final_transcript  {utterance_id, text, language, timings{...}}
//   error             {message}
class VoiceLiveRunner {
public:
    VoiceLiveRunner(
        TurnParams params,
        VoiceHooks hooks,
        std::unique_ptr<engine::runtime::IStreamingVoiceTaskSession> vad);
    ~VoiceLiveRunner();

    // Blocks until the input stream ends or the connection aborts.
    void run(const minitts::app::AudioChunkStream & stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Loads the bundled streaming Silero VAD (CPU backend; the model is tiny and
// keeping it off the GPU costs nothing). `asset_root` is the directory that
// holds assets/framework/models/silero_vad.
std::unique_ptr<engine::runtime::IStreamingVoiceTaskSession> make_vad_session(
    const std::filesystem::path & asset_root,
    const TurnParams & params);

}  // namespace minitts::server::voice
