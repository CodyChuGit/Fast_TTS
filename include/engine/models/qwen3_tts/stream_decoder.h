#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/qwen3_tts/tokenizer_speech_decoder.h"
#include "engine/models/qwen3_tts/types.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace engine::models::qwen3_tts {

struct Qwen3TTSStreamDecoderOptions {
    int64_t chunk_frames = 2;
    int64_t context_frames = 25;
};

// Stateful window around the existing Qwen speech decoder. Only the bounded
// left context and not-yet-emitted frames are retained. Each decode discards
// samples belonging to the left context, so concatenating returned buffers
// cannot duplicate a codec frame at a chunk boundary.
class Qwen3SpeechTokenizerStreamDecoder {
public:
    using DecodeFunction = std::function<runtime::AudioBuffer(const Qwen3SpeechCodes &)>;

    Qwen3SpeechTokenizerStreamDecoder(
        Qwen3SpeechTokenizerDecoderRuntime & decoder,
        int64_t code_groups,
        Qwen3TTSStreamDecoderOptions options,
        const std::optional<Qwen3SpeechCodes> & reference_codes = std::nullopt);
    Qwen3SpeechTokenizerStreamDecoder(
        DecodeFunction decode,
        int64_t code_groups,
        Qwen3TTSStreamDecoderOptions options,
        const std::optional<Qwen3SpeechCodes> & reference_codes = std::nullopt);

    std::optional<runtime::AudioBuffer> push_frame(const std::vector<int32_t> & codes);
    std::optional<runtime::AudioBuffer> flush();
    void reset(const std::optional<Qwen3SpeechCodes> & reference_codes = std::nullopt);

    int64_t pending_frames() const noexcept;
    int64_t emitted_frames() const noexcept;
    const Qwen3TTSStreamDecoderOptions & options() const noexcept;

private:
    std::optional<runtime::AudioBuffer> decode_pending();
    void set_reference_context(const std::optional<Qwen3SpeechCodes> & reference_codes);

    DecodeFunction decode_;
    int64_t code_groups_ = 0;
    Qwen3TTSStreamDecoderOptions options_;
    Qwen3SpeechCodes context_;
    Qwen3SpeechCodes pending_;
    int64_t emitted_frames_ = 0;
};

}  // namespace engine::models::qwen3_tts
