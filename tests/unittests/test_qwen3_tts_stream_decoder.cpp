#include "engine/models/qwen3_tts/stream_decoder.h"

#include "test_assert.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using engine::models::qwen3_tts::Qwen3SpeechCodes;
using engine::models::qwen3_tts::Qwen3SpeechTokenizerStreamDecoder;
using engine::models::qwen3_tts::Qwen3TTSStreamDecoderOptions;
using engine::models::qwen3_tts::kQwen3TTSSampleRate;
using engine::models::qwen3_tts::kQwen3TTSSamplesPerCodecFrame;
using engine::test::require;
using engine::test::require_eq;

constexpr int64_t kGroups = 4;

engine::runtime::AudioBuffer deterministic_decode(const Qwen3SpeechCodes & codes) {
    engine::runtime::AudioBuffer out{static_cast<int>(kQwen3TTSSampleRate), 1, {}};
    out.samples.reserve(static_cast<size_t>(codes.frames * kQwen3TTSSamplesPerCodecFrame));
    for (int64_t frame = 0; frame < codes.frames; ++frame) {
        const float marker = static_cast<float>(codes.codes[static_cast<size_t>(frame * codes.code_groups)]);
        out.samples.insert(
            out.samples.end(),
            static_cast<size_t>(kQwen3TTSSamplesPerCodecFrame),
            marker);
    }
    return out;
}

std::vector<int32_t> frame(int32_t marker) {
    return {marker, marker + 100, marker + 200, marker + 300};
}

void append(std::vector<float> & output, std::optional<engine::runtime::AudioBuffer> chunk) {
    if (!chunk.has_value()) {
        return;
    }
    output.insert(output.end(), chunk->samples.begin(), chunk->samples.end());
}

void verify_chunk_size(int64_t chunk_frames) {
    Qwen3SpeechTokenizerStreamDecoder decoder(
        deterministic_decode,
        kGroups,
        Qwen3TTSStreamDecoderOptions{chunk_frames, 25});
    std::vector<float> output;
    for (int32_t marker = 1; marker <= 11; ++marker) {
        append(output, decoder.push_frame(frame(marker)));
    }
    append(output, decoder.flush());

    require_eq(decoder.emitted_frames(), int64_t{11}, "emitted frame count");
    require_eq(decoder.pending_frames(), int64_t{0}, "pending frame count");
    require_eq(
        output.size(),
        static_cast<size_t>(11 * kQwen3TTSSamplesPerCodecFrame),
        "concatenated sample count");
    for (int32_t marker = 1; marker <= 11; ++marker) {
        const size_t begin = static_cast<size_t>((marker - 1) * kQwen3TTSSamplesPerCodecFrame);
        const size_t end = static_cast<size_t>(marker * kQwen3TTSSamplesPerCodecFrame);
        for (size_t sample = begin; sample < end; ++sample) {
            require(output[sample] == static_cast<float>(marker), "duplicate or dropped boundary samples");
        }
    }
}

void test_supported_chunk_sizes_and_flush() {
    verify_chunk_size(1);
    verify_chunk_size(2);
    verify_chunk_size(4);
    verify_chunk_size(8);
}

void test_reference_context_is_not_emitted() {
    Qwen3SpeechCodes reference;
    reference.code_groups = kGroups;
    for (int32_t marker = -40; marker < 0; ++marker) {
        const auto codes = frame(marker);
        reference.codes.insert(reference.codes.end(), codes.begin(), codes.end());
        ++reference.frames;
    }
    Qwen3SpeechTokenizerStreamDecoder decoder(
        deterministic_decode,
        kGroups,
        Qwen3TTSStreamDecoderOptions{2, 25},
        reference);
    auto none = decoder.push_frame(frame(1));
    require(!none.has_value(), "first frame should remain pending");
    auto audio = decoder.push_frame(frame(2));
    require(audio.has_value(), "second frame should decode a chunk");
    require_eq(
        audio->samples.size(),
        static_cast<size_t>(2 * kQwen3TTSSamplesPerCodecFrame),
        "reference trim sample count");
    require(audio->samples.front() == 1.0F, "reference samples leaked into output");
    require(audio->samples.back() == 2.0F, "second generated frame is missing");
}

void test_validation() {
    bool threw = false;
    try {
        Qwen3SpeechTokenizerStreamDecoder decoder(
            deterministic_decode,
            kGroups,
            Qwen3TTSStreamDecoderOptions{0, 25});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "zero chunk size was accepted");
}

}  // namespace

int main() {
    try {
        test_supported_chunk_sizes_and_flush();
        test_reference_context_is_not_emitted();
        test_validation();
        std::cout << "qwen3_tts_stream_decoder_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_tts_stream_decoder_test failed: " << error.what() << "\n";
        return 1;
    }
}
