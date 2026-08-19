#include "engine/models/qwen3_tts/stream_decoder.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::qwen3_tts {
namespace {

void validate_codes(const Qwen3SpeechCodes & codes, int64_t code_groups, const char * label) {
    if (codes.frames < 0 || codes.code_groups != code_groups) {
        throw std::runtime_error(std::string("Qwen3 streaming decoder invalid ") + label + " shape");
    }
    if (codes.frames > std::numeric_limits<int64_t>::max() / code_groups ||
        static_cast<int64_t>(codes.codes.size()) != codes.frames * code_groups) {
        throw std::runtime_error(std::string("Qwen3 streaming decoder invalid ") + label + " payload");
    }
}

Qwen3SpeechCodes tail_frames(const Qwen3SpeechCodes & source, int64_t frames) {
    Qwen3SpeechCodes out;
    out.code_groups = source.code_groups;
    out.frames = std::min(source.frames, frames);
    const int64_t first_frame = source.frames - out.frames;
    const auto first = source.codes.begin() + static_cast<std::ptrdiff_t>(first_frame * source.code_groups);
    out.codes.assign(first, source.codes.end());
    return out;
}

}  // namespace

Qwen3SpeechTokenizerStreamDecoder::Qwen3SpeechTokenizerStreamDecoder(
    Qwen3SpeechTokenizerDecoderRuntime & decoder,
    int64_t code_groups,
    Qwen3TTSStreamDecoderOptions options,
    const std::optional<Qwen3SpeechCodes> & reference_codes)
    : Qwen3SpeechTokenizerStreamDecoder(
          [&decoder](const Qwen3SpeechCodes & codes) { return decoder.decode(codes); },
          code_groups,
          options,
          reference_codes) {}

Qwen3SpeechTokenizerStreamDecoder::Qwen3SpeechTokenizerStreamDecoder(
    DecodeFunction decode,
    int64_t code_groups,
    Qwen3TTSStreamDecoderOptions options,
    const std::optional<Qwen3SpeechCodes> & reference_codes)
    : decode_(std::move(decode)), code_groups_(code_groups), options_(options) {
    if (!decode_) {
        throw std::runtime_error("Qwen3 streaming decoder requires a decode function");
    }
    if (code_groups_ <= 0) {
        throw std::runtime_error("Qwen3 streaming decoder requires positive code_groups");
    }
    if (options_.chunk_frames <= 0) {
        throw std::runtime_error("Qwen3 TTS chunk_frames must be positive");
    }
    if (options_.context_frames < 0) {
        throw std::runtime_error("Qwen3 TTS decoder_context_frames must be non-negative");
    }
    reset(reference_codes);
}

std::optional<runtime::AudioBuffer> Qwen3SpeechTokenizerStreamDecoder::push_frame(
    const std::vector<int32_t> & codes) {
    if (static_cast<int64_t>(codes.size()) != code_groups_) {
        throw std::runtime_error("Qwen3 streaming decoder codec frame width mismatch");
    }
    pending_.codes.insert(pending_.codes.end(), codes.begin(), codes.end());
    ++pending_.frames;
    if (pending_.frames < options_.chunk_frames) {
        return std::nullopt;
    }
    return decode_pending();
}

std::optional<runtime::AudioBuffer> Qwen3SpeechTokenizerStreamDecoder::flush() {
    return decode_pending();
}

void Qwen3SpeechTokenizerStreamDecoder::reset(
    const std::optional<Qwen3SpeechCodes> & reference_codes) {
    context_ = Qwen3SpeechCodes{};
    context_.code_groups = code_groups_;
    pending_ = Qwen3SpeechCodes{};
    pending_.code_groups = code_groups_;
    emitted_frames_ = 0;
    set_reference_context(reference_codes);
}

int64_t Qwen3SpeechTokenizerStreamDecoder::pending_frames() const noexcept {
    return pending_.frames;
}

int64_t Qwen3SpeechTokenizerStreamDecoder::emitted_frames() const noexcept {
    return emitted_frames_;
}

const Qwen3TTSStreamDecoderOptions & Qwen3SpeechTokenizerStreamDecoder::options() const noexcept {
    return options_;
}

void Qwen3SpeechTokenizerStreamDecoder::set_reference_context(
    const std::optional<Qwen3SpeechCodes> & reference_codes) {
    if (!reference_codes.has_value()) {
        return;
    }
    validate_codes(*reference_codes, code_groups_, "reference codec");
    context_ = tail_frames(*reference_codes, options_.context_frames);
}

std::optional<runtime::AudioBuffer> Qwen3SpeechTokenizerStreamDecoder::decode_pending() {
    if (pending_.frames == 0) {
        return std::nullopt;
    }
    Qwen3SpeechCodes window;
    window.code_groups = code_groups_;
    window.frames = context_.frames + pending_.frames;
    window.codes.reserve(context_.codes.size() + pending_.codes.size());
    window.codes.insert(window.codes.end(), context_.codes.begin(), context_.codes.end());
    window.codes.insert(window.codes.end(), pending_.codes.begin(), pending_.codes.end());

    auto audio = decode_(window);
    const int64_t context_samples = context_.frames * kQwen3TTSSamplesPerCodecFrame;
    const int64_t emitted_samples = pending_.frames * kQwen3TTSSamplesPerCodecFrame;
    if (context_samples < 0 || emitted_samples < 0 ||
        context_samples + emitted_samples > static_cast<int64_t>(audio.samples.size())) {
        throw std::runtime_error("Qwen3 streaming decoder sample trim exceeds decoded window");
    }
    audio.samples.erase(
        audio.samples.begin(),
        audio.samples.begin() + static_cast<std::ptrdiff_t>(context_samples));
    audio.samples.resize(static_cast<size_t>(emitted_samples));

    emitted_frames_ += pending_.frames;
    context_ = tail_frames(window, options_.context_frames);
    pending_.frames = 0;
    pending_.codes.clear();
    return audio;
}

}  // namespace engine::models::qwen3_tts
