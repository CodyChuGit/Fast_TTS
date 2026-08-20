#include "engine/models/qwen3_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/qwen3_tts/prompt_tts_custom_voice.h"
#include "engine/models/qwen3_tts/prompt_tts_voice_design.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_tts {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kDefaultTextChunkSize = 8192;
constexpr int64_t kDefaultStreamChunkFrames = 2;
constexpr int64_t kDefaultDecoderContextFrames = 25;

std::shared_ptr<const Qwen3TTSAssets> require_assets(std::shared_ptr<const Qwen3TTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Qwen3 TTS session requires assets");
    }
    return assets;
}

Qwen3TTSGenerationOptions generation_options_from_request_impl(
    const runtime::TaskRequest & request,
    const Qwen3TTSConfig & config) {
    Qwen3TTSGenerationOptions options;
    options.max_new_tokens = config.max_new_tokens;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("Qwen3 TTS max_tokens must be positive");
        }
        options.max_new_tokens = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        options.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::find_option(request.options, {"subtalker_do_sample"})) {
        options.subtalker_do_sample = runtime::parse_bool_option(*value, "subtalker_do_sample");
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        options.temperature = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        options.top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        options.top_p = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"repetition_penalty"})) {
        options.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"subtalker_temperature"})) {
        options.subtalker_temperature = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"subtalker_top_k"})) {
        options.subtalker_top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"subtalker_top_p"})) {
        options.subtalker_top_p = *value;
    }
    if (options.do_sample && options.temperature <= 0.0F) {
        throw std::runtime_error("Qwen3 TTS temperature must be positive when sampling");
    }
    if (options.top_k < 0) {
        throw std::runtime_error("Qwen3 TTS top_k must be non-negative");
    }
    if (options.top_p < 0.0F || options.top_p > 1.0F) {
        throw std::runtime_error("Qwen3 TTS top_p must be in [0, 1]");
    }
    if (options.repetition_penalty <= 0.0F) {
        throw std::runtime_error("Qwen3 TTS repetition_penalty must be positive");
    }
    if (options.subtalker_do_sample && options.subtalker_temperature <= 0.0F) {
        throw std::runtime_error("Qwen3 TTS subtalker_temperature must be positive when sampling");
    }
    if (options.subtalker_top_k < 0) {
        throw std::runtime_error("Qwen3 TTS subtalker_top_k must be non-negative");
    }
    if (options.subtalker_top_p < 0.0F || options.subtalker_top_p > 1.0F) {
        throw std::runtime_error("Qwen3 TTS subtalker_top_p must be in [0, 1]");
    }
    options.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    return options;
}

uint64_t fnv1a_mix(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t hash = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        hash = fnv1a_mix(hash, &bits, sizeof(bits));
    }
    return hash;
}

core::BackendConfig voice_prompt_backend_config(const runtime::SessionOptions & options) {
    core::BackendConfig config = options.backend;
    // Voice-clone prompt codes are discrete argmax outputs; keep this stage on CPU so CUDA TF32 math
    // cannot change the reference prompt that the main talker conditions on.
    config.type = core::BackendType::Cpu;
    return config;
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"qwen3_tts.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "qwen3_tts.mem_saver");
    }
    return false;
}

Qwen3TTSPerfMode perf_mode_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"qwen3_tts.perf_mode"})) {
        if (*value == "off" || *value == "standard") {
            return Qwen3TTSPerfMode::Standard;
        }
        if (*value == "flash_attention") {
            return Qwen3TTSPerfMode::FlashAttention;
        }
        throw std::runtime_error("Invalid qwen3_tts.perf_mode: " + *value);
    }
    return Qwen3TTSPerfMode::Standard;
}

bool source_contains_q8_tensor(const assets::TensorSource & source) {
    for (const auto & tensor : source.tensors()) {
        if (assets::tensor_storage_type_for_dtype(tensor.dtype) == assets::TensorStorageType::Q8_0) {
            return true;
        }
    }
    return false;
}

std::size_t voice_prompt_cache_slots_from_options(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultCacheSlots = 1;
    const int64_t slots = runtime::parse_i64_option(options.options, {"qwen3_tts.voice_prompt_cache_slots"})
        .value_or(kDefaultCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("qwen3_tts.voice_prompt_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("qwen3_tts.voice_prompt_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

std::size_t prefill_graph_cache_slots_from_options(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultCacheSlots = 1;
    const int64_t slots = runtime::parse_i64_option(options.options, {"qwen3_tts.prefill_graph_cache_slots"})
        .value_or(kDefaultCacheSlots);
    if (slots <= 0) {
        throw std::runtime_error("qwen3_tts.prefill_graph_cache_slots must be positive");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("qwen3_tts.prefill_graph_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

void validate_talker_weight_storage(engine::assets::TensorStorageType storage_type) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error("Qwen3 TTS talker_weight_type currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_conv_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, and f16");
}

class TalkerCachedStepReleaseGuard {
public:
    TalkerCachedStepReleaseGuard(Qwen3TalkerStepRuntime * runtime, bool enabled)
        : runtime_(runtime), enabled_(enabled) {}

    ~TalkerCachedStepReleaseGuard() noexcept {
        try {
            release();
        } catch (...) {
            // Cleanup must not replace the request exception during stack unwinding.
        }
    }

    void release() {
        if (!enabled_ || released_ || runtime_ == nullptr) {
            return;
        }
        const auto release_start = Clock::now();
        const int64_t released_steps = runtime_->release_cached_step_graph();
        debug::timing_log_scalar(
            "qwen3_tts.talker.cached_step_release_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
        debug::timing_log_scalar("qwen3_tts.talker.cached_step_released_steps", released_steps);
        released_ = true;
    }

private:
    Qwen3TalkerStepRuntime * runtime_ = nullptr;
    bool enabled_ = false;
    bool released_ = false;
};

Qwen3TTSStreamDecoderOptions stream_decoder_options_from_request(
    const runtime::TaskRequest & request) {
    Qwen3TTSStreamDecoderOptions out;
    out.chunk_frames = runtime::parse_i64_option(
        request.options,
        {"chunk_frames", "qwen3_tts.chunk_frames"})
        .value_or(kDefaultStreamChunkFrames);
    out.context_frames = runtime::parse_i64_option(
        request.options,
        {"decoder_context_frames", "qwen3_tts.decoder_context_frames"})
        .value_or(kDefaultDecoderContextFrames);
    if (out.chunk_frames <= 0) {
        throw std::runtime_error("Qwen3 TTS chunk_frames must be positive");
    }
    if (out.context_frames < 0) {
        throw std::runtime_error("Qwen3 TTS decoder_context_frames must be non-negative");
    }
    return out;
}

bool stream_accumulate_from_request(const runtime::TaskRequest & request) {
    if (const auto value = runtime::find_option(
            request.options,
            {"stream_accumulate", "qwen3_tts.stream_accumulate"})) {
        return runtime::parse_bool_option(*value, "stream_accumulate");
    }
    return false;
}

}  // namespace

Qwen3TTSGenerationOptions qwen3_tts_generation_options_from_request(
    const runtime::TaskRequest & request,
    const Qwen3TTSConfig & config) {
    return generation_options_from_request_impl(request, config);
}

bool Qwen3TTSSession::VoicePromptCacheKeyEqual::operator()(
    const VoicePromptCacheKey & lhs,
    const VoicePromptCacheKey & rhs) const noexcept {
    return lhs.reference_text == rhs.reference_text &&
        lhs.mode == rhs.mode &&
        lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

Qwen3TTSSession::Qwen3TTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Qwen3TTSAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      mem_saver_(mem_saver_from_options(options)),
      perf_mode_(perf_mode_from_options(options)),
      text_tokenizer_(assets_),
      talker_(assets_->config.talker),
      voice_prompt_context_(voice_prompt_backend_config(options)),
      voice_prompt_cache_(voice_prompt_cache_slots_from_options(options)) {
    std::cerr << "Qwen3-TTS startup: configuring resident runtimes...\n";
    talker_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.talker_graph_arena_mb"}, talker_graph_arena_bytes_);
    speech_encoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speech_encoder_graph_arena_mb"}, speech_encoder_graph_arena_bytes_);
    speech_decoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speech_decoder_graph_arena_mb"}, speech_decoder_graph_arena_bytes_);
    speaker_encoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speaker_encoder_graph_arena_mb"}, speaker_encoder_graph_arena_bytes_);
    talker_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.talker_constant_context_mb"}, talker_constant_context_bytes_);
    code_predictor_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.code_predictor_constant_context_mb"}, code_predictor_constant_context_bytes_);
    speech_decoder_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speech_decoder_constant_context_mb"}, speech_decoder_constant_context_bytes_);
    if (const auto it = options.options.find("qwen3_tts.weight_type"); it != options.options.end()) {
        const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(storage_type, "qwen3_tts.weight_type");
        validate_talker_weight_storage(storage_type);
        talker_weight_storage_type_ = storage_type;
    }
    if (const auto it = options.options.find("qwen3_tts.conv_weight_type"); it != options.options.end()) {
        conv_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_conv_weight_storage(conv_weight_storage_type_, "qwen3_tts.conv_weight_type");
    }
    if (const auto it = options.options.find("qwen3_tts.talker_weight_type"); it != options.options.end()) {
        talker_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_talker_weight_storage(talker_weight_storage_type_);
    }
    if (const auto it = options.options.find("qwen3_tts.speech_encoder_weight_type"); it != options.options.end()) {
        speech_encoder_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(speech_encoder_weight_storage_type_, "qwen3_tts.speech_encoder_weight_type");
    }
    if (const auto it = options.options.find("qwen3_tts.speech_decoder_weight_type"); it != options.options.end()) {
        speech_decoder_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(speech_decoder_weight_storage_type_, "qwen3_tts.speech_decoder_weight_type");
    }
    if (perf_mode_ == Qwen3TTSPerfMode::FlashAttention &&
        (!source_contains_q8_tensor(*assets_->model_weights) ||
         !source_contains_q8_tensor(*assets_->speech_tokenizer_weights))) {
        throw std::runtime_error("qwen3_tts.perf_mode=flash_attention is supported only with Q8_0 GGUF weights");
    }
    for (const auto & [key, _] : options.options) {
        if (key.rfind("qwen3_tts.", 0) == 0 &&
            key != "qwen3_tts.talker_graph_arena_mb" &&
            key != "qwen3_tts.speech_encoder_graph_arena_mb" &&
            key != "qwen3_tts.speech_decoder_graph_arena_mb" &&
            key != "qwen3_tts.speaker_encoder_graph_arena_mb" &&
            key != "qwen3_tts.talker_constant_context_mb" &&
            key != "qwen3_tts.code_predictor_constant_context_mb" &&
            key != "qwen3_tts.speech_decoder_constant_context_mb" &&
            key != "qwen3_tts.weight_type" &&
            key != "qwen3_tts.conv_weight_type" &&
            key != "qwen3_tts.talker_weight_type" &&
            key != "qwen3_tts.speech_encoder_weight_type" &&
            key != "qwen3_tts.speech_decoder_weight_type" &&
            key != "qwen3_tts.voice_prompt_cache_slots" &&
            key != "qwen3_tts.prefill_graph_cache_slots" &&
            key != "qwen3_tts.perf_mode" &&
            key != "qwen3_tts.mem_saver") {
            throw std::runtime_error("unknown Qwen3 TTS session option: " + key);
        }
    }
    std::cerr << "Qwen3-TTS startup: loading talker and code-predictor weights...\n";
    talker_weights_ = talker_.create_weights_runtime(
        assets_,
        options.backend.type,
        options.backend.device,
        std::max(1, options.backend.threads),
        talker_graph_arena_bytes_,
        talker_constant_context_bytes_,
        code_predictor_constant_context_bytes_,
        talker_weight_storage_type_,
        perf_mode_);
    std::cerr << "Qwen3-TTS startup: talker weights ready; creating step runtime...\n";
    talker_step_ = talker_.create_step_runtime(
        talker_weights_,
        assets_->config.talker.max_position_embeddings,
        assets_->config.max_new_tokens,
        prefill_graph_cache_slots_from_options(options));
    std::cerr << "Qwen3-TTS startup: step runtime ready; loading speech decoder...\n";
    speech_decoder_ = std::make_unique<Qwen3SpeechTokenizerDecoderRuntime>(
        assets_,
        execution_context(),
        speech_decoder_graph_arena_bytes_,
        speech_decoder_constant_context_bytes_,
        speech_decoder_weight_storage_type_,
        conv_weight_storage_type_,
        perf_mode_);
    std::cerr << "Qwen3-TTS startup: speech decoder ready.\n";
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Qwen3 TTS requires an offline or streaming session");
    }
    if (assets_->config.variant == Qwen3TTSVariant::Base && task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Qwen3 base TTS model only supports the Tts task");
    }
    if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign && task_.task != runtime::VoiceTaskKind::VoiceDesign) {
        throw std::runtime_error("Qwen3 voice design model only supports the VoiceDesign task");
    }
    if (assets_->config.variant == Qwen3TTSVariant::CustomVoice && task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Qwen3 custom voice model only supports the Tts task");
    }
    if (assets_->config.variant == Qwen3TTSVariant::Base) {
        std::cerr << "Qwen3-TTS startup: loading reference encoders...\n";
        speech_encoder_ = std::make_unique<Qwen3SpeechTokenizerEncoderRuntime>(
            assets_,
            voice_prompt_context_,
            speech_encoder_graph_arena_bytes_,
            speech_encoder_weight_storage_type_,
            conv_weight_storage_type_,
            perf_mode_);
        speaker_encoder_ = std::make_unique<Qwen3SpeakerEncoderRuntime>(
            assets_,
            voice_prompt_context_,
            speaker_encoder_graph_arena_bytes_,
            conv_weight_storage_type_);
        std::cerr << "Qwen3-TTS startup: reference encoders ready.\n";
    }
}

std::string Qwen3TTSSession::family() const {
    return "qwen3_tts";
}

runtime::VoiceTaskKind Qwen3TTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Qwen3TTSSession::run_mode() const {
    return task_.mode;
}

void Qwen3TTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

runtime::TaskResult Qwen3TTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("Qwen3 TTS run");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 TTS run() requires an offline session");
    }
    const auto wall_start = Clock::now();
    TalkerCachedStepReleaseGuard release_talker_cached_step_graph(talker_step_.get(), mem_saver_);
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
    if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign) {
        Qwen3TTSVoiceDesignPromptBuilder prompt_builder(
            text_tokenizer_,
            assets_->config.talker.max_position_embeddings,
            assets_->config.talker.max_position_embeddings);
        double prefill_ms = 0.0;
        double talker_ms = 0.0;
        double decoder_ms = 0.0;
        runtime::AudioBuffer merged_audio;
        for (const auto & chunk_request : chunk_requests) {
            const Qwen3TTSRequest qwen_request = make_request(chunk_request);
            const auto prefill_start = Clock::now();
            const auto prefill = prompt_builder.build_prefill(qwen_request);
            prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
            const auto talker_start = Clock::now();
            const auto codes = talker_step_->generate(
                prefill,
                qwen_request.generation,
                qwen_request.generation.repetition_penalty);
            talker_ms += engine::debug::elapsed_ms(talker_start, Clock::now());
            const auto decoder_start = Clock::now();
            runtime::append_audio_buffer(
                merged_audio,
                speech_decoder_->decode(codes.generated_codes));
            decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
        }
        release_talker_cached_step_graph.release();
        runtime::TaskResult result;
        result.audio_output = std::move(merged_audio);
        debug::timing_log_scalar("qwen3_tts.voice_design_prefill_build_ms", prefill_ms);
        debug::timing_log_scalar("qwen3_tts.talker_ms", talker_ms);
        debug::timing_log_scalar("qwen3_tts.speech_decoder_ms", decoder_ms);
        debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
        return result;
    }
    if (assets_->config.variant == Qwen3TTSVariant::CustomVoice) {
        Qwen3TTSCustomVoicePromptBuilder prompt_builder(
            text_tokenizer_,
            assets_->config.talker.max_position_embeddings,
            assets_->config.talker.max_position_embeddings);
        double prefill_ms = 0.0;
        double talker_ms = 0.0;
        double decoder_ms = 0.0;
        runtime::AudioBuffer merged_audio;
        for (const auto & chunk_request : chunk_requests) {
            const Qwen3TTSRequest qwen_request = make_request(chunk_request);
            const auto prefill_start = Clock::now();
            const auto prefill = prompt_builder.build_prefill(qwen_request);
            prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
            const auto talker_start = Clock::now();
            const auto codes = talker_step_->generate(
                prefill,
                qwen_request.generation,
                qwen_request.generation.repetition_penalty);
            talker_ms += engine::debug::elapsed_ms(talker_start, Clock::now());
            const auto decoder_start = Clock::now();
            runtime::append_audio_buffer(
                merged_audio,
                speech_decoder_->decode(codes.generated_codes));
            decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
        }
        release_talker_cached_step_graph.release();
        runtime::TaskResult result;
        result.audio_output = std::move(merged_audio);
        debug::timing_log_scalar("qwen3_tts.custom_voice_prefill_build_ms", prefill_ms);
        debug::timing_log_scalar("qwen3_tts.talker_ms", talker_ms);
        debug::timing_log_scalar("qwen3_tts.speech_decoder_ms", decoder_ms);
        debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
        return result;
    }
    const Qwen3TTSRequest first_request = make_request(chunk_requests.front());
    if (!first_request.voice_clone.has_value()) {
        throw std::runtime_error("Qwen3 base TTS requires voice clone reference audio");
    }
    if (speech_encoder_ == nullptr || speaker_encoder_ == nullptr) {
        throw std::runtime_error("Qwen3 base TTS session is missing voice clone runtimes");
    }
    Qwen3TTSVoiceClonePromptBuilder prompt_builder(
        text_tokenizer_,
        *speech_encoder_,
        *speaker_encoder_,
        assets_->config.talker.max_position_embeddings);
    double prompt_ms = 0.0;
    double prefill_ms = 0.0;
    double talker_ms = 0.0;
    double decoder_ms = 0.0;
    runtime::AudioBuffer merged_audio;
    for (const auto & chunk_request : chunk_requests) {
        const Qwen3TTSRequest qwen_request = make_request(chunk_request);
        const auto prompt_start = Clock::now();
        const auto & voice_prompt = resolve_voice_prompt(*qwen_request.voice_clone, prompt_builder);
        prompt_ms += engine::debug::elapsed_ms(prompt_start, Clock::now());
        const auto prefill_start = Clock::now();
        const auto prefill = prompt_builder.build_prefill(qwen_request, voice_prompt);
        prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
        const auto talker_start = Clock::now();
        const auto codes = talker_step_->generate(
            prefill,
            qwen_request.generation,
            qwen_request.generation.repetition_penalty);
        talker_ms += engine::debug::elapsed_ms(talker_start, Clock::now());
        const auto decoder_start = Clock::now();
        runtime::AudioBuffer decoded = voice_prompt.reference_codes.has_value()
            ? speech_decoder_->decode_and_trim_reference(
                  *voice_prompt.reference_codes,
                  codes.generated_codes)
            : speech_decoder_->decode(codes.generated_codes);
        runtime::append_audio_buffer(merged_audio, decoded);
        decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
    }
    release_talker_cached_step_graph.release();
    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    debug::timing_log_scalar("qwen3_tts.voice_prompt_ms", prompt_ms);
    debug::timing_log_scalar("qwen3_tts.prefill_build_ms", prefill_ms);
    debug::timing_log_scalar("qwen3_tts.talker_ms", talker_ms);
    debug::timing_log_scalar("qwen3_tts.speech_decoder_ms", decoder_ms);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

runtime::StreamingPolicy Qwen3TTSSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    // Events are pushed from inside the talker frame callback. Because the
    // sink is synchronous, it also supplies bounded backpressure: generation
    // cannot outrun a slow audio or network consumer.
    policy.output = runtime::StreamingOutputKind::FinalResult;
    return policy;
}

void Qwen3TTSSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Qwen3 TTS start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Qwen3 TTS start_stream() requires a streaming session");
    }
    reset();
    last_streaming_metrics_.reset();
    stream_active_ = true;
    stream_cancelled_.store(false, std::memory_order_release);

    const auto wall_start = Clock::now();
    const auto decoder_options = stream_decoder_options_from_request(request);
    const bool accumulate = stream_accumulate_from_request(request);
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
    runtime::AudioBuffer accumulated{
        static_cast<int>(kQwen3TTSSampleRate),
        1,
        {},
    };
    TalkerCachedStepReleaseGuard release_talker_cached_step_graph(talker_step_.get(), mem_saver_);
    int64_t total_codec_frames = 0;
    int64_t emitted_audio_chunks = 0;
    double decoder_ms = 0.0;
    std::optional<Clock::time_point> first_codec_at;
    std::optional<Clock::time_point> first_pcm_at;
    int64_t codec_frames_at_first_pcm = 0;
    std::unique_ptr<Qwen3SpeechTokenizerStreamDecoder> stream_decoder;

    auto emit_audio = [&](runtime::AudioBuffer audio) {
        if (audio.samples.empty()) {
            return;
        }
        if (!first_pcm_at.has_value()) {
            first_pcm_at = Clock::now();
            codec_frames_at_first_pcm = total_codec_frames;
        }
        ++emitted_audio_chunks;
        if (accumulate) {
            runtime::append_audio_buffer(accumulated, audio);
        }
        if (stream_sink_) {
            runtime::StreamEvent event;
            event.audio_output = std::move(audio);
            stream_sink_(event);
        }
    };

    auto synthesize_prefill = [&](
        const Qwen3TalkerPrefill & prefill,
        const Qwen3TTSGenerationOptions & generation,
        const std::optional<Qwen3SpeechCodes> & reference_codes) {
        if (stream_decoder == nullptr) {
            stream_decoder = std::make_unique<Qwen3SpeechTokenizerStreamDecoder>(
                *speech_decoder_,
                assets_->config.talker.num_code_groups,
                decoder_options,
                reference_codes);
        }
        (void)talker_step_->generate_streaming(
            prefill,
            generation,
            [&](const Qwen3TalkerFrame & frame) {
                if (!first_codec_at.has_value()) {
                    first_codec_at = Clock::now();
                }
                ++total_codec_frames;
                const auto decode_start = Clock::now();
                auto audio = stream_decoder->push_frame(frame.codes);
                decoder_ms += engine::debug::elapsed_ms(decode_start, Clock::now());
                if (audio.has_value()) {
                    emit_audio(std::move(*audio));
                }
                return !stream_cancelled_.load(std::memory_order_acquire);
            },
            [&] { return stream_cancelled_.load(std::memory_order_acquire); },
            generation.repetition_penalty);
    };

    try {
        if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign) {
            Qwen3TTSVoiceDesignPromptBuilder prompt_builder(
                text_tokenizer_,
                assets_->config.talker.max_position_embeddings,
                assets_->config.talker.max_position_embeddings);
            for (const auto & chunk_request : chunk_requests) {
                if (stream_cancelled_.load(std::memory_order_acquire)) {
                    break;
                }
                const auto qwen_request = make_request(chunk_request);
                synthesize_prefill(
                    prompt_builder.build_prefill(qwen_request),
                    qwen_request.generation,
                    std::nullopt);
            }
        } else if (assets_->config.variant == Qwen3TTSVariant::CustomVoice) {
            Qwen3TTSCustomVoicePromptBuilder prompt_builder(
                text_tokenizer_,
                assets_->config.talker.max_position_embeddings,
                assets_->config.talker.max_position_embeddings);
            for (const auto & chunk_request : chunk_requests) {
                if (stream_cancelled_.load(std::memory_order_acquire)) {
                    break;
                }
                const auto qwen_request = make_request(chunk_request);
                synthesize_prefill(
                    prompt_builder.build_prefill(qwen_request),
                    qwen_request.generation,
                    std::nullopt);
            }
        } else {
            const auto first_request = make_request(chunk_requests.front());
            if (!first_request.voice_clone.has_value()) {
                throw std::runtime_error("Qwen3 base TTS requires voice clone reference audio");
            }
            if (speech_encoder_ == nullptr || speaker_encoder_ == nullptr) {
                throw std::runtime_error("Qwen3 base TTS session is missing voice clone runtimes");
            }
            Qwen3TTSVoiceClonePromptBuilder prompt_builder(
                text_tokenizer_,
                *speech_encoder_,
                *speaker_encoder_,
                assets_->config.talker.max_position_embeddings);
            // Reference encoding and speaker extraction happen once for the
            // whole request, including long-text segment transitions.
            const auto & voice_prompt = resolve_voice_prompt(
                *first_request.voice_clone,
                prompt_builder);
            for (const auto & chunk_request : chunk_requests) {
                if (stream_cancelled_.load(std::memory_order_acquire)) {
                    break;
                }
                const auto qwen_request = make_request(chunk_request);
                synthesize_prefill(
                    prompt_builder.build_prefill(qwen_request, voice_prompt),
                    qwen_request.generation,
                    voice_prompt.reference_codes);
            }
        }

        if (!stream_cancelled_.load(std::memory_order_acquire) && stream_decoder != nullptr) {
            const auto decode_start = Clock::now();
            auto tail = stream_decoder->flush();
            decoder_ms += engine::debug::elapsed_ms(decode_start, Clock::now());
            if (tail.has_value()) {
                emit_audio(std::move(*tail));
            }
        }

        const auto generation_finished_at = Clock::now();
        release_talker_cached_step_graph.release();
        runtime::TaskResult result;
        if (accumulate) {
            result.audio_output = std::move(accumulated);
        }
        streaming_result_ = std::move(result);
        stream_active_ = false;

        if (stream_sink_) {
            runtime::StreamEvent event;
            event.is_final = true;
            stream_sink_(event);
        }
        const double total_ms = engine::debug::elapsed_ms(wall_start, generation_finished_at);
        Qwen3TTSStreamingMetrics metrics;
        metrics.decoder_ms = decoder_ms;
        metrics.total_ms = total_ms;
        metrics.codec_frames = total_codec_frames;
        metrics.audio_chunks = emitted_audio_chunks;
        if (first_codec_at.has_value()) {
            metrics.time_to_first_codec_ms = engine::debug::elapsed_ms(wall_start, *first_codec_at);
        }
        if (first_pcm_at.has_value()) {
            metrics.time_to_first_pcm_ms = engine::debug::elapsed_ms(wall_start, *first_pcm_at);
            metrics.first_pcm_before_generation_end = codec_frames_at_first_pcm < total_codec_frames;
        }
        last_streaming_metrics_ = metrics;
        debug::timing_log_scalar("qwen3_tts.streaming.chunk_frames", decoder_options.chunk_frames);
        debug::timing_log_scalar("qwen3_tts.streaming.decoder_context_frames", decoder_options.context_frames);
        debug::timing_log_scalar("qwen3_tts.streaming.codec_frames", total_codec_frames);
        debug::timing_log_scalar("qwen3_tts.streaming.codec_frames_at_first_pcm", codec_frames_at_first_pcm);
        debug::timing_log_scalar("qwen3_tts.streaming.audio_chunks", emitted_audio_chunks);
        debug::timing_log_scalar("qwen3_tts.streaming.decoder_ms", decoder_ms);
        debug::timing_log_scalar("qwen3_tts.streaming.total_ms", total_ms);
        if (first_codec_at.has_value()) {
            debug::timing_log_scalar(
                "qwen3_tts.streaming.time_to_first_codec_ms",
                engine::debug::elapsed_ms(wall_start, *first_codec_at));
        }
        if (first_pcm_at.has_value()) {
            debug::timing_log_scalar(
                "qwen3_tts.streaming.time_to_first_pcm_ms",
                engine::debug::elapsed_ms(wall_start, *first_pcm_at));
            debug::timing_log_scalar(
                "qwen3_tts.streaming.first_pcm_before_generation_end",
                codec_frames_at_first_pcm < total_codec_frames ? 1 : 0);
        }
        debug::timing_log_scalar("session.wall_ms", total_ms);
    } catch (...) {
        stream_active_ = false;
        streaming_result_.reset();
        throw;
    }
}

std::optional<runtime::StreamEvent> Qwen3TTSSession::next_stream_event() {
    return std::nullopt;
}

void Qwen3TTSSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_sink_ = std::move(sink);
}

void Qwen3TTSSession::cancel_stream() {
    stream_cancelled_.store(true, std::memory_order_release);
}

runtime::TaskResult Qwen3TTSSession::finish_stream() {
    if (stream_active_) {
        throw std::runtime_error("Qwen3 TTS stream is still generating");
    }
    if (!streaming_result_.has_value()) {
        throw std::runtime_error("Qwen3 TTS streaming has not been started");
    }
    auto result = std::move(*streaming_result_);
    streaming_result_.reset();
    return result;
}

void Qwen3TTSSession::reset() {
    stream_cancelled_.store(true, std::memory_order_release);
    streaming_result_.reset();
    stream_active_ = false;
}

runtime::StreamEvent Qwen3TTSSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    (void)chunk;
    throw std::runtime_error("Qwen3 TTS streaming does not consume audio chunks");
}

runtime::TaskResult Qwen3TTSSession::finalize() {
    return finish_stream();
}

const std::optional<Qwen3TTSStreamingMetrics> & Qwen3TTSSession::last_streaming_metrics() const noexcept {
    return last_streaming_metrics_;
}

const Qwen3VoiceClonePrompt & Qwen3TTSSession::resolve_voice_prompt(
    const Qwen3VoiceCloneInput & input,
    const Qwen3TTSVoiceClonePromptBuilder & prompt_builder) {
    const uint64_t sample_count = static_cast<uint64_t>(input.reference_audio.samples.size());
    const uint64_t sample_hash = hash_audio_samples(input.reference_audio);
    VoicePromptCacheKey key;
    key.reference_text = input.reference_text;
    key.mode = input.mode;
    key.sample_rate = input.reference_audio.sample_rate;
    key.channels = input.reference_audio.channels;
    key.sample_count = sample_count;
    key.sample_hash = sample_hash;
    if (auto * cached = voice_prompt_cache_.find(key)) {
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.hit", 1);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.slots", static_cast<int64_t>(voice_prompt_cache_.capacity()));
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.entries", static_cast<int64_t>(voice_prompt_cache_.size()));
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.evicted", 0);
        return cached->prompt;
    }

    VoicePromptCacheEntry entry;
    entry.prompt = prompt_builder.build_voice_prompt(input);
    if (voice_prompt_cache_.capacity() == 0) {
        uncached_voice_prompt_ = std::move(entry);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.hit", 0);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.slots", 0);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.entries", 0);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.evicted", 0);
        return uncached_voice_prompt_->prompt;
    }
    const bool will_evict = voice_prompt_cache_.size() >= voice_prompt_cache_.capacity();
    voice_prompt_cache_.put(std::move(key), std::move(entry));
    auto * cached = voice_prompt_cache_.find(VoicePromptCacheKey{
        input.reference_text,
        input.mode,
        input.reference_audio.sample_rate,
        input.reference_audio.channels,
        sample_count,
        sample_hash,
    });
    if (cached == nullptr) {
        throw std::runtime_error("Qwen3 TTS voice prompt cache insert failed");
    }
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.hit", 0);
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.slots", static_cast<int64_t>(voice_prompt_cache_.capacity()));
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.entries", static_cast<int64_t>(voice_prompt_cache_.size()));
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.evicted", will_evict ? 1 : 0);
    return cached->prompt;
}

Qwen3TTSRequest Qwen3TTSSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value()) {
        throw std::runtime_error("Qwen3 TTS requires text input");
    }
    Qwen3TTSRequest out;
    out.text = request.text_input->text;
    out.language = !request.text_input->language.empty() ? request.text_input->language : "Auto";
    out.generation = qwen3_tts_generation_options_from_request(request, assets_->config);
    if (assets_->config.variant == Qwen3TTSVariant::Base) {
        const runtime::AudioBuffer * reference_audio = nullptr;
        if (request.voice.has_value()
            && request.voice->speaker.has_value()
            && request.voice->speaker->audio.has_value()) {
            reference_audio = &*request.voice->speaker->audio;
        } else if (request.audio_input.has_value()) {
            reference_audio = &*request.audio_input;
        }
        if (reference_audio != nullptr) {
            Qwen3VoiceCloneInput voice_clone;
            voice_clone.reference_audio = *reference_audio;
            if (const auto reference_text = runtime::find_option(
                    request.options,
                    {"reference_text"})) {
                voice_clone.reference_text = *reference_text;
            }
            bool x_vector_only = false;
            if (const auto value = runtime::find_option(
                    request.options,
                    {"x_vector_only_mode", "speaker_embedding_only"})) {
                x_vector_only = runtime::parse_bool_option(*value, "speaker_embedding_only");
            }
            voice_clone.mode = x_vector_only
                ? Qwen3VoiceCloneMode::SpeakerEmbeddingOnly
                : Qwen3VoiceCloneMode::Icl;
            out.voice_clone = std::move(voice_clone);
        }
    } else if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign) {
        Qwen3VoiceDesignInput voice_design;
        voice_design.instruct = runtime::find_option(
            request.options,
            {"instruction", "instruct"})
            .value_or("");
        if (voice_design.instruct.empty()
            && request.voice.has_value()
            && request.voice->style.has_value()) {
            const auto tag = request.voice->style->tags.find("instruct");
            if (tag != request.voice->style->tags.end()) {
                voice_design.instruct = tag->second;
            }
        }
        out.voice_design = std::move(voice_design);
    } else if (assets_->config.variant == Qwen3TTSVariant::CustomVoice) {
        Qwen3CustomVoiceInput custom_voice;
        custom_voice.speaker = runtime::find_option(request.options, {"speaker"}).value_or("");
        if (custom_voice.speaker.empty() && request.voice.has_value() && request.voice->speaker.has_value()) {
            custom_voice.speaker = request.voice->speaker->cached_voice_id.value_or("");
        }
        custom_voice.instruct = runtime::find_option(
            request.options,
            {"instruction", "instruct"})
            .value_or("");
        if (custom_voice.instruct.empty()
            && request.voice.has_value()
            && request.voice->style.has_value()) {
            const auto tag = request.voice->style->tags.find("instruct");
            if (tag != request.voice->style->tags.end()) {
                custom_voice.instruct = tag->second;
            }
        }
        out.custom_voice = std::move(custom_voice);
    }
    return out;
}

}  // namespace engine::models::qwen3_tts
