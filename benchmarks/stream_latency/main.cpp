#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/qwen3_tts/session.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::optional<std::string> arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == name) return argv[i + 1];
    return std::nullopt;
}

bool flag(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i < argc; ++i) if (argv[i] == name) return true;
    return false;
}

engine::core::BackendType backend(const std::string & value) {
    if (value == "cpu") return engine::core::BackendType::Cpu;
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "vulkan") return engine::core::BackendType::Vulkan;
    if (value == "metal") return engine::core::BackendType::Metal;
    throw std::runtime_error("unsupported backend: " + value);
}

double peak_rss_mib() {
#if defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) != 0) continue;
        std::istringstream values(line.substr(6));
        double kib = 0.0;
        values >> kib;
        return kib / 1024.0;
    }
#endif
    return 0.0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto model_path = arg(argc, argv, "--model");
        if (!model_path.has_value()) {
            std::cerr << "usage: qwen3_tts_stream_latency --model PATH [--reference-audio WAV] "
                         "[--reference-text TEXT] [--speaker NAME] [--instruction TEXT] "
                         "[--voice-design] [--backend cpu|cuda] [--text TEXT] "
                         "[--max-tokens N] [--timing-file PATH]\n";
            return 2;
        }
        if (const auto timing_path = arg(argc, argv, "--timing-file")) {
            const auto path = std::filesystem::path(*timing_path);
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            engine::debug::configure_logging(engine::debug::LoggingConfig{true, *timing_path});
        }
        const bool voice_design = flag(argc, argv, "--voice-design");
        const auto load_started = std::chrono::steady_clock::now();
        engine::runtime::ModelLoadRequest load;
        load.model_path = std::filesystem::path(*model_path);
        load.family_hint = "qwen3_tts";
        auto model = engine::runtime::make_default_registry().load(load);
        const double model_load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - load_started).count();

        engine::runtime::SessionOptions options;
        options.backend.type = backend(arg(argc, argv, "--backend").value_or("cpu"));
        options.backend.device = std::stoi(arg(argc, argv, "--device").value_or("0"));
        options.backend.threads = std::stoi(arg(argc, argv, "--threads").value_or("4"));
        auto base = model->create_task_session(
            {voice_design ? engine::runtime::VoiceTaskKind::VoiceDesign
                          : engine::runtime::VoiceTaskKind::Tts,
             engine::runtime::RunMode::Streaming},
            options);
        auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(base.get());
        auto * qwen = dynamic_cast<engine::models::qwen3_tts::Qwen3TTSSession *>(base.get());
        if (streaming == nullptr || qwen == nullptr) {
            throw std::runtime_error("Qwen3 streaming session interface is unavailable");
        }

        engine::runtime::TaskRequest request;
        request.text_input = engine::runtime::Transcript{
            arg(argc, argv, "--text").value_or(
                "This deliberately long benchmark sentence verifies that the first audio chunk is emitted while later codec frames are still being generated."),
            arg(argc, argv, "--language").value_or("Auto"),
        };
        request.options["do_sample"] = "false";
        request.options["subtalker_do_sample"] = "false";
        request.options["seed"] = arg(argc, argv, "--seed").value_or("1234");
        request.options["max_tokens"] = arg(argc, argv, "--max-tokens").value_or("512");
        request.options["decoder_context_frames"] =
            arg(argc, argv, "--decoder-context-frames").value_or("25");
        if (const auto value = arg(argc, argv, "--speaker")) request.options["speaker"] = *value;
        if (const auto value = arg(argc, argv, "--instruction")) request.options["instruction"] = *value;
        if (const auto value = arg(argc, argv, "--reference-text")) request.options["reference_text"] = *value;
        if (const auto path = arg(argc, argv, "--reference-audio")) {
            const auto wav = engine::audio::read_wav_f32(std::filesystem::path(*path));
            request.voice = engine::runtime::VoiceCondition{};
            request.voice->speaker = engine::runtime::VoiceReference{};
            request.voice->speaker->audio = engine::runtime::AudioBuffer{
                wav.sample_rate, wav.channels, wav.samples};
        }
        base->prepare(engine::runtime::build_preparation_request(request));

        std::cout << "model_load_ms=" << model_load_ms << "\n";
        std::cout << "chunk_frames,ttfa_ms,codec_frames_per_sec,decoder_ms_per_chunk,audio_sec_per_sec,rtf,peak_rss_mib,true_streaming\n";
        std::cout.flush();
        for (const int chunk_frames : {1, 2, 4, 8}) {
            request.options["chunk_frames"] = std::to_string(chunk_frames);
            int64_t sample_count = 0;
            streaming->set_stream_event_sink([&](const engine::runtime::StreamEvent & event) {
                if (event.audio_output.has_value()) {
                    sample_count += static_cast<int64_t>(event.audio_output->samples.size());
                }
            });
            streaming->start_stream(request);
            (void)streaming->finish_stream();
            const auto metrics = qwen->last_streaming_metrics();
            if (!metrics.has_value() || metrics->codec_frames <= chunk_frames) {
                throw std::runtime_error("benchmark text did not generate enough frames to prove streaming");
            }
            const double wall_seconds = metrics->total_ms / 1000.0;
            const double audio_seconds = static_cast<double>(sample_count) / 24000.0;
            const double codec_fps = wall_seconds > 0.0 ? metrics->codec_frames / wall_seconds : 0.0;
            const double decoder_per_chunk = metrics->audio_chunks > 0
                ? metrics->decoder_ms / metrics->audio_chunks : 0.0;
            const double audio_per_second = wall_seconds > 0.0 ? audio_seconds / wall_seconds : 0.0;
            const double rtf = audio_seconds > 0.0 ? wall_seconds / audio_seconds : 0.0;
            std::cout << chunk_frames << ',' << metrics->time_to_first_pcm_ms << ','
                      << codec_fps << ',' << decoder_per_chunk << ',' << audio_per_second << ','
                      << rtf << ',' << peak_rss_mib() << ','
                      << (metrics->first_pcm_before_generation_end ? "true" : "false") << std::endl;
        }
        streaming->set_stream_event_sink(nullptr);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_tts_stream_latency failed: " << error.what() << "\n";
        return 1;
    }
}
