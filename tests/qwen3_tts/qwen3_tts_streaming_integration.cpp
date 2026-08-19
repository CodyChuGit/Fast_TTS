#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/qwen3_tts/session.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::optional<std::string> arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == name) return argv[i + 1];
    return std::nullopt;
}

engine::core::BackendType backend(const std::string & value) {
    if (value == "cpu") return engine::core::BackendType::Cpu;
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "vulkan") return engine::core::BackendType::Vulkan;
    if (value == "metal") return engine::core::BackendType::Metal;
    throw std::runtime_error("unsupported backend: " + value);
}

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto model_path = arg(argc, argv, "--model");
        if (!model_path.has_value()) {
            std::cerr << "usage: qwen3_tts_streaming_integration --model PATH "
                         "--variant base|custom|design [--reference-audio WAV] "
                         "[--reference-text TEXT] [--speaker NAME] [--instruction TEXT] "
                         "[--backend cpu|cuda] [--mae-tolerance 0.05]\n";
            return 2;
        }
        const std::string variant = arg(argc, argv, "--variant").value_or("base");
        require(variant == "base" || variant == "custom" || variant == "design", "invalid variant");
        const auto task = variant == "design"
            ? engine::runtime::VoiceTaskKind::VoiceDesign
            : engine::runtime::VoiceTaskKind::Tts;
        const double mae_tolerance = std::stod(arg(argc, argv, "--mae-tolerance").value_or("0.05"));

        engine::runtime::ModelLoadRequest load;
        load.model_path = std::filesystem::path(*model_path);
        load.family_hint = "qwen3_tts";
        auto model = engine::runtime::make_default_registry().load(load);
        engine::runtime::SessionOptions session_options;
        session_options.backend.type = backend(arg(argc, argv, "--backend").value_or("cpu"));
        session_options.backend.device = std::stoi(arg(argc, argv, "--device").value_or("0"));
        session_options.backend.threads = std::stoi(arg(argc, argv, "--threads").value_or("4"));

        engine::runtime::TaskRequest request;
        request.text_input = engine::runtime::Transcript{
            arg(argc, argv, "--text").value_or(
                "This integration test uses a long sentence so that several audio chunks arrive before codec generation reaches its final frame."),
            arg(argc, argv, "--language").value_or("Auto"),
        };
        request.options["do_sample"] = "false";
        request.options["subtalker_do_sample"] = "false";
        request.options["seed"] = "1234";
        request.options["max_tokens"] = arg(argc, argv, "--max-tokens").value_or("512");
        if (const auto value = arg(argc, argv, "--speaker")) request.options["speaker"] = *value;
        if (const auto value = arg(argc, argv, "--instruction")) request.options["instruction"] = *value;
        if (const auto value = arg(argc, argv, "--reference-text")) request.options["reference_text"] = *value;
        if (variant == "base") {
            const auto path = arg(argc, argv, "--reference-audio");
            require(path.has_value(), "base variant requires --reference-audio");
            const auto wav = engine::audio::read_wav_f32(std::filesystem::path(*path));
            request.voice = engine::runtime::VoiceCondition{};
            request.voice->speaker = engine::runtime::VoiceReference{};
            request.voice->speaker->audio = engine::runtime::AudioBuffer{
                wav.sample_rate, wav.channels, wav.samples};
        }
        if (variant == "custom") {
            require(request.options.find("speaker") != request.options.end(), "custom variant requires --speaker");
        }

        auto offline_base = model->create_task_session(
            {task, engine::runtime::RunMode::Offline}, session_options);
        auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(offline_base.get());
        require(offline != nullptr, "offline interface unavailable");
        offline_base->prepare(engine::runtime::build_preparation_request(request));
        const auto offline_result = offline->run(request);
        require(offline_result.audio_output.has_value(), "offline result has no audio");
        const auto offline_audio = *offline_result.audio_output;
        offline_base.reset();

        auto streaming_base = model->create_task_session(
            {task, engine::runtime::RunMode::Streaming}, session_options);
        auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(streaming_base.get());
        auto * qwen = dynamic_cast<engine::models::qwen3_tts::Qwen3TTSSession *>(streaming_base.get());
        require(streaming != nullptr && qwen != nullptr, "streaming interface unavailable");
        streaming_base->prepare(engine::runtime::build_preparation_request(request));

        for (const int chunk_frames : {1, 2, 4, 8}) {
            request.options["chunk_frames"] = std::to_string(chunk_frames);
            request.options["decoder_context_frames"] = "25";
            request.options["stream_accumulate"] = "true";
            std::vector<float> event_audio;
            size_t event_count = 0;
            streaming->set_stream_event_sink([&](const engine::runtime::StreamEvent & event) {
                if (!event.audio_output.has_value()) return;
                ++event_count;
                event_audio.insert(
                    event_audio.end(),
                    event.audio_output->samples.begin(),
                    event.audio_output->samples.end());
            });
            streaming->start_stream(request);
            const auto result = streaming->finish_stream();
            require(result.audio_output.has_value(), "stream_accumulate returned no audio");
            require(event_count > 1, "request did not produce multiple streaming events");
            require(event_audio == result.audio_output->samples, "event concatenation differs from accumulated audio");
            require(event_audio.size() == offline_audio.samples.size(), "streaming duration differs from offline output");

            double abs_error = 0.0;
            for (size_t i = 0; i < event_audio.size(); ++i) {
                abs_error += std::abs(static_cast<double>(event_audio[i]) - offline_audio.samples[i]);
            }
            const double mae = abs_error / static_cast<double>(std::max<size_t>(1, event_audio.size()));
            require(mae <= mae_tolerance, "offline/streaming MAE exceeded tolerance: " + std::to_string(mae));
            const auto metrics = qwen->last_streaming_metrics();
            require(metrics.has_value(), "streaming metrics missing");
            require(metrics->first_pcm_before_generation_end, "first PCM was not emitted before generation end");
            std::cout << "chunk_frames=" << chunk_frames << " samples=" << event_audio.size()
                      << " events=" << event_count << " mae=" << mae
                      << " ttfa_ms=" << metrics->time_to_first_pcm_ms << "\n";
        }

        request.options["chunk_frames"] = "1";
        request.options["stream_accumulate"] = "false";
        size_t cancelled_samples = 0;
        size_t cancelled_events = 0;
        streaming->set_stream_event_sink([&](const engine::runtime::StreamEvent & event) {
            if (!event.audio_output.has_value()) return;
            cancelled_samples += event.audio_output->samples.size();
            if (++cancelled_events == 2) streaming->cancel_stream();
        });
        streaming->start_stream(request);
        (void)streaming->finish_stream();
        require(cancelled_events == 2, "cancellation did not stop promptly after two events");
        require(cancelled_samples < offline_audio.samples.size(), "cancelled stream generated the complete output");

        std::cout << "cancellation_samples=" << cancelled_samples << "\n"
                  << "qwen3_tts_streaming_integration passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_tts_streaming_integration failed: " << error.what() << "\n";
        return 1;
    }
}
