#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

std::optional<std::string> arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return std::nullopt;
}

bool flag(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

engine::core::BackendType backend(const std::string & name) {
    if (name == "cpu") return engine::core::BackendType::Cpu;
    if (name == "cuda") return engine::core::BackendType::Cuda;
    if (name == "vulkan") return engine::core::BackendType::Vulkan;
    if (name == "metal") return engine::core::BackendType::Metal;
    throw std::runtime_error("unsupported backend: " + name);
}

class AudioPlayer {
public:
    explicit AudioPlayer(const std::optional<std::string> & command) {
        if (!command.has_value()) return;
#if defined(_WIN32)
        pipe_ = _popen(command->c_str(), "wb");
#else
        pipe_ = popen(command->c_str(), "w");
#endif
        if (pipe_ == nullptr) {
            throw std::runtime_error("failed to start audio player command");
        }
    }

    ~AudioPlayer() {
        if (pipe_ == nullptr) return;
#if defined(_WIN32)
        _pclose(pipe_);
#else
        pclose(pipe_);
#endif
    }

    void write(const engine::runtime::AudioBuffer & audio) {
        if (pipe_ == nullptr || audio.samples.empty()) return;
        const size_t written = std::fwrite(
            audio.samples.data(), sizeof(float), audio.samples.size(), pipe_);
        if (written != audio.samples.size()) {
            throw std::runtime_error("audio player closed while synthesis was running");
        }
        std::fflush(pipe_);
    }

private:
    FILE * pipe_ = nullptr;
};

std::optional<std::string> default_player_command(bool no_play) {
    if (no_play) return std::nullopt;
#if defined(__linux__)
    return "aplay -q -t raw -f FLOAT_LE -r 24000 -c 1";
#else
    throw std::runtime_error(
        "no default raw-f32 player is configured on this platform; pass --player-command or --no-play");
#endif
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto model_path = arg(argc, argv, "--model");
        const auto text = arg(argc, argv, "--text");
        if (!model_path.has_value() || !text.has_value()) {
            std::cerr << "usage: qwen3_tts_live --model PATH --text TEXT [--reference-audio WAV] "
                         "[--reference-text TEXT] [--speaker NAME] [--instruction TEXT] "
                         "[--voice-design] [--chunk-frames N] [--player-command CMD] [--no-play]\n";
            return 2;
        }

        engine::runtime::ModelLoadRequest load;
        load.model_path = *model_path;
        load.family_hint = "qwen3_tts";
        auto model = engine::runtime::make_default_registry().load(load);

        const bool voice_design = flag(argc, argv, "--voice-design");
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
        if (streaming == nullptr) {
            throw std::runtime_error("loaded Qwen3-TTS model does not expose streaming mode");
        }

        engine::runtime::TaskRequest request;
        request.text_input = engine::runtime::Transcript{
            *text,
            arg(argc, argv, "--language").value_or("Auto"),
        };
        request.options["chunk_frames"] = arg(argc, argv, "--chunk-frames").value_or("2");
        request.options["decoder_context_frames"] =
            arg(argc, argv, "--decoder-context-frames").value_or("25");
        if (const auto value = arg(argc, argv, "--instruction")) request.options["instruction"] = *value;
        if (const auto value = arg(argc, argv, "--speaker")) request.options["speaker"] = *value;
        if (const auto value = arg(argc, argv, "--reference-text")) request.options["reference_text"] = *value;
        if (const auto path = arg(argc, argv, "--reference-audio")) {
            const auto wav = engine::audio::read_wav_f32(std::filesystem::path(*path));
            request.voice = engine::runtime::VoiceCondition{};
            request.voice->speaker = engine::runtime::VoiceReference{};
            request.voice->speaker->audio = engine::runtime::AudioBuffer{
                wav.sample_rate, wav.channels, wav.samples};
        }

        auto player_command = arg(argc, argv, "--player-command");
        if (!player_command.has_value()) {
            player_command = default_player_command(flag(argc, argv, "--no-play"));
        }
        AudioPlayer player(player_command);
        const auto start = std::chrono::steady_clock::now();
        std::optional<std::chrono::steady_clock::time_point> first_audio;
        size_t chunks = 0;
        size_t samples = 0;
        streaming->set_stream_event_sink([&](const engine::runtime::StreamEvent & event) {
            if (!event.audio_output.has_value()) return;
            if (!first_audio.has_value()) {
                first_audio = std::chrono::steady_clock::now();
                std::cerr << "TTFA="
                          << std::chrono::duration<double, std::milli>(*first_audio - start).count()
                          << " ms\n";
            }
            ++chunks;
            samples += event.audio_output->samples.size();
            player.write(*event.audio_output);
        });
        base->prepare(engine::runtime::build_preparation_request(request));
        streaming->start_stream(request);
        (void)streaming->finish_stream();
        streaming->set_stream_event_sink(nullptr);
        const auto end = std::chrono::steady_clock::now();
        const double wall_seconds = std::chrono::duration<double>(end - start).count();
        const double audio_seconds = static_cast<double>(samples) / 24000.0;
        std::cerr << "chunks=" << chunks << " audio_seconds=" << audio_seconds
                  << " wall_seconds=" << wall_seconds
                  << " RTF=" << (audio_seconds > 0.0 ? wall_seconds / audio_seconds : 0.0) << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_tts_live failed: " << error.what() << "\n";
        return 1;
    }
}
