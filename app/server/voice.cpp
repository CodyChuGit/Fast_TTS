#include "voice.h"

#include "engine/framework/debug/trace.h"
#include "engine/models/silero_vad/session.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

namespace minitts::server::voice {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

std::string json_escape(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Query parsing
// ---------------------------------------------------------------------------

TurnParams turn_params_from_query(const std::string & query) {
    TurnParams params;
    size_t start = 0;
    while (start < query.size()) {
        size_t end = query.find('&', start);
        if (end == std::string::npos) {
            end = query.size();
        }
        const auto pair = query.substr(start, end - start);
        start = end + 1;
        const auto eq = pair.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const auto key = pair.substr(0, eq);
        const auto value = pair.substr(eq + 1);
        try {
            if (key == "vad_threshold") params.vad_threshold = std::stof(value);
            else if (key == "min_speech_ms") params.min_speech_ms = std::stoi(value);
            else if (key == "min_silence_ms") params.min_silence_ms = std::stoi(value);
            else if (key == "max_utterance_ms") params.max_utterance_ms = std::stoi(value);
            else if (key == "partial_interval_ms") params.partial_interval_ms = std::stoi(value);
            else if (key == "min_partial_audio_ms") params.min_partial_audio_ms = std::stoi(value);
            else if (key == "speech_pad_ms") params.speech_pad_ms = std::stoi(value);
            else if (key == "stable_hypothesis_count") params.stability.stable_hypothesis_count = std::stoi(value);
            else if (key == "min_stable_chars") params.stability.min_stable_chars = std::stoi(value);
            else if (key == "language") params.language = value;
        } catch (const std::exception &) {
            // Malformed tuning values fall back to defaults; the transport
            // parameters are validated separately by the route.
        }
    }
    return params;
}

// ---------------------------------------------------------------------------
// VoiceLiveRunner
// ---------------------------------------------------------------------------

struct VoiceLiveRunner::Impl {
    TurnParams params;
    VoiceHooks hooks;
    std::unique_ptr<engine::runtime::IStreamingVoiceTaskSession> vad;

    Clock::time_point t0 = Clock::now();

    // Utterance state, owned by the audio loop.
    bool speaking = false;
    int64_t utterance_id = 0;
    std::vector<float> preroll;
    std::vector<float> utterance;  // guarded by decode_mutex (worker snapshots it)
    Clock::time_point speech_started_at;
    Clock::time_point speech_ended_at;
    Clock::time_point last_partial_enqueue;
    double first_partial_ms = -1;   // written by worker, read at final
    double first_stable_ms = -1;

    // Decode worker.
    struct Job {
        int64_t utterance_id = 0;
        bool final = false;
        engine::runtime::AudioBuffer audio;
    };
    std::mutex decode_mutex;
    std::condition_variable decode_cv;
    std::condition_variable idle_cv;
    std::optional<Job> pending;
    bool worker_busy = false;
    bool quit = false;
    std::thread worker;

    StableTracker tracker{StabilityParams{}};
    std::string last_partial_emitted;

    void emit(const std::string & json) {
        if (hooks.emit) {
            hooks.emit(json);
        }
    }

    double now_ms() const {
        return elapsed_ms(t0);
    }

    void emit_event(const char * type, int64_t utt, const std::string & extra = "") {
        std::ostringstream out;
        out << "{\"type\":\"" << type << "\",\"utterance_id\":" << utt
            << ",\"t_ms\":" << static_cast<int64_t>(now_ms());
        if (!extra.empty()) {
            out << "," << extra;
        }
        out << "}";
        emit(out.str());
    }

    void worker_loop() {
        while (true) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(decode_mutex);
                decode_cv.wait(lock, [&] { return quit || pending.has_value(); });
                if (quit && !pending.has_value()) {
                    return;
                }
                job = std::move(*pending);
                pending.reset();
                worker_busy = true;
            }
            TranscribeResult result;
            bool ok = true;
            const auto decode_start = Clock::now();
            try {
                result = hooks.transcribe(job.audio);
            } catch (const std::exception & ex) {
                ok = false;
                emit(std::string("{\"type\":\"error\",\"message\":\"") + json_escape(ex.what()) + "\"}");
            }
            const double decode_ms = elapsed_ms(decode_start);
            if (ok) {
                handle_result(job, result, decode_ms);
            }
            {
                std::lock_guard<std::mutex> lock(decode_mutex);
                worker_busy = false;
            }
            idle_cv.notify_all();
        }
    }

    void handle_result(const Job & job, const TranscribeResult & result, double decode_ms) {
        engine::debug::timing_log_scalar("voice.decode_ms", decode_ms);
        if (job.final) {
            const double final_at = now_ms();
            std::ostringstream extra;
            extra << "\"text\":\"" << json_escape(result.text) << "\""
                  << ",\"language\":\"" << json_escape(result.language) << "\""
                  << ",\"timings\":{"
                  << "\"speech_started_ms\":" << static_cast<int64_t>(elapsed_ms(t0) - elapsed_ms(speech_started_at))
                  << ",\"speech_ended_ms\":" << static_cast<int64_t>(elapsed_ms(t0) - elapsed_ms(speech_ended_at))
                  << ",\"first_partial_ms\":" << static_cast<int64_t>(first_partial_ms)
                  << ",\"first_stable_ms\":" << static_cast<int64_t>(first_stable_ms)
                  << ",\"eot_to_final_ms\":" << static_cast<int64_t>(elapsed_ms(speech_ended_at))
                  << ",\"final_decode_ms\":" << static_cast<int64_t>(decode_ms)
                  << ",\"audio_ms\":" << static_cast<int64_t>(
                         1000.0 * static_cast<double>(job.audio.samples.size()) / 16000.0)
                  << "}";
            emit_event("final_transcript", job.utterance_id, extra.str());
            engine::debug::timing_log_scalar("voice.eot_to_final_ms", elapsed_ms(speech_ended_at));
            (void) final_at;
            return;
        }
        if (result.text.empty()) {
            return;
        }
        const bool stable_grew = tracker.update(result.text);
        if (stable_grew) {
            if (first_stable_ms < 0) {
                first_stable_ms = elapsed_ms(speech_started_at);
            }
            emit_event(
                "stable_transcript", job.utterance_id,
                "\"text\":\"" + json_escape(tracker.stable()) + "\"");
        }
        const auto & tentative = tracker.stable().empty() ? tracker.hypothesis() : tracker.tentative();
        if (!tentative.empty() && tentative != last_partial_emitted) {
            last_partial_emitted = tentative;
            if (first_partial_ms < 0) {
                first_partial_ms = elapsed_ms(speech_started_at);
            }
            emit_event(
                "partial_transcript", job.utterance_id,
                "\"text\":\"" + json_escape(tentative) + "\"");
        }
    }

    void enqueue_decode(bool final) {
        Job job;
        job.utterance_id = utterance_id;
        job.final = final;
        job.audio.sample_rate = 16000;
        job.audio.channels = 1;
        {
            std::unique_lock<std::mutex> lock(decode_mutex);
            if (final) {
                // The final decode must see all audio and run after any
                // in-flight partial: wait the worker idle, then queue.
                idle_cv.wait(lock, [&] { return !worker_busy; });
            } else if (worker_busy || pending.has_value()) {
                return;  // partials are best-effort; never queue behind one
            }
            job.audio.samples = utterance;
            pending = std::move(job);
        }
        decode_cv.notify_all();
    }

    void wait_worker_idle() {
        std::unique_lock<std::mutex> lock(decode_mutex);
        idle_cv.wait(lock, [&] { return !worker_busy && !pending.has_value(); });
    }

    void begin_utterance() {
        speaking = true;
        ++utterance_id;
        speech_started_at = Clock::now();
        last_partial_enqueue = Clock::now();
        first_partial_ms = -1;
        first_stable_ms = -1;
        tracker.reset();
        last_partial_emitted.clear();
        {
            std::lock_guard<std::mutex> lock(decode_mutex);
            utterance = preroll;
        }
        emit_event("speech_started", utterance_id);
    }

    void end_utterance() {
        speaking = false;
        speech_ended_at = Clock::now();
        emit_event("speech_ended", utterance_id);
        enqueue_decode(true);
        wait_worker_idle();
        {
            std::lock_guard<std::mutex> lock(decode_mutex);
            utterance.clear();
        }
    }

    void run(const minitts::app::AudioChunkStream & stream) {
        worker = std::thread([this] { worker_loop(); });
        const size_t preroll_samples =
            static_cast<size_t>(params.speech_pad_ms) * 16;  // 16 samples/ms @16 kHz
        std::vector<float> block;
        std::vector<float> vad_pending;
        int64_t stream_sample = 0;
        int64_t vad_sample = 0;
        try {
            while (!hooks.aborted || !hooks.aborted()) {
                if (!stream.read(320, block)) {  // 20 ms
                    break;
                }
                if (block.empty()) {
                    continue;
                }
                stream_sample += static_cast<int64_t>(block.size());

                // Silero's streaming session consumes exactly 512-sample
                // (32 ms) windows; aggregate the transport blocks.
                vad_pending.insert(vad_pending.end(), block.begin(), block.end());
                engine::runtime::StreamEvent vad_event;
                while (vad_pending.size() >= 512) {
                    engine::runtime::AudioChunk chunk;
                    chunk.sample_rate = 16000;
                    chunk.channels = 1;
                    chunk.start_sample = vad_sample;
                    chunk.samples.assign(vad_pending.begin(), vad_pending.begin() + 512);
                    vad_pending.erase(vad_pending.begin(), vad_pending.begin() + 512);
                    vad_sample += 512;
                    auto window_event = vad->process_audio_chunk(chunk);
                    vad_event.voice_activity.insert(
                        vad_event.voice_activity.end(),
                        window_event.voice_activity.begin(),
                        window_event.voice_activity.end());
                }

                if (speaking) {
                    std::lock_guard<std::mutex> lock(decode_mutex);
                    utterance.insert(utterance.end(), block.begin(), block.end());
                } else {
                    preroll.insert(preroll.end(), block.begin(), block.end());
                    if (preroll.size() > preroll_samples) {
                        preroll.erase(
                            preroll.begin(),
                            preroll.begin() + static_cast<std::ptrdiff_t>(preroll.size() - preroll_samples));
                    }
                }

                for (const auto & activity : vad_event.voice_activity) {
                    using Kind = engine::runtime::VoiceActivityEvent::Kind;
                    if (activity.kind == Kind::SpeechStart && !speaking) {
                        begin_utterance();
                    } else if (activity.kind == Kind::SpeechEnd && speaking) {
                        end_utterance();
                    }
                }

                if (speaking) {
                    const double speech_ms = elapsed_ms(speech_started_at);
                    if (speech_ms > params.max_utterance_ms) {
                        // Runaway turn: close it out and let the VAD's next
                        // SpeechStart open a fresh one.
                        end_utterance();
                        vad->reset();
                        continue;
                    }
                    if (elapsed_ms(last_partial_enqueue) >= params.partial_interval_ms) {
                        size_t utterance_samples = 0;
                        {
                            std::lock_guard<std::mutex> lock(decode_mutex);
                            utterance_samples = utterance.size();
                        }
                        if (utterance_samples >= static_cast<size_t>(params.min_partial_audio_ms) * 16) {
                            last_partial_enqueue = Clock::now();
                            enqueue_decode(false);
                        }
                    }
                }
            }
            // Stream over: a turn cut off by the client still deserves its
            // final transcript.
            if (speaking) {
                end_utterance();
            }
        } catch (const std::exception & ex) {
            emit(std::string("{\"type\":\"error\",\"message\":\"") + json_escape(ex.what()) + "\"}");
        }
        {
            std::lock_guard<std::mutex> lock(decode_mutex);
            quit = true;
        }
        decode_cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }
};

VoiceLiveRunner::VoiceLiveRunner(
    TurnParams params,
    VoiceHooks hooks,
    std::unique_ptr<engine::runtime::IStreamingVoiceTaskSession> vad)
    : impl_(std::make_unique<Impl>()) {
    impl_->params = params;
    impl_->hooks = std::move(hooks);
    impl_->vad = std::move(vad);
    impl_->tracker = StableTracker(params.stability);
}

VoiceLiveRunner::~VoiceLiveRunner() = default;

void VoiceLiveRunner::run(const minitts::app::AudioChunkStream & stream) {
    impl_->run(stream);
}

// ---------------------------------------------------------------------------
// VAD session construction
// ---------------------------------------------------------------------------

std::unique_ptr<engine::runtime::IStreamingVoiceTaskSession> make_vad_session(
    const std::filesystem::path & asset_root,
    const TurnParams & params) {
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path =
        asset_root / "assets" / "framework" / "models" / "silero_vad";
    auto model = engine::models::silero_vad::load_silero_vad_model(load_request);

    engine::runtime::SessionOptions options;
    options.backend.type = engine::core::BackendType::Cpu;
    options.options["threshold"] = std::to_string(params.vad_threshold);
    options.options["min_speech_duration_ms"] = std::to_string(params.min_speech_ms);
    options.options["min_silence_duration_ms"] = std::to_string(params.min_silence_ms);
    options.options["speech_pad_ms"] = "0";  // the runner keeps its own preroll

    auto session = model->create_task_session(
        engine::runtime::TaskSpec{engine::runtime::VoiceTaskKind::Vad, engine::runtime::RunMode::Streaming},
        options);
    auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(session.get());
    if (streaming == nullptr) {
        throw std::runtime_error("Silero VAD did not provide a streaming session");
    }
    engine::runtime::AudioBuffer hint;
    hint.sample_rate = 16000;
    hint.channels = 1;
    session->prepare(engine::runtime::build_preparation_request(hint));
    session.release();
    return std::unique_ptr<engine::runtime::IStreamingVoiceTaskSession>(streaming);
}

}  // namespace minitts::server::voice
