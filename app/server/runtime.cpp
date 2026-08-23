#include "runtime.h"

#include "base64.h"
#include "llm_client.h"
#include "segmenter.h"
#include "multipart.h"
#include "chat_hygiene.h"
#include "ui_assets.h"

#include "../cli/request.h"
#include "../streaming/pcm_source.h"
#include "../streaming/streaming.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <deque>
#include <condition_variable>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace minitts::server {
namespace {

using engine::io::json::Value;

using Clock = std::chrono::steady_clock;

void materialize_embedded_demo_voices(const std::filesystem::path & voice_dir) {
    std::filesystem::create_directories(voice_dir);
    for (const auto & voice : embedded_demo_voices()) {
        std::ofstream wav(
            voice_dir / (std::string(voice.name) + ".wav"),
            std::ios::binary | std::ios::trunc);
        if (!wav) {
            throw std::runtime_error("failed to create embedded demo voice: " + std::string(voice.name));
        }
        wav.write(voice.wav_bytes.data(), static_cast<std::streamsize>(voice.wav_bytes.size()));
        if (!wav) {
            throw std::runtime_error("failed to write embedded demo voice: " + std::string(voice.name));
        }
    }
    const auto prompt_text = embedded_demo_voice_prompt_text();
    std::ofstream prompts(voice_dir / "prompt_text", std::ios::binary | std::ios::trunc);
    if (!prompts) {
        throw std::runtime_error("failed to create embedded demo voice prompt_text");
    }
    prompts.write(prompt_text.data(), static_cast<std::streamsize>(prompt_text.size()));
    if (!prompts) {
        throw std::runtime_error("failed to write embedded demo voice prompt_text");
    }
}

// Per-request override for the busy timeout. Absent means "use the model's
// configured ceiling"; a value is clamped to that ceiling by resolve_busy_timeout_ms
// so a client can shorten its own wait but never weaken the guard.
std::optional<int> parse_busy_timeout_override(const Value & body) {
    const auto * value = body.find("busy_timeout_ms");
    if (value == nullptr) {
        return std::nullopt;
    }
    const auto requested = engine::io::json::optional_i32(body, "busy_timeout_ms", 0);
    if (requested < 0) {
        throw std::runtime_error("busy_timeout_ms must be >= 0 (0 means no client-side bound)");
    }
    return requested;
}

bool model_accepts_request_option(std::string_view family, std::string_view option) {
    const auto contract = engine::model_spec::model_contract(family);
    if (!contract.has_value()) {
        return true;
    }
    return contract->request_option_keys.find(std::string(option)) != contract->request_option_keys.end();
}

std::string json_quote(std::string_view value) {
    return engine::io::json::stringify_string(value);
}

std::filesystem::path resolve_path(const std::filesystem::path & base, const std::filesystem::path & path) {
    return path.is_absolute() ? path : base / path;
}

// Looks up `<voice_name>` in `<voice_dir>/prompt_text`, a mapping file with one
// `<basename>|<transcript>` line per built-in voice (same format the webui uses).
std::optional<std::string> load_voice_library_text(
    const std::filesystem::path & voice_dir, const std::string & voice_name) {
    std::ifstream f(voice_dir / "prompt_text");
    std::string line;
    while (std::getline(f, line)) {
        const auto sep = line.find('|');
        if (sep == std::string::npos) continue;
        std::string name = line.substr(0, sep);
        // trim trailing whitespace from name
        while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) {
            name.pop_back();
        }
        if (name == voice_name) {
            return line.substr(sep + 1);
        }
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> resolve_voice_library_wav(
    const std::filesystem::path & voice_dir,
    const std::string & voice_name) {
    const std::filesystem::path name_path(voice_name);
    if (voice_name.empty() ||
        name_path.has_root_name() ||
        name_path.has_root_directory() ||
        name_path.has_parent_path() ||
        name_path.filename().string() != voice_name ||
        voice_name == "." ||
        voice_name == "..") {
        return std::nullopt;
    }
    const auto wav = voice_dir / (voice_name + ".wav");
    std::error_code ec;
    if (!std::filesystem::is_regular_file(wav, ec)) {
        return std::nullopt;
    }
    return wav;
}

std::unordered_map<std::string, std::string> options_from_object(const Value * value);

std::string safe_upload_name(std::string value) {
    value = std::filesystem::path(value).filename().string();
    for (char & ch : value) {
        const auto uch = static_cast<unsigned char>(ch);
        if (!(std::isalnum(uch) || ch == '.' || ch == '-' || ch == '_')) {
            ch = '_';
        }
    }
    if (value.empty() || value == "." || value == "..") {
        value = "audio.wav";
    }
    return value;
}

ServerModelConfig model_config_from_json(
    const Value & body,
    const std::filesystem::path & request_base,
    bool lazy) {
    ServerModelConfig model;
    model.id = engine::io::json::require_string(body, "id");
    model.path = resolve_path(request_base, engine::io::json::require_string(body, "path"));
    model.family = engine::io::json::require_string(body, "family");
    model.task = engine::io::json::optional_string(body, "task", model.task);
    model.mode = engine::io::json::optional_string(body, "mode", model.mode);
    model.lazy = lazy;
    if (const auto * value = body.find("model_spec_override")) {
        model.model_spec_override = resolve_path(request_base, value->as_string());
    }
    if (const auto * value = body.find("config")) {
        model.config_id = value->as_string();
    }
    if (const auto * value = body.find("weight")) {
        model.weight_id = value->as_string();
    }
    model.load_options = options_from_object(body.find("load_options"));
    model.session_options = options_from_object(body.find("session_options"));
    return model;
}

// Minimal application/x-www-form-urlencoded query string lookup, e.g.
// query_param("model=pocket-tts&foo=bar", "model") -> "pocket-tts".
std::string query_param(const std::string & query, const std::string & key) {
    size_t pos = 0;
    while (pos < query.size()) {
        const size_t amp = query.find('&', pos);
        const std::string pair = query.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
        const auto eq = pair.find('=');
        const std::string name = pair.substr(0, eq);
        if (name == key) {
            return eq == std::string::npos ? "" : pair.substr(eq + 1);
        }
        if (amp == std::string::npos) {
            break;
        }
        pos = amp + 1;
    }
    return {};
}

const char * backend_name(engine::core::BackendType type) {
    switch (type) {
        case engine::core::BackendType::Cpu:
            return "cpu";
        case engine::core::BackendType::Cuda:
            return "cuda";
        case engine::core::BackendType::Hip:
            return "hip";
        case engine::core::BackendType::Vulkan:
            return "vulkan";
        case engine::core::BackendType::Metal:
            return "metal";
        case engine::core::BackendType::BestAvailable:
            return "best";
    }
    return "unknown";
}

HttpResponse ui_service_worker_retirement_response() {
    HttpResponse response;
    response.status = 200;
    response.content_type = "application/javascript; charset=utf-8";
    response.body = R"JS(
// Retire service workers left by applications that previously used this origin.
self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});
self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const cacheNames = await caches.keys();
    await Promise.all(cacheNames.map((name) => caches.delete(name)));
    await self.clients.claim();
    const windows = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    await self.registration.unregister();
    await Promise.all(windows.map((client) => client.navigate(client.url)));
  })());
});
)JS";
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0";
    response.headers["Clear-Site-Data"] = "\"cache\"";
    response.headers["Expires"] = "0";
    response.headers["Pragma"] = "no-cache";
    response.headers["Service-Worker-Allowed"] = "/";
    response.headers["X-Content-Type-Options"] = "nosniff";
    return response;
}

std::unordered_map<std::string, std::string> options_from_object(const Value * value) {
    return minitts::cli::json_options_map(value);
}

void add_option_from_json(
    std::unordered_map<std::string, std::string> & options,
    const Value & object,
    const std::string & field,
    const std::string & option_key) {
    const auto * value = object.find(field);
    if (value != nullptr && !value->is_null()) {
        options[option_key] = minitts::cli::json_option_string(*value);
    }
}

std::vector<uint8_t> encode_pcm16_wav(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("audio output sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("audio output channel count must be positive");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio output sample count must be divisible by channel count");
    }

    const uint16_t channels = static_cast<uint16_t>(audio.channels);
    const uint16_t bits_per_sample = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(audio.samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_bytes;
    const uint32_t byte_rate = static_cast<uint32_t>(audio.sample_rate) * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;

    std::vector<uint8_t> out;
    out.reserve(44 + data_bytes);
    auto append_bytes = [&](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    auto append_u16 = [&](uint16_t value) { append_bytes(&value, sizeof(value)); };
    auto append_u32 = [&](uint32_t value) { append_bytes(&value, sizeof(value)); };

    out.insert(out.end(), {'R', 'I', 'F', 'F'});
    append_u32(riff_size);
    out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    append_u32(16);
    append_u16(1);
    append_u16(channels);
    append_u32(static_cast<uint32_t>(audio.sample_rate));
    append_u32(byte_rate);
    append_u16(block_align);
    append_u16(bits_per_sample);
    out.insert(out.end(), {'d', 'a', 't', 'a'});
    append_u32(data_bytes);
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        append_bytes(&pcm, sizeof(pcm));
    }
    return out;
}

std::vector<uint8_t> encode_pcm16_samples(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("audio output sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("audio output channel count must be positive");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio output sample count must be divisible by channel count");
    }

    std::vector<uint8_t> out;
    out.reserve(audio.samples.size() * sizeof(int16_t));
    auto append_bytes = [&](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        append_bytes(&pcm, sizeof(pcm));
    }
    return out;
}

void write_sse(HttpStreamWriter & writer, const std::string & json) {
    writer.write("data: " + json + "\n\n");
}

void write_sse_done(HttpStreamWriter & writer) {
    writer.write("data: [DONE]\n\n");
}

bool bool_field(const Value & object, const std::string & key, bool default_value) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return default_value;
    }
    if (value->is_bool()) {
        return value->as_bool();
    }
    if (value->is_string()) {
        const auto str = value->as_string();
        if (str == "true" || str == "1") {
            return true;
        }
        if (str == "false" || str == "0") {
            return false;
        }
    }
    throw std::runtime_error(key + " must be a boolean");
}

HttpResponse sse_response(std::function<void(HttpStreamWriter &)> stream) {
    HttpResponse response;
    response.status = 200;
    response.content_type = "text/event-stream; charset=utf-8";
    response.headers.emplace("X-Accel-Buffering", "no");
    response.stream_body = std::move(stream);
    return response;
}

HttpResponse chunked_audio_response(std::function<void(HttpStreamWriter &)> stream) {
    HttpResponse response;
    response.status = 200;
    response.content_type = "application/octet-stream";
    response.stream_body = std::move(stream);
    return response;
}

bool is_wav_upload_filename(const std::string & filename) {
    std::string ext = std::filesystem::path(filename).extension().string();
    for (char & ch : ext) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return ext.empty() || ext == ".wav";
}

std::string lower_ascii(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string_view trim_ascii(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.remove_suffix(1);
    }
    return value;
}

bool is_json_content_type(const HttpRequest & request) {
    const auto it = request.headers.find("content-type");
    if (it == request.headers.end()) {
        return false;
    }
    const std::string content_type = lower_ascii(it->second);
    const auto media_type = trim_ascii(std::string_view(content_type).substr(0, content_type.find(';')));
    return media_type == "application/json" ||
        (media_type.size() > 5 && media_type.substr(media_type.size() - 5) == "+json");
}

std::string request_content_type(const HttpRequest & request) {
    const auto it = request.headers.find("content-type");
    return it == request.headers.end() ? "" : it->second;
}

void log_request_body_if_enabled(const ServerConfig & config, const HttpRequest & request) {
    if (!config.log_request_body || !engine::debug::log_enabled()) {
        return;
    }
    if (request.method != "POST") {
        return;
    }
    if (!request.body.empty() && is_json_content_type(request)) {
        engine::debug::log_message("[REQUEST_BODY] " + request.method + " " + request.path);
        engine::debug::log_message(request.body);
        return;
    }
    if (request.body.empty() && request.body_stream == nullptr) {
        return;
    }
    std::ostringstream out;
    out << "[REQUEST_BODY_SKIPPED] " << request.method << " " << request.path
        << " content_type=" << json_quote(request_content_type(request));
    if (request.body_stream != nullptr) {
        out << " body=stream";
    } else {
        out << " body_bytes=" << request.body.size();
    }
    if (!request.query.empty()) {
        out << " query=" << json_quote(request.query);
    }
    engine::debug::log_message(out.str());
}

void log_multipart_request_summary_if_enabled(
    const ServerConfig & config,
    const std::vector<MultipartPart> & parts) {
    if (!config.log_request_body || !engine::debug::log_enabled()) {
        return;
    }
    for (const auto & part : parts) {
        if (part.filename.empty()) {
            continue;
        }
        std::ostringstream out;
        out << "[REQUEST_BODY_SKIPPED] multipart_file"
            << " field=" << json_quote(part.name)
            << " filename=" << json_quote(part.filename)
            << " bytes=" << part.data.size();
        engine::debug::log_message(out.str());
    }
}

double elapsed_ms(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

double audio_duration_ms(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        return 0.0;
    }
    return 1000.0 * static_cast<double>(audio.samples.size()) /
        static_cast<double>(audio.sample_rate * audio.channels);
}

double audio_rtf(double wall_ms, double duration_ms) {
    return duration_ms > 0.0 ? wall_ms / duration_ms : 0.0;
}

std::string timing_json(double wall_ms) {
    std::ostringstream out;
    out << "{\"wall_ms\":" << wall_ms << "}";
    return out.str();
}

std::string timing_json(double wall_ms, const engine::runtime::AudioBuffer & audio) {
    const double duration_ms = audio_duration_ms(audio);
    std::ostringstream out;
    out << "{\"wall_ms\":" << wall_ms
        << ",\"audio_duration_ms\":" << duration_ms
        << ",\"rtf\":" << audio_rtf(wall_ms, duration_ms) << "}";
    return out.str();
}

std::string ttft_timing_json(double ttft_ms) {
    std::ostringstream out;
    out << "{\"ttft_ms\":" << ttft_ms << "}";
    return out.str();
}

bool stream_event_has_output(const engine::runtime::StreamEvent & event) {
    return (event.partial_text.has_value() && !event.partial_text->text.empty()) ||
        event.audio_output.has_value() ||
        !event.named_audio_outputs.empty();
}

bool task_result_has_output(const engine::runtime::TaskResult & result) {
    return result.text_output.has_value() ||
        result.audio_output.has_value() ||
        !result.named_audio_outputs.empty() ||
        result.artifact_output.has_value() ||
        !result.output_artifacts.empty();
}

const char * artifact_kind_name(engine::runtime::ArtifactKind kind) {
    using engine::runtime::ArtifactKind;
    switch (kind) {
    case ArtifactKind::SpeakerEmbedding: return "speaker_embedding";
    case ArtifactKind::StyleEmbedding: return "style_embedding";
    case ArtifactKind::PromptEmbedding: return "prompt_embedding";
    case ArtifactKind::AcousticTokens: return "acoustic_tokens";
    case ArtifactKind::Midi: return "midi";
    case ArtifactKind::TranscriptAlignment: return "transcript_alignment";
    case ArtifactKind::DiarizationState: return "diarization_state";
    case ArtifactKind::VadState: return "vad_state";
    case ArtifactKind::Custom: return "custom";
    }
    return "custom";
}

double require_ttft_ms(const std::optional<double> & ttft_ms) {
    if (!ttft_ms.has_value()) {
        throw std::runtime_error("streaming response produced no TTFT event");
    }
    return *ttft_ms;
}

std::unordered_map<std::string, std::string> timing_headers(
    double wall_ms,
    const engine::runtime::AudioBuffer & audio) {
    const double duration_ms = audio_duration_ms(audio);
    return {
        {"X-AudioCPP-Wall-Ms", std::to_string(wall_ms)},
        {"X-AudioCPP-Audio-Duration-Ms", std::to_string(duration_ms)},
        {"X-AudioCPP-RTF", std::to_string(audio_rtf(wall_ms, duration_ms))},
    };
}

std::string task_result_json_with_timing(
    const engine::runtime::TaskResult & result,
    const std::string & timing) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    auto field = [&](const std::string & name) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << json_quote(name) << ":";
    };

    if (result.text_output.has_value()) {
        field("text");
        out << json_quote(result.text_output->text);
        if (!result.text_output->language.empty()) {
            field("language");
            out << json_quote(result.text_output->language);
        }
    }
    if (result.audio_output.has_value()) {
        const auto wav = encode_pcm16_wav(*result.audio_output);
        field("audio");
        out << json_quote(base64_encode(wav));
        field("sample_rate");
        out << result.audio_output->sample_rate;
        field("channels");
        out << result.audio_output->channels;
    }
    if (!result.named_audio_outputs.empty()) {
        field("named_audio_outputs");
        out << "[";
        for (size_t i = 0; i < result.named_audio_outputs.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto wav = encode_pcm16_wav(result.named_audio_outputs[i].audio);
            out << "{\"id\":" << json_quote(result.named_audio_outputs[i].id)
                << ",\"audio\":" << json_quote(base64_encode(wav))
                << ",\"sample_rate\":" << result.named_audio_outputs[i].audio.sample_rate
                << ",\"channels\":" << result.named_audio_outputs[i].audio.channels
                << "}";
        }
        out << "]";
    }
    if (result.artifact_output.has_value() || !result.output_artifacts.empty()) {
        field("artifacts");
        out << "[";
        bool first_artifact = true;
        auto write_artifact = [&](const engine::runtime::VoiceArtifact & artifact) {
            if (!first_artifact) out << ",";
            first_artifact = false;
            out << "{\"id\":" << json_quote(artifact.id)
                << ",\"kind\":" << json_quote(artifact_kind_name(artifact.kind))
                << ",\"payload\":" << json_quote(base64_encode(artifact.payload))
                << ",\"meta\":{";
            bool first_meta = true;
            for (const auto & [key, value] : artifact.meta) {
                if (!first_meta) out << ",";
                first_meta = false;
                out << json_quote(key) << ":" << json_quote(value);
            }
            out << "}}";
        };
        if (result.artifact_output.has_value()) write_artifact(*result.artifact_output);
        for (const auto & artifact : result.output_artifacts) write_artifact(artifact);
        out << "]";
    }
    if (!result.speech_segments.empty()) {
        field("segments");
        out << "[";
        for (size_t i = 0; i < result.speech_segments.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & segment = result.speech_segments[i];
            out << "{\"start_sample\":" << segment.span.start_sample
                << ",\"end_sample\":" << segment.span.end_sample
                << ",\"confidence\":" << segment.confidence;
            if (!segment.text.empty()) {
                out << ",\"text\":" << json_quote(segment.text);
            }
            out << "}";
        }
        out << "]";
    }
    if (!result.speaker_turns.empty()) {
        field("speaker_turns");
        out << "[";
        for (size_t i = 0; i < result.speaker_turns.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & turn = result.speaker_turns[i];
            out << "{\"start_sample\":" << turn.span.start_sample
                << ",\"end_sample\":" << turn.span.end_sample
                << ",\"speaker_id\":" << json_quote(turn.speaker_id)
                << ",\"confidence\":" << turn.confidence;
            if (!turn.text.empty()) {
                out << ",\"text\":" << json_quote(turn.text);
            }
            out << "}";
        }
        out << "]";
    }
    if (!result.word_timestamps.empty()) {
        field("words");
        out << "[";
        for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & word = result.word_timestamps[i];
            out << "{\"word\":" << json_quote(word.word)
                << ",\"start_sample\":" << word.span.start_sample
                << ",\"end_sample\":" << word.span.end_sample
                << ",\"confidence\":" << word.confidence << "}";
        }
        out << "]";
    }
    field("timing");
    out << timing;
    out << "}";
    return out.str();
}

std::string task_result_json(const engine::runtime::TaskResult & result, double wall_ms) {
    if (result.audio_output.has_value()) {
        return task_result_json_with_timing(result, timing_json(wall_ms, *result.audio_output));
    }
    if (result.named_audio_outputs.size() == 1) {
        return task_result_json_with_timing(result, timing_json(wall_ms, result.named_audio_outputs.front().audio));
    }
    return task_result_json_with_timing(result, timing_json(wall_ms));
}

std::string streaming_task_result_json(
    const engine::runtime::TaskResult & result,
    const std::optional<double> & ttft_ms) {
    return task_result_json_with_timing(result, ttft_timing_json(require_ttft_ms(ttft_ms)));
}

std::string stream_event_json(const engine::runtime::StreamEvent & event) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    auto field = [&](const char * name) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << name << "\":";
    };
    if (event.partial_text.has_value()) {
        field("partial_text");
        out << "{\"text\":" << json_quote(event.partial_text->text)
            << ",\"language\":" << json_quote(event.partial_text->language)
            << "}";
    }
    if (event.audio_output.has_value()) {
        const auto wav = encode_pcm16_wav(*event.audio_output);
        field("audio");
        out << json_quote(base64_encode(wav));
    }
    if (!event.named_audio_outputs.empty()) {
        field("named_audio_outputs");
        out << "[";
        for (size_t i = 0; i < event.named_audio_outputs.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto wav = encode_pcm16_wav(event.named_audio_outputs[i].audio);
            out << "{\"id\":" << json_quote(event.named_audio_outputs[i].id)
                << ",\"audio\":" << json_quote(base64_encode(wav))
                << ",\"format\":\"wav\"}";
        }
        out << "]";
    }
    if (!event.word_timestamps.empty()) {
        field("word_timestamps");
        out << "[";
        for (size_t i = 0; i < event.word_timestamps.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            out << "{\"start_sample\":" << event.word_timestamps[i].span.start_sample
                << ",\"end_sample\":" << event.word_timestamps[i].span.end_sample
                << ",\"word\":" << json_quote(event.word_timestamps[i].word)
                << ",\"confidence\":" << event.word_timestamps[i].confidence
                << "}";
        }
        out << "]";
    }
    field("is_final");
    out << (event.is_final ? "true" : "false");
    out << "}";
    return out.str();
}

// Converts a voice reference to the mono 24 kHz form the Qwen3 encoders use
// internally, once, at load time. Every request otherwise carries, copies, and
// hashes the raw upload -- a stereo 44.1 kHz recording is nearly 4x the bytes
// for the same conditioning.
engine::runtime::AudioBuffer normalize_voice_reference(const engine::runtime::AudioBuffer & audio) {
    constexpr int kTargetRate = 24000;
    if (audio.sample_rate == kTargetRate && audio.channels == 1) {
        return audio;
    }
    engine::runtime::AudioBuffer out;
    out.sample_rate = kTargetRate;
    out.channels = 1;
    out.samples = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        kTargetRate);
    return out;
}

const engine::runtime::AudioBuffer & select_audio_output(const engine::runtime::TaskResult & result) {
    if (result.audio_output.has_value()) {
        return *result.audio_output;
    }
    if (result.named_audio_outputs.size() == 1) {
        return result.named_audio_outputs.front().audio;
    }
    throw std::runtime_error("model result did not contain exactly one audio output");
}

engine::runtime::TaskRequest build_openai_transcription_request(
    const Value & body,
    const std::filesystem::path & base_dir,
    const std::string * uploaded_audio_bytes = nullptr) {
    const auto * audio = body.find("audio");
    if (audio == nullptr) {
        audio = body.find("audio_path");
    }
    if (audio == nullptr) {
        audio = body.find("file");
    }
    if (uploaded_audio_bytes == nullptr && (audio == nullptr || !audio->is_string())) {
        throw std::runtime_error("transcription request requires audio, audio_path, or file path");
    }

    engine::runtime::TaskRequest request;
    if (uploaded_audio_bytes == nullptr) {
        request.audio_input = minitts::cli::read_audio_buffer(resolve_path(base_dir, audio->as_string()));
    } else {
        request.audio_input = minitts::cli::read_audio_buffer(std::string_view(*uploaded_audio_bytes));
    }
    request.options = options_from_object(body.find("options"));
    std::string language;
    if (const auto * value = body.find("language")) {
        language = value->as_string();
        request.options["language"] = language;
    }
    std::string context;
    if (const auto * value = body.find("text")) {
        context = value->as_string();
    }
    if (!language.empty() || !context.empty()) {
        request.text_input = engine::runtime::Transcript{std::move(context), std::move(language)};
    }
    return request;
}

template <typename Predicate>
std::optional<std::filesystem::path> find_ancestor(
    std::filesystem::path start,
    Predicate predicate) {
    std::error_code ec;
    start = std::filesystem::absolute(std::move(start), ec).lexically_normal();
    if (ec) {
        return std::nullopt;
    }
    for (;;) {
        if (predicate(start)) {
            return start;
        }
        const auto parent = start.parent_path();
        if (parent.empty() || parent == start) {
            return std::nullopt;
        }
        start = parent;
    }
}

template <typename Predicate>
std::optional<std::filesystem::path> find_from_roots(
    const std::filesystem::path & request_base,
    const std::filesystem::path & resource_anchor,
    Predicate predicate) {
    if (auto found = find_ancestor(request_base, predicate)) {
        return found;
    }
    if (!resource_anchor.empty()) {
        return find_ancestor(resource_anchor, predicate);
    }
    return std::nullopt;
}

}  // namespace

ServerState::ServerState(
    ServerConfig config,
    std::filesystem::path request_base,
    std::filesystem::path ui_resource_anchor)
    : config_(std::move(config)),
      request_base_(std::filesystem::absolute(std::move(request_base)).lexically_normal()) {
    if (config_.backend != engine::core::BackendType::Cuda) {
        std::cerr
            << "audio.cpp is optimized for CUDA. The "
            << backend_name(config_.backend)
            << " server backend is intended for portability and testing, but performance and model coverage may be lower than CUDA.\n";
    }
    load_models();

    // The active character survives restarts, so both the WebUI and MCP callers
    // keep speaking with the chosen voice. A store that fails to load falls back
    // to the default character rather than failing startup: speech must come up
    // even when a customization file is broken, and the log says why F is back.
    character_dir_ = config_.character_dir.value_or(request_base_ / "character");
    try {
        character_ = load_character(character_dir_);
    } catch (const std::exception & ex) {
        std::cerr << "character store ignored (" << ex.what() << "); using the default character\n";
        character_ = default_character();
    }
    try {
        llm_settings_ = load_llm_settings(character_dir_);
    } catch (const std::exception & ex) {
        std::cerr << "LLM settings ignored (" << ex.what() << "); using roleplay defaults\n";
        llm_settings_ = default_llm_settings();
    }

    // With a model registry in the config, this server owns the sidecar: spawn
    // the persisted choice (falling back to the configured default, then the
    // first entry) and let the launcher's health poll cover the load time.
    if (!config_.llm_models.empty() && config_.llm_port > 0 && !config_.llm_server_exe.empty()) {
        const LlmModelSpec * spec = find_llm_spec(llm_settings_.model);
        if (spec == nullptr) {
            spec = find_llm_spec(config_.llm_default);
        }
        if (spec == nullptr) {
            spec = &config_.llm_models.front();
        }
        llm_manager_ = std::make_unique<LlmManager>(
            config_.llm_host, config_.llm_port, config_.llm_server_exe,
            config_.llm_log_dir.empty() ? request_base_ : config_.llm_log_dir);
        std::string llm_error;
        if (!llm_manager_->start(*spec, llm_error)) {
            std::cerr << "LLM sidecar not started (" << llm_error << "); chat will be unavailable\n";
            llm_manager_.reset();
        } else {
            std::cerr << "LLM sidecar starting: " << spec->name << "\n";
        }
    }
    // Prime the sidecar's prompt cache with the active character's system
    // prompt as soon as the model is up: the first message of the first
    // conversation then pays only its own prefill. Harmless no-op when no
    // LLM is configured; also covers a launcher-owned sidecar.
    warm_llm_system_prompt();
    if (auto * model = find_speech_model()) {
        try {
            apply_character(*model, character_);
        } catch (const std::exception & ex) {
            std::cerr << "character voice not applied (" << ex.what() << "); using the default character\n";
            character_ = default_character();
            apply_character(*model, character_);
        }
        // Seed the library with the active character so a fresh install shows
        // its default entry and switching away is always one click reversible.
        try {
            save_character(character_dir_, character_);
            store_character_in_library(character_);
        } catch (const std::exception & ex) {
            std::cerr << "character library not seeded (" << ex.what() << ")" << std::endl;
        }
    }
}

ServerState::~ServerState() {
    // Voice browser sessions own detachable-looking but joined threads;
    // close them all so shutdown cannot race a live transcription.
    std::vector<std::shared_ptr<VoiceBrowserSession>> sessions;
    {
        std::lock_guard<std::mutex> lock(voice_sessions_mutex_);
        for (auto & [id, session] : voice_sessions_) {
            sessions.push_back(session);
        }
        voice_sessions_.clear();
    }
    for (auto & session : sessions) {
        close_voice_session(*session);
    }
}

HttpResponse ServerState::handle(const HttpRequest & request) {
  HttpResponse response;
  const std::string allowed_origin = get_allowed_origin(request);
  try {
    log_request_body_if_enabled(config_, request);
    if (request.method == "OPTIONS" && (!allowed_origin.empty() || config_.ui_enabled)) {
        response.status = 204;
        response.content_type = "text/plain";
        response.headers["Access-Control-Allow-Headers"] = "*";
        response.headers["Access-Control-Allow-Methods"] = "GET, POST";
    }
    else if (request.method == "GET" && (request.path == "/" || request.path == "/index.html")) {
        response = handle_ui_asset();
    }
    else if (
        request.method == "GET" && config_.ui_enabled &&
        (request.path == "/sw.js" || request.path == "/service-worker.js")) {
        response = ui_service_worker_retirement_response();
    }
    else if (request.method == "GET" && request.path == "/favicon.ico") {
        response.status = 204;
        response.content_type = "image/x-icon";
    }
    else if (request.method == "GET" && request.path == "/health") {
        size_t model_count = 0;
        {
            std::lock_guard<std::mutex> lock(models_mutex_);
            model_count = models_.size();
        }
        response = json_response(
            "{\"status\":\"ok\",\"backend\":\"" +
            std::string(backend_name(config_.backend)) +
            "\",\"llm\":" + std::string(config_.llm_port > 0 ? "true" : "false") +
            ",\"models\":" +
            std::to_string(model_count) +
            ",\"ui\":" + (config_.ui_enabled ? "true" : "false") +
            ",\"ui_management\":" + (config_.ui_management ? "true" : "false") +
            "}");
    }
    else if (request.method == "GET" && request.path == "/v1/models") {
        response = json_response(models_json());
    }
    else if (request.method == "GET" && request.path == "/v1/character") {
        response = handle_character_get();
    }
    else if ((request.method == "POST" || request.method == "PUT") && request.path == "/v1/character") {
        response = handle_character_set(request);
    }
    else if (request.method == "GET" && request.path == "/v1/characters") {
        response = handle_characters_list();
    }
    else if (request.method == "GET" && request.path == "/v1/character/voice") {
        response = handle_character_voice();
    }
    else if (request.method == "POST" && request.path == "/v1/characters/activate") {
        response = handle_character_activate(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/characters/delete") {
        response = handle_character_delete(request.body);
    }
    else if (request.path == "/mcp") {
        response = handle_mcp(request);
    }
    else if (request.method == "POST" && request.path == "/v1/chat/speak") {
        response = handle_chat_speak(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/llm/chat") {
        response = handle_llm_chat(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/telemetry/typing") {
        response = handle_typing_telemetry(request.body);
    }
    else if (request.method == "GET" && request.path == "/v1/llm-settings") {
        response = handle_llm_settings_get();
    }
    else if (request.method == "POST" && request.path == "/v1/llm-settings") {
        response = handle_llm_settings_set(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/audio/speech") {
        response = handle_speech(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/voice/live") {
        response = handle_voice_live(request);
    }
    else if (request.method == "POST" && request.path == "/v1/voice/sessions") {
        response = handle_voice_session_create(request);
    }
    else if (request.method == "POST" && request.path == "/v1/voice/sessions/audio") {
        response = handle_voice_session_audio(request);
    }
    else if (request.method == "GET" && request.path == "/v1/voice/sessions/events") {
        response = handle_voice_session_events(request);
    }
    else if (request.method == "POST" && request.path == "/v1/voice/sessions/stop") {
        response = handle_voice_session_stop(request);
    }
    else {
        response = error_response(404, "unknown endpoint: " + request.path, "not_found");
    }
  } catch (const engine::runtime::CapacityError & ex) {
    // The request is too big for the device, which is the caller's to fix --
    // reporting it as 500 sends them looking for a server fault that is not
    // there. Checked before ServerBusyError only because both are
    // runtime_error; the two conditions are disjoint.
    response = error_response(400, ex.what(), "invalid_request_error");
  } catch (const ServerBusyError & ex) {
    // Non-streaming requests surface the busy state as 503 before any response is
    // sent. (Streaming requests acquire the lock inside the stream body, after
    // headers are sent, so there it becomes a stream error event instead.)
    response = error_response(503, ex.what(), "server_busy");
  }
  if (!allowed_origin.empty()) {
      response.headers["Access-Control-Allow-Origin"] = allowed_origin;
  }
  return response;
}

void ServerState::load_models() {
    for (auto & config : config_.models) {
        auto loaded = make_model(std::move(config));
        if (!model_index_.emplace(loaded->config.id, models_.size()).second) {
            throw std::runtime_error("duplicate server model id: " + loaded->config.id);
        }
        if (!loaded->config.lazy) {
            ensure_model_loaded_locked(*loaded);
        }
        models_.push_back(std::move(loaded));
    }
}

std::unique_ptr<ServerState::LoadedModel> ServerState::make_model(ServerModelConfig config) {
    auto loaded = std::make_unique<LoadedModel>();
    loaded->config = std::move(config);
    loaded->task = engine::runtime::TaskSpec{
        engine::runtime::parse_voice_task_kind(loaded->config.task),
        engine::runtime::parse_run_mode(loaded->config.mode),
    };
    load_voice_presets(*loaded);
    return loaded;
}

HttpResponse ServerState::handle_ui_asset() const {
    if (!config_.ui_enabled) {
        return error_response(404, "WebUI is disabled", "not_found");
    }
    HttpResponse response;
    response.status = 200;
    response.content_type = "text/html; charset=utf-8";
    const auto html = embedded_ui_html();
    response.body.assign(html.data(), html.size());
    // The WebUI shares the server origin (usually localhost:8080) with any app
    // that previously occupied that port. Never let an old shell survive a
    // server upgrade, and ask the browser to discard only cached resources.
    // Deliberately omit the Clear-Site-Data "storage" directive: saved voices,
    // the selected models folder, and UI preferences live in local storage.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0";
    response.headers["Clear-Site-Data"] = "\"cache\"";
    response.headers["Expires"] = "0";
    response.headers["Pragma"] = "no-cache";
    response.headers["Content-Security-Policy"] =
        "default-src 'self' 'unsafe-inline' blob: data:; connect-src 'self'; media-src 'self' blob: data:";
    response.headers["X-Content-Type-Options"] = "nosniff";
    return response;
}

ServerState::LoadedModel::RuntimeVoicePreset ServerState::load_runtime_voice_preset(
    const ServerModelConfig::VoicePreset & preset) const {
    LoadedModel::RuntimeVoicePreset out;
    out.voice_id = preset.voice_id;
    out.reference_text = preset.reference_text;
    if (preset.voice_ref.has_value()) {
        out.audio = normalize_voice_reference(minitts::cli::read_audio_buffer(*preset.voice_ref));
    }
    return out;
}

void ServerState::load_voice_presets(LoadedModel & model) const {
    for (const auto & [name, preset] : model.config.voice_presets) {
        auto [it, inserted] = model.voice_presets.emplace(name, load_runtime_voice_preset(preset));
        if (!inserted) {
            throw std::runtime_error("duplicate runtime voice preset for model " + model.config.id + ": " + name);
        }
        (void) it;
    }
    if (model.config.default_voice_preset_id.has_value()) {
        const auto it = model.voice_presets.find(*model.config.default_voice_preset_id);
        if (it == model.voice_presets.end()) {
            throw std::runtime_error(
                "default_voice_preset for model " + model.config.id +
                " was not loaded: " +
                *model.config.default_voice_preset_id);
        }
        model.default_voice_preset = it->second;
    } else if (model.config.default_voice_preset.has_value()) {
        model.default_voice_preset = load_runtime_voice_preset(*model.config.default_voice_preset);
    }
}

void ServerState::ensure_model_loaded_locked(LoadedModel & model) {
    if (model.session != nullptr) {
        return;
    }
    auto registry = engine::runtime::make_default_registry();

    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model.config.path;
    load_request.model_spec_override = model.config.model_spec_override.has_value()
        ? model.config.model_spec_override
        : config_.model_spec_override;
    load_request.family_hint = model.config.family;
    load_request.config_id = model.config.config_id;
    load_request.weight_id = model.config.weight_id;
    load_request.options = model.config.load_options;

    engine::runtime::SessionOptions session_options;
    session_options.backend.type = config_.backend;
    session_options.backend.device = config_.device;
    session_options.backend.threads = config_.threads;
    session_options.options = model.config.session_options;

    engine::debug::trace_log_scalar("server.model.id", model.config.id);
    engine::debug::trace_log_scalar("server.model.path", model.config.path.string());
    engine::debug::trace_log_scalar("server.model.family", model.config.family);
    engine::debug::trace_log_scalar(
        "server.model.task",
        std::string_view(engine::runtime::to_string(model.task.task)));
    engine::debug::trace_log_scalar(
        "server.model.mode",
        std::string_view(engine::runtime::to_string(model.task.mode)));
    engine::debug::trace_log_scalar("server.model.backend", std::string_view(backend_name(session_options.backend.type)));
    engine::debug::trace_log_scalar("server.model.device", int64_t{session_options.backend.device});
    engine::debug::trace_log_scalar("server.model.threads", int64_t{session_options.backend.threads});
    engine::debug::trace_log_scalar(
        "server.model.session_option_count",
        static_cast<int64_t>(session_options.options.size()));
    for (const auto & [key, value] : session_options.options) {
        engine::debug::trace_log_scalar("server.model.session_options." + key, value);
    }

    auto loaded_model = registry.load(load_request);
    auto session = loaded_model->create_task_session(model.task, session_options);
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
    auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(session.get());
    if (model.task.mode == engine::runtime::RunMode::Offline && offline == nullptr) {
        throw std::runtime_error("configured model does not provide offline execution: " + model.config.id);
    }
    if (model.task.mode == engine::runtime::RunMode::Streaming && streaming == nullptr) {
        throw std::runtime_error("configured model does not provide streaming execution: " + model.config.id);
    }
    model.model = std::move(loaded_model);
    model.session = std::move(session);
    model.offline = offline;
    model.streaming = streaming;
    model.loaded.store(true);
}

LiveIngestLimits ServerState::live_ingest_limits(const HttpRequest & request) const {
    const std::string model_id = query_param(request.query, "model");
    if (model_id.empty()) {
        return config_.live_ingest;
    }
    const auto it = model_index_.find(model_id);
    if (it == model_index_.end()) {
        return config_.live_ingest;
    }
    return resolve_live_ingest_limits(config_.live_ingest, models_.at(it->second)->config.live_ingest);
}

ServerState::LoadedModel & ServerState::require_model(const Value & body) {
    const std::string id = engine::io::json::require_string(body, "model");
    std::lock_guard<std::mutex> state_lock(models_mutex_);
    const auto it = model_index_.find(id);
    if (it == model_index_.end()) {
        throw std::runtime_error("unknown model id: " + id);
    }
    return *models_.at(it->second);
}

const ServerState::LoadedModel::RuntimeVoicePreset * ServerState::select_voice_preset(
    const LoadedModel & model,
    const Value & body,
    bool & voice_field_is_preset) const {
    voice_field_is_preset = false;
    if (const auto * value = body.find("voice")) {
        const auto it = model.voice_presets.find(value->as_string());
        if (it != model.voice_presets.end()) {
            voice_field_is_preset = true;
            return &it->second;
        }
        return nullptr;
    }
    if (body.find("voice_ref") != nullptr) {
        return nullptr;
    }
    return model.default_voice_preset.has_value() ? &*model.default_voice_preset : nullptr;
}

engine::runtime::TaskRequest ServerState::build_speech_request(const LoadedModel & model, const Value & body) const {
    std::shared_lock<std::shared_mutex> metadata_lock(model.metadata_mutex);
    engine::runtime::TaskRequest request;
    const auto * input = body.find("input");
    request.text_input = engine::runtime::Transcript{
        input != nullptr ? input->as_string() : engine::io::json::require_string(body, "text"),
        engine::io::json::optional_string(body, "language", ""),
    };

    request.options = options_from_object(body.find("options"));
    add_option_from_json(request.options, body, "seed", "seed");
    add_option_from_json(request.options, body, "temperature", "temperature");
    add_option_from_json(request.options, body, "top_k", "top_k");
    add_option_from_json(request.options, body, "top_p", "top_p");
    add_option_from_json(request.options, body, "max_tokens", "max_tokens");
    add_option_from_json(request.options, body, "max_steps", "max_steps");
    add_option_from_json(request.options, body, "repetition_penalty", "repetition_penalty");
    add_option_from_json(request.options, body, "guidance_scale", "guidance_scale");
    add_option_from_json(request.options, body, "num_inference_steps", "num_inference_steps");
    add_option_from_json(request.options, body, "chunk_frames", "chunk_frames");
    add_option_from_json(request.options, body, "decoder_context_frames", "decoder_context_frames");
    add_option_from_json(request.options, body, "speaker_embedding_only", "speaker_embedding_only");
    add_option_from_json(request.options, body, "stream_accumulate", "stream_accumulate");
    if (const auto * value = body.find("instructions")) {
        request.options["instruction"] = value->as_string();
    }
    if (const auto * value = body.find("instruction")) {
        request.options["instruction"] = value->as_string();
    }

    bool voice_field_is_preset = false;
    const auto * preset = select_voice_preset(model, body, voice_field_is_preset);
    const bool can_inject_reference_text =
        model_accepts_request_option(model.config.family, "reference_text");

    engine::runtime::VoiceCondition voice;
    bool has_voice = false;
    if (preset != nullptr) {
        if (preset->voice_id.has_value()) {
            voice.speaker = engine::runtime::VoiceReference{};
            voice.speaker->cached_voice_id = *preset->voice_id;
            has_voice = true;
        }
        if (preset->audio.has_value()) {
            if (!voice.speaker.has_value()) {
                voice.speaker = engine::runtime::VoiceReference{};
            }
            voice.speaker->audio = *preset->audio;
            has_voice = true;
        }
        if (can_inject_reference_text &&
            preset->reference_text.has_value() &&
            request.options.find("reference_text") == request.options.end()) {
            request.options["reference_text"] = *preset->reference_text;
        }
    }
    const bool has_explicit_voice_ref = body.find("voice_ref") != nullptr;
    if (const auto * value = body.find("voice"); value != nullptr && !voice_field_is_preset) {
        // Voice library: "voice" may name a wav in the configured voice_dir. When it
        // does, that audio becomes the cloning reference and the transcript from
        // prompt_text is injected unless the request already sets reference_text.
        bool voice_library_resolved = false;
        if (config_.voice_dir.has_value() && !has_explicit_voice_ref) {
            const std::string voice_name = value->as_string();
            if (const auto wav = resolve_voice_library_wav(*config_.voice_dir, voice_name)) {
                voice.speaker = engine::runtime::VoiceReference{};
                voice.speaker->audio = minitts::cli::read_audio_buffer(*wav);
                has_voice = true;
                voice_library_resolved = true;
                if (can_inject_reference_text && request.options.find("reference_text") == request.options.end()) {
                    auto text = load_voice_library_text(*config_.voice_dir, voice_name);
                    if (text.has_value()) {
                        request.options["reference_text"] = *text;
                    }
                }
            }
        }
        // Names that do not resolve to a wav keep the cached_voice_id behavior.
        if (!voice_library_resolved) {
            if (!voice.speaker.has_value()) {
                voice.speaker = engine::runtime::VoiceReference{};
            }
            voice.speaker->cached_voice_id = value->as_string();
            has_voice = true;
        }
    }
    if (const auto * value = body.find("voice_ref")) {
        if (!voice.speaker.has_value()) {
            voice.speaker = engine::runtime::VoiceReference{};
        }
        if (value->is_string()) {
            voice.speaker->audio = minitts::cli::read_audio_buffer(resolve_path(request_base_, value->as_string()));
        } else if (value->is_object()) {
            const auto & type = engine::io::json::require_string(*value, "type");
            if (type == "path") {
                voice.speaker->audio = minitts::cli::read_audio_buffer(
                    resolve_path(request_base_, engine::io::json::require_string(*value, "path")));
            } else if (type == "base64") {
                // Bound the inline reference audio so a huge base64 payload cannot
                // blow up host RAM through decode + f32 expansion (~3x its size).
                constexpr size_t kMaxVoiceRefBytes = size_t{5} * 1024 * 1024;
                // 4/3 expansion plus slack for a data URI prefix and whitespace.
                constexpr size_t kMaxVoiceRefB64Length = ((kMaxVoiceRefBytes + 2) / 3) * 4 + 4096;
                const auto & data = engine::io::json::require_string(*value, "data");
                if (data.size() > kMaxVoiceRefB64Length) {
                    throw std::runtime_error("voice_ref base64 data exceeds the 5 MiB limit");
                }
                const auto bytes = base64_decode(data);
                if (bytes.empty()) {
                    throw std::runtime_error("voice_ref base64 data decoded to an empty payload");
                }
                if (bytes.size() > kMaxVoiceRefBytes) {
                    throw std::runtime_error("voice_ref base64 data exceeds the 5 MiB limit");
                }
                voice.speaker->audio = minitts::cli::read_audio_buffer(
                    std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
            } else {
                throw std::runtime_error("voice_ref type must be \"path\" or \"base64\"");
            }
        } else {
            throw std::runtime_error("voice_ref must be a path string or an object with type \"path\" or \"base64\"");
        }
        has_voice = true;
    }
    if (const auto * value = body.find("reference_text")) {
        request.options["reference_text"] = value->as_string();
    }
    if (has_voice) {
        request.voice = std::move(voice);
    }
    return apply_default_request_options(model, std::move(request));
}

engine::runtime::TaskRequest ServerState::apply_default_request_options(
    const LoadedModel & model,
    engine::runtime::TaskRequest request) const {
    if (model.config.default_request_options.empty()) {
        return request;
    }
    auto options = model.config.default_request_options;
    for (auto & [key, value] : request.options) {
        options[key] = std::move(value);
    }
    request.options = std::move(options);
    return request;
}

struct ServerState::TimedTaskResult {
    engine::runtime::TaskResult result;
    double wall_ms = 0.0;
    std::optional<double> ttft_ms;
};

engine::runtime::RunMode ServerState::model_run_mode(const LoadedModel & model) const {
    std::shared_lock<std::shared_mutex> metadata_lock(model.metadata_mutex);
    return model.task.mode;
}

BusyGuard::Lock ServerState::acquire_model_run(
    LoadedModel & model,
    std::optional<int> request_timeout_ms) {
    int timeout_ms = 0;
    std::string model_id;
    {
        std::shared_lock<std::shared_mutex> metadata_lock(model.metadata_mutex);
        timeout_ms = resolve_busy_timeout_ms(
            model.config.busy_timeout_ms.value_or(config_.busy_timeout_ms),
            request_timeout_ms);
        model_id = model.config.id;
    }
    return model.busy.acquire(timeout_ms, model_id);
}

ServerState::TimedTaskResult ServerState::run_model(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request,
    std::optional<int> busy_timeout_ms) {
    BusyGuard::Lock lock = acquire_model_run(model, busy_timeout_ms);
    ensure_model_loaded_locked(model);
    if (model.offline == nullptr) {
        throw std::runtime_error("configured model does not provide offline execution: " + model.config.id);
    }
    const auto started = Clock::now();
    model.session->prepare(engine::runtime::build_preparation_request(request));
    auto result = model.offline->run(request);
    return TimedTaskResult{std::move(result), elapsed_ms(started), std::nullopt};
}

// `audio` selects where the samples come from: null means request.audio_input, as
// every caller did before live ingest existed; non-null pulls them from a stream
// as they arrive. Everything else — locking, preparation, timing — is identical,
// so both entry points share this body rather than drifting apart.
ServerState::TimedTaskResult ServerState::run_streaming_model_impl(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request,
    const minitts::app::AudioChunkStream * audio,
    const std::function<void(const engine::runtime::StreamEvent &)> & event_sink,
    std::optional<int> busy_timeout_ms) {
    BusyGuard::Lock lock = acquire_model_run(model, busy_timeout_ms);
    ensure_model_loaded_locked(model);
    if (model.streaming == nullptr) {
        throw std::runtime_error("configured model does not provide streaming execution: " + model.config.id);
    }
    const auto started = Clock::now();
    model.session->prepare(engine::runtime::build_preparation_request(request));
    TimedTaskResult timed_result;
    const auto sink = [&](const engine::runtime::StreamEvent & event) {
        if (!timed_result.ttft_ms.has_value() && stream_event_has_output(event)) {
            timed_result.ttft_ms = elapsed_ms(started);
        }
        if (event_sink) {
            event_sink(event);
        }
    };
    auto result = audio != nullptr
        ? minitts::app::run_streaming_task(*model.streaming, request, sink, *audio)
        : minitts::app::run_streaming_task(*model.streaming, request, sink);
    timed_result.result = std::move(result);
    timed_result.wall_ms = elapsed_ms(started);
    if (!timed_result.ttft_ms.has_value() && task_result_has_output(timed_result.result)) {
        timed_result.ttft_ms = timed_result.wall_ms;
    }
    return timed_result;
}

ServerState::TimedTaskResult ServerState::run_streaming_model(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request,
    const std::function<void(const engine::runtime::StreamEvent &)> & event_sink,
    std::optional<int> busy_timeout_ms) {
    return run_streaming_model_impl(model, request, nullptr, event_sink, busy_timeout_ms);
}

HttpResponse ServerState::handle_speech(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_speech_model(body);
    auto request = build_speech_request(model, body);
    if (body.find("stream_format") != nullptr || bool_field(body, "stream", false)) {
        return handle_speech_stream(model, std::move(request), body);
    }
    const auto busy_timeout_ms = parse_busy_timeout_override(body);
    if (model_run_mode(model) == engine::runtime::RunMode::Streaming) {
        // A non-streaming OpenAI-compatible request still expects a completed
        // WAV response. Streaming Qwen sessions otherwise keep no whole-output
        // buffer by design.
        request.options["stream_accumulate"] = "true";
    }
    const auto timed_result = model_run_mode(model) == engine::runtime::RunMode::Streaming
        ? run_streaming_model(model, request, {}, busy_timeout_ms)
        : run_model(model, request, busy_timeout_ms);
    const auto & audio = select_audio_output(timed_result.result);
    const auto wav = encode_pcm16_wav(audio);
    const auto response_format = engine::io::json::optional_string(body, "response_format", "wav");
    if (response_format == "json" || response_format == "b64_json") {
        return json_response(
            "{\"audio\":" + json_quote(base64_encode(wav)) +
            ",\"format\":\"wav\",\"timing\":" + timing_json(timed_result.wall_ms, audio) + "}");
    }
    HttpResponse response;
    response.status = 200;
    response.content_type = "audio/wav";
    response.body = std::string(reinterpret_cast<const char *>(wav.data()), wav.size());
    response.headers = timing_headers(timed_result.wall_ms, audio);
    return response;
}

HttpResponse ServerState::handle_speech_stream(
    LoadedModel & model,
    engine::runtime::TaskRequest request,
    const Value & body) {
    if (model_run_mode(model) != engine::runtime::RunMode::Streaming) {
        throw std::runtime_error("speech streaming requires a model configured with mode=streaming");
    }
    const auto stream_format = engine::io::json::optional_string(body, "stream_format", "sse");
    const auto response_format = engine::io::json::optional_string(body, "response_format", "pcm");
    if (response_format != "pcm") {
        throw std::runtime_error("streaming speech currently supports response_format=pcm");
    }
    if (stream_format != "sse" && stream_format != "audio") {
        throw std::runtime_error("streaming speech stream_format must be sse or audio");
    }

    const auto busy_timeout_ms = parse_busy_timeout_override(body);
    LoadedModel * model_ptr = &model;
    if (stream_format == "sse") {
        return sse_response(
            [this, model_ptr, request = std::move(request), busy_timeout_ms](HttpStreamWriter & writer) {
                bool wrote_audio = false;
                const auto timed_result = run_streaming_model(
                    *model_ptr,
                    request,
                    [&](const engine::runtime::StreamEvent & event) {
                        std::vector<engine::runtime::AudioBuffer> buffers;
                        if (event.audio_output.has_value()) {
                            buffers.push_back(*event.audio_output);
                        }
                        for (const auto & named : event.named_audio_outputs) {
                            buffers.push_back(named.audio);
                        }
                        for (const auto & audio : buffers) {
                            const auto pcm = encode_pcm16_samples(audio);
                            write_sse(
                                writer,
                                "{\"type\":\"speech.audio.delta\",\"audio\":" +
                                    json_quote(base64_encode(pcm)) +
                                    "}");
                            wrote_audio = true;
                        }
                    },
                    busy_timeout_ms);
                if (!wrote_audio) {
                    throw std::runtime_error("streaming speech model produced no audio delta events");
                }
                write_sse(
                    writer,
                    "{\"type\":\"speech.audio.done\",\"timing\":" +
                        ttft_timing_json(require_ttft_ms(timed_result.ttft_ms)) +
                        "}");
                write_sse_done(writer);
            });
    }
    return chunked_audio_response([this, model_ptr, request = std::move(request), busy_timeout_ms](HttpStreamWriter & writer) {
        bool wrote_audio = false;
        (void)run_streaming_model(
            *model_ptr,
            request,
            [&](const engine::runtime::StreamEvent & event) {
                if (event.audio_output.has_value()) {
                    const auto pcm = encode_pcm16_samples(*event.audio_output);
                    writer.write(std::string(reinterpret_cast<const char *>(pcm.data()), pcm.size()));
                    wrote_audio = true;
                }
                for (const auto & named : event.named_audio_outputs) {
                    const auto pcm = encode_pcm16_samples(named.audio);
                    writer.write(std::string(reinterpret_cast<const char *>(pcm.data()), pcm.size()));
                    wrote_audio = true;
                }
            },
            busy_timeout_ms);
        if (!wrote_audio) {
            throw std::runtime_error("streaming speech model produced no audio delta events");
        }
    });
}

// Accepts the same multipart/form-data shape OpenAI's Whisper API (and clients built against it,
// e.g. Open WebUI) send: a "file" part with the audio bytes, plus "model" and optional "language"
// fields. audio.cpp's native JSON request only takes a server-local path, so the uploaded bytes are
// spooled to a temp file and routed through the existing JSON request builder.
// Live PCM ingest. The client streams raw interleaved samples in a chunked request
// body while transcript deltas stream back as SSE on the same connection, so
// partials track capture instead of waiting for a finished upload. Same event shape
// as the file-backed `stream=true` path, so a client can share one SSE reader.
//
// Deliberately a single request rather than open/append/finish session endpoints:
// the busy lock is held for the length of a run, so a session spread across
// separate requests would pin the model while idling between a client's appends —
// and wedge it outright if that client vanished. One request bounds the lock by the
// lifetime of the connection.
ServerState::LoadedModel & ServerState::require_speech_model(const Value & body) {
    if (body.find("model") != nullptr) {
        return require_model(body);
    }
    // This app's clients -- the Speak page and MCP callers -- talk to one
    // character on one model and do not know model ids. A speech request that
    // names no model gets the unambiguous one, matching what /v1/audio/voices
    // already does; only a config with several TTS models still requires the
    // field.
    std::lock_guard<std::mutex> state_lock(models_mutex_);
    if (models_.size() == 1) {
        return *models_.front();
    }
    LoadedModel * only_tts = nullptr;
    for (const auto & model : models_) {
        if (model->config.task != "tts") {
            continue;
        }
        if (only_tts != nullptr) {
            throw std::runtime_error(
                "missing required json key: model (several TTS models are configured)");
        }
        only_tts = model.get();
    }
    if (only_tts == nullptr) {
        throw std::runtime_error("missing required json key: model");
    }
    return *only_tts;
}

ServerState::LoadedModel * ServerState::find_speech_model() {
    std::lock_guard<std::mutex> state_lock(models_mutex_);
    for (const auto & model : models_) {
        if (model->config.task == "tts") {
            // Entries are never removed, only unloaded, so the pointer stays
            // valid after the lock is released.
            return model.get();
        }
    }
    return nullptr;
}

void ServerState::apply_character(LoadedModel & model, const CharacterConfig & character) {
    LoadedModel::RuntimeVoicePreset preset;
    if (character.is_custom()) {
        // Read outside the metadata lock; only the assignment needs it.
        preset.audio = normalize_voice_reference(
            minitts::cli::read_audio_buffer(character_dir_ / character.voice_file));
        if (!character.transcript.empty()) {
            preset.reference_text = character.transcript;
        }
    }
    std::unique_lock<std::shared_mutex> metadata_lock(model.metadata_mutex);
    if (!character.is_custom()) {
        const auto it = model.voice_presets.find(character.preset);
        if (it == model.voice_presets.end()) {
            throw std::runtime_error(
                "character preset '" + character.preset + "' is not a configured voice preset");
        }
        preset = it->second;
    }
    model.default_voice_preset = std::move(preset);
    metadata_lock.unlock();
}

HttpResponse ServerState::handle_character_get() {
    CharacterConfig character;
    {
        std::lock_guard<std::mutex> lock(character_mutex_);
        character = character_;
    }
    std::string presets = "[";
    if (auto * model = find_speech_model()) {
        std::shared_lock<std::shared_mutex> metadata_lock(model->metadata_mutex);
        std::vector<std::string> names;
        names.reserve(model->voice_presets.size());
        for (const auto & [name, preset] : model->voice_presets) {
            names.push_back(name);
        }
        std::sort(names.begin(), names.end());
        for (size_t index = 0; index < names.size(); ++index) {
            presets += (index == 0 ? "" : ",") + json_quote(names[index]);
        }
    }
    presets += "]";

    std::string body = "{\"name\":" + json_quote(character.name) +
        ",\"source\":" + std::string(character.is_custom() ? "\"custom\"" : "\"preset\"");
    if (!character.is_custom()) {
        body += ",\"preset\":" + json_quote(character.preset);
    }
    if (!character.transcript.empty()) {
        body += ",\"transcript\":" + json_quote(character.transcript);
    }
    if (!character.persona.empty()) {
        body += ",\"persona\":" + json_quote(character.persona);
    }
    body += ",\"available_presets\":" + presets + "}";
    return json_response(body);
}

HttpResponse ServerState::handle_character_set(const HttpRequest & request) {
    auto * model = find_speech_model();
    if (model == nullptr) {
        return error_response(400, "no TTS model is configured to give the character a voice", "invalid_request_error");
    }

    CharacterConfig character;
    std::string content_type;
    if (const auto it = request.headers.find("content-type"); it != request.headers.end()) {
        content_type = it->second;
    }
    try {
        if (const auto boundary = extract_multipart_boundary(content_type)) {
            // A custom voice: name, transcript, and a WAV recording, stored under
            // a fixed filename so replacing the character never accumulates files.
            // The persona is inherited unless the form replaces it -- a new
            // recording changes how the character sounds, not who they are.
            {
                std::lock_guard<std::mutex> lock(character_mutex_);
                character.persona = character_.persona;
            }
            const auto parts = parse_multipart_body(request.body, *boundary);
            const MultipartPart * file_part = nullptr;
            for (const auto & part : parts) {
                if (part.name == "file" || part.name == "voice") {
                    file_part = &part;
                } else if (part.name == "name") {
                    character.name = part.data;
                } else if (part.name == "transcript" || part.name == "reference_text") {
                    character.transcript = part.data;
                } else if (part.name == "persona") {
                    character.persona = part.data;
                }
            }
            if (file_part == nullptr || file_part->data.empty()) {
                throw std::runtime_error("a custom character voice requires a non-empty 'file' upload");
            }
            if (!is_wav_upload_filename(file_part->filename)) {
                throw std::runtime_error("only WAV uploads are supported for the character voice");
            }
            character.name = sanitize_character_name(character.name);
            std::filesystem::create_directories(character_dir_);
            const auto voice_path = character_dir_ / "voice.wav";
            {
                std::ofstream out(voice_path, std::ios::binary | std::ios::trunc);
                if (!out) {
                    throw std::runtime_error("could not write " + voice_path.string());
                }
                out.write(file_part->data.data(), static_cast<std::streamsize>(file_part->data.size()));
                if (!out) {
                    throw std::runtime_error("could not write " + voice_path.string());
                }
            }
            character.voice_file = "voice.wav";
        } else {
            // JSON: {name, preset} switches to a bundled voice; {name} alone (or
            // {name, transcript} for a custom character) edits the character
            // while keeping its saved voice, so a rename never demands the
            // recording be uploaded again.
            const auto body = engine::io::json::parse(request.body);
            {
                std::lock_guard<std::mutex> lock(character_mutex_);
                character = character_;
            }
            character.name = sanitize_character_name(engine::io::json::require_string(body, "name"));
            if (const auto * preset = body.find("preset");
                preset != nullptr && preset->is_string() && !preset->as_string().empty()) {
                character.preset = preset->as_string();
                character.voice_file.clear();
                character.transcript.clear();
            } else if (const auto * transcript = body.find("transcript");
                transcript != nullptr && transcript->is_string() && character.is_custom()) {
                character.transcript = transcript->as_string();
            }
            if (const auto * persona = body.find("persona");
                persona != nullptr && persona->is_string()) {
                character.persona = persona->as_string();
            }
            if (character.preset.empty() && !character.is_custom()) {
                throw std::runtime_error("the character needs a voice: name a 'preset' or upload a recording");
            }
        }
        apply_character(*model, character);
    } catch (const std::exception & ex) {
        return error_response(400, ex.what(), "invalid_request_error");
    }

    save_character(character_dir_, character);
    store_character_in_library(character);
    {
        std::lock_guard<std::mutex> lock(character_mutex_);
        character_ = character;
    }
    // The system prompt just changed with the character; re-prime the
    // sidecar's cache so the next fresh conversation starts fast.
    warm_llm_system_prompt();
    return handle_character_get();
}

void ServerState::store_character_in_library(const CharacterConfig & character) {
    const auto entry_dir = character_dir_ / "library" / character_slug(character.name);
    std::filesystem::create_directories(entry_dir);
    if (character.is_custom()) {
        // The active slot holds the recording; the library entry gets its own
        // copy so deleting or replacing the active character never orphans it.
        std::filesystem::copy_file(
            character_dir_ / character.voice_file,
            entry_dir / "voice.wav",
            std::filesystem::copy_options::overwrite_existing);
        CharacterConfig stored = character;
        stored.voice_file = "voice.wav";
        save_character(entry_dir, stored);
    } else {
        save_character(entry_dir, character);
    }
}

HttpResponse ServerState::handle_character_voice() {
    CharacterConfig current;
    {
        std::lock_guard<std::mutex> lock(character_mutex_);
        current = character_;
    }
    if (!current.is_custom()) {
        return error_response(404, "the active character uses a bundled voice preset", "not_found");
    }
    std::ifstream in(character_dir_ / current.voice_file, std::ios::binary);
    if (!in) {
        return error_response(404, "the character recording is missing", "not_found");
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    HttpResponse response;
    response.status = 200;
    response.content_type = "audio/wav";
    response.body = buffer.str();
    // The file changes whenever the character does; never let a browser cache
    // yesterday's voice.
    response.headers["Cache-Control"] = "no-store";
    return response;
}

HttpResponse ServerState::handle_characters_list() {
    CharacterConfig current;
    {
        std::lock_guard<std::mutex> lock(character_mutex_);
        current = character_;
    }
    const auto active_id = character_slug(current.name);
    std::string body = "{\"active_id\":" + json_quote(active_id) + ",\"characters\":[";
    bool first = true;
    for (const auto & entry : list_character_library(character_dir_)) {
        if (!first) {
            body += ",";
        }
        first = false;
        body += "{\"id\":" + json_quote(entry.id) +
            ",\"name\":" + json_quote(entry.config.name) +
            ",\"source\":" + std::string(entry.config.is_custom() ? "\"custom\"" : "\"preset\"");
        if (!entry.config.is_custom()) {
            body += ",\"preset\":" + json_quote(entry.config.preset);
        }
        body += ",\"active\":" + std::string(entry.id == active_id ? "true" : "false") + "}";
    }
    body += "]}";
    return json_response(body);
}

HttpResponse ServerState::handle_character_activate(const std::string & body_text) {
    auto * model = find_speech_model();
    if (model == nullptr) {
        return error_response(400, "no TTS model is configured to give the character a voice", "invalid_request_error");
    }
    const auto body = engine::io::json::parse(body_text);
    const auto id = engine::io::json::require_string(body, "id");
    if (!is_valid_character_id(id)) {
        return error_response(400, "invalid character id", "invalid_request_error");
    }
    const auto entry_dir = character_dir_ / "library" / id;
    if (!std::filesystem::exists(entry_dir / "character.json")) {
        return error_response(400, "no saved character with id '" + id + "'", "invalid_request_error");
    }

    CharacterConfig character;
    try {
        character = load_character(entry_dir);
        if (character.is_custom()) {
            // Copy the recording into the active slot so the active character
            // stays self-contained even if the library entry is later deleted.
            std::filesystem::copy_file(
                entry_dir / character.voice_file,
                character_dir_ / "voice.wav",
                std::filesystem::copy_options::overwrite_existing);
            character.voice_file = "voice.wav";
        }
        apply_character(*model, character);
    } catch (const std::exception & ex) {
        return error_response(400, ex.what(), "invalid_request_error");
    }

    save_character(character_dir_, character);
    {
        std::lock_guard<std::mutex> lock(character_mutex_);
        character_ = character;
    }
    // The system prompt just changed with the character; re-prime the
    // sidecar's cache so the next fresh conversation starts fast.
    warm_llm_system_prompt();
    return handle_character_get();
}

HttpResponse ServerState::handle_character_delete(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    const auto id = engine::io::json::require_string(body, "id");
    if (!is_valid_character_id(id)) {
        return error_response(400, "invalid character id", "invalid_request_error");
    }
    const auto entry_dir = character_dir_ / "library" / id;
    if (!std::filesystem::exists(entry_dir)) {
        return error_response(400, "no saved character with id '" + id + "'", "invalid_request_error");
    }
    // The active slot holds its own copy of everything, so deleting a library
    // entry never silences the currently speaking character.
    std::filesystem::remove_all(entry_dir);
    return handle_characters_list();
}

mcp::SpeakOutcome ServerState::run_mcp_speak(const std::string & text, long long seed) {
    mcp::SpeakOutcome outcome;
    try {
        auto * model = find_speech_model();
        if (model == nullptr) {
            throw std::runtime_error("no TTS model is configured");
        }
        Value::Object fields;
        fields.emplace("input", Value::make_string(text));
        if (seed >= 0) {
            fields.emplace("seed", Value::make_number(static_cast<double>(seed)));
        }
        const auto body = Value::make_object(std::move(fields));
        auto request = build_speech_request(*model, body);
        if (model_run_mode(*model) == engine::runtime::RunMode::Streaming) {
            // MCP tool results carry a complete clip, so the streaming session
            // has to keep the whole output it would otherwise discard.
            request.options["stream_accumulate"] = "true";
        }
        const auto timed_result = model_run_mode(*model) == engine::runtime::RunMode::Streaming
            ? run_streaming_model(*model, request)
            : run_model(*model, request);
        const auto & audio = select_audio_output(timed_result.result);
        const auto wav = encode_pcm16_wav(audio);
        outcome.wav_base64 = base64_encode(wav);
        outcome.audio_seconds = audio_duration_ms(audio) / 1000.0;
        outcome.ok = true;
    } catch (const std::exception & ex) {
        outcome.error_text = ex.what();
    }
    return outcome;
}

HttpResponse ServerState::handle_mcp(const HttpRequest & request) {
    if (request.method != "POST") {
        // No server-initiated stream and no session state, so GET and DELETE
        // have nothing to do here; the spec allows refusing both.
        HttpResponse response;
        response.status = 405;
        response.headers["Allow"] = "POST";
        response.body = "{\"error\":\"POST a JSON-RPC message to this MCP endpoint\"}";
        return response;
    }
    const auto reply = mcp::handle_mcp_message(
        request.body,
        [this](const std::string & text, long long seed) { return run_mcp_speak(text, seed); });
    HttpResponse response;
    response.status = reply.status;
    response.body = reply.body;
    return response;
}

std::string ServerState::fresh_llm_warm_body() const {
    CharacterConfig character;
    LlmSettings settings;
    {
        std::lock_guard<std::mutex> lock(character_mutex_);
        character = character_;
        settings = llm_settings_;
    }
    const std::string system =
        render_master_prompt(settings, character.name, character.persona) +
        hygiene::kNoteRule +
        length_guidance(settings, 0);
    // The near-empty user turn keeps every chat template happy (Gemma folds
    // the system text into the first user turn); the real first message
    // shares the rendered system prefix and diverges only at its own text.
    return "{\"stream\":false,\"cache_prompt\":true,\"max_tokens\":1,\"messages\":["
           "{\"role\":\"system\",\"content\":" + json_quote(system) + "},"
           "{\"role\":\"user\",\"content\":\" \"}]}";
}

void ServerState::warm_llm_system_prompt() {
    if (config_.llm_port <= 0) {
        return;
    }
    std::thread([this, host = config_.llm_host, port = config_.llm_port,
                 body = fresh_llm_warm_body()] {
        // The sidecar may still be loading its model (startup, or a switch in
        // flight); wait for health quietly, then prime and go away.
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(300);
        while (std::chrono::steady_clock::now() < deadline) {
            std::string health;
            if (llm::http_get_status(host, port, "/health", health) == 200) {
                queue_llm_warm(body);
                return;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }).detach();
}

void ServerState::queue_llm_warm(std::string body) {
    std::lock_guard<std::mutex> lock(llm_warm_mutex_);
    llm_warm_pending_ = std::move(body);
    llm_warm_cv_.notify_all();
    if (llm_warm_worker_running_) {
        return;
    }
    llm_warm_worker_running_ = true;
    std::thread([this] {
        std::unique_lock<std::mutex> lock(llm_warm_mutex_);
        while (true) {
            if (llm_warm_pending_.empty()) {
                llm_warm_worker_running_ = false;
                return;
            }
            if (llm_generations_active_ > 0) {
                llm_warm_cv_.wait(lock);
                continue;
            }
            // Let the draft settle before spending sidecar time: a prefill
            // cannot be interrupted once sent, and the send most often lands
            // within half a second of the pause that queued this warm. A
            // newer body or a generation arriving during the wait replaces
            // or defers this dispatch for free.
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            lock.lock();
            if (llm_warm_pending_.empty() || llm_generations_active_ > 0) {
                continue;
            }
            const auto body = std::move(llm_warm_pending_);
            llm_warm_pending_.clear();
            lock.unlock();
            std::string response;
            llm::http_post_status(config_.llm_host, config_.llm_port,
                                  "/v1/chat/completions", body, response);
            lock.lock();
        }
    }).detach();
}

void ServerState::begin_llm_generation() {
    std::lock_guard<std::mutex> lock(llm_warm_mutex_);
    ++llm_generations_active_;
}

void ServerState::end_llm_generation() {
    {
        std::lock_guard<std::mutex> lock(llm_warm_mutex_);
        --llm_generations_active_;
    }
    llm_warm_cv_.notify_all();
}

void ServerState::note_llm_decode_rate(double predicted_n, double predicted_ms) {
    // Tiny generations measure mostly overhead; skip them.
    if (predicted_n < 8.0 || predicted_ms <= 0.0) {
        return;
    }
    const double rate = 1000.0 * predicted_n / predicted_ms;
    const double previous = llm_decode_tps_ema_.load();
    llm_decode_tps_ema_.store(previous <= 0.0 ? rate : 0.7 * previous + 0.3 * rate);
}

std::string ServerState::llm_settings_json() const {
    LlmSettings settings;
    {
        std::lock_guard<std::mutex> lock(character_mutex_);
        settings = llm_settings_;
    }
    std::ostringstream out;
    out << "{\"master_prompt\":" << json_quote(settings.master_prompt)
        << ",\"temperature\":" << settings.temperature
        << ",\"top_p\":" << settings.top_p
        << ",\"repeat_penalty\":" << settings.repeat_penalty
        << ",\"max_tokens\":" << settings.max_tokens
        << ",\"length_ramp\":" << (settings.length_ramp ? "true" : "false");
    // The switchable-model registry: which model is running now and what else
    // could be. Empty registry (launcher-owned sidecar) reports no models and
    // the UI hides the picker.
    std::string current = llm_manager_ ? llm_manager_->current_model_id() : std::string();
    out << ",\"model\":" << json_quote(current) << ",\"models\":[";
    bool first = true;
    for (const auto & spec : config_.llm_models) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "{\"id\":" << json_quote(spec.id)
            << ",\"name\":" << json_quote(spec.name)
            << ",\"installed\":"
            << (std::filesystem::exists(spec.path) ? "true" : "false") << "}";
    }
    out << "]}";
    return out.str();
}

HttpResponse ServerState::handle_typing_telemetry(const std::string & body_text) {
    if (body_text.size() > 512 * 1024) {
        return error_response(400, "typing episode is too large", "invalid_request_error");
    }
    const auto body = engine::io::json::parse(body_text);
    const auto * events = body.find("events");
    if (events == nullptr || !events->is_array() || events->as_array().empty()) {
        return error_response(400, "a typing episode needs a non-empty 'events' array", "invalid_request_error");
    }
    const auto * outcome = body.find("outcome");
    if (outcome == nullptr || !outcome->is_string()) {
        return error_response(400, "a typing episode needs an 'outcome' string", "invalid_request_error");
    }

    namespace fs = std::filesystem;
    const fs::path dir = fs::path("data");
    const fs::path file = dir / "typing_episodes.jsonl";
    std::lock_guard<std::mutex> lock(typing_log_mutex_);
    std::error_code ec;
    fs::create_directories(dir, ec);
    // A runaway client must not fill the disk; 256 MB of episodes is years
    // of typing for one person.
    if (fs::exists(file, ec) && fs::file_size(file, ec) > 256ull * 1024 * 1024) {
        return error_response(507, "typing log is full; archive data/typing_episodes.jsonl", "storage_full");
    }
    std::ofstream out(file, std::ios::app | std::ios::binary);
    if (!out) {
        return error_response(500, "could not open the typing log", "internal_error");
    }
    // One line per episode, stamped server-side so offline analysis can
    // order collection sessions without trusting client clocks.
    out << "{\"received_unix_ms\":"
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count()
        << ",\"episode\":" << body_text << "}\n";
    return json_response("{\"status\":\"recorded\"}");
}

bool ServerState::active_llm_cache_rollback() const {
    if (llm_manager_ == nullptr) {
        return true;
    }
    const auto * spec = find_llm_spec(llm_manager_->current_model_id());
    return spec == nullptr || spec->cache_rollback;
}

const LlmModelSpec * ServerState::find_llm_spec(const std::string & id) const {
    if (id.empty()) {
        return nullptr;
    }
    for (const auto & spec : config_.llm_models) {
        if (spec.id == id) {
            return &spec;
        }
    }
    return nullptr;
}

void ServerState::switch_llm_model(const LlmModelSpec & spec) {
    std::lock_guard<std::mutex> lock(llm_switch_mutex_);
    if (llm_manager_ == nullptr) {
        throw std::runtime_error("this server does not manage the LLM sidecar");
    }
    const std::string previous = llm_manager_->current_model_id();
    if (previous == spec.id && llm_manager_->process_running()) {
        return;
    }
    llm_manager_->stop();
    // A different model writes at a different speed; let the opener sizing
    // re-learn instead of inheriting the previous model's rate.
    llm_decode_tps_ema_.store(0.0);
    std::string error;
    if (llm_manager_->start(spec, error) && llm_manager_->wait_ready(300, error)) {
        return;
    }
    // The requested model failed; a dead sidecar helps nobody, so put the
    // previous one back before reporting what went wrong.
    llm_manager_->stop();
    if (const LlmModelSpec * fallback = find_llm_spec(previous)) {
        std::string restore_error;
        if (llm_manager_->start(*fallback, restore_error)) {
            llm_manager_->wait_ready(300, restore_error);
        }
    }
    throw std::runtime_error("could not switch to " + spec.name + ": " + error);
}

HttpResponse ServerState::handle_llm_settings_get() {
    return json_response(llm_settings_json());
}

HttpResponse ServerState::handle_llm_settings_set(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    LlmSettings settings;
    if (const auto * reset = body.find("reset"); reset != nullptr && reset->is_bool() && reset->as_bool()) {
        settings = default_llm_settings();
    } else {
        {
            std::lock_guard<std::mutex> lock(character_mutex_);
            settings = llm_settings_;
        }
        if (const auto * value = body.find("master_prompt"); value != nullptr && value->is_string()) {
            settings.master_prompt = value->as_string();
        }
        if (const auto * value = body.find("temperature"); value != nullptr && value->is_number()) {
            settings.temperature = value->as_number();
        }
        if (const auto * value = body.find("top_p"); value != nullptr && value->is_number()) {
            settings.top_p = value->as_number();
        }
        if (const auto * value = body.find("repeat_penalty"); value != nullptr && value->is_number()) {
            settings.repeat_penalty = value->as_number();
        }
        if (const auto * value = body.find("max_tokens"); value != nullptr && value->is_number()) {
            settings.max_tokens = value->as_i64();
        }
        if (const auto * value = body.find("length_ramp"); value != nullptr && value->is_bool()) {
            settings.length_ramp = value->as_bool();
        }
    }

    // A model change restarts the sidecar, which takes seconds to minutes; the
    // request blocks until the new model answers /health so the client knows
    // exactly when chat is live again. Validated before anything is persisted.
    const LlmModelSpec * switch_to = nullptr;
    if (const auto * value = body.find("model"); value != nullptr && value->is_string()) {
        const std::string requested = value->as_string();
        if (!requested.empty()) {
            switch_to = find_llm_spec(requested);
            if (switch_to == nullptr) {
                return error_response(400, "unknown chat model: " + requested, "invalid_request_error");
            }
            if (!std::filesystem::exists(switch_to->path)) {
                return error_response(
                    400, switch_to->name + " is not installed yet", "invalid_request_error");
            }
            settings.model = switch_to->id;
        }
    }
    if (switch_to != nullptr) {
        try {
            switch_llm_model(*switch_to);
        } catch (const std::exception & ex) {
            return error_response(500, ex.what(), "llm_unavailable");
        }
    }

    try {
        save_llm_settings(character_dir_, settings);
    } catch (const std::exception & ex) {
        return error_response(400, ex.what(), "invalid_request_error");
    }
    {
        std::lock_guard<std::mutex> lock(character_mutex_);
        llm_settings_ = settings;
    }
    // The master prompt (or the sidecar itself) may just have changed;
    // re-prime the cache for the next fresh conversation.
    warm_llm_system_prompt();
    return json_response(llm_settings_json());
}

namespace {

// Trailing whitespace and sentence-final punctuation, ASCII and CJK: the
// part of a draft most often typed in the last instant before send.
std::string strip_terminal_tail(std::string text) {
    static const std::vector<std::string> tails = {
        " ", "\t", "\n", ".", "!", "?", "~", "…", "。", "！", "？", "～"};
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto & tail : tails) {
            if (text.size() >= tail.size() &&
                text.compare(text.size() - tail.size(), tail.size(), tail) == 0) {
                text.resize(text.size() - tail.size());
                changed = true;
                break;
            }
        }
    }
    return text;
}

// The speculation key: the conversation's role/content pairs, with the
// newest user message's terminal punctuation stripped. Predictive fill: a
// reply generated for "what should i cook tonight" must still attach when
// the user sends "...tonight?" -- the question mark typed in the last
// moment does not change what they asked.
std::string speculation_key(
    const engine::io::json::Value::Array & list, size_t last_user_index) {
    std::string key;
    for (size_t i = 0; i < list.size(); ++i) {
        key += list[i].find("role")->as_string();
        key += '\x1f';
        std::string content = list[i].find("content")->as_string();
        if (i == last_user_index) {
            content = strip_terminal_tail(std::move(content));
        }
        key += content;
        key += '\x1e';
    }
    return key;
}

}  // namespace

HttpResponse ServerState::handle_chat_speak(const std::string & body_text) {
    if (config_.llm_port <= 0) {
        return error_response(
            503,
            "no LLM sidecar is configured; start the server with --llm-port",
            "llm_unavailable");
    }
    auto * model = find_speech_model();
    if (model == nullptr) {
        return error_response(400, "no TTS model is configured", "invalid_request_error");
    }

    const auto body = engine::io::json::parse(body_text);
    const auto * messages = body.find("messages");
    if (messages == nullptr || !messages->is_array() || messages->as_array().empty()) {
        return error_response(400, "chat requires a non-empty 'messages' array", "invalid_request_error");
    }
    // Bound the history before it reaches the sidecar: an oversized payload
    // would otherwise turn into an opaque context-overflow error mid-stream.
    // ~24k characters comfortably fills the useful part of Peach's 8k-token
    // context; the client trims its own history well before this.
    size_t total_content = 0;
    for (const auto & message : messages->as_array()) {
        if (const auto * content = message.find("content");
            content != nullptr && content->is_string()) {
            total_content += content->as_string().size();
        }
    }
    if (total_content > 24000) {
        return error_response(
            400,
            "chat history is too large (" + std::to_string(total_content) +
                " characters; the limit is 24000) -- trim older messages",
            "invalid_request_error");
    }

    CharacterConfig character;
    LlmSettings settings;
    {
        std::lock_guard<std::mutex> lock(character_mutex_);
        character = character_;
        settings = llm_settings_;
    }
    // Stored roleplay settings are the defaults; a request may still override
    // any of them per call.
    const double temperature = engine::io::json::optional_f32(
        body, "temperature", static_cast<float>(settings.temperature));
    const double top_p = engine::io::json::optional_f32(
        body, "top_p", static_cast<float>(settings.top_p));
    const double repeat_penalty = engine::io::json::optional_f32(
        body, "repeat_penalty", static_cast<float>(settings.repeat_penalty));
    // The opener stays short because first audio waits on it; once replies are
    // playing there is time for more, so the budget grows with each completed
    // assistant turn toward the configured ceiling. An explicit max_tokens in
    // the request bypasses the ramp entirely.
    int64_t assistant_turns = 0;
    for (const auto & message : messages->as_array()) {
        if (const auto * role = message.find("role");
            role != nullptr && role->is_string() && role->as_string() == "assistant") {
            ++assistant_turns;
        }
    }
    const auto * explicit_max = body.find("max_tokens");
    const int64_t max_tokens = explicit_max != nullptr && explicit_max->is_number()
        ? explicit_max->as_i64()
        : ramped_max_tokens(settings, assistant_turns);
    const int64_t llm_seed = engine::io::json::optional_i64(body, "seed", -1);
    const int64_t tts_seed = engine::io::json::optional_i64(body, "tts_seed", -1);
    // A prewarm request is the same conversation the client is ABOUT to send:
    // the draft the user is still typing rides as the newest message, and the
    // sidecar prefills it in the background. Each successive prewarm extends
    // the cached prefix by only the newly typed characters, so when the real
    // message arrives its prompt is already resident and the first token is
    // one decode step away. The small-batch tail prefill this hides runs at
    // decode speed on CPU experts -- several hundred milliseconds per turn.
    const bool prewarm = [&] {
        const auto * flag = body.find("prewarm");
        return flag != nullptr && flag->is_bool() && flag->as_bool();
    }();
    // Speculation goes one step past prewarming: generate the WHOLE reply
    // while the user hovers over a finished-looking draft, so send attaches
    // to a buffer instead of starting a generation.
    const bool speculate = [&] {
        const auto * flag = body.find("speculate");
        return flag != nullptr && flag->is_bool() && flag->as_bool();
    }();

    // The system prompt is deliberately STATIC across turns (master prompt,
    // the bracketed-note contract, and the ramp-tier guidance): per-turn
    // steering rides inside the newest user message instead, so the prompt
    // cache keeps first-token latency low.
    const std::string system_prompt =
        render_master_prompt(settings, character.name, character.persona) +
        hygiene::kNoteRule +
        length_guidance(settings, assistant_turns);

    // Validate roles first, then collect what the steering note needs: the
    // sanitized assistant history and the newest user message.
    const auto & list = messages->as_array();
    for (const auto & message : list) {
        const auto * role = message.find("role");
        const auto * content = message.find("content");
        if (role == nullptr || !role->is_string() || content == nullptr || !content->is_string()) {
            return error_response(400, "each message needs string 'role' and 'content'", "invalid_request_error");
        }
        const auto & role_name = role->as_string();
        if (role_name != "system" && role_name != "user" && role_name != "assistant") {
            return error_response(400, "message roles must be system, user, or assistant", "invalid_request_error");
        }
    }
    std::vector<std::string> assistant_history;
    size_t last_user_index = list.size();
    for (size_t i = 0; i < list.size(); ++i) {
        const auto & role_name = list[i].find("role")->as_string();
        if (role_name == "assistant") {
            assistant_history.push_back(hygiene::sanitize_assistant_text(
                list[i].find("content")->as_string()));
        } else if (role_name == "user") {
            last_user_index = i;
        }
    }

    // Build the llama.cpp request by hand: cache_prompt keeps the chat prefix
    // KV resident across turns, so each turn's prefill covers only what is
    // new. The stop strings end generation at role leakage (even Peach's
    // habit of continuing the dialogue as the user) and at the blank line
    // where a drifting model starts re-answering old turns.
    std::string llama_body = std::string("{\"stream\":") +
        (prewarm ? "false" : "true") + ",\"cache_prompt\":true";
    llama_body += ",\"stop\":[\"<|im_end|>\",\"<|im_start|>\",\"\\n\\n\"]";
    // The DRY sampler stops the model from recycling whole sentences it wrote
    // dozens of turns ago (goodnight sign-offs were coming back verbatim);
    // allowed_length 4 leaves ordinary short phrases untouched.
    llama_body += ",\"dry_multiplier\":0.8,\"dry_base\":1.75"
                  ",\"dry_allowed_length\":4,\"dry_penalty_last_n\":4096";
    llama_body += ",\"temperature\":" + std::to_string(temperature);
    llama_body += ",\"top_p\":" + std::to_string(top_p);
    llama_body += ",\"repeat_penalty\":" + std::to_string(repeat_penalty);
    llama_body += ",\"max_tokens\":" + std::to_string(prewarm ? 1 : max_tokens);
    if (llm_seed >= 0) {
        llama_body += ",\"seed\":" + std::to_string(llm_seed);
    }
    llama_body += ",\"messages\":[";
    // Built alongside: the canonical next-turn request (bare last user
    // message, next ramp tier) that re-primes the prompt cache once the reply
    // is known. See chat_orchestrate.
    const std::string next_system =
        render_master_prompt(settings, character.name, character.persona) +
        hygiene::kNoteRule +
        length_guidance(settings, assistant_turns + 1);
    std::string warm_prefix =
        "{\"stream\":false,\"cache_prompt\":true,\"max_tokens\":1,\"messages\":[";
    bool first_message = true;
    const bool client_has_system = list[0].find("role")->as_string() == "system";
    if (!client_has_system) {
        llama_body += "{\"role\":\"system\",\"content\":" + json_quote(system_prompt) + "}";
        warm_prefix += "{\"role\":\"system\",\"content\":" + json_quote(next_system) + "}";
        first_message = false;
    }
    size_t assistant_seen = 0;
    for (size_t i = 0; i < list.size(); ++i) {
        const auto & role_name = list[i].find("role")->as_string();
        std::string content_text = list[i].find("content")->as_string();
        if (role_name == "assistant") {
            // The model's own past, cleaned: feeding it raw teaches it to
            // imitate its own markup and drift.
            content_text = assistant_history[assistant_seen++];
        }
        const std::string separator = first_message ? "" : ",";
        warm_prefix += separator +
            "{\"role\":" + json_quote(role_name) +
            ",\"content\":" + json_quote(content_text) + "}";
        if (role_name == "user" && i == last_user_index) {
            content_text += "\n[" +
                hygiene::turn_anchor(content_text, assistant_history) + "]";
        }
        llama_body += separator +
            "{\"role\":" + json_quote(role_name) +
            ",\"content\":" + json_quote(content_text) + "}";
        first_message = false;
    }
    llama_body += "]}";
    const std::string spec_key = speculation_key(list, last_user_index);

    // A model whose cache cannot roll back pays a FULL prompt pass for every
    // draft revision: successive prewarms and speculations each diverge from
    // the last one's cache and jam the single llama slot for seconds
    // (measured: a mid-conversation turn behind that queue took 4.9 s to
    // first token). Plain sends extend the post-turn canonical warm for a
    // few dozen tokens, so the cheap-refresh triggers are skipped -- EXCEPT
    // the highest-value one: a send-shaped draft (sentence-terminal, the
    // send imminent) is worth one full pass, because a hit turns the whole
    // turn into pre-generated audio. Coalescing (same-key keep, newest
    // wins) bounds the churn to roughly one live speculation at a time.
    if (!active_llm_cache_rollback()) {
        const bool send_shaped = speculate && last_user_index < list.size() &&
            ends_with_sentence_terminal(list[last_user_index].find("content")->as_string());
        if (prewarm || (speculate && !send_shaped)) {
            return json_response("{\"status\":\"skipped\"}");
        }
    }
    if (speculate) {
        // A draft ending like a sentence is the client's highest-confidence
        // tier -- the send is imminent, so the first sentence's audio is
        // worth synthesizing the moment it exists.
        const bool early_tts = last_user_index < list.size() &&
            ends_with_sentence_terminal(list[last_user_index].find("content")->as_string());
        start_chat_speculation(llama_body, spec_key, model, tts_seed, early_tts);
        return json_response("{\"status\":\"speculating\"}");
    }
    if (prewarm) {
        // A prewarm for a DIFFERENT draft means the user typed past the
        // speculated text: that speculation is dead, and letting it finish
        // would only hold the llama slot against this prewarm.
        {
            std::lock_guard<std::mutex> lock(chat_speculation_mutex_);
            if (chat_speculation_ != nullptr && chat_speculation_->key != spec_key) {
                chat_speculation_->abort.store(true);
                chat_speculation_ = nullptr;
            }
        }
        // Coalesced: only the newest draft's prefill is worth running, and
        // never in front of a real generation.
        queue_llm_warm(llama_body);
        return json_response("{\"status\":\"warming\"}");
    }

    {
        // Remember what is being served: a speculate request for this exact
        // conversation arriving from here on lost its race with the send and
        // must not run (see llm_last_send_key_).
        std::lock_guard<std::mutex> lock(chat_speculation_mutex_);
        llm_last_send_key_ = spec_key;
        llm_last_send_at_ = std::chrono::steady_clock::now();
    }
    auto speculation = take_matching_speculation(spec_key);

    LoadedModel * model_ptr = model;
    return sse_response([this, model_ptr, llama_body, warm_prefix, tts_seed,
                         speculation](HttpStreamWriter & writer) {
        chat_orchestrate(*model_ptr, llama_body, warm_prefix, tts_seed, writer, speculation);
    });
}

HttpResponse ServerState::handle_llm_chat(const std::string & body_text) {
    if (config_.llm_port <= 0) {
        return error_response(
            503,
            "no LLM sidecar is configured; start the server with --llm-port",
            "llm_unavailable");
    }
    const auto body = engine::io::json::parse(body_text);
    const auto * messages = body.find("messages");
    if (messages == nullptr || !messages->is_array() || messages->as_array().empty()) {
        return error_response(400, "chat requires a non-empty 'messages' array", "invalid_request_error");
    }
    const auto & list = messages->as_array();
    size_t total_content = 0;
    for (const auto & message : list) {
        const auto * role = message.find("role");
        const auto * content = message.find("content");
        if (role == nullptr || !role->is_string() || content == nullptr || !content->is_string()) {
            return error_response(400, "each message needs string 'role' and 'content'", "invalid_request_error");
        }
        const auto & role_name = role->as_string();
        if (role_name != "system" && role_name != "user" && role_name != "assistant") {
            return error_response(400, "message roles must be system, user, or assistant", "invalid_request_error");
        }
        total_content += content->as_string().size();
    }
    if (total_content > 24000) {
        return error_response(
            400,
            "chat history is too large (" + std::to_string(total_content) +
                " characters; the limit is 24000) -- trim older messages",
            "invalid_request_error");
    }

    CharacterConfig character;
    LlmSettings settings;
    {
        std::lock_guard<std::mutex> lock(character_mutex_);
        character = character_;
        settings = llm_settings_;
    }
    const double temperature = engine::io::json::optional_f32(
        body, "temperature", static_cast<float>(settings.temperature));
    const double top_p = engine::io::json::optional_f32(
        body, "top_p", static_cast<float>(settings.top_p));
    const double repeat_penalty = engine::io::json::optional_f32(
        body, "repeat_penalty", static_cast<float>(settings.repeat_penalty));
    // No length ramp here: that machinery exists to keep first AUDIO snappy.
    // Pure text gets the configured ceiling unless the request says otherwise.
    const int64_t max_tokens = engine::io::json::optional_i64(
        body, "max_tokens", std::max<int64_t>(settings.max_tokens, 256));
    const int64_t llm_seed = engine::io::json::optional_i64(body, "seed", -1);
    // cache_prompt defaults ON (the pipeline's real configuration). Sending
    // false forces a full re-prefill -- the honest way to measure raw
    // prompt-processing speed instead of the cached-prefix fast path.
    const bool cache_prompt = bool_field(body, "cache_prompt", true);

    // The same brain the character runs -- master prompt, persona, samplers,
    // stop strings -- minus everything that exists only to serve the TTS
    // (steering notes, length ramp, sentence segmentation, scrubbing).
    std::string llama_body = std::string("{\"stream\":true,\"cache_prompt\":") +
        (cache_prompt ? "true" : "false");
    llama_body += ",\"stop\":[\"<|im_end|>\",\"<|im_start|>\",\"\\n\\n\"]";
    llama_body += ",\"dry_multiplier\":0.8,\"dry_base\":1.75"
                  ",\"dry_allowed_length\":4,\"dry_penalty_last_n\":4096";
    llama_body += ",\"temperature\":" + std::to_string(temperature);
    llama_body += ",\"top_p\":" + std::to_string(top_p);
    llama_body += ",\"repeat_penalty\":" + std::to_string(repeat_penalty);
    llama_body += ",\"max_tokens\":" + std::to_string(max_tokens);
    if (llm_seed >= 0) {
        llama_body += ",\"seed\":" + std::to_string(llm_seed);
    }
    llama_body += ",\"messages\":[";
    bool first_message = true;
    if (list[0].find("role")->as_string() != "system") {
        const std::string system_prompt =
            render_master_prompt(settings, character.name, character.persona);
        llama_body += "{\"role\":\"system\",\"content\":" + json_quote(system_prompt) + "}";
        first_message = false;
    }
    for (const auto & message : list) {
        llama_body += (first_message ? "" : ",");
        llama_body += "{\"role\":" + json_quote(message.find("role")->as_string()) +
            ",\"content\":" + json_quote(message.find("content")->as_string()) + "}";
        first_message = false;
    }
    llama_body += "]}";

    return sse_response([this, llama_body](HttpStreamWriter & writer) {
        const auto started = Clock::now();
        double first_token_ms = -1.0;
        bool client_gone = false;
        begin_llm_generation();
        llm::ChatResult result;
        try {
            write_sse(writer, "{\"type\":\"start\"}");
            result = llm::stream_chat(
                config_.llm_host, config_.llm_port, llama_body,
                [&](const std::string & delta) {
                    if (first_token_ms < 0) {
                        first_token_ms = elapsed_ms(started);
                    }
                    try {
                        write_sse(writer, "{\"type\":\"token\",\"text\":" + json_quote(delta) + "}");
                    } catch (const std::exception &) {
                        client_gone = true;
                        return false;
                    }
                    return true;
                });
        } catch (const std::exception &) {
            client_gone = true;
        }
        end_llm_generation();
        note_llm_decode_rate(result.predicted_n, result.predicted_ms);
        if (client_gone) {
            return;
        }
        try {
            if (!result.ok && !result.error.empty()) {
                write_sse(writer, "{\"type\":\"error\",\"message\":" + json_quote(result.error) + "}");
            }
            std::ostringstream done;
            done << "{\"type\":\"done\",\"stats\":{"
                 << "\"first_token_ms\":" << first_token_ms
                 << ",\"wall_ms\":" << elapsed_ms(started)
                 << ",\"prompt_n\":" << result.prompt_n
                 << ",\"prompt_ms\":" << result.prompt_ms
                 << ",\"predicted_n\":" << result.predicted_n
                 << ",\"predicted_ms\":" << result.predicted_ms
                 << ",\"finish_reason\":" << json_quote(result.finish_reason)
                 << "}}";
            write_sse(writer, done.str());
            write_sse_done(writer);
        } catch (const std::exception &) {
            // The client left between the last token and the summary.
        }
    });
}

void ServerState::start_chat_speculation(
    const std::string & llama_body, const std::string & spec_key,
    LoadedModel * model, long long tts_seed, bool early_tts) {
    auto spec = std::make_shared<ChatSpeculation>();
    spec->key = spec_key;
    {
        std::lock_guard<std::mutex> lock(chat_speculation_mutex_);
        if (spec_key == llm_last_send_key_ &&
            std::chrono::steady_clock::now() - llm_last_send_at_ < std::chrono::seconds(30)) {
            // The real send for this draft already arrived; this speculate
            // lost the timer-vs-enter race and would only duplicate it.
            return;
        }
        if (chat_speculation_ != nullptr) {
            if (chat_speculation_->key == spec->key) {
                // Same draft already speculating; keep it.
                return;
            }
            chat_speculation_->abort.store(true);
        }
        chat_speculation_ = spec;
    }
    std::thread([this, host = config_.llm_host, port = config_.llm_port, spec,
                 body = llama_body, model, tts_seed, early_tts] {
        // Real users mostly attach EARLY (send lands while this worker is
        // still generating), and with no hesitation audio covering the air
        // the first segment sets time-to-first-sound. 28 characters buys a
        // fuller clause than the miss path's 16 without pushing the first
        // sound noticeably later; a settled hover-hit plays it instantly
        // either way.
        SentenceSegmenter segmenter(SentenceSegmenter::Options{28, 56, 300});
        hygiene::StreamScrubber scrubber;
        // Synthesizes one sentence into the speculation buffer unless another
        // trigger got there first (the audio_state handoff below). The settle
        // delay lets an abort land before a synthesis is burned -- a run
        // cannot be cancelled once started, which bounds a wrong guess's TTS
        // cost at roughly one short sentence.
        auto synthesize_first = [this, spec, model, tts_seed](
                                    std::string sentence, int settle_ms) {
            if (model == nullptr) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
            if (spec->abort.load()) {
                return;
            }
            const auto spoken = strip_speech_markup(sentence);
            if (spoken.empty()) {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(spec->mutex);
                if (spec->audio_state != ChatSpeculation::AudioState::None) {
                    return;
                }
                spec->audio_state = ChatSpeculation::AudioState::Running;
            }
            try {
                Value::Object fields;
                fields.emplace("input", Value::make_string(spoken));
                // Never roll a random seed for the character's speech: short
                // chat sentences are a per-seed quality lottery (muffle,
                // noise, off-voice deliveries), and a different roll per
                // sentence makes her timbre drift within one reply.
                fields.emplace("seed", Value::make_number(
                    static_cast<double>(tts_seed >= 0 ? tts_seed : 777)));
                const auto speech_body = Value::make_object(std::move(fields));
                auto request = build_speech_request(*model, speech_body);
                request.options["stream_accumulate"] = "false";
                run_streaming_model(*model, request, [&](const engine::runtime::StreamEvent & stream_event) {
                    std::vector<engine::runtime::AudioBuffer> buffers;
                    if (stream_event.audio_output.has_value()) {
                        buffers.push_back(*stream_event.audio_output);
                    }
                    for (const auto & named : stream_event.named_audio_outputs) {
                        buffers.push_back(named.audio);
                    }
                    for (const auto & audio : buffers) {
                        auto pcm = encode_pcm16_samples(audio);
                        if (pcm.empty()) {
                            continue;
                        }
                        {
                            std::lock_guard<std::mutex> lock(spec->mutex);
                            spec->first_audio_chunks.push_back(std::move(pcm));
                        }
                        spec->cv.notify_all();
                    }
                });
                {
                    std::lock_guard<std::mutex> lock(spec->mutex);
                    spec->audio_state = ChatSpeculation::AudioState::Done;
                }
            } catch (const std::exception &) {
                std::lock_guard<std::mutex> lock(spec->mutex);
                spec->audio_state = ChatSpeculation::AudioState::Failed;
                spec->first_audio_chunks.clear();
            }
            spec->cv.notify_all();
        };
        bool early_tts_started = false;
        auto emit = [&](const std::string & clean) {
            if (clean.empty()) {
                return;
            }
            auto sentences = segmenter.feed(clean);
            {
                std::lock_guard<std::mutex> lock(spec->mutex);
                spec->text += clean;
                for (auto & sentence : sentences) {
                    spec->sentences.push_back(std::move(sentence));
                }
            }
            spec->cv.notify_all();
            // The frontier of an early attach is TTS time: the moment the
            // first sentence exists, synthesize it while the rest of the
            // reply is still being written, so a send landing mid-generation
            // finds its audio already streaming instead of paying the full
            // synthesis after replay. Send-shaped drafts only.
            if (early_tts && !early_tts_started) {
                std::string first;
                {
                    std::lock_guard<std::mutex> lock(spec->mutex);
                    if (!spec->sentences.empty()) {
                        first = spec->sentences.front();
                    }
                }
                if (!first.empty()) {
                    early_tts_started = true;
                    std::thread(synthesize_first, std::move(first), 400).detach();
                }
            }
        };
        std::unique_lock<std::mutex> stream_lock(llm_spec_stream_mutex_);
        if (spec->abort.load()) {
            return;  // superseded while waiting for the stream slot
        }
        begin_llm_generation();
        const auto result = llm::stream_chat(host, port, body,
            [&](const std::string & delta) {
                emit(scrubber.feed(delta));
                return !spec->abort.load();
            });
        end_llm_generation();
        stream_lock.unlock();
        emit(scrubber.flush());
        std::string first_sentence;
        {
            std::lock_guard<std::mutex> lock(spec->mutex);
            spec->rest = segmenter.flush();
            spec->ok = result.ok;
            spec->error = result.error;
            spec->finish_reason = result.finish_reason;
            spec->prompt_n = result.prompt_n;
            spec->prompt_ms = result.prompt_ms;
            spec->predicted_n = result.predicted_n;
            spec->predicted_ms = result.predicted_ms;
            spec->llm_done = true;
            if (!spec->sentences.empty()) {
                first_sentence = spec->sentences.front();
            } else if (!spec->rest.empty()) {
                first_sentence = spec->rest;
            }
        }
        spec->cv.notify_all();

        // Fallback for replies whose only sentence appeared at flush time
        // (the early trigger above never saw one). The audio_state handoff
        // inside synthesize_first makes double-synthesis impossible.
        if (!result.ok || spec->abort.load()) {
            return;
        }
        synthesize_first(std::move(first_sentence), 300);
    }).detach();
}

std::shared_ptr<ChatSpeculation> ServerState::take_matching_speculation(
    const std::string & key) {
    std::lock_guard<std::mutex> lock(chat_speculation_mutex_);
    if (chat_speculation_ == nullptr) {
        return nullptr;
    }
    auto spec = std::move(chat_speculation_);
    chat_speculation_ = nullptr;
    if (spec->key != key || spec->abort.load()) {
        if (spec->key != key) {
            size_t diff = 0;
            while (diff < spec->key.size() && diff < key.size() &&
                   spec->key[diff] == key[diff]) {
                ++diff;
            }
            std::cerr << "chat speculation key mismatch: spec=" << spec->key.size()
                      << "B real=" << key.size() << "B first_diff=" << diff
                      << " spec[..]=" << spec->key.substr(diff, 60)
                      << " real[..]=" << key.substr(diff, 60) << "\n";
        }
        spec->abort.store(true);
        // An aborted speculation whose worker died mid-stream is useless; a
        // fresh normal request is both simpler and correct.
        return nullptr;
    }
    // A speculation that ran but FAILED (sidecar hiccup) must not silently
    // become an empty reply; fall back to the normal path.
    {
        std::lock_guard<std::mutex> spec_lock(spec->mutex);
        if (spec->llm_done && !spec->ok) {
            return nullptr;
        }
    }
    return spec;
}

void ServerState::chat_orchestrate(
    LoadedModel & model,
    const std::string & llama_body,
    const std::string & warm_body_prefix,
    long long tts_seed,
    HttpStreamWriter & writer,
    std::shared_ptr<ChatSpeculation> speculation) {
    // Producer/consumer: the LLM reader thread turns the sidecar's SSE stream
    // into token and sentence events; this (writer) thread is the only one
    // touching the HTTP stream. Tokens flow out immediately; each completed
    // sentence is synthesized here, its PCM interleaved as audio events, while
    // the LLM keeps generating on the other side -- text runs ahead, audio
    // catches up.
    struct ChatEvent {
        enum class Kind { Token, Sentence, LlmDone } kind = Kind::Token;
        std::string text;
        bool ok = true;
        std::string error;
    };
    std::deque<ChatEvent> queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> abort{false};
    // llama's own per-request timings; written by the reader thread before it
    // pushes LlmDone, read by the writer thread after llm_finished.
    double llm_prompt_n = -1, llm_prompt_ms = -1, llm_predicted_n = -1, llm_predicted_ms = -1;
    auto push = [&](ChatEvent event) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            queue.push_back(std::move(event));
        }
        queue_cv.notify_all();
    };

    std::thread llm_thread([&] {
        if (speculation != nullptr) {
            // Speculative hit: the reply was generated (or is being generated)
            // while the user hovered over send. Replay the buffer and then
            // follow it live; the first token is a memory read.
            size_t text_pos = 0;
            size_t sentence_pos = 0;
            while (!abort.load()) {
                std::string new_text;
                std::vector<std::string> new_sentences;
                std::string rest;
                std::string finish;
                bool finished = false;
                {
                    std::unique_lock<std::mutex> lock(speculation->mutex);
                    speculation->cv.wait_for(lock, std::chrono::milliseconds(250), [&] {
                        return speculation->text.size() > text_pos ||
                               speculation->sentences.size() > sentence_pos ||
                               speculation->llm_done;
                    });
                    if (speculation->text.size() > text_pos) {
                        new_text = speculation->text.substr(text_pos);
                        text_pos = speculation->text.size();
                    }
                    for (size_t i = sentence_pos; i < speculation->sentences.size(); ++i) {
                        new_sentences.push_back(speculation->sentences[i]);
                    }
                    sentence_pos = speculation->sentences.size();
                    if (speculation->llm_done &&
                        text_pos == speculation->text.size() &&
                        sentence_pos == speculation->sentences.size()) {
                        finished = true;
                        rest = speculation->rest;
                        finish = speculation->finish_reason;
                        llm_prompt_n = speculation->prompt_n;
                        llm_prompt_ms = speculation->prompt_ms;
                        llm_predicted_n = speculation->predicted_n;
                        llm_predicted_ms = speculation->predicted_ms;
                    }
                }
                if (!new_text.empty()) {
                    push({ChatEvent::Kind::Token, std::move(new_text)});
                }
                for (auto & sentence : new_sentences) {
                    push({ChatEvent::Kind::Sentence, std::move(sentence)});
                }
                if (finished) {
                    const bool truncated_tail = finish == "length" &&
                        !ends_with_sentence_terminal(strip_speech_markup(rest));
                    if (!rest.empty() && !truncated_tail) {
                        push({ChatEvent::Kind::Sentence, std::move(rest)});
                    }
                    bool ok;
                    std::string error;
                    {
                        std::lock_guard<std::mutex> lock(speculation->mutex);
                        ok = speculation->ok;
                        error = speculation->error;
                    }
                    push({ChatEvent::Kind::LlmDone, finish, ok, error});
                    return;
                }
            }
            // The client went away mid-replay; release the worker too.
            speculation->abort.store(true);
            return;
        }
        // The scrubber cleans the stream before ANYONE sees it: the transcript
        // the client renders, the sentences the TTS speaks, and (next turn)
        // the history the model re-reads all stay free of roleplay markup.
        // The opening clause is cut early: every buffered character is
        // audible silence at decode speed, and only the first segment is
        // compromised -- every later one is a whole sentence. The cut
        // adapts to the active model's decode rate, and the constraint is
        // CONTINUITY, not just latency: the opener's spoken duration must
        // outlast the time the model needs to write what follows it -- and
        // once synthesis starts, a dense model's decode drops further from
        // GPU contention with the TTS (measured 17 -> 10.6 t/s), so short
        // openers drain into an audible gap right after the first words.
        // Tried and rejected for slow decoders: 8-char cuts (half a second
        // of runway, fragment prosody -- the TTS's worst case) and 24-char
        // cuts (still outrun at the contended rate). What works is no cut
        // at all: the first COMPLETE sentence generates at full speed
        // before synthesis exists to contend with, plays ~4 s of runway,
        // and carries natural prosody. ~1 s later to first sound, clean
        // from the first word on.
        // One opener for every model: the 16-char clause cut. Slow decoders
        // used to need a longer opener, but the real thief was synthesis
        // sprinting ahead of playback and starving decode on the shared GPU
        // -- the realtime-aware throttle in the synthesis sink fixes the
        // cause, and the client's paced reserve absorbs what remains.
        SentenceSegmenter segmenter(SentenceSegmenter::Options{16, 28, 300});
        hygiene::StreamScrubber scrubber;
        auto emit = [&](const std::string & clean) {
            if (clean.empty()) {
                return;
            }
            push({ChatEvent::Kind::Token, clean});
            for (auto & sentence : segmenter.feed(clean)) {
                push({ChatEvent::Kind::Sentence, std::move(sentence)});
            }
        };
        begin_llm_generation();
        const auto result = llm::stream_chat(
            config_.llm_host,
            config_.llm_port,
            llama_body,
            [&](const std::string & delta) {
                emit(scrubber.feed(delta));
                return !abort.load();
            });
        end_llm_generation();
        emit(scrubber.flush());
        llm_prompt_n = result.prompt_n;
        llm_prompt_ms = result.prompt_ms;
        llm_predicted_n = result.predicted_n;
        llm_predicted_ms = result.predicted_ms;
        note_llm_decode_rate(result.predicted_n, result.predicted_ms);
        auto rest = segmenter.flush();
        // A reply cut off by the token ceiling ends mid-thought. The dangling
        // fragment still displays as text, but speaking it would stop the
        // character mid-word -- keep it out of the audio unless it happens to
        // end like a sentence anyway.
        const bool truncated_tail = result.finish_reason == "length" &&
            !ends_with_sentence_terminal(strip_speech_markup(rest));
        if (!rest.empty() && !truncated_tail) {
            push({ChatEvent::Kind::Sentence, std::move(rest)});
        }
        push({ChatEvent::Kind::LlmDone, result.finish_reason, result.ok, result.error});
    });

    const auto started = Clock::now();
    auto elapsed = [&] { return elapsed_ms(started); };
    double first_token_ms = -1.0;
    double first_sentence_ms = -1.0;
    double first_audio_ms = -1.0;
    size_t audio_bytes = 0;
    bool llm_finished = false;
    bool llm_ok = true;
    std::string llm_error;

    std::string reply_text;
    bool completed = false;
    auto write_token = [&](const std::string & delta) {
        if (first_token_ms < 0) {
            first_token_ms = elapsed();
        }
        reply_text += delta;
        write_sse(writer, "{\"type\":\"token\",\"text\":" + json_quote(delta) + "}");
    };
    // Called from inside the TTS sink so the transcript keeps streaming while a
    // sentence is being spoken.
    auto drain_tokens = [&] {
        std::vector<ChatEvent> tokens;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            while (!queue.empty() && queue.front().kind == ChatEvent::Kind::Token) {
                tokens.push_back(std::move(queue.front()));
                queue.pop_front();
            }
        }
        for (const auto & token : tokens) {
            write_token(token.text);
        }
    };

    try {
        // A slow decoder cannot keep the TTS fed at full surplus: sentence
        // handoffs arrive with less margin, and dozens of small underruns
        // beat one deeper client reserve. Tell the client to hold ~600 ms
        // of audio before the first sample instead of its default 120.
        {
            const double pace_tps = llm_decode_tps_ema_.load();
            const bool slow_pace = pace_tps > 0.0
                ? pace_tps < 30.0
                : !active_llm_cache_rollback();
            write_sse(writer, slow_pace
                ? "{\"type\":\"start\",\"reserve_ms\":600}"
                : "{\"type\":\"start\"}");
        }
        while (true) {
            ChatEvent event;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [&] { return !queue.empty() || llm_finished; });
                if (queue.empty()) {
                    break;
                }
                event = std::move(queue.front());
                queue.pop_front();
            }
            if (event.kind == ChatEvent::Kind::Token) {
                write_token(event.text);
            } else if (event.kind == ChatEvent::Kind::Sentence) {
                const auto spoken = strip_speech_markup(event.text);
                if (spoken.empty()) {
                    continue;
                }
                const bool use_spec_audio = first_sentence_ms < 0 && speculation != nullptr &&
                    [&] {
                        std::lock_guard<std::mutex> lock(speculation->mutex);
                        if (speculation->audio_state == ChatSpeculation::AudioState::None) {
                            // This thread is about to synthesize the first
                            // sentence itself; claim the slot so the worker's
                            // early-TTS trigger cannot start a duplicate
                            // synthesis behind it and hold the model against
                            // sentence two.
                            speculation->audio_state = ChatSpeculation::AudioState::Failed;
                            return false;
                        }
                        return speculation->audio_state == ChatSpeculation::AudioState::Running ||
                               speculation->audio_state == ChatSpeculation::AudioState::Done;
                    }();
                if (first_sentence_ms < 0) {
                    first_sentence_ms = elapsed();
                }
                write_sse(writer, "{\"type\":\"sentence\",\"text\":" + json_quote(spoken) + "}");
                if (use_spec_audio) {
                    // The first sentence's audio was synthesized while the user
                    // hovered over send: stream the stored chunks (and any that
                    // are still arriving) instead of synthesizing again.
                    size_t chunk_pos = 0;
                    while (true) {
                        std::vector<std::vector<uint8_t>> chunks;
                        bool finished = false;
                        {
                            std::unique_lock<std::mutex> lock(speculation->mutex);
                            speculation->cv.wait_for(lock, std::chrono::milliseconds(250), [&] {
                                return speculation->first_audio_chunks.size() > chunk_pos ||
                                       speculation->audio_state != ChatSpeculation::AudioState::Running;
                            });
                            for (size_t i = chunk_pos; i < speculation->first_audio_chunks.size(); ++i) {
                                chunks.push_back(speculation->first_audio_chunks[i]);
                            }
                            chunk_pos = speculation->first_audio_chunks.size();
                            finished = speculation->audio_state != ChatSpeculation::AudioState::Running &&
                                       chunk_pos == speculation->first_audio_chunks.size();
                            if (speculation->audio_state == ChatSpeculation::AudioState::Failed) {
                                chunks.clear();
                                finished = true;
                            }
                        }
                        for (const auto & pcm : chunks) {
                            drain_tokens();
                            if (first_audio_ms < 0) {
                                first_audio_ms = elapsed();
                            }
                            audio_bytes += pcm.size();
                            write_sse(writer, "{\"type\":\"audio\",\"audio\":" + json_quote(base64_encode(pcm)) + "}");
                        }
                        if (finished) {
                            break;
                        }
                    }
                    bool had_audio;
                    {
                        std::lock_guard<std::mutex> lock(speculation->mutex);
                        had_audio = !speculation->first_audio_chunks.empty() &&
                                    speculation->audio_state == ChatSpeculation::AudioState::Done;
                    }
                    if (had_audio) {
                        continue;
                    }
                    // Synthesis failed or produced nothing; fall through and
                    // synthesize this sentence normally.
                }
                Value::Object fields;
                fields.emplace("input", Value::make_string(spoken));
                // Pinned for the same reason as the speculative synthesis
                // above: random seeds make short sentences a quality lottery.
                fields.emplace("seed", Value::make_number(
                    static_cast<double>(tts_seed >= 0 ? tts_seed : 777)));
                const auto speech_body = Value::make_object(std::move(fields));
                auto request = build_speech_request(model, speech_body);
                request.options["stream_accumulate"] = "false";
                run_streaming_model(model, request, [&](const engine::runtime::StreamEvent & stream_event) {
                    drain_tokens();
                    std::vector<engine::runtime::AudioBuffer> buffers;
                    if (stream_event.audio_output.has_value()) {
                        buffers.push_back(*stream_event.audio_output);
                    }
                    for (const auto & named : stream_event.named_audio_outputs) {
                        buffers.push_back(named.audio);
                    }
                    for (const auto & audio : buffers) {
                        const auto pcm = encode_pcm16_samples(audio);
                        if (pcm.empty()) {
                            continue;
                        }
                        if (first_audio_ms < 0) {
                            first_audio_ms = elapsed();
                        }
                        audio_bytes += pcm.size();
                        write_sse(writer, "{\"type\":\"audio\",\"audio\":" + json_quote(base64_encode(pcm)) + "}");
                    }
                    // Realtime-aware pacing: the speakers play 1 s of audio
                    // per second no matter how fast it is synthesized, and a
                    // dense LLM's decode drops ~40% while synthesis hogs the
                    // GPU (measured 17 -> 10.6 t/s). Once a comfortable lead
                    // over playback exists, napping between codec steps hands
                    // the GPU back to the model that is still WRITING the
                    // reply -- which is what the next sentence is waiting on.
                    if (first_audio_ms >= 0 && !llm_finished) {
                        const double lead_seconds =
                            static_cast<double>(audio_bytes) / 48000.0 -
                            (elapsed() - first_audio_ms) / 1000.0;
                        if (lead_seconds > 2.5) {
                            drain_tokens();
                            std::this_thread::sleep_for(std::chrono::milliseconds(90));
                        }
                    }
                });
            } else {
                llm_finished = true;
                llm_ok = event.ok;
                llm_error = event.error;
                // The reply's text is final here even though audio is still
                // being synthesized; the client uses this to start
                // speculating on the user's next draft while the character
                // is still speaking.
                write_sse(writer, "{\"type\":\"llm_done\"}");
                // Queue the canonical re-prime NOW rather than at stream end:
                // synthesis (throttled to playback pace) can outlast the
                // generation by many seconds, and a no-rollback model needs
                // the warm finished before the next send to stay cheap.
                if (llm_ok && !warm_body_prefix.empty() && !reply_text.empty()) {
                    queue_llm_warm(warm_body_prefix +
                        ",{\"role\":\"assistant\",\"content\":" +
                        json_quote(hygiene::sanitize_assistant_text(reply_text)) + "}]}");
                }
            }
        }
        if (!llm_ok && !llm_error.empty()) {
            write_sse(writer, "{\"type\":\"error\",\"message\":" + json_quote(llm_error) + "}");
        }
        std::ostringstream done;
        done << "{\"type\":\"done\",\"stats\":{"
             << "\"first_token_ms\":" << first_token_ms
             << ",\"first_sentence_ms\":" << first_sentence_ms
             << ",\"first_audio_ms\":" << first_audio_ms
             << ",\"wall_ms\":" << elapsed()
             << ",\"audio_seconds\":" << (static_cast<double>(audio_bytes) / 48000.0)
             // Per-stage telemetry: how much prompt llama actually reprocessed
             // this turn and what each stage cost. The waterfall that guides
             // every latency decision.
             << ",\"llm_prompt_n\":" << llm_prompt_n
             << ",\"llm_prompt_ms\":" << llm_prompt_ms
             << ",\"llm_predicted_n\":" << llm_predicted_n
             << ",\"llm_predicted_ms\":" << llm_predicted_ms
             << ",\"speculative_hit\":" << (speculation != nullptr ? "true" : "false")
             << "}}";
        write_sse(writer, done.str());
        write_sse_done(writer);
        completed = true;
    } catch (...) {
        // The client went away or synthesis failed mid-stream; stop the LLM
        // and unwind. The connection is already unusable, so there is nothing
        // to report to.
        abort.store(true);
    }
    abort.store(true);
    llm_thread.join();

    // The canonical re-prime was queued at llm_done (see above); nothing
    // left to do here. An aborted stream skipped it naturally: the abort
    // fires before llm_done is processed, or the coalescer replaces the
    // stale body on the next real activity.
}

// ============================== Live voice (STT) ==============================

namespace {

std::unordered_map<std::string, std::string> parse_query_params(const std::string & query) {
    std::unordered_map<std::string, std::string> out;
    size_t start = 0;
    while (start < query.size()) {
        size_t end = query.find('&', start);
        if (end == std::string::npos) {
            end = query.size();
        }
        const auto pair = query.substr(start, end - start);
        start = end + 1;
        const auto eq = pair.find('=');
        if (eq != std::string::npos) {
            out[pair.substr(0, eq)] = pair.substr(eq + 1);
        }
    }
    return out;
}

// The bundled Silero VAD ships under assets/; the server may be launched
// from the repo root or with paths resolved against the request base.
std::filesystem::path voice_asset_root(const std::filesystem::path & request_base) {
    const auto bundled = std::filesystem::path("assets") / "framework" / "models" / "silero_vad";
    std::error_code ec;
    if (std::filesystem::exists(request_base / bundled, ec)) {
        return request_base;
    }
    return std::filesystem::current_path();
}

}  // namespace

ServerState::LoadedModel * ServerState::find_asr_model() {
    for (auto & model : models_) {
        if (model->config.task == "asr") {
            std::lock_guard<std::mutex> lock(models_mutex_);
            ensure_model_loaded_locked(*model);
            return model.get();
        }
    }
    return nullptr;
}

voice::TranscribeResult ServerState::run_voice_transcribe(
    LoadedModel & model,
    const engine::runtime::AudioBuffer & audio,
    const std::string & language) {
    engine::runtime::TaskRequest request;
    request.audio_input = audio;
    request.options["audio_chunk_mode"] = "none";
    if (!language.empty()) {
        request.options["language"] = language;
        request.text_input = engine::runtime::Transcript{"", language};
    }
    auto timed = run_model(model, request);
    voice::TranscribeResult out;
    if (timed.result.text_output.has_value()) {
        out.text = timed.result.text_output->text;
        out.language = timed.result.text_output->language;
    }
    return out;
}

HttpResponse ServerState::handle_voice_live(const HttpRequest & request) {
    auto * asr = find_asr_model();
    if (asr == nullptr) {
        return error_response(
            503, "no ASR model is configured; add a model with task \"asr\"", "voice_unavailable");
    }
    if (request.body_stream == nullptr) {
        return error_response(
            400,
            "voice live requires a raw PCM body sent with Transfer-Encoding: chunked",
            "invalid_request_error");
    }
    const auto q = parse_query_params(request.query);
    const int sample_rate = q.count("sample_rate") != 0 ? std::atoi(q.at("sample_rate").c_str()) : 16000;
    const int channels = q.count("channels") != 0 ? std::atoi(q.at("channels").c_str()) : 1;
    const std::string sample_format = q.count("sample_format") != 0 ? q.at("sample_format") : "s16le";
    if (sample_rate != 16000 || channels != 1) {
        return error_response(
            400, "voice live requires sample_rate=16000 and channels=1", "invalid_request_error");
    }
    const auto params = voice::turn_params_from_query(request.query);
    auto * body_stream = request.body_stream;
    const auto asset_root = voice_asset_root(request_base_);
    return sse_response([this, asr, params, body_stream, sample_format, asset_root](HttpStreamWriter & writer) {
        std::mutex emit_mutex;
        auto aborted = std::make_shared<std::atomic<bool>>(false);
        auto emit = [&writer, &emit_mutex, aborted](const std::string & json) {
            if (aborted->load()) {
                return;
            }
            std::lock_guard<std::mutex> lock(emit_mutex);
            try {
                write_sse(writer, json);
            } catch (...) {
                aborted->store(true);
            }
        };
        voice::VoiceHooks hooks;
        hooks.emit = emit;
        hooks.aborted = [aborted] { return aborted->load(); };
        hooks.transcribe = [this, asr, language = params.language](const engine::runtime::AudioBuffer & audio) {
            return run_voice_transcribe(*asr, audio, language);
        };
        try {
            auto vad = voice::make_vad_session(asset_root, params);
            const auto stream = minitts::app::make_pcm_chunk_stream(
                *body_stream,
                minitts::app::AudioStreamFormat{16000, 1},
                minitts::app::parse_pcm_sample_format(sample_format));
            voice::VoiceLiveRunner runner(params, std::move(hooks), std::move(vad));
            emit("{\"type\":\"session\"}");
            runner.run(stream);
        } catch (const std::exception & ex) {
            emit(std::string("{\"type\":\"error\",\"message\":") + json_quote(ex.what()) + "}");
        }
        if (!aborted->load()) {
            std::lock_guard<std::mutex> lock(emit_mutex);
            try {
                write_sse_done(writer);
            } catch (...) {
            }
        }
    });
}

// One browser voice session: the mic tab cannot stream a request body over
// HTTP/1.1, so audio arrives as small sequential POSTs into this queue while
// a long-lived runner consumes it, and events leave over a separate SSE
// connection. The runner thread is always joined (stop, events-reader loss,
// or server shutdown), never detached.
struct ServerState::VoiceBrowserSession {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<float> audio;
    bool closed = false;
    std::deque<std::string> events;
    bool done = false;
    std::thread thread;
    std::chrono::steady_clock::time_point created = std::chrono::steady_clock::now();
};

void ServerState::close_voice_session(VoiceBrowserSession & session) {
    {
        std::lock_guard<std::mutex> lock(session.mutex);
        session.closed = true;
    }
    session.cv.notify_all();
    if (session.thread.joinable()) {
        session.thread.join();
    }
}

HttpResponse ServerState::handle_voice_session_create(const HttpRequest & request) {
    auto * asr = find_asr_model();
    if (asr == nullptr) {
        return error_response(
            503, "no ASR model is configured; add a model with task \"asr\"", "voice_unavailable");
    }
    const auto params = voice::turn_params_from_query(request.query);
    const auto asset_root = voice_asset_root(request_base_);
    auto session = std::make_shared<VoiceBrowserSession>();
    std::string id;
    {
        std::lock_guard<std::mutex> lock(voice_sessions_mutex_);
        // Sweep sessions that finished and were drained (or abandoned).
        for (auto it = voice_sessions_.begin(); it != voice_sessions_.end();) {
            bool expired = false;
            {
                std::lock_guard<std::mutex> slock(it->second->mutex);
                const auto age = std::chrono::steady_clock::now() - it->second->created;
                expired = (it->second->done && it->second->events.empty()) ||
                          age > std::chrono::minutes(30);
            }
            if (expired) {
                close_voice_session(*it->second);
                it = voice_sessions_.erase(it);
            } else {
                ++it;
            }
        }
        id = "v" + std::to_string(++voice_session_counter_);
        voice_sessions_[id] = session;
    }

    voice::VoiceHooks hooks;
    hooks.emit = [session](const std::string & json) {
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            session->events.push_back(json);
            if (session->events.size() > 512) {
                session->events.pop_front();
            }
        }
        session->cv.notify_all();
    };
    hooks.aborted = [session] {
        std::lock_guard<std::mutex> lock(session->mutex);
        return session->closed && session->audio.empty();
    };
    hooks.transcribe = [this, asr, language = params.language](const engine::runtime::AudioBuffer & audio) {
        return run_voice_transcribe(*asr, audio, language);
    };

    session->thread = std::thread([session, params, hooks, asset_root] {
        try {
            auto vad = voice::make_vad_session(asset_root, params);
            minitts::app::AudioChunkStream stream;
            stream.format = minitts::app::AudioStreamFormat{16000, 1};
            stream.read = [session](int64_t max_samples, std::vector<float> & out) {
                std::unique_lock<std::mutex> lock(session->mutex);
                session->cv.wait(lock, [&] { return session->closed || !session->audio.empty(); });
                if (session->audio.empty()) {
                    return false;
                }
                const size_t take = std::min<size_t>(
                    static_cast<size_t>(std::max<int64_t>(1, max_samples)), session->audio.size());
                out.assign(session->audio.begin(), session->audio.begin() + static_cast<std::ptrdiff_t>(take));
                session->audio.erase(session->audio.begin(), session->audio.begin() + static_cast<std::ptrdiff_t>(take));
                return true;
            };
            voice::VoiceLiveRunner runner(params, hooks, std::move(vad));
            runner.run(stream);
        } catch (const std::exception & ex) {
            hooks.emit(std::string("{\"type\":\"error\",\"message\":") + json_quote(ex.what()) + "}");
        }
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            session->done = true;
        }
        session->cv.notify_all();
    });
    return json_response("{\"id\":\"" + id + "\"}");
}

HttpResponse ServerState::handle_voice_session_audio(const HttpRequest & request) {
    const auto q = parse_query_params(request.query);
    const auto id = q.count("id") != 0 ? q.at("id") : "";
    std::shared_ptr<VoiceBrowserSession> session;
    {
        std::lock_guard<std::mutex> lock(voice_sessions_mutex_);
        const auto it = voice_sessions_.find(id);
        if (it != voice_sessions_.end()) {
            session = it->second;
        }
    }
    if (session == nullptr) {
        return error_response(404, "unknown voice session: " + id, "not_found");
    }
    const auto & body = request.body;  // raw s16le mono 16 kHz
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->closed) {
            return error_response(409, "voice session is closed", "invalid_request_error");
        }
        for (size_t i = 0; i + 1 < body.size(); i += 2) {
            const int16_t sample = static_cast<int16_t>(
                static_cast<uint8_t>(body[i]) | (static_cast<uint8_t>(body[i + 1]) << 8));
            session->audio.push_back(static_cast<float>(sample) / 32768.0F);
        }
        // A stalled runner must not let a chatty client grow this unbounded:
        // cap at ~30 s of queued audio.
        while (session->audio.size() > 16000u * 30u) {
            session->audio.pop_front();
        }
    }
    session->cv.notify_all();
    return json_response("{\"ok\":true}");
}

HttpResponse ServerState::handle_voice_session_events(const HttpRequest & request) {
    const auto q = parse_query_params(request.query);
    const auto id = q.count("id") != 0 ? q.at("id") : "";
    std::shared_ptr<VoiceBrowserSession> session;
    {
        std::lock_guard<std::mutex> lock(voice_sessions_mutex_);
        const auto it = voice_sessions_.find(id);
        if (it != voice_sessions_.end()) {
            session = it->second;
        }
    }
    if (session == nullptr) {
        return error_response(404, "unknown voice session: " + id, "not_found");
    }
    return sse_response([session](HttpStreamWriter & writer) {
        try {
            while (true) {
                std::vector<std::string> batch;
                bool finished = false;
                {
                    std::unique_lock<std::mutex> lock(session->mutex);
                    session->cv.wait_for(lock, std::chrono::seconds(15), [&] {
                        return session->done || !session->events.empty();
                    });
                    while (!session->events.empty()) {
                        batch.push_back(std::move(session->events.front()));
                        session->events.pop_front();
                    }
                    finished = session->done && session->events.empty();
                }
                for (const auto & event : batch) {
                    write_sse(writer, event);
                }
                if (batch.empty() && !finished) {
                    writer.write(": ping\n\n");
                }
                if (finished) {
                    break;
                }
            }
            write_sse_done(writer);
        } catch (...) {
            // The browser went away; release the runner so the session can
            // be swept.
            {
                std::lock_guard<std::mutex> lock(session->mutex);
                session->closed = true;
            }
            session->cv.notify_all();
        }
    });
}

HttpResponse ServerState::handle_voice_session_stop(const HttpRequest & request) {
    const auto q = parse_query_params(request.query);
    const auto id = q.count("id") != 0 ? q.at("id") : "";
    std::shared_ptr<VoiceBrowserSession> session;
    {
        std::lock_guard<std::mutex> lock(voice_sessions_mutex_);
        const auto it = voice_sessions_.find(id);
        if (it != voice_sessions_.end()) {
            session = it->second;
        }
    }
    if (session == nullptr) {
        return error_response(404, "unknown voice session: " + id, "not_found");
    }
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->closed = true;
    }
    session->cv.notify_all();
    return json_response("{\"ok\":true}");
}

// Cached-voice discovery for the "voice"/cached_voice_id request field. Families that
// support voice presets (e.g. pocket_tts, see assets.cpp: model_root/embeddings/<id>.safetensors)
// keep them under an "embeddings" directory next to the model weights; other families simply
// have no such directory and report no voices. Used by clients (llama-swap's playground, and
// potentially Open WebUI) that call GET /v1/audio/voices?model=<id> to populate a voice picker
// instead of guessing generic names like "alloy"/"nova".
std::string ServerState::models_json() const {
    std::lock_guard<std::mutex> state_lock(models_mutex_);
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < models_.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        const auto & model = *models_[i];
        std::shared_lock<std::shared_mutex> metadata_lock(model.metadata_mutex);
        out << "{\"id\":" << json_quote(model.config.id)
            << ",\"object\":\"model\""
            << ",\"owned_by\":\"engine\""
            << ",\"family\":" << json_quote(model.config.family)
            << ",\"task\":" << json_quote(engine::runtime::to_string(model.task.task))
            << ",\"mode\":" << json_quote(engine::runtime::to_string(model.task.mode))
            << ",\"loaded\":" << (model.loaded.load() ? "true" : "false")
            << ",\"path\":" << json_quote(model.config.path.string())
            << "}";
    }
    out << "]}";
    return out.str();
}

std::string ServerState::get_allowed_origin(const HttpRequest & request) const {
    // TODO: Handle lists of specific origins.
    if (config_.cors_origins == "*") {
        if (const auto it = request.headers.find("origin"); it != request.headers.end()) {
            return it->second;
        }
    }
    return "";
}

void ServerState::LoadedModel::unload() {
	offline = nullptr;
    streaming = nullptr;
    session.reset();
    model.reset();
}

}  // namespace minitts::server
