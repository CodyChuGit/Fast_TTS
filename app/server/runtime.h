#pragma once

#include "busy_guard.h"
#include "character.h"
#include "config.h"
#include "http.h"
#include "llm_manager.h"
#include "mcp.h"
#include "voice.h"
#include "voice_identity.h"

#include "../streaming/streaming.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace minitts::server {

// A reply generated ahead of the user pressing send. While they pause over a
// finished-looking draft, the whole generation runs in the background; if the
// draft arrives unchanged, the real request attaches to this buffer and the
// first token is a memory read. Filled by a detached worker under `mutex`,
// consumed by chat_orchestrate; `abort` tears the worker down when the draft
// changes. No thread handle lives here -- workers are detached and hold their
// own shared_ptr, so whoever drops the last reference frees the state.
struct ChatSpeculation {
    // The "messages":[...] portion of the llama body; prewarm, speculative,
    // and real requests for the same draft share it byte-for-byte.
    std::string key;
    std::atomic<bool> abort{false};
    std::mutex mutex;
    std::condition_variable cv;
    std::string text;                    // scrubbed text so far
    std::vector<std::string> sentences;  // completed speech segments
    std::string rest;                    // segmenter tail at end of stream
    bool llm_done = false;
    bool ok = false;
    std::string error;
    std::string finish_reason;
    double prompt_n = -1, prompt_ms = -1, predicted_n = -1, predicted_ms = -1;
    // The first sentence's audio, synthesized speculatively once the reply
    // has settled: raw PCM chunks in emission order. On a hit these stream to
    // the client instantly, collapsing first audio to the client's buffer.
    enum class AudioState { None, Running, Done, Failed };
    AudioState audio_state = AudioState::None;
    std::vector<std::vector<uint8_t>> first_audio_chunks;
};


class ServerState final : public IHttpHandler {
public:
    ServerState(
        ServerConfig config,
        std::filesystem::path request_base,
        std::filesystem::path ui_resource_anchor = {});
    ~ServerState() override;

    HttpResponse handle(const HttpRequest & request) override;

    // Server-level `live_ingest` policy with this request's model override applied.
    // Deliberately does not reject an unknown or non-streaming model: it runs before
    // the handler, and rejecting here would turn a client's mistake into a dropped
    // connection instead of the 400 handle_transcription_live already produces.
    LiveIngestLimits live_ingest_limits(const HttpRequest & request) const override;

private:
    struct LoadedModel {
        struct RuntimeVoicePreset {
            std::optional<std::string> voice_id;
            std::optional<engine::runtime::AudioBuffer> audio;
            std::optional<std::string> reference_text;
        };

        ServerModelConfig config;
        engine::runtime::TaskSpec task;
        std::unique_ptr<engine::runtime::ILoadedVoiceModel> model;
        std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
        engine::runtime::IOfflineVoiceTaskSession * offline = nullptr;
        engine::runtime::IStreamingVoiceTaskSession * streaming = nullptr;
        std::atomic<bool> loaded{false};
        mutable std::shared_mutex metadata_mutex;
        std::unordered_map<std::string, RuntimeVoicePreset> voice_presets;
        std::optional<RuntimeVoicePreset> default_voice_preset;
        // Serializes runs on this model and bounds how long a caller waits for its
        // turn; see BusyGuard.
        BusyGuard busy;

        // Release the loaded model and session from memory (frees VRAM on GPU backends).
        // The next request will trigger a reload via ensure_model_loaded_locked().
        void unload();
    };

    // Acquire the model's run guard. `request_timeout_ms` is the caller-supplied
    // override, clamped by this model's configured ceiling. Throws ServerBusyError
    // (-> HTTP 503) once the effective timeout has elapsed.
    BusyGuard::Lock acquire_model_run(LoadedModel & model, std::optional<int> request_timeout_ms);

    // Server policy for this model: its own busy_timeout_ms if set, else the
    // top-level config value.
    engine::runtime::RunMode model_run_mode(const LoadedModel & model) const;

    void load_models();
    std::unique_ptr<LoadedModel> make_model(ServerModelConfig config);
    HttpResponse handle_ui_asset() const;
    LoadedModel::RuntimeVoicePreset load_runtime_voice_preset(const ServerModelConfig::VoicePreset & preset) const;
    void load_voice_presets(LoadedModel & model) const;
    void ensure_model_loaded_locked(LoadedModel & model);
    LoadedModel & require_model(const engine::io::json::Value & body);
    const LoadedModel::RuntimeVoicePreset * select_voice_preset(
        const LoadedModel & model,
        const engine::io::json::Value & body,
        bool & voice_field_is_preset) const;
    engine::runtime::TaskRequest build_speech_request(
        const LoadedModel & model,
        const engine::io::json::Value & body) const;
    engine::runtime::TaskRequest apply_default_request_options(
        const LoadedModel & model,
        engine::runtime::TaskRequest request) const;
    struct TimedTaskResult;
    // `busy_timeout_ms` on each of these is the per-request override parsed from the
    // request body; nullopt means "use the model's configured ceiling".
    TimedTaskResult run_model(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        std::optional<int> busy_timeout_ms = std::nullopt);
    TimedTaskResult run_streaming_model(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        const std::function<void(const engine::runtime::StreamEvent &)> & event_sink = {},
        std::optional<int> busy_timeout_ms = std::nullopt);
    // Shared body of the two entry points above/below; `audio` selects the source.
    TimedTaskResult run_streaming_model_impl(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        const minitts::app::AudioChunkStream * audio,
        const std::function<void(const engine::runtime::StreamEvent &)> & event_sink,
        std::optional<int> busy_timeout_ms);
    HttpResponse handle_speech(const std::string & body_text);
    HttpResponse handle_speech_stream(
        LoadedModel & model,
        engine::runtime::TaskRequest request,
        const engine::io::json::Value & body);
    // The active character: the display name plus default voice used whenever a
    // request names no voice, shared by the WebUI and MCP callers.
    HttpResponse handle_character_get();
    HttpResponse handle_character_set(const HttpRequest & request);
    // The saved-character library: every save lands here under an id derived
    // from the name, so any saved character can be re-activated in one click.
    HttpResponse handle_characters_list();
    // The active character's reference recording, for auditioning what the
    // clone is conditioned on.
    HttpResponse handle_character_voice();
    HttpResponse handle_character_activate(const std::string & body_text);
    HttpResponse handle_character_delete(const std::string & body_text);
    // Copies the active character into library/<slug>, recording included.
    void store_character_in_library(const CharacterConfig & character);
    // Resolves the stored character into a runtime voice preset and installs it
    // as the speech model's default under its metadata lock.
    void apply_character(LoadedModel & model, const CharacterConfig & character);
    // The single model MCP speak and the character apply to: the first
    // configured TTS model.
    LoadedModel * find_speech_model();
    // require_model, except a body naming no model resolves to the unambiguous
    // configured (TTS) model -- the Speak page and MCP callers do not know
    // model ids.
    LoadedModel & require_speech_model(const engine::io::json::Value & body);

    // The roleplay master prompt and sampling, persisted beside the character.
    HttpResponse handle_llm_settings_get();
    HttpResponse handle_llm_settings_set(const std::string & body_text);
    std::string llm_settings_json() const;

    // The registered chat model matching `id`, or null. Registry is immutable
    // after construction, so no lock is needed.
    const LlmModelSpec * find_llm_spec(const std::string & id) const;
    // True unless the ACTIVE sidecar model is flagged cache_rollback=false.
    bool active_llm_cache_rollback() const;
    // The active character's name, for spoken-name normalization.
    std::string character_name_for_speech() const;
    // What the active character is supposed to sound like, summarised from
    // their reference recording and cached until the character changes.
    const VoiceProfile & reference_voice_profile() const;
    // Stops the running sidecar, starts `spec`, and blocks until its /health
    // reports the model loaded. On failure the previous model is restarted.
    // Serialized by llm_switch_mutex_.
    void switch_llm_model(const LlmModelSpec & spec);

    // MCP endpoint: JSON-RPC over streamable HTTP with one speak tool.
    HttpResponse handle_mcp(const HttpRequest & request);

    // Chat with the character: streams the llama.cpp sidecar's reply as token
    // events while completed sentences are synthesized and interleaved as
    // audio events on the same SSE stream. The LLM keeps generating while the
    // TTS model speaks, so text runs ahead and audio catches up.
    HttpResponse handle_chat_speak(const std::string & body_text);
    // Text-only chat straight through the sidecar: no TTS, no speculation, no
    // prewarming -- the benchmark surface for the LLM leg by itself. The done
    // event carries llama's own prefill/decode telemetry.
    HttpResponse handle_llm_chat(const std::string & body_text);
    // Typing-episode collector: the client records raw keystroke timing and
    // draft evolution while the user types, and posts finished episodes here.
    // Episodes land in data/typing_episodes.jsonl as the training corpus for
    // the send predictor. Raw events, not features: features are computed
    // offline, so the collector never needs re-instrumenting.
    HttpResponse handle_typing_telemetry(const std::string & body_text);
    // `warm_body_prefix`, when non-empty, is the canonical next-turn request
    // up to (not including) the reply message: once the reply is complete it
    // is appended and the whole thing sent to the sidecar as a one-token
    // priming request, so the next turn's prefill covers only the user's new
    // message. Without it the per-turn steering note (which the client never
    // stores) makes the prompt cache diverge a full turn early.
    void chat_orchestrate(
        LoadedModel & model,
        const std::string & llama_body,
        const std::string & warm_body_prefix,
        long long tts_seed,
        HttpStreamWriter & writer,
        std::shared_ptr<ChatSpeculation> speculation = nullptr);
    // Launches a detached worker generating the reply for `llama_body` into a
    // ChatSpeculation buffer, replacing (and aborting) any previous one. Once
    // the reply settles, the worker also synthesizes the first sentence's
    // audio on `model`. `spec_key` is the punctuation-tolerant conversation
    // key the real send will be matched against. With `early_tts`, the first
    // sentence synthesizes the moment it exists (draft looked send-shaped,
    // so the attach is near-certain); otherwise synthesis waits for the
    // settled reply -- a synthesis cannot be cancelled, and burning one per
    // mid-thought speculation measurably starves the real send's TTS.
    void start_chat_speculation(
        const std::string & llama_body, const std::string & spec_key,
        LoadedModel * model, long long tts_seed, bool early_tts);
    // Hands over the current speculation when its key matches; otherwise
    // aborts it. Null when there is nothing to attach to.
    std::shared_ptr<ChatSpeculation> take_matching_speculation(const std::string & key);
    // Primes the sidecar's prompt cache with the active character's rendered
    // system prompt on a background thread, so the first message of a fresh
    // conversation pays only its own prefill. Called at startup and whenever
    // the system prompt changes (character or settings edits, model switch).
    void warm_llm_system_prompt();
    // Every fire-and-forget prefill (typing prewarm, post-reply rewarm,
    // system-prompt warm) funnels through here: only the NEWEST pending body
    // is kept, at most one is in flight, and none dispatch while a real
    // generation (send or speculation) needs the sidecar's single slot.
    // Real-user testing showed a burst of mid-typing prewarms otherwise
    // queues seconds of prefill in front of the actual send.
    void queue_llm_warm(std::string body);
    void begin_llm_generation();
    void end_llm_generation();
    // Feed one finished generation's telemetry into llm_decode_tps_ema_.
    void note_llm_decode_rate(double predicted_n, double predicted_ms);
    std::string fresh_llm_warm_body() const;
    mcp::SpeakOutcome run_mcp_speak(const std::string & text, long long seed);

    // Live voice (streaming STT). Two transports over one core:
    // /v1/voice/live streams chunked PCM in and SSE events out on a single
    // connection (ffmpeg, CLI); the /v1/voice/sessions endpoints carry the
    // same events for browsers, which cannot stream a request body over
    // HTTP/1.1 -- audio arrives as small sequential POSTs instead.
    HttpResponse handle_voice_live(const HttpRequest & request);
    HttpResponse handle_voice_session_create(const HttpRequest & request);
    HttpResponse handle_voice_session_audio(const HttpRequest & request);
    HttpResponse handle_voice_session_events(const HttpRequest & request);
    HttpResponse handle_voice_session_stop(const HttpRequest & request);
    // The configured ASR model (task "asr"), loaded; null when voice is not
    // configured.
    LoadedModel * find_asr_model();
    voice::TranscribeResult run_voice_transcribe(
        LoadedModel & model,
        const engine::runtime::AudioBuffer & audio,
        const std::string & language);
    struct VoiceBrowserSession;
    // Marks the session closed, wakes its runner, and joins its thread.
    void close_voice_session(VoiceBrowserSession & session);

    std::string models_json() const;
    std::string get_allowed_origin(const HttpRequest & request) const;

    ServerConfig config_;
    std::filesystem::path request_base_;
    std::vector<std::unique_ptr<LoadedModel>> models_;
    std::unordered_map<std::string, size_t> model_index_;
    mutable std::mutex models_mutex_;
    // Guards character_; the voice preset it resolves to lives on the model and
    // is guarded by that model's metadata_mutex.
    mutable std::mutex character_mutex_;
    CharacterConfig character_;
    LlmSettings llm_settings_;
    std::filesystem::path character_dir_;
    // Owns the llama.cpp sidecar when the config carries a model registry;
    // null when the launcher (or nobody) runs the LLM. Model switches are
    // serialized by llm_switch_mutex_ -- a switch stops and restarts the
    // sidecar, and two interleaved switches would race the port.
    std::unique_ptr<LlmManager> llm_manager_;
    std::mutex llm_switch_mutex_;
    std::shared_ptr<ChatSpeculation> chat_speculation_;
    std::mutex chat_speculation_mutex_;
    // The conversation key of the most recent real send. A speculate request
    // landing at or after its own send (the client's timer racing the enter
    // key) must be dropped, or it becomes a zombie that generates the same
    // reply again and steals the sidecar and TTS from the next turn.
    // Guarded by chat_speculation_mutex_.
    std::string llm_last_send_key_;
    std::chrono::steady_clock::time_point llm_last_send_at_{};
    // Prewarm coalescing state; see queue_llm_warm.
    std::mutex llm_warm_mutex_;
    std::condition_variable llm_warm_cv_;
    std::string llm_warm_pending_;
    bool llm_warm_worker_running_ = false;
    int llm_generations_active_ = 0;
    // Serializes appends to data/typing_episodes.jsonl.
    std::mutex typing_log_mutex_;
    mutable std::mutex reference_voice_mutex_;
    mutable std::string reference_voice_key_;
    mutable VoiceProfile reference_voice_profile_;
    // Rolling decode rate of the active sidecar model (tokens/sec), fed by
    // every finished generation. The opener segmenter reads it: a slow
    // decoder gets a shorter first cut so time-to-first-audio stays bounded
    // instead of scaling with the model's writing speed. Zero = unmeasured.
    std::atomic<double> llm_decode_tps_ema_{0.0};
    // One speculative stream at a time, newest wins. A superseded worker
    // that has not started streaming yet sees its abort flag while waiting
    // here and never reaches the sidecar; without this, a burst of
    // mid-typing speculations queues whole prefills in front of the real
    // send (measured: first token past 8 s for stall-heavy typists).
    std::mutex llm_spec_stream_mutex_;
    // Browser voice sessions; see handle_voice_session_create.
    std::unordered_map<std::string, std::shared_ptr<VoiceBrowserSession>> voice_sessions_;
    std::mutex voice_sessions_mutex_;
    uint64_t voice_session_counter_ = 0;
};

}  // namespace minitts::server
