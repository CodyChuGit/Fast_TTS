#include "config.h"
#include "http.h"
#include "runtime.h"

#ifdef AUDIOCPP_SERVER_HAS_CUDA_KEEPALIVE
#include "cuda_keepalive.h"
#endif

#include "engine/framework/debug/trace.h"

#include <csignal>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

volatile std::sig_atomic_t g_shutdown_requested = 0;

void request_shutdown(int) {
    g_shutdown_requested = 1;
}

bool shutdown_requested() {
    return g_shutdown_requested != 0;
}

std::optional<std::string> arg_value(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

bool has_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

std::filesystem::path executable_directory(const char * argv0) {
    if (argv0 == nullptr || *argv0 == '\0') {
        return std::filesystem::current_path();
    }
    std::error_code ec;
    auto path = std::filesystem::absolute(std::filesystem::path(argv0), ec);
    if (ec) {
        return std::filesystem::current_path();
    }
    path = path.lexically_normal();
    if (std::filesystem::is_regular_file(path, ec)) {
        return path.parent_path();
    }
    return std::filesystem::current_path();
}

void print_help() {
    std::cout
        << "audiocpp_server --config <server.json> [--host <ip>] [--port <port>]\n"
        << "                [--llm-host <ip>] [--llm-port <port>] [--character-dir <dir>]\n"
        << "                [--voice-dir <dir>] [--cuda-keepalive-ms <ms>] [--busy-timeout-ms <ms>]\n"
        << "                [--log] [--log-file <path>] [--cors-origins <origins>]\n"
        << "\n"
        << "The Super Fast TTS character-voice server: Qwen3-TTS speech plus a\n"
        << "llama.cpp chat sidecar, one character, streamed end to end.\n"
        << "\n"
        << "Endpoints:\n"
        << "  GET  /                     the app (Speak, Chat, Settings)\n"
        << "  GET  /health               liveness, backend, LLM sidecar state\n"
        << "  GET  /v1/models            configured model state\n"
        << "  POST /v1/audio/speech      OpenAI-compatible TTS; no voice field -> the character\n"
        << "  POST /v1/chat/speak        SSE: LLM tokens + interleaved character audio\n"
        << "  GET  /v1/character         the active character\n"
        << "  POST /v1/character         replace it: JSON {name, preset, persona} or\n"
        << "                             multipart name/transcript/persona/file\n"
        << "  GET  /v1/character/voice   the active custom recording\n"
        << "  GET  /v1/characters        the saved-character library\n"
        << "  POST /v1/characters/activate | /v1/characters/delete   {id}\n"
        << "  POST /mcp                  Model Context Protocol (streamable HTTP), speak tool\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (has_arg(argc, argv, "--help") || has_arg(argc, argv, "-h")) {
            print_help();
            return 0;
        }
        const auto config_path = arg_value(argc, argv, "--config");
        const bool ui_requested = has_arg(argc, argv, "--ui");
        if (!config_path.has_value() && !ui_requested) {
            throw std::runtime_error("missing required --config argument (or use --ui for the native WebUI)");
        }
        const auto log_file = arg_value(argc, argv, "--log-file");
        engine::debug::configure_logging(engine::debug::LoggingConfig{
            has_arg(argc, argv, "--log") || log_file.has_value(),
            log_file,
        });
        std::signal(SIGINT, request_shutdown);
        std::signal(SIGTERM, request_shutdown);
#ifdef SIGPIPE
        // Writing to a socket whose peer has already disconnected (for example a
        // client that closed an SSE/chunked stream early) would otherwise deliver
        // SIGPIPE and terminate the whole server. Ignore it so the failed send
        // surfaces as an EPIPE error on that single request thread, which
        // handle_client already unwinds cleanly, instead of taking the process down.
        std::signal(SIGPIPE, SIG_IGN);
#endif

        auto config = config_path.has_value()
            ? minitts::server::load_server_config(*config_path)
            : minitts::server::ServerConfig{};
        if (!config_path.has_value()) {
            config.ui_management = true;
            config.lazy_load = true;
        }
        if (ui_requested) {
            config.ui_enabled = true;
        }
        if (has_arg(argc, argv, "--no-ui")) {
            config.ui_enabled = false;
        }
        if (has_arg(argc, argv, "--ui-management")) {
            config.ui_management = true;
        }
        if (const auto host = arg_value(argc, argv, "--host")) {
            config.host = *host;
        }
        if (const auto port = arg_value(argc, argv, "--port")) {
            config.port = std::stoi(*port);
        }
        if (const auto cors_origins = arg_value(argc, argv, "--cors-origins")) {
            config.cors_origins = *cors_origins;
        }
        if (const auto backend = arg_value(argc, argv, "--backend")) {
            config.backend = minitts::server::parse_server_backend(*backend);
        }
        if (const auto device = arg_value(argc, argv, "--device")) {
            config.device = std::stoi(*device);
        }
        if (const auto threads = arg_value(argc, argv, "--threads")) {
            config.threads = std::stoi(*threads);
        }
        if (const auto busy_timeout = arg_value(argc, argv, "--busy-timeout-ms")) {
            config.busy_timeout_ms = std::stoi(*busy_timeout);
        }
        int cuda_keepalive_ms = 0;
        if (const auto cuda_keepalive = arg_value(argc, argv, "--cuda-keepalive-ms")) {
            cuda_keepalive_ms = std::stoi(*cuda_keepalive);
        }
        int cuda_keepalive_work_ms = 50;
        if (const auto cuda_keepalive_work = arg_value(argc, argv, "--cuda-keepalive-work-ms")) {
            cuda_keepalive_work_ms = std::stoi(*cuda_keepalive_work);
        }
        if (const auto model_spec = arg_value(argc, argv, "--model-spec-override")) {
            config.model_spec_override = std::filesystem::path(*model_spec);
        }
        if (const auto voice_dir = arg_value(argc, argv, "--voice-dir")) {
            config.voice_dir = std::filesystem::path(*voice_dir);
        }
        if (const auto character_dir = arg_value(argc, argv, "--character-dir")) {
            config.character_dir = std::filesystem::path(*character_dir);
        }
        if (const auto llm_host = arg_value(argc, argv, "--llm-host")) {
            config.llm_host = *llm_host;
        }
        if (const auto llm_port = arg_value(argc, argv, "--llm-port")) {
            config.llm_port = std::stoi(*llm_port);
        }
        if (!(config.cors_origins == "*" || config.cors_origins == "")) {
            throw std::runtime_error("--cors-origins must be '*' (allow all origins) or '' (disabled)");
        }
        if (config.threads <= 0) {
            throw std::runtime_error("--threads must be positive");
        }
        if (config.busy_timeout_ms < 0) {
            throw std::runtime_error("--busy-timeout-ms must be >= 0 (0 disables the guard)");
        }
        if (cuda_keepalive_ms < 0) {
            throw std::runtime_error("--cuda-keepalive-ms must be >= 0 (0 disables)");
        }
        if (cuda_keepalive_work_ms <= 0) {
            throw std::runtime_error("--cuda-keepalive-work-ms must be positive");
        }
        if (cuda_keepalive_ms > 0 && config.backend != engine::core::BackendType::Cuda) {
            throw std::runtime_error("--cuda-keepalive-ms requires --backend cuda");
        }

        const auto ui_resource_anchor = executable_directory(argc > 0 ? argv[0] : nullptr);
        minitts::server::ServerState state(
            config,
            std::filesystem::current_path(),
            ui_resource_anchor);
#ifdef AUDIOCPP_SERVER_HAS_CUDA_KEEPALIVE
        auto cuda_keepalive = minitts::server::start_cuda_keepalive(
            config.device,
            cuda_keepalive_ms,
            cuda_keepalive_work_ms);
#else
        if (cuda_keepalive_ms > 0) {
            throw std::runtime_error("--cuda-keepalive-ms requires a CUDA-enabled server build");
        }
#endif
        minitts::server::serve_http(config.host, config.port, state, shutdown_requested, config.max_request_body_bytes);
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "audiocpp_server failed: " << ex.what() << "\n";
        return 1;
    }
}
