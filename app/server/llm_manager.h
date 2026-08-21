#pragma once

#include "config.h"

#include <filesystem>
#include <mutex>
#include <string>

namespace minitts::server {

// Owns the llama.cpp sidecar process so the LLM can be switched from Settings:
// stop the child, start it with another model, wait for its health. The child
// is placed in a kill-on-close job object, so however this server exits --
// clean shutdown, crash, task kill -- the sidecar and its VRAM go with it.
class LlmManager {
public:
    LlmManager(std::string host, int port, std::filesystem::path server_exe, std::filesystem::path log_dir);
    ~LlmManager();

    LlmManager(const LlmManager &) = delete;
    LlmManager & operator=(const LlmManager &) = delete;

    // Spawns llama-server for the spec. Does not wait for the model to load;
    // wait_ready() does. Returns false with `error` set when the process could
    // not even be created.
    bool start(const LlmModelSpec & spec, std::string & error);

    // Polls the sidecar's /health until the model is loaded or the deadline
    // passes. A 26B file takes tens of seconds to read from disk; the deadline
    // reflects that.
    bool wait_ready(int timeout_seconds, std::string & error);

    void stop();

    // Empty when no sidecar is running.
    std::string current_model_id() const;

    bool process_running() const;

private:
    std::string host_;
    int port_ = 0;
    std::filesystem::path server_exe_;
    std::filesystem::path log_dir_;
    mutable std::mutex mutex_;
    std::string current_id_;
    void * process_handle_ = nullptr;  // HANDLE on Windows; pid_t via intptr elsewhere.
    void * job_handle_ = nullptr;
};

}  // namespace minitts::server
