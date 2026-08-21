#include "llm_manager.h"

#include "llm_client.h"

#include <chrono>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace minitts::server {

LlmManager::LlmManager(
    std::string host,
    int port,
    std::filesystem::path server_exe,
    std::filesystem::path log_dir)
    : host_(std::move(host)),
      port_(port),
      server_exe_(std::move(server_exe)),
      log_dir_(std::move(log_dir)) {}

LlmManager::~LlmManager() {
    stop();
#ifdef _WIN32
    if (job_handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(job_handle_));
    }
#endif
}

bool LlmManager::start(const LlmModelSpec & spec, std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (process_handle_ != nullptr) {
        error = "a sidecar is already running; stop it first";
        return false;
    }
    if (!std::filesystem::exists(server_exe_)) {
        error = "llama-server not found at " + server_exe_.string();
        return false;
    }
    if (!std::filesystem::exists(spec.path)) {
        error = "model file not found: " + spec.path.string();
        return false;
    }

    std::vector<std::string> args = {
        server_exe_.string(),
        "-m", spec.path.string(),
        "--host", host_,
        "--port", std::to_string(port_),
        "-ngl", "99",
        "-c", "8192",
        "--jinja",
        "--no-webui",
        "--cache-reuse", "256",
        "--parallel", "1",
        "-t", "6",
    };
    for (const auto & extra : spec.extra_args) {
        args.push_back(extra);
    }

#ifdef _WIN32
    if (job_handle_ == nullptr) {
        HANDLE job = CreateJobObjectW(nullptr, nullptr);
        if (job != nullptr) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
            job_handle_ = job;
        }
    }

    std::wstring command_line;
    for (const auto & arg : args) {
        std::wstring wide(arg.begin(), arg.end());
        // Quote when the argument has spaces or quotes; embedded quotes are
        // backslash-escaped so JSON arguments (--chat-template-kwargs) arrive
        // intact through CommandLineToArgvW.
        const bool needs_quotes = arg.find_first_of(" \"") != std::string::npos;
        if (!command_line.empty()) {
            command_line += L' ';
        }
        if (needs_quotes) {
            command_line += L'"';
            for (const wchar_t ch : wide) {
                if (ch == L'"') {
                    command_line += L'\\';
                }
                command_line += ch;
            }
            command_line += L'"';
        } else {
            command_line += wide;
        }
    }

    SECURITY_ATTRIBUTES inheritable{};
    inheritable.nLength = sizeof(inheritable);
    inheritable.bInheritHandle = TRUE;
    const auto stdout_path = (log_dir_ / "llama_server.stdout.log").wstring();
    const auto stderr_path = (log_dir_ / "llama_server.stderr.log").wstring();
    HANDLE out_handle = CreateFileW(
        stdout_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &inheritable,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE err_handle = CreateFileW(
        stderr_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &inheritable,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = out_handle;
    startup.hStdError = err_handle;
    startup.hStdInput = INVALID_HANDLE_VALUE;

    PROCESS_INFORMATION process{};
    std::wstring mutable_command = command_line;
    const auto working_dir = server_exe_.parent_path().wstring();
    const BOOL created = CreateProcessW(
        nullptr,
        mutable_command.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW | CREATE_SUSPENDED,
        nullptr,
        working_dir.empty() ? nullptr : working_dir.c_str(),
        &startup,
        &process);
    if (out_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(out_handle);
    }
    if (err_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(err_handle);
    }
    if (!created) {
        error = "could not start llama-server (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    // Assign to the job BEFORE the process runs, so there is no window in
    // which a crash of this server leaks a running sidecar.
    if (job_handle_ != nullptr) {
        AssignProcessToJobObject(static_cast<HANDLE>(job_handle_), process.hProcess);
    }
    ResumeThread(process.hThread);
    CloseHandle(process.hThread);
    process_handle_ = process.hProcess;
#else
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (auto & arg : args) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    pid_t pid = 0;
    const int rc = posix_spawn(&pid, server_exe_.c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0) {
        error = "could not start llama-server (errno " + std::to_string(rc) + ")";
        return false;
    }
    process_handle_ = reinterpret_cast<void *>(static_cast<intptr_t>(pid));
#endif
    current_id_ = spec.id;
    return true;
}

bool LlmManager::wait_ready(int timeout_seconds, std::string & error) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (!process_running()) {
            error = "llama-server exited during model load; see llama_server.stderr.log";
            return false;
        }
        std::string body;
        if (llm::http_get_status(host_, port_, "/health", body) == 200) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    error = "llama-server did not become healthy in " + std::to_string(timeout_seconds) + " s";
    return false;
}

void LlmManager::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (process_handle_ == nullptr) {
        return;
    }
#ifdef _WIN32
    HANDLE handle = static_cast<HANDLE>(process_handle_);
    TerminateProcess(handle, 0);
    WaitForSingleObject(handle, 15000);
    CloseHandle(handle);
#else
    const pid_t pid = static_cast<pid_t>(reinterpret_cast<intptr_t>(process_handle_));
    kill(pid, SIGTERM);
    int status = 0;
    waitpid(pid, &status, 0);
#endif
    process_handle_ = nullptr;
    current_id_.clear();
}

std::string LlmManager::current_model_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_id_;
}

bool LlmManager::process_running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (process_handle_ == nullptr) {
        return false;
    }
#ifdef _WIN32
    return WaitForSingleObject(static_cast<HANDLE>(process_handle_), 0) == WAIT_TIMEOUT;
#else
    const pid_t pid = static_cast<pid_t>(reinterpret_cast<intptr_t>(process_handle_));
    return kill(pid, 0) == 0;
#endif
}

}  // namespace minitts::server
