#include "llm_client.h"

#include "engine/framework/io/json.h"

#include <cctype>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using SocketHandle = SOCKET;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using SocketHandle = int;
constexpr SocketHandle kInvalidSocket = -1;
#endif

namespace minitts::server::llm {
namespace {

void close_socket(SocketHandle socket_handle) {
#ifdef _WIN32
    closesocket(socket_handle);
#else
    close(socket_handle);
#endif
}

struct SocketGuard {
    SocketHandle handle = kInvalidSocket;
    ~SocketGuard() {
        if (handle != kInvalidSocket) {
            close_socket(handle);
        }
    }
};

bool send_all(SocketHandle socket_handle, const std::string & data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const int rc = ::send(
            socket_handle,
            data.data() + sent,
            static_cast<int>(data.size() - sent),
            0);
        if (rc <= 0) {
            return false;
        }
        sent += static_cast<size_t>(rc);
    }
    return true;
}

std::string lower_ascii(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

}  // namespace

bool ChunkedBodyDecoder::feed(const char * data, size_t size, std::string & out) {
    size_t index = 0;
    while (index < size && state_ != State::Done) {
        const char ch = data[index];
        switch (state_) {
        case State::Size:
            if (ch == '\r') {
                state_ = State::SizeLf;
            } else {
                size_line_ += ch;
            }
            ++index;
            break;
        case State::SizeLf: {
            ++index;  // consume '\n'
            // Chunk extensions after ';' are legal; ignore them.
            const auto semicolon = size_line_.find(';');
            const std::string digits = semicolon == std::string::npos
                ? size_line_ : size_line_.substr(0, semicolon);
            remaining_ = 0;
            for (const char digit : digits) {
                const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(digit)));
                if (lower >= '0' && lower <= '9') {
                    remaining_ = remaining_ * 16 + static_cast<size_t>(lower - '0');
                } else if (lower >= 'a' && lower <= 'f') {
                    remaining_ = remaining_ * 16 + static_cast<size_t>(lower - 'a' + 10);
                }
            }
            size_line_.clear();
            state_ = remaining_ == 0 ? State::TrailerCr : State::Data;
            break;
        }
        case State::Data: {
            const size_t take = std::min(remaining_, size - index);
            out.append(data + index, take);
            index += take;
            remaining_ -= take;
            if (remaining_ == 0) {
                state_ = State::DataCr;
            }
            break;
        }
        case State::DataCr:
            ++index;  // '\r'
            state_ = State::DataLf;
            break;
        case State::DataLf:
            ++index;  // '\n'
            state_ = State::Size;
            break;
        case State::TrailerCr:
            // Consume everything through the final blank line; trailers are
            // not used. The first LF after the zero-chunk's CRLF pair ends it.
            if (ch == '\n') {
                state_ = State::Done;
            }
            ++index;
            break;
        case State::Done:
            break;
        }
    }
    return state_ == State::Done;
}

SseDeltaParser::Event SseDeltaParser::parse_payload(const std::string & payload) const {
    Event event;
    if (payload == "[DONE]") {
        event.done = true;
        return event;
    }
    try {
        const auto value = engine::io::json::parse(payload);
        if (const auto * error = value.find("error")) {
            if (const auto * message = error->find("message"); message != nullptr && message->is_string()) {
                event.error = message->as_string();
            } else {
                event.error = "LLM stream error";
            }
            return event;
        }
        if (const auto * timings = value.find("timings")) {
            const auto number = [&](const char * key) -> double {
                const auto * v = timings->find(key);
                return v != nullptr && v->is_number() ? v->as_number() : -1;
            };
            event.prompt_n = number("prompt_n");
            event.prompt_ms = number("prompt_ms");
            event.predicted_n = number("predicted_n");
            event.predicted_ms = number("predicted_ms");
        }
        if (const auto * choices = value.find("choices");
            choices != nullptr && choices->is_array() && !choices->as_array().empty()) {
            const auto & choice = choices->as_array()[0];
            if (const auto * delta = choice.find("delta")) {
                if (const auto * content = delta->find("content");
                    content != nullptr && content->is_string()) {
                    event.delta = content->as_string();
                }
            }
            if (const auto * reason = choice.find("finish_reason");
                reason != nullptr && reason->is_string()) {
                event.finish_reason = reason->as_string();
            }
        }
    } catch (const std::exception &) {
        // A malformed frame is dropped rather than killing the stream; the
        // [DONE] sentinel or the connection close still terminates cleanly.
    }
    return event;
}

std::vector<SseDeltaParser::Event> SseDeltaParser::feed(const std::string & bytes) {
    std::vector<Event> events;
    buffer_ += bytes;
    size_t start = 0;
    while (true) {
        const size_t newline = buffer_.find('\n', start);
        if (newline == std::string::npos) {
            break;
        }
        std::string line = buffer_.substr(start, newline - start);
        start = newline + 1;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            // Blank line dispatches the accumulated event.
            if (!event_data_.empty()) {
                events.push_back(parse_payload(event_data_));
                event_data_.clear();
            }
            continue;
        }
        if (line.rfind("data:", 0) == 0) {
            std::string payload = line.substr(5);
            if (!payload.empty() && payload.front() == ' ') {
                payload.erase(0, 1);
            }
            if (!event_data_.empty()) {
                event_data_ += '\n';
            }
            event_data_ += payload;
        }
        // Comment lines (":") and other fields are ignored.
    }
    buffer_.erase(0, start);
    return events;
}

int http_get_status(
    const std::string & host,
    int port,
    const std::string & path,
    std::string & body) {
    body.clear();
    SocketGuard guard;
    guard.handle = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (guard.handle == kInvalidSocket) {
        return -1;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<unsigned short>(port));
    if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1 ||
        ::connect(guard.handle, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        return -1;
    }
    {
#ifdef _WIN32
        DWORD timeout = 5000;
        setsockopt(guard.handle, SOL_SOCKET, SO_RCVTIMEO,
            reinterpret_cast<const char *>(&timeout), sizeof(timeout));
#else
        timeval timeout{5, 0};
        setsockopt(guard.handle, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
    }
    const std::string request =
        "GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";
    if (!send_all(guard.handle, request)) {
        return -1;
    }
    char buffer[8192];
    std::string response;
    while (response.size() < 64 * 1024) {
        const int received = ::recv(guard.handle, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            break;
        }
        response.append(buffer, static_cast<size_t>(received));
    }
    if (response.size() < 12 || response.compare(0, 5, "HTTP/") != 0) {
        return -1;
    }
    const int status = std::atoi(response.c_str() + 9);
    const auto header_end = response.find("\r\n\r\n");
    if (header_end != std::string::npos) {
        body = response.substr(header_end + 4, 4096);
    }
    return status;
}

int http_post_status(
    const std::string & host,
    int port,
    const std::string & path,
    const std::string & request_body,
    std::string & response) {
    response.clear();
    SocketGuard guard;
    guard.handle = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (guard.handle == kInvalidSocket) {
        return -1;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<unsigned short>(port));
    if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1 ||
        ::connect(guard.handle, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        return -1;
    }
    {
        // Cache-priming prefills of a long tail can take tens of seconds on a
        // busy GPU; be patient, the caller runs on a background thread.
#ifdef _WIN32
        DWORD timeout = 60000;
        setsockopt(guard.handle, SOL_SOCKET, SO_RCVTIMEO,
            reinterpret_cast<const char *>(&timeout), sizeof(timeout));
#else
        timeval timeout{60, 0};
        setsockopt(guard.handle, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
    }
    const std::string request =
        "POST " + path + " HTTP/1.1\r\nHost: " + host +
        "\r\nContent-Type: application/json\r\nContent-Length: " +
        std::to_string(request_body.size()) + "\r\nConnection: close\r\n\r\n" +
        request_body;
    if (!send_all(guard.handle, request)) {
        return -1;
    }
    char buffer[8192];
    std::string raw;
    while (raw.size() < 256 * 1024) {
        const int received = ::recv(guard.handle, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            break;
        }
        raw.append(buffer, static_cast<size_t>(received));
    }
    if (raw.size() < 12 || raw.compare(0, 5, "HTTP/") != 0) {
        return -1;
    }
    const int status = std::atoi(raw.c_str() + 9);
    const auto header_end = raw.find("\r\n\r\n");
    if (header_end != std::string::npos) {
        response = raw.substr(header_end + 4, 4096);
    }
    return status;
}

ChatResult stream_chat(
    const std::string & host,
    int port,
    const std::string & body_json,
    const std::function<bool(const std::string & delta)> & on_delta) {
    ChatResult result;

    SocketGuard guard;
    guard.handle = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (guard.handle == kInvalidSocket) {
        result.error = "could not create a socket to the LLM server";
        return result;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<unsigned short>(port));
    if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
        result.error = "invalid LLM server address: " + host;
        return result;
    }
    if (::connect(guard.handle, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        result.error = "the LLM server is not reachable at " + host + ":" + std::to_string(port);
        return result;
    }

    // Tokens are tiny; never let Nagle sit on one.
    {
        int yes = 1;
        setsockopt(guard.handle, IPPROTO_TCP, TCP_NODELAY,
            reinterpret_cast<const char *>(&yes), sizeof(yes));
    }
    // A wedged generation must not park this thread forever. Five minutes
    // exceeds any legitimate inter-token gap by orders of magnitude.
    {
#ifdef _WIN32
        DWORD timeout = 300000;
        setsockopt(guard.handle, SOL_SOCKET, SO_RCVTIMEO,
            reinterpret_cast<const char *>(&timeout), sizeof(timeout));
#else
        timeval timeout{300, 0};
        setsockopt(guard.handle, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
    }

    std::string request =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: " + host + ":" + std::to_string(port) + "\r\n"
        "Content-Type: application/json\r\n"
        "Accept: text/event-stream\r\n"
        "Connection: close\r\n"
        "Content-Length: " + std::to_string(body_json.size()) + "\r\n"
        "\r\n" + body_json;
    if (!send_all(guard.handle, request)) {
        result.error = "could not send the request to the LLM server";
        return result;
    }

    // Read and split the response headers.
    std::string headers;
    std::string carry;
    char buffer[16384];
    while (headers.find("\r\n\r\n") == std::string::npos) {
        const int received = ::recv(guard.handle, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            result.error = "the LLM server closed the connection before responding";
            return result;
        }
        headers.append(buffer, static_cast<size_t>(received));
        if (headers.size() > 64 * 1024) {
            result.error = "the LLM server sent an oversized response header";
            return result;
        }
    }
    const size_t header_end = headers.find("\r\n\r\n");
    carry = headers.substr(header_end + 4);
    headers.resize(header_end);

    const auto lowered = lower_ascii(headers);
    int status = 0;
    if (headers.size() > 12) {
        status = std::atoi(headers.c_str() + 9);
    }
    const bool chunked = lowered.find("transfer-encoding: chunked") != std::string::npos;

    if (status != 200) {
        // Collect a little body for the error message.
        std::string body = carry;
        while (body.size() < 4096) {
            const int received = ::recv(guard.handle, buffer, sizeof(buffer), 0);
            if (received <= 0) {
                break;
            }
            body.append(buffer, static_cast<size_t>(received));
        }
        result.error = "LLM server answered HTTP " + std::to_string(status) +
            (body.empty() ? "" : (": " + body.substr(0, 512)));
        return result;
    }

    ChunkedBodyDecoder decoder;
    SseDeltaParser parser;
    bool done = false;
    auto consume = [&](const char * data, size_t size) -> bool {
        std::string payload;
        bool final_chunk = false;
        if (chunked) {
            final_chunk = decoder.feed(data, size, payload);
        } else {
            payload.assign(data, size);
        }
        for (const auto & event : parser.feed(payload)) {
            if (!event.error.empty()) {
                result.error = event.error;
                return false;
            }
            if (!event.finish_reason.empty()) {
                result.finish_reason = event.finish_reason;
            }
            if (event.prompt_n >= 0) {
                result.prompt_n = event.prompt_n;
                result.prompt_ms = event.prompt_ms;
                result.predicted_n = event.predicted_n;
                result.predicted_ms = event.predicted_ms;
            }
            if (event.done) {
                done = true;
                return false;
            }
            if (!event.delta.empty() && !on_delta(event.delta)) {
                result.error = "aborted";
                return false;
            }
        }
        return !final_chunk;
    };

    bool keep_reading = consume(carry.data(), carry.size());
    while (keep_reading) {
        const int received = ::recv(guard.handle, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            break;  // Connection: close ends non-chunked streams this way.
        }
        keep_reading = consume(buffer, static_cast<size_t>(received));
    }

    if (result.error.empty()) {
        result.ok = true;
    } else if (result.error == "aborted") {
        result.error.clear();
        result.ok = false;
    }
    (void)done;
    return result;
}

}  // namespace minitts::server::llm
