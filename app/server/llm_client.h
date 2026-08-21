#pragma once

#include <functional>
#include <string>
#include <vector>

namespace minitts::server::llm {

// Decodes an HTTP/1.1 chunked transfer body fed as raw bytes. Payload bytes
// accumulate into the string passed to feed; feed returns true once the final
// zero-size chunk has been consumed.
class ChunkedBodyDecoder {
public:
    bool feed(const char * data, size_t size, std::string & out);
    bool finished() const { return state_ == State::Done; }

private:
    enum class State { Size, SizeLf, Data, DataCr, DataLf, TrailerCr, Done };
    State state_ = State::Size;
    std::string size_line_;
    size_t remaining_ = 0;
};

// Extracts OpenAI-style streaming deltas from a text/event-stream body. Feed
// arbitrary byte slices; complete events come back in order. `done` marks the
// [DONE] sentinel; `error` carries a server-reported error payload.
class SseDeltaParser {
public:
    struct Event {
        std::string delta;
        std::string finish_reason;
        bool done = false;
        std::string error;
    };

    std::vector<Event> feed(const std::string & bytes);

private:
    Event parse_payload(const std::string & payload) const;

    std::string buffer_;
    std::string event_data_;
};

struct ChatResult {
    bool ok = false;
    std::string error;
    std::string finish_reason;
};

// Plain HTTP GET against the sidecar (health polls). Returns the status code,
// or -1 when the server is unreachable; `body` receives up to a few KB.
int http_get_status(
    const std::string & host,
    int port,
    const std::string & path,
    std::string & body);

// Plain HTTP POST against the sidecar (cache-priming requests). Returns the
// status code, or -1 when the server is unreachable; `response` receives up
// to a few KB.
int http_post_status(
    const std::string & host,
    int port,
    const std::string & path,
    const std::string & request_body,
    std::string & response);

// Streams a chat completion from a llama.cpp server on the loopback,
// invoking `on_delta` once per content token. Returning false from the
// callback aborts the generation by closing the connection -- llama-server
// stops the slot when its client disappears. Blocking; call from a worker
// thread.
ChatResult stream_chat(
    const std::string & host,
    int port,
    const std::string & body_json,
    const std::function<bool(const std::string & delta)> & on_delta);

}  // namespace minitts::server::llm
