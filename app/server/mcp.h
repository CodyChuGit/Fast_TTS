#pragma once

#include <functional>
#include <string>

namespace minitts::server::mcp {

// Outcome of running the `speak` tool. On success `wav_base64` holds a complete
// WAV file and `audio_seconds` its duration; on failure `error_text` carries a
// message the calling agent can read. Tool-execution failures are reported
// inside the tool result (`isError`) rather than as JSON-RPC errors, per the
// MCP spec, so the agent can see and react to them.
struct SpeakOutcome {
    bool ok = false;
    std::string wav_base64;
    double audio_seconds = 0.0;
    std::string error_text;
};

// Runs text-to-speech for a tools/call. `seed` < 0 means unseeded.
using SpeakFn = std::function<SpeakOutcome(const std::string & text, long long seed)>;

// One HTTP reply for a POST /mcp message. `status` 202 with an empty body is
// the accepted-notification case; everything else is a JSON body.
struct McpReply {
    int status = 200;
    std::string body;
};

// Handles one JSON-RPC message POSTed to the MCP endpoint (Streamable HTTP
// transport, single-message bodies -- batching was removed from the spec).
// Responses are plain JSON rather than SSE, which the transport permits and
// every client must accept.
McpReply handle_mcp_message(const std::string & body_text, const SpeakFn & speak);

}  // namespace minitts::server::mcp
