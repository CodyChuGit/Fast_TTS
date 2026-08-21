#include "mcp.h"

#include "engine/framework/io/json.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace minitts::server::mcp {
namespace {

using engine::io::json::Value;

// Newest first; the first entry is offered when the client asks for a revision
// this server does not know.
constexpr const char * kSupportedProtocolVersions[] = {
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
};

constexpr const char * kServerName = "qwen3-voice-mcp";
constexpr const char * kServerVersion = "1.0.0";

// A tool call is one utterance, not a document reader. Past this the model's
// token ceiling truncates mid-sentence anyway, so refuse with advice instead.
constexpr size_t kMaxSpeakChars = 4000;

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (ch < 0x20) {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
                out += buffer;
            } else {
                out += static_cast<char>(ch);
            }
        }
    }
    return out;
}

std::string quoted(const std::string & value) {
    return "\"" + json_escape(value) + "\"";
}

// JSON-RPC ids may be strings or numbers and must be echoed exactly, so they
// are kept pre-serialized rather than coerced to one type.
std::string serialize_id(const Value * id) {
    if (id == nullptr || id->is_null()) {
        return "null";
    }
    if (id->is_string()) {
        return quoted(id->as_string());
    }
    if (id->is_number()) {
        const double number = id->as_number();
        if (number == std::floor(number)) {
            return std::to_string(static_cast<long long>(number));
        }
        return std::to_string(number);
    }
    return "null";
}

McpReply jsonrpc_error(int http_status, const std::string & id, int code, const std::string & message) {
    return McpReply{
        http_status,
        "{\"jsonrpc\":\"2.0\",\"id\":" + id +
            ",\"error\":{\"code\":" + std::to_string(code) +
            ",\"message\":" + quoted(message) + "}}",
    };
}

McpReply jsonrpc_result(const std::string & id, const std::string & result_json) {
    return McpReply{
        200,
        "{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"result\":" + result_json + "}",
    };
}

std::string negotiated_protocol_version(const Value * params) {
    if (params != nullptr) {
        if (const auto * requested = params->find("protocolVersion");
            requested != nullptr && requested->is_string()) {
            for (const char * version : kSupportedProtocolVersions) {
                if (requested->as_string() == version) {
                    return version;
                }
            }
        }
    }
    return kSupportedProtocolVersions[0];
}

std::string initialize_result(const Value * params) {
    return "{\"protocolVersion\":" + quoted(negotiated_protocol_version(params)) +
        ",\"capabilities\":{\"tools\":{}}" +
        ",\"serverInfo\":{\"name\":" + quoted(kServerName) +
        ",\"title\":" + quoted("Super Fast TTS MCP Server") +
        ",\"version\":" + quoted(kServerVersion) + "}" +
        ",\"instructions\":" + quoted(std::string(
            "This server turns text into spoken audio with a fast streaming TTS engine. "
            "Call the speak tool with the text to say; it returns a complete WAV clip. "
            "Keep each call to one utterance -- a sentence or short paragraph -- and "
            "make several calls for longer passages.")) +
        "}";
}

std::string tools_list_result() {
    return std::string("{\"tools\":[{"
        "\"name\":\"speak\","
        "\"title\":\"Speak\","
        "\"description\":") + quoted(
            "Convert text to spoken audio and return it as a WAV clip. "
            "Best for one utterance per call.") + ","
        "\"inputSchema\":{"
            "\"type\":\"object\","
            "\"properties\":{"
                "\"text\":{\"type\":\"string\",\"description\":\"The text to speak aloud.\"},"
                "\"seed\":{\"type\":\"integer\",\"description\":\"Optional sampling seed for a reproducible delivery.\"}"
            "},"
            "\"required\":[\"text\"]"
        "}}]}";
}

std::string tool_error_result(const std::string & message) {
    return "{\"content\":[{\"type\":\"text\",\"text\":" + quoted(message) + "}],\"isError\":true}";
}

McpReply handle_tools_call(
    const std::string & id,
    const Value * params,
    const SpeakFn & speak) {
    if (params == nullptr || !params->is_object()) {
        return jsonrpc_error(200, id, -32602, "tools/call requires params with a tool name");
    }
    const auto * name = params->find("name");
    if (name == nullptr || !name->is_string()) {
        return jsonrpc_error(200, id, -32602, "tools/call requires a string 'name'");
    }
    if (name->as_string() != "speak") {
        return jsonrpc_error(200, id, -32602, "Unknown tool: " + name->as_string());
    }

    const auto * arguments = params->find("arguments");
    const Value * text = arguments != nullptr ? arguments->find("text") : nullptr;
    if (text == nullptr || !text->is_string() || text->as_string().empty()) {
        return jsonrpc_error(200, id, -32602, "speak requires a non-empty string 'text' argument");
    }
    if (text->as_string().size() > kMaxSpeakChars) {
        return jsonrpc_result(id, tool_error_result(
            "Text is too long for one utterance (" + std::to_string(text->as_string().size()) +
            " characters; the limit is " + std::to_string(kMaxSpeakChars) +
            "). Split it into sentences and call speak once per sentence."));
    }

    long long seed = -1;
    if (arguments != nullptr) {
        if (const auto * value = arguments->find("seed"); value != nullptr && value->is_number()) {
            seed = static_cast<long long>(value->as_i64());
        }
    }

    const SpeakOutcome outcome = speak(text->as_string(), seed);
    if (!outcome.ok) {
        return jsonrpc_result(id, tool_error_result("Speech generation failed: " + outcome.error_text));
    }

    char summary[64];
    std::snprintf(summary, sizeof(summary), "Spoke %.1f seconds of audio.", outcome.audio_seconds);
    return jsonrpc_result(id,
        "{\"content\":[{\"type\":\"audio\",\"data\":\"" + outcome.wav_base64 +
        "\",\"mimeType\":\"audio/wav\"},{\"type\":\"text\",\"text\":" + quoted(summary) +
        "}],\"isError\":false}");
}

}  // namespace

McpReply handle_mcp_message(const std::string & body_text, const SpeakFn & speak) {
    Value message = Value::make_null();
    try {
        message = engine::io::json::parse(body_text);
    } catch (const std::exception &) {
        return jsonrpc_error(400, "null", -32700, "Parse error");
    }
    if (!message.is_object()) {
        return jsonrpc_error(400, "null", -32600, "Expected a single JSON-RPC message object");
    }
    const auto * method = message.find("method");
    if (method == nullptr || !method->is_string()) {
        return jsonrpc_error(400, serialize_id(message.find("id")), -32600, "Missing method");
    }

    // A message without an id is a notification: accept it and reply with
    // nothing. This covers notifications/initialized and notifications/cancelled
    // without the server having to know each one.
    if (message.find("id") == nullptr) {
        return McpReply{202, ""};
    }

    const std::string id = serialize_id(message.find("id"));
    const std::string & name = method->as_string();
    const auto * params = message.find("params");

    if (name == "initialize") {
        return jsonrpc_result(id, initialize_result(params));
    }
    if (name == "ping") {
        return jsonrpc_result(id, "{}");
    }
    if (name == "tools/list") {
        return jsonrpc_result(id, tools_list_result());
    }
    if (name == "tools/call") {
        return handle_tools_call(id, params, speak);
    }
    return jsonrpc_error(200, id, -32601, "Method not found: " + name);
}

}  // namespace minitts::server::mcp
