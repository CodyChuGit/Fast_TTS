#include "mcp.h"

#include "engine/framework/io/json.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using engine::io::json::Value;
using minitts::server::mcp::McpReply;
using minitts::server::mcp::SpeakOutcome;
using minitts::server::mcp::handle_mcp_message;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

SpeakOutcome ok_speak(const std::string & text, long long seed) {
    (void)text;
    (void)seed;
    SpeakOutcome outcome;
    outcome.ok = true;
    outcome.wav_base64 = "UklGRg==";
    outcome.audio_seconds = 1.5;
    return outcome;
}

SpeakOutcome failing_speak(const std::string &, long long) {
    SpeakOutcome outcome;
    outcome.error_text = "device exploded";
    return outcome;
}

McpReply call(const std::string & body, const minitts::server::mcp::SpeakFn & speak = ok_speak) {
    return handle_mcp_message(body, "F", speak);
}

// Every reply is itself JSON; parse it so assertions read fields rather than
// substrings.
Value parsed(const McpReply & reply) {
    return engine::io::json::parse(reply.body);
}

void test_initialize_negotiates_the_protocol_version() {
    const auto reply = call(
        R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"t","version":"0"}}})");
    require(reply.status == 200, "initialize answers 200");
    const auto body = parsed(reply);
    const auto & result = body.require("result");
    require(
        engine::io::json::require_string(result, "protocolVersion") == "2025-03-26",
        "a supported requested version is echoed");
    require(result.require("capabilities").find("tools") != nullptr, "tools capability is declared");
    require(
        engine::io::json::require_string(result, "instructions").find("\"F\"") != std::string::npos,
        "instructions name the character");

    const auto future = parsed(call(
        R"({"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2099-01-01"}})"));
    require(
        engine::io::json::require_string(future.require("result"), "protocolVersion") == "2025-06-18",
        "an unknown version gets the newest supported one");
}

void test_notifications_are_accepted_without_a_body() {
    const auto reply = call(R"({"jsonrpc":"2.0","method":"notifications/initialized"})");
    require(reply.status == 202, "notifications answer 202");
    require(reply.body.empty(), "notifications carry no body");
}

void test_tools_list_names_the_character() {
    const auto body = parsed(call(R"({"jsonrpc":"2.0","id":3,"method":"tools/list"})"));
    const auto & tools = body.require("result").require("tools").as_array();
    require(tools.size() == 1, "exactly one tool");
    require(engine::io::json::require_string(tools[0], "name") == "speak", "the tool is speak");
    require(
        engine::io::json::require_string(tools[0], "description").find("\"F\"") != std::string::npos,
        "the description names the character");
    const auto & schema = tools[0].require("inputSchema");
    require(schema.require("required").as_array().size() == 1, "only text is required");
}

void test_tools_call_returns_audio_content() {
    const auto body = parsed(call(
        R"({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"speak","arguments":{"text":"Hello.","seed":7}}})"));
    const auto & result = body.require("result");
    require(!engine::io::json::require_bool(result, "isError"), "success is not an error");
    const auto & content = result.require("content").as_array();
    require(content.size() == 2, "audio plus a text summary");
    require(engine::io::json::require_string(content[0], "type") == "audio", "first item is audio");
    require(engine::io::json::require_string(content[0], "mimeType") == "audio/wav", "wav mime type");
    require(engine::io::json::require_string(content[0], "data") == "UklGRg==", "audio data passes through");
    require(
        engine::io::json::require_string(content[1], "text").find("F") != std::string::npos,
        "the summary names the character");
}

void test_tool_failures_are_tool_results_not_protocol_errors() {
    const auto body = parsed(call(
        R"({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"speak","arguments":{"text":"Hi"}}})",
        failing_speak));
    const auto & result = body.require("result");
    require(engine::io::json::require_bool(result, "isError"), "tool failure sets isError");
    require(
        engine::io::json::require_string(result.require("content").as_array()[0], "text")
            .find("device exploded") != std::string::npos,
        "the agent sees the failure reason");
}

void test_protocol_errors_use_jsonrpc_codes() {
    require(
        parsed(call(R"({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"paint","arguments":{}}})"))
            .require("error").require("code").as_i64() == -32602,
        "an unknown tool is invalid params");
    require(
        parsed(call(R"({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"speak","arguments":{}}})"))
            .require("error").require("code").as_i64() == -32602,
        "missing text is invalid params");
    require(
        parsed(call(R"({"jsonrpc":"2.0","id":8,"method":"resources/list"})"))
            .require("error").require("code").as_i64() == -32601,
        "an unsupported method is method-not-found");
    const auto garbage = call("{not json");
    require(garbage.status == 400, "unparseable bodies answer 400");
    require(parsed(garbage).require("error").require("code").as_i64() == -32700, "with a parse error");
    require(
        parsed(call(R"({"jsonrpc":"2.0","id":"str-id","method":"ping"})"))
            .require("id").as_string() == "str-id",
        "string ids are echoed as strings");
}

void test_overlong_text_is_refused_with_advice() {
    const std::string text(5000, 'a');
    const auto body = parsed(call(
        std::string(R"({"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"speak","arguments":{"text":")") +
        text + R"("}}})"));
    const auto & result = body.require("result");
    require(engine::io::json::require_bool(result, "isError"), "overlong text is a tool error");
    require(
        engine::io::json::require_string(result.require("content").as_array()[0], "text")
            .find("once per sentence") != std::string::npos,
        "the error tells the agent how to proceed");
}

}  // namespace

int main() {
    try {
        test_initialize_negotiates_the_protocol_version();
        test_notifications_are_accepted_without_a_body();
        test_tools_list_names_the_character();
        test_tools_call_returns_audio_content();
        test_tool_failures_are_tool_results_not_protocol_errors();
        test_protocol_errors_use_jsonrpc_codes();
        test_overlong_text_is_refused_with_advice();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_mcp_test passed\n";
    return 0;
}
