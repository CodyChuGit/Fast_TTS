#include "llm_client.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using minitts::server::llm::ChunkedBodyDecoder;
using minitts::server::llm::SseDeltaParser;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_chunked_bodies_decode_across_arbitrary_splits() {
    const std::string wire = "5\r\nHello\r\n8\r\n, world!\r\n0\r\n\r\n";
    // Feed one byte at a time: TCP owes no alignment whatsoever.
    ChunkedBodyDecoder decoder;
    std::string out;
    bool finished = false;
    for (const char ch : wire) {
        finished = decoder.feed(&ch, 1, out);
    }
    require(out == "Hello, world!", "payload reassembled");
    require(finished, "final chunk recognized");
}

void test_chunk_extensions_and_hex_sizes() {
    const std::string wire = "A;ext=1\r\n0123456789\r\n0\r\n\r\n";
    ChunkedBodyDecoder decoder;
    std::string out;
    require(decoder.feed(wire.data(), wire.size(), out), "decoded to completion");
    require(out == "0123456789", "hex size with extension decoded");
}

void test_sse_deltas_parse_split_frames() {
    SseDeltaParser parser;
    std::string frame =
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n";
    // Split mid-frame to prove buffering.
    auto events = parser.feed(frame.substr(0, 25));
    for (auto & event : parser.feed(frame.substr(25))) {
        events.push_back(event);
    }
    require(events.size() == 2, "two delta events");
    require(events[0].delta == "Hel" && events[1].delta == "lo", "deltas in order");

    auto done = parser.feed("data: [DONE]\n\n");
    require(done.size() == 1 && done[0].done, "the DONE sentinel terminates");
}

void test_sse_errors_and_finish_reason_surface() {
    SseDeltaParser parser;
    auto events = parser.feed(
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n");
    require(events.size() == 1 && events[0].finish_reason == "stop", "finish reason surfaces");

    auto errors = parser.feed("data: {\"error\":{\"message\":\"slot exhausted\"}}\n\n");
    require(errors.size() == 1 && errors[0].error == "slot exhausted", "server errors surface");

    // Malformed frames are dropped, not fatal.
    auto garbage = parser.feed("data: {nope\n\n");
    require(garbage.size() == 1 && garbage[0].delta.empty() && garbage[0].error.empty(),
        "a malformed frame degrades to an empty event");
}

}  // namespace

int main() {
    try {
        test_chunked_bodies_decode_across_arbitrary_splits();
        test_chunk_extensions_and_hex_sizes();
        test_sse_deltas_parse_split_frames();
        test_sse_errors_and_finish_reason_surface();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_llm_client_test passed\n";
    return 0;
}
