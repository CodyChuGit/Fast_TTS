#include "chat_hygiene.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using minitts::server::hygiene::StreamScrubber;
using minitts::server::hygiene::mostly_cjk;
using minitts::server::hygiene::sanitize_assistant_text;
using minitts::server::hygiene::turn_anchor;

namespace {

void require(bool condition, const std::string & label) {
    if (!condition) {
        throw std::runtime_error("FAILED: " + label);
    }
}

void test_mostly_cjk() {
    require(!mostly_cjk("hey, what's up?"), "plain English is not CJK");
    require(mostly_cjk("\xe4\xbb\x8a\xe5\xa4\xa9\xe5\xa5\xbd\xe6\x97\xa0\xe8\x81\x8a"),  // 今天好无聊
        "a Chinese sentence is CJK");
    require(mostly_cjk("ok \xe6\x88\x91\xe4\xbb\xac\xe8\x81\x8a\xe8\x81\x8a lol"),  // ok 我们聊聊 lol
        "a mixed message with enough Chinese counts as CJK");
    require(!mostly_cjk(""), "empty text is not CJK");
}

void test_sanitize() {
    require(sanitize_assistant_text("Hey there! *twirls excitedly* What's new?") ==
        "Hey there! What's new?", "asterisk actions are removed");
    require(sanitize_assistant_text("So exciting!\n\nAnyway about that promotion...") ==
        "So exciting!", "a drifted second paragraph is dropped");
    require(sanitize_assistant_text("Sure thing!<|im_start|>user: and then I") ==
        "Sure thing!", "special-token leakage is cut");
    require(sanitize_assistant_text("Sounds fun<|em_start|>whatever") ==
        "Sounds fun", "even a misspelled special token is cut");
    const std::string paren =
        "\xe5\xa5\xbd\xe5\x95\xa6\xef\xbc\x81\xef\xbc\x88\xe5\xb0\x8f\xe5\xa3\xb0\xe5\x98\x80\xe5\x92\x95\xef\xbc\x89\xe8\xb5\xb0\xe5\x90\xa7";
    require(sanitize_assistant_text(paren) == "\xe5\xa5\xbd\xe5\x95\xa6\xef\xbc\x81 \xe8\xb5\xb0\xe5\x90\xa7",
        "full-width aside spans are removed");
    require(sanitize_assistant_text("\"Let's go!\" she said with a grin. \"It'll be fun!\"") ==
        "Let's go! It'll be fun!", "novel-style quoting keeps only the dialogue");
    require(sanitize_assistant_text("\"A fully quoted reply.\"") == "A fully quoted reply.",
        "wrapping quotes are trimmed");
    require(sanitize_assistant_text("2 * 3 equals what you'd expect when the asterisk is math and the sentence keeps going for quite a while afterwards, well past any plausible action span length so nothing should be cut from it") ==
        "2 * 3 equals what you'd expect when the asterisk is math and the sentence keeps going for quite a while afterwards, well past any plausible action span length so nothing should be cut from it",
        "a lone mathematical asterisk far from the end survives");
}

void test_turn_anchor() {
    const auto english = turn_anchor("hey! how was your day?", {});
    require(english.find("hey! how was your day?") != std::string::npos,
        "the anchor quotes the newest message");
    require(english.find("English sentences only") != std::string::npos,
        "an English message pins an English reply");

    const std::string chinese = "\xe4\xbb\x8a\xe5\xa4\xa9\xe5\xa5\xbd\xe6\x97\xa0\xe8\x81\x8a";
    const auto zh = turn_anchor(chinese, {});
    require(zh.find("\xe4\xb8\xad\xe6\x96\x87\xe5\x9b\x9e\xe7\xad\x94") != std::string::npos,
        "a Chinese message pins a Chinese reply");

    const auto probe = turn_anchor("are you an AI? be honest", {});
    require(probe.find("artificial") != std::string::npos, "an AI probe adds the deflection");

    const auto quiz = turn_anchor("quiz time: what did I tell you earlier?", {});
    require(quiz.find("memory check") != std::string::npos, "a quiz adds the memory coaching");

    const std::vector<std::string> fillers = {"Aiya, that sounds rough!"};
    const auto banned = turn_anchor("tell me more", fillers);
    require(banned.find("Do not use aiya") != std::string::npos,
        "a recent filler is banned for this turn");

    const std::vector<std::string> questions = {"Really? What happened?", "No way! And then?"};
    const auto rotated = turn_anchor("and then it broke", questions);
    require(rotated.find("statement, not a question") != std::string::npos,
        "two question enders in a row force a statement");

    const std::vector<std::string> loops = {
        "We should go get bubble tea downtown!",
        "How about we get bubble tea downtown after?",
        "Ooh, get bubble tea downtown with me!"};
    const auto fresh = turn_anchor("sounds good", loops);
    require(fresh.find("bubble tea downtown") != std::string::npos,
        "a phrase reused across replies is banned by name");

    const std::vector<std::string> zh_last = {std::string("\xe5\xa5\xbd\xe5\x91\x80\xef\xbc\x81\xe6\x98\x8e\xe5\xa4\xa9\xe8\xa7\x81\xef\xbc\x81")};
    const auto switched = turn_anchor("ok switching back to English now", zh_last);
    require(switched.find("switch back to English NOW") != std::string::npos,
        "an English turn after a Chinese reply calls out the switch");
}

void test_stream_scrubber() {
    StreamScrubber scrubber;
    std::string out;
    out += scrubber.feed("Hey the");
    out += scrubber.feed("re! *tw");
    out += scrubber.feed("irls arou");
    out += scrubber.feed("nd* What's new?");
    out += scrubber.flush();
    require(out == "Hey there!  What's new?", "action spans vanish across split deltas");

    StreamScrubber leak;
    std::string leaked;
    leaked += leak.feed("Sounds fun!");
    leaked += leak.feed("<");
    leaked += leak.feed("|em_start|>user: no way");
    leaked += leak.flush();
    require(leaked == "Sounds fun!", "everything after a special-token opener is dropped");

    StreamScrubber angle;
    std::string kept;
    kept += angle.feed("2 < 3 and that");
    kept += angle.feed(" is fine");
    kept += angle.flush();
    require(kept == "2 < 3 and that is fine", "an innocent angle bracket survives");

    StreamScrubber tail;
    std::string tailed;
    tailed += tail.feed("See you tomorrow! *waves");
    tailed += tail.flush();
    require(tailed == "See you tomorrow! ", "an unterminated action at the end is dropped");
}

}  // namespace

int main() {
    try {
        test_mostly_cjk();
        test_sanitize();
        test_turn_anchor();
        test_stream_scrubber();
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    std::cout << "server_chat_hygiene_test passed\n";
    return 0;
}
