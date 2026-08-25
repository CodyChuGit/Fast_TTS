#include "segmenter.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using minitts::server::SentenceSegmenter;
using minitts::server::ends_with_sentence_terminal;
using minitts::server::strip_speech_markup;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Feeds text in small pieces the way LLM deltas arrive.
std::vector<std::string> feed_tokens(SentenceSegmenter & segmenter, const std::string & text, size_t piece = 4) {
    std::vector<std::string> out;
    for (size_t index = 0; index < text.size(); index += piece) {
        for (auto & segment : segmenter.feed(text.substr(index, piece))) {
            out.push_back(std::move(segment));
        }
    }
    return out;
}

void test_sentences_are_emitted_as_they_complete() {
    SentenceSegmenter segmenter;
    auto segments = feed_tokens(segmenter,
        "The rain had finally stopped. She stepped outside and looked up. ");
    require(segments.size() == 2, "two sentences emitted");
    require(segments[0] == "The rain had finally stopped.", "first sentence intact");
    require(segments[1] == "She stepped outside and looked up.", "second sentence intact");
    require(segmenter.flush().empty(), "nothing pending");
}

void test_the_first_segment_cuts_early_at_a_clause() {
    // Time-to-first-audio depends on this: a long opening sentence must yield a
    // speakable clause before its period arrives.
    SentenceSegmenter segmenter;
    auto segments = feed_tokens(segmenter,
        "When the storm finally cleared, the villagers came out to see what "
        "was left of the harbor and of the boats they had pulled ashore.");
    require(!segments.empty(), "an early segment was emitted");
    require(segments[0].back() == ',', "the early cut lands on the clause boundary");
    require(segments[0].size() >= 14, "the early segment is a speakable clause, not a fragment");
    const auto rest = segmenter.flush();
    require(!rest.empty(), "the remainder is still pending at end of stream");
}

void test_abbreviations_and_numbers_do_not_split() {
    SentenceSegmenter segmenter;
    auto segments = feed_tokens(segmenter, "Pi is 3.14159 to five places! ");
    require(segments.size() == 1, "the decimal point did not split");
    require(segments[0] == "Pi is 3.14159 to five places!", "sentence intact");
}

void test_cjk_sentences_split_without_spaces() {
    SentenceSegmenter segmenter;
    const std::string text = "\xe4\xbd\xa0\xe5\xa5\xbd\xe3\x80\x82"  // ni hao .
                             "\xe5\x86\x8d\xe8\xa7\x81\xef\xbc\x81"; // zai jian !
    auto segments = feed_tokens(segmenter, text, 3);
    require(segments.size() == 2, "CJK enders split without trailing spaces");
}

void test_runons_are_force_split_at_a_space() {
    SentenceSegmenter::Options options;
    options.max_segment_chars = 40;
    SentenceSegmenter segmenter(options);
    std::string runon;
    for (int index = 0; index < 20; ++index) {
        runon += "word word word ";
    }
    auto segments = feed_tokens(segmenter, runon);
    require(!segments.empty(), "a boundary-free run-on still gets split");
    for (const auto & segment : segments) {
        require(segment.size() <= 41, "force-split segments respect the ceiling");
    }
}

void test_speech_markup_is_stripped_for_tts() {
    require(strip_speech_markup("*smiles warmly* Hello there.") == "Hello there.",
        "roleplay actions are not read aloud");
    require(strip_speech_markup("I *really* mean it.") == "I  mean it.",
        "emphasis markers vanish");
    require(strip_speech_markup("## Heading `code`") == "Heading code",
        "markdown noise vanishes");
    // An unbalanced asterisk must not swallow the reply.
    const auto unbalanced = strip_speech_markup("*She leans in. Are you coming or not?");
    require(unbalanced.find("Are you coming") != std::string::npos,
        "an unbalanced action marker does not silence the speech");
    // Emoji are written, never spoken: read aloud they become a word, or a
    // token the synthesiser has no sound for at all.
    require(strip_speech_markup("Yay, boba! \xF0\x9F\x92\x96\xE2\x9C\xA8") == "Yay, boba!",
        "emoji and pictographs are not read aloud");
    require(strip_speech_markup("Sun \xE2\x98\x80\xEF\xB8\x8F today.") == "Sun  today.",
        "a variation selector leaves nothing behind either");
    require(strip_speech_markup("Cost is 50\xE2\x82\xAC today.") == "Cost is 50\xE2\x82\xAC today.",
        "ordinary non-Latin characters still speak");
}

void test_a_runon_opener_is_cut_at_a_word_boundary() {
    // No clause boundary anywhere in the opener: time-to-first-audio must
    // still be bounded, so a word-boundary cut fires once enough streamed.
    SentenceSegmenter segmenter;
    auto segments = feed_tokens(segmenter,
        "Well now that is truly one of the most remarkable things anyone has ever walked in here and said to me");
    require(!segments.empty(), "the run-on opener still produced an early segment");
    require(segments[0].size() >= 14 && segments[0].size() <= 60,
        "the word cut lands in the bounded window");
    require(segments[0].find(' ') != std::string::npos, "the cut is at a word boundary, not mid-word");
}

void test_terminal_detection_for_truncated_tails() {
    require(ends_with_sentence_terminal("It ends here."), "a period terminates");
    require(ends_with_sentence_terminal("Really?!"), "stacked enders terminate");
    require(ends_with_sentence_terminal("\"So it goes.\""), "trailing quotes are ignored");
    require(ends_with_sentence_terminal(strip_speech_markup("Done. *smiles*")),
        "a trailing action is stripped before the check at the call site");
    require(!ends_with_sentence_terminal("and then she was about to"), "mid-thought text does not");
    require(!ends_with_sentence_terminal("half a word, then a comma,"), "a comma does not terminate");
    require(!ends_with_sentence_terminal("   "), "whitespace alone does not");
    require(ends_with_sentence_terminal("\xe5\xa5\xbd\xe3\x80\x82"), "CJK enders terminate");
}

}  // namespace

int main() {
    try {
        test_sentences_are_emitted_as_they_complete();
        test_the_first_segment_cuts_early_at_a_clause();
        test_abbreviations_and_numbers_do_not_split();
        test_cjk_sentences_split_without_spaces();
        test_runons_are_force_split_at_a_space();
        test_speech_markup_is_stripped_for_tts();
        test_a_runon_opener_is_cut_at_a_word_boundary();
        test_terminal_detection_for_truncated_tails();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_segmenter_test passed\n";
    return 0;
}
