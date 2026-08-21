#pragma once

#include <string>
#include <vector>

namespace minitts::server {

// Turns an LLM token stream into speakable segments for the TTS pipeline.
//
// The unit of speech is the sentence: synthesizing mid-sentence fragments ruins
// prosody, but waiting for the whole reply wastes the seconds the model spends
// writing sentence two while sentence one could already be playing. So segments
// are emitted the moment a sentence completes -- with one deliberate exception:
// the FIRST segment may be cut early at a clause boundary once enough text has
// accumulated, trading a little prosody on the opening clause for a much
// earlier first audio. Time-to-first-audio is set by the first segment alone;
// every later segment is synthesized while the previous one is still playing,
// so only the first one is worth compromising for.
class SentenceSegmenter {
public:
    struct Options {
        // Once the pending first segment reaches this many bytes, a clause
        // boundary (comma, semicolon, dash) is good enough to cut at.
        size_t first_segment_min_chars = 60;
        // Safety valve: a run-on with no boundaries at all is force-split at
        // the last space past this size rather than buffered forever.
        size_t max_segment_chars = 300;
    };

    SentenceSegmenter() = default;
    explicit SentenceSegmenter(Options options) : options_(options) {}

    // Appends one token delta; returns every segment completed by it, in order.
    std::vector<std::string> feed(const std::string & delta);

    // Returns whatever is still pending at end of stream, or empty.
    std::string flush();

private:
    std::vector<std::string> take_ready();

    Options options_;
    std::string pending_;
    bool first_segment_emitted_ = false;
};

// Strips markup that reads badly aloud -- roleplay asterisk actions and
// markdown emphasis -- while leaving the text itself intact. The display copy
// keeps the original; only the TTS input goes through this.
std::string strip_speech_markup(const std::string & text);

}  // namespace minitts::server
