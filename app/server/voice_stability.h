#pragma once

#include <string>
#include <vector>

namespace minitts::server::voice {

// ---------------------------------------------------------------------------
// Stable-prefix tracking.
//
// Streaming ASR re-decodes the growing utterance, so consecutive hypotheses
// may revise each other ("turn the lights" -> "turn the light" -> "turn the
// lights in the bedroom"). Downstream consumers (display, speculative LLM
// prefill) need a prefix that has stopped moving, separate from the tail that
// is still in flux. A token survives into the stable prefix once it has been
// identical -- ignoring case and punctuation -- across the last
// `stable_hypothesis_count` hypotheses; the committed prefix only ever grows.
// ---------------------------------------------------------------------------

struct StabilityParams {
    // Consecutive hypotheses a token must agree across before it commits.
    int stable_hypothesis_count = 2;
    // Never commit anything shorter than this many characters.
    int min_stable_chars = 2;
};

class StableTracker {
public:
    explicit StableTracker(StabilityParams params = {});

    void reset();

    // Feed the newest full-utterance hypothesis. Returns true when the
    // committed prefix grew.
    bool update(const std::string & hypothesis);

    // The committed prefix, in the newest hypothesis's raw spelling.
    const std::string & stable() const noexcept { return stable_; }
    // The newest hypothesis past the committed prefix.
    const std::string & tentative() const noexcept { return tentative_; }
    // The newest full hypothesis.
    const std::string & hypothesis() const noexcept { return hypothesis_; }

private:
    struct Token {
        std::string normalized;
        size_t raw_end = 0;  // byte offset past this token in the raw text
    };
    static std::vector<Token> tokenize(const std::string & text);

    StabilityParams params_;
    std::vector<std::vector<Token>> history_;
    size_t committed_tokens_ = 0;
    std::string stable_;
    std::string tentative_;
    std::string hypothesis_;
};

}  // namespace minitts::server::voice
