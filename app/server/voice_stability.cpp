#include "voice_stability.h"

#include <algorithm>
#include <cctype>

namespace minitts::server::voice {

namespace {

bool is_cjk_lead(unsigned char byte) {
    // Three-byte UTF-8 leads covering the CJK blocks this app speaks.
    return byte >= 0xE3 && byte <= 0xE9;
}

}  // namespace

StableTracker::StableTracker(StabilityParams params) : params_(params) {
    if (params_.stable_hypothesis_count < 1) {
        params_.stable_hypothesis_count = 1;
    }
}

void StableTracker::reset() {
    history_.clear();
    committed_tokens_ = 0;
    stable_.clear();
    tentative_.clear();
    hypothesis_.clear();
}

std::vector<StableTracker::Token> StableTracker::tokenize(const std::string & text) {
    // ASCII words tokenize per whitespace run with case and punctuation
    // stripped, so "Lights," and "lights" agree; CJK has no whitespace, so
    // each character is its own token. Punctuation-only runs vanish -- a
    // hypothesis that only re-punctuates must not un-stabilize the prefix.
    std::vector<Token> tokens;
    std::string current;
    size_t i = 0;
    auto flush = [&](size_t end) {
        if (!current.empty()) {
            tokens.push_back(Token{current, end});
            current.clear();
        }
    };
    while (i < text.size()) {
        const unsigned char byte = static_cast<unsigned char>(text[i]);
        if (byte < 0x80) {
            if (std::isspace(byte)) {
                flush(i);
                ++i;
                continue;
            }
            if (std::isalnum(byte) || byte == '\'') {
                current.push_back(static_cast<char>(std::tolower(byte)));
            }
            // ASCII punctuation: ignored for agreement, but it still ends up
            // inside the raw span of the token it trails.
            ++i;
            if (i == text.size() || std::isspace(static_cast<unsigned char>(text[i]))) {
                flush(i);
            }
            continue;
        }
        const size_t len = (byte & 0xF8) == 0xF0 ? 4 : (byte & 0xF0) == 0xE0 ? 3 : (byte & 0xE0) == 0xC0 ? 2 : 1;
        const size_t end = std::min(text.size(), i + len);
        if (len == 3 && is_cjk_lead(byte)) {
            flush(i);
            const std::string ch = text.substr(i, end - i);
            // The CJK symbol/punctuation block (、。「」...) lives at 0xE3 0x80;
            // ideographs become tokens, punctuation is skipped.
            if (static_cast<unsigned char>(ch[0]) == 0xE3 && static_cast<unsigned char>(ch[1]) == 0x80) {
                i = end;
                continue;
            }
            tokens.push_back(Token{ch, end});
            i = end;
            continue;
        }
        // Fullwidth punctuation (！？，… in 0xEF): ends the current token and
        // is otherwise ignored for agreement.
        if (len == 3 && static_cast<unsigned char>(byte) == 0xEF) {
            flush(i);
            i = end;
            continue;
        }
        // Other multibyte (accented latin etc.): part of the current word.
        current += text.substr(i, end - i);
        i = end;
        if (i == text.size()) {
            flush(i);
        }
    }
    flush(text.size());
    return tokens;
}

bool StableTracker::update(const std::string & hypothesis) {
    hypothesis_ = hypothesis;
    auto tokens = tokenize(hypothesis);
    history_.push_back(tokens);
    while (history_.size() > static_cast<size_t>(params_.stable_hypothesis_count)) {
        history_.erase(history_.begin());
    }

    size_t agreed = committed_tokens_;
    if (history_.size() == static_cast<size_t>(params_.stable_hypothesis_count)) {
        // Longest token prefix identical across every retained hypothesis.
        size_t limit = history_.front().size();
        for (const auto & h : history_) {
            limit = std::min(limit, h.size());
        }
        size_t common = 0;
        while (common < limit) {
            const auto & first = history_.front()[common].normalized;
            bool all = true;
            for (const auto & h : history_) {
                if (h[common].normalized != first) {
                    all = false;
                    break;
                }
            }
            if (!all) {
                break;
            }
            ++common;
        }
        agreed = std::max(agreed, common);
    }

    const bool grew = agreed > committed_tokens_;
    committed_tokens_ = agreed;

    // Express the committed prefix in the NEWEST hypothesis's raw spelling.
    // If the newest hypothesis rolled back below the commit point, keep the
    // previous stable text -- commitment is a promise to the consumer.
    if (committed_tokens_ > 0 && tokens.size() >= committed_tokens_) {
        const size_t raw_end = tokens[committed_tokens_ - 1].raw_end;
        if (raw_end >= static_cast<size_t>(params_.min_stable_chars)) {
            stable_ = hypothesis.substr(0, raw_end);
            tentative_ = hypothesis.substr(std::min(hypothesis.size(), raw_end));
            while (!tentative_.empty() && tentative_.front() == ' ') {
                tentative_.erase(tentative_.begin());
            }
            return grew;
        }
    }
    if (stable_.empty()) {
        tentative_ = hypothesis;
    }
    return false;
}

}  // namespace minitts::server::voice
