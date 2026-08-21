#include "segmenter.h"

#include <algorithm>
#include <cctype>

namespace minitts::server {
namespace {

bool is_space_byte(char ch) {
    return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

// True when text[index] ends a sentence. ASCII enders need a following space,
// quote, or end-of-buffer so "3.5" or "Dr." mid-word does not split; CJK
// enders are unambiguous single characters (multi-byte UTF-8 sequences).
size_t sentence_end_length(const std::string & text, size_t index) {
    const char ch = text[index];
    if (ch == '.' || ch == '!' || ch == '?') {
        // Swallow runs like "?!" or "..." as one ending.
        size_t end = index;
        while (end + 1 < text.size() &&
            (text[end + 1] == '.' || text[end + 1] == '!' || text[end + 1] == '?')) {
            ++end;
        }
        const size_t after = end + 1;
        if (after >= text.size()) {
            return 0;  // Might still be growing ("..." could continue) -- wait.
        }
        if (is_space_byte(text[after]) || text[after] == '"' || text[after] == '\'' ||
            text[after] == ')' || text[after] == '*') {
            return end - index + 1;
        }
        return 0;
    }
    if (ch == '\n') {
        return 1;
    }
    // CJK sentence enders (UTF-8): 。 ！ ？ ； and ellipsis …
    static const char * kCjk[] = {"\xe3\x80\x82", "\xef\xbc\x81", "\xef\xbc\x9f", "\xef\xbc\x9b", "\xe2\x80\xa6"};
    for (const char * ender : kCjk) {
        const size_t len = 3;
        if (index + len <= text.size() && text.compare(index, len, ender) == 0) {
            return len;
        }
    }
    return 0;
}

bool is_clause_break(const std::string & text, size_t index) {
    const char ch = text[index];
    if (ch == ',' || ch == ';' || ch == ':') {
        return index + 1 >= text.size() || is_space_byte(text[index + 1]);
    }
    // CJK comma 、 and ，
    static const char * kCjk[] = {"\xe3\x80\x81", "\xef\xbc\x8c"};
    for (const char * br : kCjk) {
        if (index + 3 <= text.size() && text.compare(index, 3, br) == 0) {
            return true;
        }
    }
    return false;
}

std::string trimmed(std::string text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && is_space_byte(text[begin])) {
        ++begin;
    }
    while (end > begin && is_space_byte(text[end - 1])) {
        --end;
    }
    return text.substr(begin, end - begin);
}

}  // namespace

std::vector<std::string> SentenceSegmenter::feed(const std::string & delta) {
    pending_ += delta;
    return take_ready();
}

std::vector<std::string> SentenceSegmenter::take_ready() {
    std::vector<std::string> ready;
    while (true) {
        size_t cut = std::string::npos;

        // A completed sentence always cuts, wherever it is.
        for (size_t index = 0; index < pending_.size(); ++index) {
            const size_t len = sentence_end_length(pending_, index);
            if (len > 0) {
                cut = index + len;
                break;
            }
        }

        // Before the first emission, a clause boundary is taken instead of
        // waiting for the period: it is the difference between first audio at
        // "...and so," and first audio a full sentence later. The largest
        // clause prefix available is used, with a floor so a bare "Hi," is not
        // synthesized alone.
        constexpr size_t kMinClauseChars = 14;
        if (cut == std::string::npos && !first_segment_emitted_ &&
            pending_.size() >= options_.first_segment_min_chars) {
            for (size_t index = pending_.size(); index-- > kMinClauseChars;) {
                if (is_clause_break(pending_, index)) {
                    const char ch = pending_[index];
                    cut = index + ((ch == ',' || ch == ';' || ch == ':') ? 1 : 3);
                    break;
                }
            }
        }

        // Still nothing and the opener keeps running: take a word boundary.
        // A breath mid-clause costs less than another half second of silence
        // before the character says anything at all.
        if (cut == std::string::npos && !first_segment_emitted_ &&
            pending_.size() >= options_.first_segment_word_cut_chars) {
            const size_t space = pending_.rfind(' ');
            if (space != std::string::npos && space >= kMinClauseChars) {
                cut = space;
            }
        }

        // Force-split a boundary-free run-on at its last space.
        if (cut == std::string::npos && pending_.size() >= options_.max_segment_chars) {
            const size_t space = pending_.rfind(' ', options_.max_segment_chars);
            if (space != std::string::npos && space > 0) {
                cut = space;
            } else {
                cut = options_.max_segment_chars;
            }
        }

        if (cut == std::string::npos) {
            return ready;
        }
        auto segment = trimmed(pending_.substr(0, cut));
        pending_.erase(0, cut);
        if (!segment.empty()) {
            ready.push_back(std::move(segment));
            first_segment_emitted_ = true;
        }
    }
}

std::string SentenceSegmenter::flush() {
    auto segment = trimmed(pending_);
    pending_.clear();
    return segment;
}

bool ends_with_sentence_terminal(const std::string & text) {
    size_t end = text.size();
    while (end > 0) {
        const char ch = text[end - 1];
        if (is_space_byte(ch) || ch == '"' || ch == '\'' || ch == ')' || ch == '*') {
            --end;
            continue;
        }
        break;
    }
    if (end == 0) {
        return false;
    }
    const char last = text[end - 1];
    if (last == '.' || last == '!' || last == '?') {
        return true;
    }
    // CJK enders are 3-byte UTF-8 sequences.
    if (end >= 3) {
        static const char * kCjk[] = {"\xe3\x80\x82", "\xef\xbc\x81", "\xef\xbc\x9f", "\xe2\x80\xa6"};
        for (const char * ender : kCjk) {
            if (text.compare(end - 3, 3, ender) == 0) {
                return true;
            }
        }
    }
    return false;
}

std::string strip_speech_markup(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    bool in_action = false;
    for (size_t index = 0; index < text.size(); ++index) {
        const char ch = text[index];
        if (ch == '*') {
            // Roleplay convention: *actions* are stage directions, not speech.
            // Toggling on '*' both drops the emphasis markers of *word* and
            // skips the whole action body.
            in_action = !in_action;
            continue;
        }
        if (in_action) {
            continue;
        }
        if (ch == '#' || ch == '`') {
            continue;  // Markdown noise.
        }
        out += ch;
    }
    // An unbalanced '*' would have swallowed real speech; better to read the
    // markup than to silence half the reply.
    if (in_action) {
        size_t kept = 0;
        for (const char ch : text) {
            if (ch != '*' && ch != '#' && ch != '`') {
                ++kept;
            }
        }
        if (out.size() * 2 < kept) {
            out.clear();
            for (const char ch : text) {
                if (ch != '*' && ch != '#' && ch != '`') {
                    out += ch;
                }
            }
        }
    }
    return trimmed(out);
}

}  // namespace minitts::server
