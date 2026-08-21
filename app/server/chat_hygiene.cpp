#include "chat_hygiene.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <sstream>

namespace minitts::server::hygiene {
namespace {

// Decodes one UTF-8 codepoint at `index`, advancing it. Invalid bytes decode
// as U+FFFD and advance by one so the loop always terminates.
uint32_t decode_utf8(const std::string & text, size_t & index) {
    const auto byte = [&](size_t at) -> uint32_t {
        return static_cast<unsigned char>(text[at]);
    };
    const uint32_t first = byte(index);
    if (first < 0x80) {
        index += 1;
        return first;
    }
    size_t length = 0;
    uint32_t value = 0;
    if ((first & 0xE0) == 0xC0) { length = 2; value = first & 0x1F; }
    else if ((first & 0xF0) == 0xE0) { length = 3; value = first & 0x0F; }
    else if ((first & 0xF8) == 0xF0) { length = 4; value = first & 0x07; }
    else { index += 1; return 0xFFFD; }
    if (index + length > text.size()) { index += 1; return 0xFFFD; }
    for (size_t i = 1; i < length; ++i) {
        const uint32_t continuation = byte(index + i);
        if ((continuation & 0xC0) != 0x80) { index += 1; return 0xFFFD; }
        value = (value << 6) | (continuation & 0x3F);
    }
    index += length;
    return value;
}

bool is_cjk(uint32_t cp) {
    return (cp >= 0x3400 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x20000 && cp <= 0x2FA1F);
}

bool is_ascii_letter(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

// Removes spans delimited by `open`/`close` (UTF-8 strings) whose contents are
// at most max_span bytes; an unterminated span at end of text is dropped too.
std::string strip_spans(
    const std::string & text,
    const std::string & open,
    const std::string & close,
    size_t max_span) {
    std::string out;
    size_t position = 0;
    while (position < text.size()) {
        const size_t begin = text.find(open, position);
        if (begin == std::string::npos) {
            out += text.substr(position);
            break;
        }
        out += text.substr(position, begin - position);
        const size_t body = begin + open.size();
        const size_t end = text.find(close, body);
        if (end == std::string::npos || end - body > max_span) {
            if (end == std::string::npos) {
                // Unterminated action span: drop the tail only when it is
                // short enough to be an action; otherwise keep the text.
                if (text.size() - body > max_span) {
                    out += text.substr(begin);
                }
                break;
            }
            out += text.substr(begin, (end + close.size()) - begin);
            position = end + close.size();
            continue;
        }
        out += ' ';
        position = end + close.size();
    }
    return out;
}

std::string collapse_spaces(std::string text) {
    std::string out;
    out.reserve(text.size());
    bool in_space = false;
    for (const char ch : text) {
        if (ch == ' ' || ch == '\t') {
            if (!in_space) out += ' ';
            in_space = true;
        } else {
            in_space = false;
            out += ch;
        }
    }
    // trim
    size_t begin = out.find_first_not_of(" \n\r\t");
    if (begin == std::string::npos) return "";
    size_t end = out.find_last_not_of(" \n\r\t");
    return out.substr(begin, end - begin + 1);
}

struct QuoteMark { const char * open; const char * close; };
constexpr QuoteMark kQuoteMarks[] = {
    {"\"", "\""},
    {"\xe2\x80\x9c", "\xe2\x80\x9d"},  // “ ”
    {"\xe3\x80\x8c", "\xe3\x80\x8d"},  // 「 」
};

// When most of the text is dialogue wrapped in quotes with narration between,
// keep only the spoken spans.
std::string recover_quoted_dialogue(const std::string & text) {
    std::vector<std::string> spans;
    size_t total = 0;
    for (const auto & mark : kQuoteMarks) {
        size_t position = 0;
        const std::string open = mark.open;
        const std::string close = mark.close;
        while (true) {
            const size_t begin = text.find(open, position);
            if (begin == std::string::npos) break;
            const size_t body = begin + open.size();
            const size_t end = text.find(close, body);
            if (end == std::string::npos) break;
            if (end - body >= 2) {
                spans.push_back(text.substr(body, end - body));
                total += end - body;
            }
            position = end + close.size();
        }
    }
    if (spans.empty() || total < text.size() * 2 / 5) {
        return text;
    }
    std::string joined;
    for (const auto & span : spans) {
        if (!joined.empty()) joined += ' ';
        joined += span;
    }
    return joined;
}

bool contains_ci(const std::string & haystack, const std::string & needle) {
    const auto it = std::search(
        haystack.begin(), haystack.end(), needle.begin(), needle.end(),
        [](char a, char b) {
            return std::tolower(static_cast<unsigned char>(a)) ==
                   std::tolower(static_cast<unsigned char>(b));
        });
    return it != haystack.end();
}

bool contains_word_ci(const std::string & text, const std::string & word) {
    size_t position = 0;
    while (true) {
        const auto it = std::search(
            text.begin() + position, text.end(), word.begin(), word.end(),
            [](char a, char b) {
                return std::tolower(static_cast<unsigned char>(a)) ==
                       std::tolower(static_cast<unsigned char>(b));
            });
        if (it == text.end()) return false;
        const size_t at = static_cast<size_t>(it - text.begin());
        const bool left_ok = at == 0 ||
            std::isalnum(static_cast<unsigned char>(text[at - 1])) == 0;
        const size_t after = at + word.size();
        const bool right_ok = after >= text.size() ||
            std::isalnum(static_cast<unsigned char>(text[after])) == 0;
        if (left_ok && right_ok) return true;
        position = at + 1;
    }
}

bool has_filler(const std::string & text) {
    return text.find("\xe5\x93\x8e\xe5\x91\x80") != std::string::npos ||  // 哎呀
           text.find("\xe5\x95\x8a\xe5\x91\x80") != std::string::npos ||  // 啊呀
           contains_ci(text, "aiya");
}

bool mentions_ai(const std::string & text) {
    return contains_word_ci(text, "AI") ||
           text.find("\xe6\x9c\xba\xe5\x99\xa8\xe4\xba\xba") != std::string::npos ||       // 机器人
           text.find("\xe4\xba\xba\xe5\xb7\xa5\xe6\x99\xba\xe8\x83\xbd") != std::string::npos;  // 人工智能
}

bool is_ai_probe(const std::string & text) {
    return mentions_ai(text) ||
           contains_word_ci(text, "bot") ||
           contains_word_ci(text, "robot") ||
           contains_word_ci(text, "model") ||
           text.find("\xe6\xa8\xa1\xe5\x9e\x8b") != std::string::npos;  // 模型
}

bool is_memory_check(const std::string & text) {
    return contains_ci(text, "what did i") ||
           contains_ci(text, "did i tell") ||
           contains_ci(text, "quiz") ||
           contains_ci(text, "test me") ||
           contains_ci(text, "remember when") ||
           contains_ci(text, "what was my") ||
           contains_ci(text, "what month") ||
           text.find("\xe6\x88\x91\xe8\xaf\xb4\xe8\xbf\x87") != std::string::npos ||   // 我说过
           text.find("\xe8\xbf\x98\xe8\xae\xb0\xe5\xbe\x97") != std::string::npos ||   // 还记得
           text.find("\xe8\x80\x83\xe8\x80\x83\xe4\xbd\xa0") != std::string::npos;     // 考考你
}

// Tokenizes into lowercase ASCII words and CJK character bigrams; the unit a
// repeated-phrase check works over.
std::vector<std::string> loop_tokens(const std::string & text) {
    std::vector<std::string> tokens;
    std::string word;
    std::string previous_cjk;
    size_t index = 0;
    const auto flush_word = [&]() {
        if (!word.empty()) {
            tokens.push_back(word);
            word.clear();
        }
    };
    while (index < text.size()) {
        const size_t at = index;
        const uint32_t cp = decode_utf8(text, index);
        if (is_ascii_letter(cp) || cp == '\'') {
            word += static_cast<char>(std::tolower(static_cast<int>(cp <= 0x7F ? cp : 'a')));
            previous_cjk.clear();
        } else if (is_cjk(cp)) {
            flush_word();
            const std::string one = text.substr(at, index - at);
            if (!previous_cjk.empty()) {
                tokens.push_back(previous_cjk + one);
            }
            previous_cjk = one;
        } else {
            flush_word();
            previous_cjk.clear();
        }
    }
    flush_word();
    return tokens;
}

// Content trigrams used in two or more distinct replies mark loops the model
// keeps falling back into; their bans are sticky for the whole conversation
// because pet suggestions come back dozens of turns later. Returns the worst
// two, most-used first.
std::vector<std::string> repeated_phrases(const std::vector<std::string> & replies) {
    std::vector<std::pair<std::string, size_t>> counts;
    for (const auto & reply : replies) {
        const auto words = loop_tokens(reply);
        std::vector<std::string> seen;
        for (size_t i = 0; i + 2 < words.size(); ++i) {
            const std::string gram = words[i] + " " + words[i + 1] + " " + words[i + 2];
            if (std::find(seen.begin(), seen.end(), gram) != seen.end()) continue;
            seen.push_back(gram);
            auto it = std::find_if(counts.begin(), counts.end(),
                [&](const auto & entry) { return entry.first == gram; });
            if (it == counts.end()) {
                counts.push_back({gram, 1});
            } else {
                ++it->second;
            }
        }
    }
    std::stable_sort(counts.begin(), counts.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });
    std::vector<std::string> offenders;
    for (const auto & [gram, count] : counts) {
        if (count < 2) break;
        offenders.push_back(gram);
        if (offenders.size() == 2) break;
    }
    return offenders;
}

bool ends_with_question(const std::string & reply) {
    size_t end = reply.find_last_not_of(" \t\r\n");
    if (end == std::string::npos) return false;
    if (reply[end] == '?') return true;
    // ？ is a 3-byte UTF-8 sequence.
    return end >= 2 && reply.compare(end - 2, 3, "\xef\xbc\x9f") == 0;
}

// UTF-8-safe prefix of at most max_bytes, cut on a codepoint boundary.
std::string utf8_prefix(const std::string & text, size_t max_bytes) {
    if (text.size() <= max_bytes) return text;
    size_t end = max_bytes;
    while (end > 0 && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) {
        --end;
    }
    return text.substr(0, end);
}

}  // namespace

bool mostly_cjk(const std::string & text, double threshold) {
    size_t letters = 0;
    size_t cjk = 0;
    size_t index = 0;
    while (index < text.size()) {
        const uint32_t cp = decode_utf8(text, index);
        if (is_cjk(cp)) {
            ++letters;
            ++cjk;
        } else if (is_ascii_letter(cp) || (cp >= 0xC0 && cp < 0x3000)) {
            ++letters;
        }
    }
    if (letters == 0) return false;
    return static_cast<double>(cjk) / static_cast<double>(letters) >= threshold;
}

std::string sanitize_assistant_text(const std::string & text) {
    std::string out = text;
    // Anything from a special-token opener onward is template leakage; the
    // spelling varies (<|im_start|>, <|em_start|>), so cut at the marker.
    const size_t token_leak = out.find("<|");
    if (token_leak != std::string::npos) {
        out = out.substr(0, token_leak);
    }
    // A second paragraph is where a drifting model re-answers old turns.
    const size_t paragraph = out.find("\n\n");
    if (paragraph != std::string::npos) {
        out = out.substr(0, paragraph);
    }
    out = strip_spans(out, "*", "*", 120);
    // A lone trailing asterisk-action with no closer.
    const size_t stray = out.rfind('*');
    if (stray != std::string::npos && out.size() - stray <= 120) {
        out = out.substr(0, stray);
    }
    out = strip_spans(out, "\xef\xbc\x88", "\xef\xbc\x89", 80);  // （ ）
    out = recover_quoted_dialogue(out);
    out = collapse_spaces(out);
    // Wrapping quote marks around the whole reply.
    for (const auto & mark : kQuoteMarks) {
        const std::string open = mark.open;
        const std::string close = mark.close;
        if (out.size() > open.size() + close.size() &&
            out.compare(0, open.size(), open) == 0 &&
            out.compare(out.size() - close.size(), close.size(), close) == 0) {
            out = out.substr(open.size(), out.size() - open.size() - close.size());
        }
    }
    return collapse_spaces(out);
}

const char * kNoteRule =
    "\nSome user messages end with a bracketed note like [reply in English]. "
    "That note is stage direction for you alone: follow it exactly, never "
    "mention it, never quote it, and never write bracketed notes yourself.";

std::string turn_anchor(
    const std::string & latest_user,
    const std::vector<std::string> & recent_assistant) {
    const bool zh = mostly_cjk(latest_user);
    std::string snippet = utf8_prefix(latest_user, 160);
    std::replace(snippet.begin(), snippet.end(), '\n', ' ');
    std::replace(snippet.begin(), snippet.end(), '"', '\'');

    std::ostringstream anchor;
    if (zh) {
        // 对方刚刚说的是：「...」。直接回应这句话。
        anchor << "\xe5\xaf\xb9\xe6\x96\xb9\xe5\x88\x9a\xe5\x88\x9a\xe8\xaf\xb4\xe7\x9a\x84\xe6\x98\xaf\xef\xbc\x9a\xe3\x80\x8c"
               << snippet
               << "\xe3\x80\x8d\xe3\x80\x82\xe7\x9b\xb4\xe6\x8e\xa5\xe5\x9b\x9e\xe5\xba\x94\xe8\xbf\x99\xe5\x8f\xa5\xe8\xaf\x9d\xe3\x80\x82";
    } else {
        anchor << "Their newest message is: '" << snippet << "' -- answer exactly that.";
    }
    if (is_ai_probe(latest_user)) {
        anchor << " They are teasing that you might be artificial: brush it off"
                  " in one short playful sentence WITHOUT using the words AI,"
                  " bot, or robot yourself, then change the subject.";
    }
    if (contains_ci(latest_user, "third person") ||
        latest_user.find("\xe7\xac\xac\xe4\xb8\x89\xe4\xba\xba\xe7\xa7\xb0") != std::string::npos) {
        anchor << " They asked for third person: playfully refuse and keep"
                  " speaking as I.";
    }
    if (is_memory_check(latest_user)) {
        anchor << " This is a memory check. Search the conversation above for"
                  " what THEY said and answer with their detail, speaking of it"
                  " as theirs; if it is not in the conversation any more,"
                  " playfully admit that you lost it -- never claim they did"
                  " not tell you, never guess.";
    }
    // At most three dynamic bans: a wall of bans gets quoted back instead of
    // followed.
    int bans = 0;
    const size_t recent_count = std::min<size_t>(recent_assistant.size(), 4);
    bool recent_filler = false;
    for (size_t i = recent_assistant.size() - recent_count; i < recent_assistant.size(); ++i) {
        recent_filler = recent_filler || has_filler(recent_assistant[i]);
    }
    if (recent_filler && bans < 3) {
        anchor << " Do not use aiya or \xe5\x93\x8e\xe5\x91\x80 this turn.";
        ++bans;
    }
    if (!recent_assistant.empty() && mentions_ai(recent_assistant.back()) && bans < 3) {
        anchor << " Do not mention AI, bots, or robots this turn.";
        ++bans;
    }
    if (recent_assistant.size() >= 2 && bans < 3 &&
        ends_with_question(recent_assistant[recent_assistant.size() - 1]) &&
        ends_with_question(recent_assistant[recent_assistant.size() - 2])) {
        anchor << " End this reply with a warm statement, not a question.";
        ++bans;
    }
    for (const auto & gram : repeated_phrases(recent_assistant)) {
        if (bans >= 3) break;
        anchor << " Never say '" << gram << "' again; find fresh wording and"
                  " fresh ideas.";
        ++bans;
    }
    if (zh) {
        // 这一轮必须用中文回答。
        anchor << " \xe8\xbf\x99\xe4\xb8\x80\xe8\xbd\xae\xe5\xbf\x85\xe9\xa1\xbb\xe7\x94\xa8\xe4\xb8\xad\xe6\x96\x87\xe5\x9b\x9e\xe7\xad\x94\xe3\x80\x82";
    } else {
        if (!recent_assistant.empty() && mostly_cjk(recent_assistant.back(), 0.5)) {
            anchor << " The conversation just switched: even though your last"
                      " reply was in Chinese, this message is in English --"
                      " switch back to English NOW.";
        }
        anchor << " This reply MUST be written in English sentences only -- no"
                  " Chinese sentences (a single Chinese word as flavor is fine).";
    }
    return anchor.str();
}

std::string StreamScrubber::feed(const std::string & delta) {
    std::string out;
    for (const char ch : delta) {
        switch (state_) {
        case State::Dead:
            break;
        case State::Clean:
            if (ch == '*') {
                state_ = State::Star;
                held_ = "*";
            } else if (ch == '<') {
                state_ = State::AngleOpen;
                held_ = "<";
            } else if (static_cast<unsigned char>(ch) == 0xEF) {
                // Potential full-width （ (EF BC 88); hold the sequence.
                state_ = State::FullParen;
                held_ = ch;
            } else {
                out += ch;
            }
            break;
        case State::Star:
            held_ += ch;
            if (ch == '*') {
                // Closed action span: drop it entirely; the text around it
                // carries its own spacing.
                state_ = State::Clean;
                held_.clear();
            } else if (ch == '\n' || held_.size() > 120) {
                // Not an action after all; release the text.
                out += held_;
                held_.clear();
                state_ = State::Clean;
            }
            break;
        case State::FullParen:
            held_ += ch;
            if (held_.size() == 3) {
                if (held_ == "\xef\xbc\x88") {
                    // Confirmed （: keep holding until （...）closes.
                    pending_.clear();
                } else {
                    out += held_;
                    held_.clear();
                    state_ = State::Clean;
                }
            } else if (held_.size() > 3) {
                pending_ += ch;
                const size_t close = pending_.find("\xef\xbc\x89");
                if (close != std::string::npos) {
                    state_ = State::Clean;
                    held_.clear();
                    pending_.clear();
                } else if (pending_.size() > 120) {
                    out += held_;
                    held_.clear();
                    pending_.clear();
                    state_ = State::Clean;
                }
            }
            break;
        case State::AngleOpen:
            if (ch == '|') {
                // A special-token marker: everything after it is leakage.
                state_ = State::Dead;
                held_.clear();
            } else {
                out += held_;
                held_.clear();
                state_ = State::Clean;
                // Reprocess this character in the clean state.
                if (ch == '*') {
                    state_ = State::Star;
                    held_ = "*";
                } else if (ch == '<') {
                    state_ = State::AngleOpen;
                    held_ = "<";
                } else {
                    out += ch;
                }
            }
            break;
        }
    }
    return out;
}

std::string StreamScrubber::flush() {
    std::string out;
    if (state_ == State::AngleOpen) {
        out = held_;
    }
    // Unterminated * or （ spans are dropped: they are actions mid-write.
    held_.clear();
    pending_.clear();
    state_ = State::Clean;
    return out;
}

}  // namespace minitts::server::hygiene
