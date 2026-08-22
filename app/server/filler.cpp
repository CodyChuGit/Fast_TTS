#include "filler.h"

#include <algorithm>
#include <cctype>

namespace minitts::server::filler {

const std::vector<Entry> & library() {
    // Written to be speakable and character-neutral: hesitations, not
    // content. Short covers a prewarmed miss (~0.5-0.8 s of air), medium a
    // typical cold turn, long a fresh conversation's first exchange. Kinds
    // let the pick answer the message: React to excitement, Think at a
    // question, Ack at a statement. Append new entries at the END --
    // scripts/filler_qa.py maps cache files to entries by index.
    static const std::vector<Entry> entries = {
        // English, short
        {Size::Short, false, Kind::Ack, "Mmm."},
        {Size::Short, false, Kind::Think, "Uhh..."},
        {Size::Short, false, Kind::React, "Ooh.", false},
        {Size::Short, false, Kind::React, "Hmm?", false},
        {Size::Short, false, Kind::React, "Oh!", false},
        {Size::Short, false, Kind::Ack, "Right..."},
        // English, medium
        {Size::Medium, false, Kind::Ack, "Ummm, okay so..."},
        {Size::Medium, false, Kind::Think, "Hmm, let me think..."},
        {Size::Medium, false, Kind::React, "Oh, that? Well..."},
        {Size::Medium, false, Kind::Ack, "Yeah, yeah, okay..."},
        {Size::Medium, false, Kind::Think, "Mmm, good question..."},
        {Size::Medium, false, Kind::Think, "Wait, let me see..."},
        // English, long
        {Size::Long, false, Kind::Think, "Okay okay, hold on... let me actually think about that for a second..."},
        {Size::Long, false, Kind::Think, "Hmm... that's actually such a good question, give me a sec..."},
        {Size::Long, false, Kind::React, "Ooh, um, okay... so, how do I put this..."},
        {Size::Long, false, Kind::Think, "Mmm, wait wait... okay, I think I know what I want to say..."},
        // Chinese, short
        {Size::Short, true, Kind::Ack, "嗯…"},
        {Size::Short, true, Kind::React, "哦？", false},
        {Size::Short, true, Kind::Think, "呃…"},
        {Size::Short, true, Kind::Ack, "嗯哼。"},
        {Size::Short, true, Kind::React, "哦——", false},
        // Chinese, medium
        {Size::Medium, true, Kind::Think, "嗯…让我想想哦…"},
        {Size::Medium, true, Kind::React, "哎呀，这个嘛…"},
        {Size::Medium, true, Kind::React, "哦，那个呀…"},
        {Size::Medium, true, Kind::Think, "嗯嗯，好问题…"},
        {Size::Medium, true, Kind::Think, "等一下哦,我想想…"},
        // Chinese, long
        {Size::Long, true, Kind::Think, "哎呀等一下等一下,让我好好想一想这个问题哦…"},
        {Size::Long, true, Kind::Think, "嗯——这个问题还挺有意思的,让我想想看…"},
        {Size::Long, true, Kind::Think, "唔,怎么说呢…让我组织一下语言哈…"},
        // Appended after the first build (new entries synthesize on boot).
        {Size::Medium, true, Kind::Ack, "嗯嗯，这样啊…"},
    };
    return entries;
}

namespace {

uint64_t fnv1a(uint64_t seed, const void * data, size_t size) {
    uint64_t hash = seed ? seed : 1469598103934665603ull;
    const auto * bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

bool contains_any(const std::string & text, const std::vector<const char *> & needles) {
    for (const auto * needle : needles) {
        if (text.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool starts_with_word(const std::string & text, const std::vector<const char *> & words) {
    for (const auto * word : words) {
        const size_t n = std::char_traits<char>::length(word);
        if (text.size() > n && text.compare(0, n, word) == 0 &&
            (text[n] == ' ' || text[n] == '\'')) {
            return true;
        }
    }
    return false;
}

}  // namespace

uint64_t clip_key(uint64_t voice_fingerprint, const std::string & text) {
    return fnv1a(voice_fingerprint, text.data(), text.size());
}

Kind classify(const std::string & user_text) {
    std::string lower;
    lower.reserve(user_text.size());
    for (const char c : user_text) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    // Questions want thinking, and they outrank excitement: an excited
    // question ("whoa really? why...") is still a question.
    const bool question = lower.find('?') != std::string::npos ||
        user_text.find("？") != std::string::npos ||
        contains_any(user_text, {"吗", "什么", "怎么", "为什么", "哪", "多少", "谁"}) ||
        starts_with_word(lower, {"what", "why", "how", "when", "where", "who", "which",
                                 "can", "could", "would", "should", "do", "does", "did",
                                 "is", "are", "will", "have", "has"});
    if (question) {
        return Kind::Think;
    }
    // A long message is a lot to take in even without a question mark;
    // thinking reads more attentive than a flat acknowledgment.
    if (user_text.size() > 120) {
        return Kind::Think;
    }
    const bool excited = lower.find('!') != std::string::npos ||
        user_text.find("！") != std::string::npos ||
        contains_any(lower, {"wow", "omg", "oh my", "no way", "guess what", "crazy",
                             "insane", "unbelievable"}) ||
        contains_any(user_text, {"哇", "天啊", "天哪", "我的天", "没想到", "居然"});
    if (excited) {
        return Kind::React;
    }
    return Kind::Ack;
}

std::vector<int> candidates(Size size, bool chinese, Kind kind, bool chain_only) {
    const auto & entries = library();
    std::vector<int> pool;
    std::vector<int> matched;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto & entry = entries[i];
        if (entry.size != size || entry.chinese != chinese || (chain_only && !entry.chain)) {
            continue;
        }
        pool.push_back(static_cast<int>(i));
        if (entry.kind == kind) {
            matched.push_back(static_cast<int>(i));
        }
    }
    return matched.empty() ? pool : matched;
}

uint64_t mix(uint64_t seed) {
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdull;
    seed ^= seed >> 33;
    return seed;
}

std::string strip_leading(std::string text) {
    while (true) {
        // The longest matching entry wins: some entries are prefixes of
        // others ("嗯…" opens "嗯…让我想想哦…"), and taking the short one
        // would leave the tail of the long one behind.
        size_t best = 0;
        for (const auto & entry : library()) {
            if (entry.text.size() > best &&
                text.compare(0, entry.text.size(), entry.text) == 0) {
                best = entry.text.size();
            }
        }
        if (best == 0) {
            return text;
        }
        while (best < text.size() && text[best] == ' ') {
            ++best;
        }
        text.erase(0, best);
    }
}

}  // namespace minitts::server::filler
