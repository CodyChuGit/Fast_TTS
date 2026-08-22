#include "filler.h"

namespace minitts::server::filler {

const std::vector<Entry> & library() {
    // Written to be speakable and character-neutral: hesitations, not
    // content. Short covers a prewarmed miss (~0.5-0.8 s of air), medium a
    // typical cold turn, long a fresh conversation's first exchange.
    static const std::vector<Entry> entries = {
        // English, short
        {Size::Short, false, "Mmm."},
        {Size::Short, false, "Uhh..."},
        {Size::Short, false, "Ooh."},
        {Size::Short, false, "Hmm?"},
        {Size::Short, false, "Oh!"},
        {Size::Short, false, "Right..."},
        // English, medium
        {Size::Medium, false, "Ummm, okay so..."},
        {Size::Medium, false, "Hmm, let me think..."},
        {Size::Medium, false, "Oh, that? Well..."},
        {Size::Medium, false, "Yeah, yeah, okay..."},
        {Size::Medium, false, "Mmm, good question..."},
        {Size::Medium, false, "Wait, let me see..."},
        // English, long
        {Size::Long, false, "Okay okay, hold on... let me actually think about that for a second..."},
        {Size::Long, false, "Hmm... that's actually such a good question, give me a sec..."},
        {Size::Long, false, "Ooh, um, okay... so, how do I put this..."},
        {Size::Long, false, "Mmm, wait wait... okay, I think I know what I want to say..."},
        // Chinese, short
        {Size::Short, true, "嗯…"},
        {Size::Short, true, "哦？"},
        {Size::Short, true, "呃…"},
        {Size::Short, true, "嗯哼。"},
        {Size::Short, true, "哦——"},
        // Chinese, medium
        {Size::Medium, true, "嗯…让我想想哦…"},
        {Size::Medium, true, "哎呀，这个嘛…"},
        {Size::Medium, true, "哦，那个呀…"},
        {Size::Medium, true, "嗯嗯，好问题…"},
        {Size::Medium, true, "等一下哦,我想想…"},
        // Chinese, long
        {Size::Long, true, "哎呀等一下等一下,让我好好想一想这个问题哦…"},
        {Size::Long, true, "嗯——这个问题还挺有意思的,让我想想看…"},
        {Size::Long, true, "唔,怎么说呢…让我组织一下语言哈…"},
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

}  // namespace

uint64_t clip_key(uint64_t voice_fingerprint, const std::string & text) {
    return fnv1a(voice_fingerprint, text.data(), text.size());
}

int pick(Size size, bool chinese, uint64_t random_seed) {
    const auto & entries = library();
    std::vector<int> candidates;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].size == size && entries[i].chinese == chinese) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    if (candidates.empty()) {
        return -1;
    }
    return candidates[static_cast<size_t>(random_seed % candidates.size())];
}

}  // namespace minitts::server::filler
