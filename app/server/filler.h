#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace minitts::server::filler {

// A conversational hesitation spoken while the real reply is still being
// generated: "Ummm, let me think..." buys the pipeline its first-audio time
// and makes the turn feel answered the instant send is pressed. Sized by how
// much air there is to fill.
enum class Size { Short, Medium, Long };

struct Entry {
    Size size;
    bool chinese;
    std::string text;
};

// The library of filler texts, stable across builds: the synthesized PCM is
// cached on disk keyed by (voice, model, text), so editing a text re-renders
// only that clip.
const std::vector<Entry> & library();

// A stable fingerprint for one filler's synthesized audio: mixes the voice
// fingerprint with the text so cache files survive restarts and voice edits
// invalidate cleanly.
uint64_t clip_key(uint64_t voice_fingerprint, const std::string & text);

// Picks a random library index of the wanted size and language; -1 when the
// library has no such entry.
int pick(Size size, bool chinese, uint64_t random_seed);

}  // namespace minitts::server::filler
