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

// What the hesitation is doing, so it can match the message it answers: a
// question earns thinking, an excited message earns a reaction, a plain
// statement earns an acknowledgment. A "good question..." before a
// non-question is worse than silence.
enum class Kind { React, Think, Ack };

struct Entry {
    Size size;
    bool chinese;
    Kind kind;
    std::string text;
    // Whether the line works as a follow-up when the reply outlasts the
    // first hesitation. Realization openers ("Oh!") sound wrong mid-wait.
    bool chain = true;
};

// The library of filler texts, stable across builds: the synthesized PCM is
// cached on disk keyed by (voice, text), so editing a text re-renders only
// that clip. Append new entries at the END -- scripts/filler_qa.py maps
// cache files to entries by index.
const std::vector<Entry> & library();

// A stable fingerprint for one filler's synthesized audio: mixes the voice
// fingerprint with the text so cache files survive restarts and voice edits
// invalidate cleanly.
uint64_t clip_key(uint64_t voice_fingerprint, const std::string & text);

// Classifies the user's newest message into the hesitation kind that answers
// it. Question signals win over excitement (an excited question still wants
// thinking); everything else is an acknowledgment.
Kind classify(const std::string & user_text);

// Library indexes matching size and language (and, with chain_only, only
// entries that work as follow-ups). Entries of the wanted kind are returned
// when any exist; otherwise the whole size bucket, so a sparse kind never
// leaves a turn without a hesitation.
std::vector<int> candidates(Size size, bool chinese, Kind kind, bool chain_only);

// Scrambles a clock-derived seed; tick granularity patterns the low bits,
// which a bare modulo would turn into favoritism.
uint64_t mix(uint64_t seed);

// Removes filler texts this server prepended to an assistant reply. The
// transcript keeps them -- they were really spoken -- but the LLM must not
// see them, or a history where every past reply opens with a hesitation
// teaches it to write hesitations of its own on top of the spoken ones.
std::string strip_leading(std::string text);

}  // namespace minitts::server::filler
