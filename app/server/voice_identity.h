#pragma once

#include <cstdint>
#include <vector>

namespace minitts::server {

// Does this audio still sound like the character?
//
// Below roughly a second and a half of speech the TTS sometimes loses hold of
// WHO is speaking: measured across repeats, one-second utterances drift to a
// different voice in a third to a half of runs, while anything longer is
// steady. The drift is not a seed lottery -- the same text and seed produce it
// intermittently -- so it cannot be curated away, only detected.
//
// A voice is summarised as the mean log-mel spectrum of its loud frames, which
// describes the vocal tract rather than the pitch. (Pitch is the obvious
// measure and the wrong one: autocorrelation octave-doubles on short creaky
// clips and calls a correct voice an impostor.) Two summaries are compared by
// cosine similarity: the same speaker scores above ~0.93, a different speaker
// around 0.75.
struct VoiceProfile {
    std::vector<float> mel_mean;

    bool empty() const { return mel_mean.empty(); }
};

// Summarises PCM. Any sample rate is accepted; frames quieter than a fraction
// of the clip's own peak are ignored so silence cannot dominate the average.
VoiceProfile voice_profile(const std::vector<float> & samples, int64_t sample_rate);

// Cosine similarity of two profiles, or -1 when either is empty.
float voice_similarity(const VoiceProfile & a, const VoiceProfile & b);

}  // namespace minitts::server
