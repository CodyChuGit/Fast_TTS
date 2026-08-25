#include "voice_identity.h"

#include "engine/framework/audio/dsp.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace minitts::server {

namespace {

// The extractor wants 16 kHz. Linear resampling is plenty: this compares the
// coarse shape of a spectrum, not audio anyone will hear.
std::vector<float> to_16k(const std::vector<float> & samples, int64_t sample_rate) {
    if (sample_rate == 16000 || samples.empty() || sample_rate <= 0) {
        return samples;
    }
    const double ratio = 16000.0 / static_cast<double>(sample_rate);
    const auto out_count = static_cast<size_t>(static_cast<double>(samples.size()) * ratio);
    std::vector<float> out(out_count);
    for (size_t i = 0; i < out_count; ++i) {
        const double src = static_cast<double>(i) / ratio;
        const auto lo = static_cast<size_t>(src);
        const size_t hi = std::min(lo + 1, samples.size() - 1);
        const auto frac = static_cast<float>(src - static_cast<double>(lo));
        out[i] = samples[lo] * (1.0F - frac) + samples[hi] * frac;
    }
    return out;
}

}  // namespace

VoiceProfile voice_profile(const std::vector<float> & samples, int64_t sample_rate) {
    VoiceProfile profile;
    const auto audio = to_16k(samples, sample_rate);
    // Under ~0.3 s there is not enough spectrum to characterise anything.
    if (audio.size() < 4800) {
        return profile;
    }

    static const engine::audio::WhisperLogMelExtractor extractor{
        engine::audio::WhisperLogMelConfig{}};
    const auto features = extractor.compute(audio);
    if (features.mel_bins <= 0 || features.frames <= 0) {
        return profile;
    }

    // Frame energy, so quiet frames can be left out of the average: a short
    // clip is mostly silence, and silence is the same for every speaker.
    const auto bins = static_cast<size_t>(features.mel_bins);
    const auto frames = static_cast<size_t>(features.frames);
    std::vector<float> frame_energy(frames, 0.0F);
    for (size_t f = 0; f < frames; ++f) {
        float sum = 0.0F;
        for (size_t b = 0; b < bins; ++b) {
            sum += features.values[b * frames + f];
        }
        frame_energy[f] = sum / static_cast<float>(bins);
    }
    auto sorted = frame_energy;
    std::sort(sorted.begin(), sorted.end());
    const float loud = sorted[static_cast<size_t>(static_cast<double>(sorted.size()) * 0.6)];

    std::vector<double> acc(bins, 0.0);
    size_t counted = 0;
    for (size_t f = 0; f < frames; ++f) {
        if (frame_energy[f] < loud) {
            continue;
        }
        for (size_t b = 0; b < bins; ++b) {
            acc[b] += features.values[b * frames + f];
        }
        ++counted;
    }
    if (counted == 0) {
        return profile;
    }
    profile.mel_mean.resize(bins);
    for (size_t b = 0; b < bins; ++b) {
        profile.mel_mean[b] = static_cast<float>(acc[b] / static_cast<double>(counted));
    }
    // Remove the overall level so loudness cannot masquerade as identity.
    const float mean = std::accumulate(profile.mel_mean.begin(), profile.mel_mean.end(), 0.0F) /
                       static_cast<float>(bins);
    for (float & v : profile.mel_mean) {
        v -= mean;
    }
    return profile;
}

float voice_similarity(const VoiceProfile & a, const VoiceProfile & b) {
    if (a.empty() || b.empty() || a.mel_mean.size() != b.mel_mean.size()) {
        return -1.0F;
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.mel_mean.size(); ++i) {
        dot += static_cast<double>(a.mel_mean[i]) * b.mel_mean[i];
        na += static_cast<double>(a.mel_mean[i]) * a.mel_mean[i];
        nb += static_cast<double>(b.mel_mean[i]) * b.mel_mean[i];
    }
    if (na <= 0.0 || nb <= 0.0) {
        return -1.0F;
    }
    return static_cast<float>(dot / std::sqrt(na * nb));
}

}  // namespace minitts::server
