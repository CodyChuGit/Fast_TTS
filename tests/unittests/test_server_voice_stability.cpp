// Stable-prefix tracker: the contract is that committed text never moves
// backwards, punctuation/case churn does not destabilize, and revision of the
// tail is confined to the tentative part.

#include "voice_stability.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void expect_eq(const std::string & actual, const std::string & expected, const char * label) {
    if (actual != expected) {
        std::printf("FAIL %s\n  expected: %s\n  actual:   %s\n", label, expected.c_str(), actual.c_str());
        ++failures;
    }
}

void expect_true(bool value, const char * label) {
    if (!value) {
        std::printf("FAIL %s\n", label);
        ++failures;
    }
}

using minitts::server::voice::StabilityParams;
using minitts::server::voice::StableTracker;

void growing_hypotheses_commit_common_prefix() {
    StableTracker tracker(StabilityParams{2, 2});
    tracker.update("turn on the light");
    expect_eq(tracker.stable(), "", "nothing stable after one hypothesis");
    tracker.update("turn on the lights");
    // "turn on the" agrees; the light/lights token does not.
    expect_eq(tracker.stable(), "turn on the", "agreeing prefix commits");
    tracker.update("turn on the lights in");
    expect_eq(tracker.stable(), "turn on the lights", "growth extends the commit");
    tracker.update("turn on the lights in the kitchen");
    expect_eq(tracker.stable(), "turn on the lights in", "steady growth");
    expect_eq(tracker.tentative(), "the kitchen", "tail stays tentative");
}

void punctuation_and_case_do_not_destabilize() {
    StableTracker tracker(StabilityParams{2, 2});
    tracker.update("can you check the weather");
    const bool grew = tracker.update("Can you check the weather in");
    expect_true(grew, "case change still agrees");
    expect_eq(tracker.stable(), "Can you check the weather", "commit spans case change");
    tracker.update("Can you check the weather, in New");
    // "in" agreed across the last two hypotheses, so it commits too; the
    // re-punctuated raw form is taken from the newest hypothesis.
    expect_eq(tracker.stable(), "Can you check the weather, in", "punctuation rides along");
}

void rollback_keeps_commitment() {
    StableTracker tracker(StabilityParams{2, 2});
    tracker.update("play some jazz music");
    tracker.update("play some jazz music");
    expect_eq(tracker.stable(), "play some jazz music", "full agreement commits all");
    // A wild rollback must not shrink the committed prefix.
    tracker.update("play some");
    expect_true(tracker.stable().rfind("play some jazz music", 0) == 0 ||
                    tracker.stable() == "play some jazz music",
                "commit survives rollback");
}

void cjk_tokenizes_per_character() {
    StableTracker tracker(StabilityParams{2, 1});
    tracker.update("你好吗");
    const bool grew = tracker.update("你好吗今天");
    expect_true(grew, "CJK prefix agreement commits");
    expect_eq(tracker.stable(), "你好吗", "CJK commit boundary");
    expect_eq(tracker.tentative(), "今天", "CJK tentative tail");
}

void duplicates_are_stable() {
    StableTracker tracker(StabilityParams{3, 2});
    tracker.update("hello there");
    tracker.update("hello there");
    expect_eq(tracker.stable(), "", "needs three agreeing with count=3");
    tracker.update("hello there");
    expect_eq(tracker.stable(), "hello there", "third agreement commits");
}

void reset_clears_everything() {
    StableTracker tracker(StabilityParams{2, 2});
    tracker.update("one two three");
    tracker.update("one two three");
    tracker.reset();
    expect_eq(tracker.stable(), "", "reset clears stable");
    tracker.update("brand new turn");
    expect_eq(tracker.stable(), "", "no leakage across reset");
}

}  // namespace

int main() {
    growing_hypotheses_commit_common_prefix();
    punctuation_and_case_do_not_destabilize();
    rollback_keeps_commitment();
    cjk_tokenizes_per_character();
    duplicates_are_stable();
    reset_clears_everything();
    if (failures == 0) {
        std::printf("server_voice_stability_test: all tests passed\n");
        return 0;
    }
    std::printf("server_voice_stability_test: %d failure(s)\n", failures);
    return 1;
}
