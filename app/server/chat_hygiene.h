#pragma once

#include <string>
#include <vector>

namespace minitts::server::hygiene {

// True when at least `threshold` of the text's letters are CJK. Used to pin
// the reply language to the user's newest message.
bool mostly_cjk(const std::string & text, double threshold = 0.3);

// Removes the artifacts a roleplay model slips into a spoken reply before the
// text re-enters the conversation window: a drifted second paragraph, *action*
// spans, full-width （aside） spans, and novel-style narration around quoted
// dialogue. Feeding the model a clean version of its own past keeps it from
// imitating its own bad habits.
std::string sanitize_assistant_text(const std::string & text);

// The per-turn steering note carried inside the LATEST user message (the
// system prompt must stay static or every turn re-pays the whole prefill).
// Quotes the newest message so the model answers THAT, deflects AI probes and
// third-person bait, coaches memory checks, bans the model's own recent tics
// (filler interjections, question-ender streaks, phrases it keeps reusing),
// and pins the reply language last, where a model weighs it most.
std::string turn_anchor(
    const std::string & latest_user,
    const std::vector<std::string> & recent_assistant);

// The standing instruction that makes bracketed notes work; appended to the
// master prompt by the server so custom prompts keep working.
extern const char * kNoteRule;

// Cleans the model's token stream as it is produced, so the transcript the
// user sees and the text the TTS speaks never contain roleplay markup.
// *action* and （aside） spans are held back and dropped when they close (or
// prove unterminated at flush); everything from a special-token opener "<|"
// onward is discarded. Feed returns the text safe to emit now.
class StreamScrubber {
public:
    std::string feed(const std::string & delta);
    // End of stream: releases held text that turned out not to be markup.
    std::string flush();

private:
    enum class State { Clean, Star, FullParen, AngleOpen, Dead };
    State state_ = State::Clean;
    std::string held_;
    std::string pending_;
};

}  // namespace minitts::server::hygiene
