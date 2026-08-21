#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace minitts::server {

// The one voice this app speaks with. A character is a display name plus a
// voice source: either a named server voice preset (the bundled demo voices)
// or a custom reference recording stored in the character directory. It is
// server state, not browser state, so the WebUI and MCP clients always agree
// on who is speaking.
struct CharacterConfig {
    std::string name = "F";
    // Named voice preset, when the character uses a bundled voice.
    std::string preset;
    // Filename of a custom reference recording inside the character directory,
    // when the character uses an uploaded or recorded voice. Mutually exclusive
    // with `preset`.
    std::string voice_file;
    // Transcript of the custom recording, injected as reference_text for
    // cloning quality.
    std::string transcript;
    // Who the character IS, not just how they sound: used as the LLM system
    // prompt in chat, so the roleplay model plays this character while the TTS
    // model voices them. Empty means a generic in-character prompt is built
    // from the name.
    std::string persona;

    bool is_custom() const { return !voice_file.empty(); }
};

// The out-of-the-box character: "F", voiced by the bundled demo_3_woman
// preset.
CharacterConfig default_character();

// Trims and bounds a display name. Throws std::runtime_error when nothing
// usable remains.
std::string sanitize_character_name(const std::string & name);

// Reads `character.json` from the directory; the default character when the
// file does not exist. Throws on a file that exists but cannot be parsed --
// silently reverting a customized character to the default would be worse than
// failing loudly.
CharacterConfig load_character(const std::filesystem::path & directory);

// Writes `character.json`, creating the directory if needed.
void save_character(const std::filesystem::path & directory, const CharacterConfig & character);

// A library entry: the id is the directory name under `library/`, derived from
// the character's name, so saving under the same name replaces the entry.
struct SavedCharacter {
    std::string id;
    CharacterConfig config;
};

// Filesystem- and URL-safe id for a character name: lowercased ASCII
// letters/digits with runs of everything else collapsed to single dashes. A
// name with no usable ASCII (for example a fully non-Latin one) gets a stable
// hash-based id instead, so distinct names do not all collapse onto one entry.
std::string character_slug(const std::string & name);

// True only for ids character_slug can produce. Ids arrive from clients and
// become path components, so anything else -- separators, dots, empties -- is
// rejected outright.
bool is_valid_character_id(const std::string & id);

// How the roleplay LLM behaves: the master prompt every character is played
// through, and the sampling that shapes delivery. App-level state, persisted
// beside the character store, editable from Settings; chat requests may still
// override the sampling per call.
struct LlmSettings {
    // {name} and {persona} are substituted from the active character.
    std::string master_prompt;
    double temperature = 0.55;
    double top_p = 0.75;
    double repeat_penalty = 1.1;
    int64_t max_tokens = 140;
    // Grow replies over the conversation: the opener stays snappy because
    // first audio depends on it, then later turns get more room -- max_tokens
    // is the ceiling the ramp grows toward.
    bool length_ramp = true;
    // Which registered chat model to run; empty means the server default.
    std::string model;
};

// The shipped roleplay-tuned defaults.
LlmSettings default_llm_settings();

// Reads `llm.json` from the directory; defaults when missing. Throws on a file
// that exists but cannot be parsed.
LlmSettings load_llm_settings(const std::filesystem::path & directory);

// Writes `llm.json`, creating the directory if needed. Values are validated --
// out-of-range sampling is refused rather than silently clamped.
void save_llm_settings(const std::filesystem::path & directory, const LlmSettings & settings);

// The reply-length budget for a turn: ramps from a short opener toward the
// configured max_tokens ceiling as the conversation accumulates assistant
// turns. With the ramp disabled, the ceiling applies from turn one.
int64_t ramped_max_tokens(const LlmSettings & settings, int64_t assistant_turns);

// A per-turn sentence-count hint appended to the system prompt. Token ceilings
// truncate mid-sentence -- awful when spoken -- so the model is asked for the
// length; the ceiling only backstops it. Empty when the ramp is disabled.
std::string length_guidance(const LlmSettings & settings, int64_t assistant_turns);

// The master prompt with {name} and {persona} substituted.
std::string render_master_prompt(
    const LlmSettings & settings,
    const std::string & name,
    const std::string & persona);

// Every saved character under `<root>/library`, sorted by name. Entries that
// fail to load (broken JSON, missing recording) are skipped: one damaged entry
// must not hide the rest of the library.
std::vector<SavedCharacter> list_character_library(const std::filesystem::path & root);

}  // namespace minitts::server
