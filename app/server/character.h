#pragma once

#include <filesystem>
#include <string>

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

}  // namespace minitts::server
