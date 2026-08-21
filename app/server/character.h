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

// Every saved character under `<root>/library`, sorted by name. Entries that
// fail to load (broken JSON, missing recording) are skipped: one damaged entry
// must not hide the rest of the library.
std::vector<SavedCharacter> list_character_library(const std::filesystem::path & root);

}  // namespace minitts::server
