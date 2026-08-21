#include "character.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace minitts::server {
namespace {

constexpr const char * kCharacterFile = "character.json";
constexpr size_t kMaxNameLength = 64;

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (ch < 0x20) {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
                out += buffer;
            } else {
                out += static_cast<char>(ch);
            }
        }
    }
    return out;
}

}  // namespace

CharacterConfig default_character() {
    CharacterConfig character;
    character.name = "F";
    character.preset = "demo_3_woman";
    return character;
}

std::string sanitize_character_name(const std::string & name) {
    size_t begin = 0;
    size_t end = name.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(name[begin])) != 0) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(name[end - 1])) != 0) {
        --end;
    }
    std::string trimmed = name.substr(begin, end - begin);
    if (trimmed.empty()) {
        throw std::runtime_error("character name must not be empty");
    }
    if (trimmed.size() > kMaxNameLength) {
        throw std::runtime_error(
            "character name must be at most " + std::to_string(kMaxNameLength) + " characters");
    }
    return trimmed;
}

namespace {

// Parses `character.json` in the directory. Throws on every problem, including
// a custom recording the store names but the disk no longer has -- callers
// decide whether that means "fall back" (the active slot) or "skip" (the
// library).
CharacterConfig read_character_file(const std::filesystem::path & directory) {
    const auto path = directory / kCharacterFile;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("character store missing: " + path.string());
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();

    const auto value = engine::io::json::parse(buffer.str());
    CharacterConfig character;
    character.name = sanitize_character_name(engine::io::json::require_string(value, "name"));
    character.preset = engine::io::json::optional_string(value, "preset", "");
    character.voice_file = engine::io::json::optional_string(value, "voice_file", "");
    character.transcript = engine::io::json::optional_string(value, "transcript", "");
    character.persona = engine::io::json::optional_string(value, "persona", "");
    if (character.preset.empty() && character.voice_file.empty()) {
        throw std::runtime_error(
            "character store " + path.string() + " names neither a preset nor a voice_file");
    }
    if (character.is_custom() && !std::filesystem::exists(directory / character.voice_file)) {
        throw std::runtime_error(
            "character recording missing: " + (directory / character.voice_file).string());
    }
    return character;
}

}  // namespace

CharacterConfig load_character(const std::filesystem::path & directory) {
    if (!std::filesystem::exists(directory / kCharacterFile)) {
        return default_character();
    }
    try {
        return read_character_file(directory);
    } catch (const std::exception & ex) {
        // A recording that vanished is a cosmetic loss; bricking startup over it
        // would be worse, so the default character takes over. Anything else --
        // corrupt JSON, missing keys -- stays loud.
        if (std::string(ex.what()).find("recording missing") != std::string::npos) {
            return default_character();
        }
        throw;
    }
}

std::string character_slug(const std::string & name) {
    std::string slug;
    bool pending_dash = false;
    for (const unsigned char ch : name) {
        if (std::isalnum(ch) != 0 && ch < 0x80) {
            if (pending_dash && !slug.empty()) {
                slug += '-';
            }
            pending_dash = false;
            slug += static_cast<char>(std::tolower(ch));
        } else {
            pending_dash = true;
        }
    }
    if (!slug.empty()) {
        return slug;
    }
    // No usable ASCII at all: derive a stable id from the bytes so two
    // different non-Latin names keep two different entries.
    uint64_t hash = 1469598103934665603ull;
    for (const unsigned char ch : name) {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    char buffer[24];
    std::snprintf(buffer, sizeof(buffer), "c-%08x", static_cast<unsigned>(hash & 0xffffffffu));
    return buffer;
}

bool is_valid_character_id(const std::string & id) {
    if (id.empty() || id.size() > 80) {
        return false;
    }
    for (const unsigned char ch : id) {
        if (std::isalnum(ch) == 0 && ch != '-') {
            return false;
        }
        if (ch >= 0x80 || std::isupper(ch) != 0) {
            return false;
        }
    }
    return true;
}

std::vector<SavedCharacter> list_character_library(const std::filesystem::path & root) {
    std::vector<SavedCharacter> entries;
    const auto library = root / "library";
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(library, ec)) {
        if (!entry.is_directory()) {
            continue;
        }
        const auto id = entry.path().filename().string();
        if (!is_valid_character_id(id)) {
            continue;
        }
        if (!std::filesystem::exists(entry.path() / kCharacterFile)) {
            continue;
        }
        try {
            entries.push_back({id, read_character_file(entry.path())});
        } catch (const std::exception &) {
            // One damaged entry must not hide the rest of the library.
        }
    }
    std::sort(entries.begin(), entries.end(), [](const SavedCharacter & a, const SavedCharacter & b) {
        return a.config.name < b.config.name;
    });
    return entries;
}

void save_character(const std::filesystem::path & directory, const CharacterConfig & character) {
    std::filesystem::create_directories(directory);
    std::ostringstream out;
    out << "{\"name\":\"" << json_escape(character.name) << "\"";
    if (!character.preset.empty()) {
        out << ",\"preset\":\"" << json_escape(character.preset) << "\"";
    }
    if (!character.voice_file.empty()) {
        out << ",\"voice_file\":\"" << json_escape(character.voice_file) << "\"";
    }
    if (!character.transcript.empty()) {
        out << ",\"transcript\":\"" << json_escape(character.transcript) << "\"";
    }
    if (!character.persona.empty()) {
        out << ",\"persona\":\"" << json_escape(character.persona) << "\"";
    }
    out << "}";

    const auto path = directory / kCharacterFile;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("could not write character store: " + path.string());
    }
    file << out.str();
    if (!file) {
        throw std::runtime_error("could not write character store: " + path.string());
    }
}

}  // namespace minitts::server
