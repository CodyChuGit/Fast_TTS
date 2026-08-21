#include "character.h"

#include "engine/framework/io/json.h"

#include <cctype>
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

CharacterConfig load_character(const std::filesystem::path & directory) {
    const auto path = directory / kCharacterFile;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return default_character();
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();

    const auto value = engine::io::json::parse(buffer.str());
    CharacterConfig character;
    character.name = sanitize_character_name(engine::io::json::require_string(value, "name"));
    character.preset = engine::io::json::optional_string(value, "preset", "");
    character.voice_file = engine::io::json::optional_string(value, "voice_file", "");
    character.transcript = engine::io::json::optional_string(value, "transcript", "");
    if (character.preset.empty() && character.voice_file.empty()) {
        throw std::runtime_error(
            "character store " + path.string() + " names neither a preset nor a voice_file");
    }
    // A custom recording named by the store but missing from disk means the
    // directory was partially copied or cleaned. Failing loudly here would brick
    // startup over a cosmetic customization; the recorded intent is gone either
    // way, so fall back to the default character.
    if (character.is_custom() && !std::filesystem::exists(directory / character.voice_file)) {
        return default_character();
    }
    return character;
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
