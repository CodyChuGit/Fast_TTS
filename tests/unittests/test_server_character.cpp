#include "character.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using minitts::server::CharacterConfig;
using minitts::server::default_character;
using minitts::server::load_character;
using minitts::server::sanitize_character_name;
using minitts::server::save_character;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Each test gets its own scratch directory under the system temp path so runs
// never interfere with each other or with a developer's real character store.
struct ScratchDir {
    std::filesystem::path path;
    explicit ScratchDir(const std::string & tag)
        : path(std::filesystem::temp_directory_path() / ("audiocpp-character-test-" + tag)) {
        std::filesystem::remove_all(path);
    }
    ~ScratchDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

void test_missing_store_yields_the_default_character() {
    ScratchDir dir("missing");
    const auto character = load_character(dir.path);
    require(character.name == "F", "the default character is F");
    require(character.preset == "demo_3_woman", "F speaks with the bundled female demo voice");
    require(!character.is_custom(), "the default is a preset, not a recording");
}

void test_preset_character_roundtrips() {
    ScratchDir dir("preset");
    CharacterConfig character;
    character.name = "Mira";
    character.preset = "demo_1_man";
    save_character(dir.path, character);

    const auto loaded = load_character(dir.path);
    require(loaded.name == "Mira", "name roundtrips");
    require(loaded.preset == "demo_1_man", "preset roundtrips");
    require(!loaded.is_custom(), "preset source survives");
}

void test_custom_character_roundtrips_with_its_recording() {
    ScratchDir dir("custom");
    std::filesystem::create_directories(dir.path);
    {
        std::ofstream wav(dir.path / "voice.wav", std::ios::binary);
        wav << "RIFF....";
    }
    CharacterConfig character;
    character.name = "Kai \"the voice\"";
    character.voice_file = "voice.wav";
    character.transcript = "A line the recording actually says.";
    save_character(dir.path, character);

    const auto loaded = load_character(dir.path);
    require(loaded.name == character.name, "quoted names survive the JSON roundtrip");
    require(loaded.is_custom(), "custom source survives");
    require(loaded.transcript == character.transcript, "transcript roundtrips");
}

void test_a_custom_character_missing_its_recording_falls_back() {
    ScratchDir dir("orphan");
    CharacterConfig character;
    character.name = "Ghost";
    character.voice_file = "voice.wav";
    save_character(dir.path, character);
    // The store names voice.wav but the file was never written: the intent is
    // unrecoverable, so startup gets the default rather than a crash.
    const auto loaded = load_character(dir.path);
    require(loaded.name == "F", "an orphaned custom character falls back to F");
}

void test_broken_store_fails_loudly() {
    ScratchDir dir("broken");
    std::filesystem::create_directories(dir.path);
    {
        std::ofstream file(dir.path / "character.json", std::ios::binary);
        file << "{not json";
    }
    bool threw = false;
    try {
        load_character(dir.path);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "a corrupt store is an error, not a silent reset");
}

void test_name_sanitization() {
    require(sanitize_character_name("  F  ") == "F", "surrounding whitespace is trimmed");
    bool threw = false;
    try {
        sanitize_character_name("   ");
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "an all-whitespace name is refused");
    threw = false;
    try {
        sanitize_character_name(std::string(200, 'x'));
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "an absurdly long name is refused");
}

}  // namespace

int main() {
    try {
        test_missing_store_yields_the_default_character();
        test_preset_character_roundtrips();
        test_custom_character_roundtrips_with_its_recording();
        test_a_custom_character_missing_its_recording_falls_back();
        test_broken_store_fails_loudly();
        test_name_sanitization();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_character_test passed\n";
    return 0;
}
