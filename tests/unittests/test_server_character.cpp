#include "character.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using minitts::server::CharacterConfig;
using minitts::server::character_slug;
using minitts::server::default_character;
using minitts::server::is_valid_character_id;
using minitts::server::list_character_library;
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

void test_slugs_are_safe_and_stable() {
    require(character_slug("F") == "f", "a single letter lowercases");
    require(character_slug("Nova Prime") == "nova-prime", "spaces become dashes");
    require(character_slug("  A!!B  ") == "a-b", "punctuation collapses to one dash");
    require(character_slug("Kai \"the voice\"") == "kai-the-voice", "quotes vanish");
    const std::string han_name = "\xe5\xb0\x8f\xe6\x98\x8e";  // Han name, bytes spelled out for MSVC
    const std::string other_han = "\xe9\x9f\xb3\xe5\xa3\xb0";
    const auto hashed = character_slug(han_name);
    require(hashed.rfind("c-", 0) == 0 && is_valid_character_id(hashed),
        "a fully non-Latin name gets a stable hashed id");
    require(hashed != character_slug(other_han),
        "two different non-Latin names get different ids");

    require(is_valid_character_id("nova-prime"), "slugs validate");
    require(!is_valid_character_id(""), "empty ids are rejected");
    require(!is_valid_character_id("../escape"), "path traversal is rejected");
    require(!is_valid_character_id("a b"), "spaces are rejected");
    require(!is_valid_character_id("UPPER"), "ids are lowercase only");
}

void test_library_lists_saved_characters_and_skips_broken_ones() {
    ScratchDir dir("library");
    const auto library = dir.path / "library";

    CharacterConfig preset;
    preset.name = "Marcus";
    preset.preset = "demo_1_man";
    save_character(library / character_slug(preset.name), preset);

    CharacterConfig custom;
    custom.name = "Nova";
    custom.voice_file = "voice.wav";
    custom.transcript = "A line.";
    save_character(library / character_slug(custom.name), custom);
    {
        std::ofstream wav(library / "nova" / "voice.wav", std::ios::binary);
        wav << "RIFF....";
    }

    // An orphaned custom entry (recording gone) and a corrupt one must both be
    // skipped without hiding the healthy entries.
    CharacterConfig orphan;
    orphan.name = "Ghost";
    orphan.voice_file = "voice.wav";
    save_character(library / character_slug(orphan.name), orphan);
    std::filesystem::create_directories(library / "broken");
    {
        std::ofstream file(library / "broken" / "character.json", std::ios::binary);
        file << "{not json";
    }

    const auto entries = list_character_library(dir.path);
    require(entries.size() == 2, "two healthy entries listed");
    require(entries[0].config.name == "Marcus" && entries[0].id == "marcus", "sorted by name with slug ids");
    require(entries[1].config.name == "Nova" && entries[1].config.is_custom(), "custom entry listed");

    require(list_character_library(dir.path / "nowhere").empty(), "a missing library is just empty");
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
        test_slugs_are_safe_and_stable();
        test_library_lists_saved_characters_and_skips_broken_ones();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_character_test passed\n";
    return 0;
}
