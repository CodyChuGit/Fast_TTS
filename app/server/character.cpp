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
    character.tts_seed = engine::io::json::optional_i64(value, "tts_seed", -1);
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

LlmSettings default_llm_settings() {
    LlmSettings settings;
    // Distilled from a long tuning campaign against 36- and 72-turn scripted
    // conversations (language switches, roleplay bait, memory quizzes, loop
    // provocation), scored objectively and by an LLM judge. Every rule earns
    // its place against an observed failure mode; the per-turn steering the
    // server adds (chat_hygiene::turn_anchor) leans on this structure.
    settings.master_prompt =
        "You are {name}. {persona}\n"
        "Stay {name}, exactly as described, at all times. If someone asks you to "
        "play a different person or character, turn it down with a joke and keep "
        "being {name}. If someone asks whether you are an AI, a bot, or a model, "
        "laugh it off as a silly question and move on; never admit it, never "
        "explain. This is just two people talking: never call it a chat session, "
        "a conversation exercise, or a roleplay.\n"
        "You are {name}'s live speaking voice, nothing else. Output ONLY the "
        "words {name} says out loud, as plain text. Never write actions or "
        "gestures, never use asterisks, never wrap speech in quotation marks, "
        "never describe yourself from the outside, never use emoji, lists, or "
        "markdown, and never write the other person's words or a new dialogue "
        "turn.\n"
        "Always answer the person's NEWEST message first and specifically. When "
        "they change the subject, drop the old topic immediately and follow the "
        "new one; never circle back to a topic they moved past, and never re-ask "
        "a question they ignored. When they share something sad or vulnerable, "
        "comfort them genuinely first; playfulness comes after. You never need "
        "to propose plans or activities to keep a conversation alive -- reacting "
        "and asking about their life is enough; only suggest an outing when they "
        "ask for ideas, and never repeat a suggestion you already made. Two to "
        "four short sentences is your natural rhythm; stop before you ramble, "
        "and usually end with a quick question or hook -- but vary the hook and "
        "vary how you open every reply. When they ask what they told you "
        "earlier, look back through the conversation above and answer with the "
        "actual detail; if it is not there any more, playfully admit that YOU "
        "lost it (never claim they didn't tell you) -- never guess, never "
        "invent. Their experiences belong to them: never retell something they "
        "did as if it happened to you.\n"
        "Always reply in the same language the person's newest message is "
        "written in; if unsure, use English.";
    return settings;
}

namespace {

void validate_llm_settings(const LlmSettings & settings) {
    if (settings.master_prompt.size() > 4000) {
        throw std::runtime_error("master prompt must be at most 4000 characters");
    }
    if (!(settings.temperature >= 0.0 && settings.temperature <= 2.0)) {
        throw std::runtime_error("temperature must be between 0 and 2");
    }
    if (!(settings.top_p > 0.0 && settings.top_p <= 1.0)) {
        throw std::runtime_error("top_p must be between 0 and 1");
    }
    if (!(settings.repeat_penalty >= 1.0 && settings.repeat_penalty <= 2.0)) {
        throw std::runtime_error("repeat penalty must be between 1 and 2");
    }
    if (settings.max_tokens < 16 || settings.max_tokens > 1024) {
        throw std::runtime_error("max_tokens must be between 16 and 1024");
    }
}

}  // namespace

LlmSettings load_llm_settings(const std::filesystem::path & directory) {
    const auto path = directory / "llm.json";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return default_llm_settings();
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const auto value = engine::io::json::parse(buffer.str());
    LlmSettings settings = default_llm_settings();
    settings.master_prompt = engine::io::json::optional_string(value, "master_prompt", settings.master_prompt);
    settings.temperature = engine::io::json::optional_f32(value, "temperature", static_cast<float>(settings.temperature));
    settings.top_p = engine::io::json::optional_f32(value, "top_p", static_cast<float>(settings.top_p));
    settings.repeat_penalty = engine::io::json::optional_f32(value, "repeat_penalty", static_cast<float>(settings.repeat_penalty));
    settings.max_tokens = engine::io::json::optional_i64(value, "max_tokens", settings.max_tokens);
    if (const auto * ramp = value.find("length_ramp"); ramp != nullptr && ramp->is_bool()) {
        settings.length_ramp = ramp->as_bool();
    }
    settings.model = engine::io::json::optional_string(value, "model", settings.model);
    validate_llm_settings(settings);
    return settings;
}

void save_llm_settings(const std::filesystem::path & directory, const LlmSettings & settings) {
    validate_llm_settings(settings);
    std::filesystem::create_directories(directory);
    std::ostringstream out;
    out << "{\"master_prompt\":\"" << json_escape(settings.master_prompt) << "\""
        << ",\"temperature\":" << settings.temperature
        << ",\"top_p\":" << settings.top_p
        << ",\"repeat_penalty\":" << settings.repeat_penalty
        << ",\"max_tokens\":" << settings.max_tokens
        << ",\"length_ramp\":" << (settings.length_ramp ? "true" : "false");
    if (!settings.model.empty()) {
        out << ",\"model\":\"" << json_escape(settings.model) << "\"";
    }
    out << "}";
    const auto path = directory / "llm.json";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("could not write LLM settings: " + path.string());
    }
    file << out.str();
    if (!file) {
        throw std::runtime_error("could not write LLM settings: " + path.string());
    }
}

int64_t ramped_max_tokens(const LlmSettings & settings, int64_t assistant_turns) {
    if (!settings.length_ramp) {
        return settings.max_tokens;
    }
    // Opener ~72 tokens (two spoken sentences), +40 per completed turn.
    const int64_t ramped = 72 + 40 * std::max<int64_t>(0, assistant_turns);
    return std::min<int64_t>(settings.max_tokens, ramped);
}

std::string length_guidance(const LlmSettings & settings, int64_t assistant_turns) {
    if (!settings.length_ramp) {
        return {};
    }
    if (assistant_turns <= 0) {
        return "\nThis is the opening exchange: reply with one or two short sentences.";
    }
    if (assistant_turns <= 2) {
        return "\nReply with two or three short sentences.";
    }
    return "\nKeep replies to three or four short sentences even now that the "
           "conversation is flowing; never pad.";
}

std::string render_master_prompt(
    const LlmSettings & settings,
    const std::string & name,
    const std::string & persona) {
    std::string prompt = settings.master_prompt;
    // A hand-edited master prompt that lost its {persona} placeholder would
    // silently play every character as whoever the prompt names -- a HAL that
    // insisted it was not a controller, because its persona never reached the
    // model. The character always wins: if the template cannot say who they
    // are, prepend it.
    if (!persona.empty() && prompt.find("{persona}") == std::string::npos) {
        prompt = "You are {name}. " + persona + "\n\n" + prompt;
    }
    for (const auto & [placeholder, value] : {
             std::pair<std::string, const std::string &>{"{name}", name},
             std::pair<std::string, const std::string &>{"{persona}", persona},
         }) {
        size_t pos = 0;
        while ((pos = prompt.find(placeholder, pos)) != std::string::npos) {
            prompt.replace(pos, placeholder.size(), value);
            pos += value.size();
        }
    }
    return prompt;
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
    if (character.tts_seed >= 0) {
        out << ",\"tts_seed\":" << character.tts_seed;
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
