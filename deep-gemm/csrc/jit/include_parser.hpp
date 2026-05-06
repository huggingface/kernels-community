#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "../utils/format.hpp"
#include "../utils/system.hpp"

namespace deep_gemm {

class IncludeParser {
    std::unordered_map<std::string, std::optional<std::string>> cache;

    // Manual scanner replacing `std::regex` — the regex implementation in some
    // libstdc++ builds segfaults inside `std::codecvt<char16_t,char>::do_unshift`
    // when an ABI mismatch is present at load time. Hand-rolled parsing also
    // dodges locale facets entirely and is faster.
    static std::vector<std::string> get_includes(const std::string& code, const std::filesystem::path& file_path = "") {
        std::vector<std::string> includes;
        constexpr std::string_view kInclude = "include";
        const size_t n = code.size();
        size_t pos = 0;

        while (pos < n) {
            // Find next `#` that starts (or is at) a line.
            const size_t hash = code.find('#', pos);
            if (hash == std::string::npos) break;
            if (hash != 0 && code[hash - 1] != '\n') {
                pos = hash + 1;
                continue;
            }

            // Skip horizontal whitespace after `#`.
            size_t i = hash + 1;
            while (i < n && (code[i] == ' ' || code[i] == '\t')) ++i;

            // Must be "include".
            if (i + kInclude.size() > n || code.compare(i, kInclude.size(), kInclude) != 0) {
                pos = hash + 1;
                continue;
            }
            i += kInclude.size();

            // Skip horizontal whitespace before the opening bracket/quote.
            while (i < n && (code[i] == ' ' || code[i] == '\t')) ++i;
            if (i >= n) break;

            const char open = code[i];
            char close;
            if (open == '<') close = '>';
            else if (open == '"') close = '"';
            else { pos = i; continue; }  // not a recognised include form

            const size_t end = code.find(close, i + 1);
            if (end == std::string::npos) break;
            const std::string filename = code.substr(i + 1, end - i - 1);

            if (open == '<') {
                // Upstream only ingests `<deep_gemm/...>`; other angle-bracket
                // system includes (e.g. `<cuda_fp16.h>`) are silently skipped.
                if (filename.compare(0, 9, "deep_gemm") == 0) {
                    includes.push_back(filename);
                }
            } else {
                // Quoted form is treated as non-standard (matches upstream).
                std::string err = fmt::format("Non-standard include: #include \"{}\"", filename);
                if (file_path != "") err += fmt::format(" ({})", file_path.string());
                DG_HOST_UNREACHABLE(err);
            }
            pos = end + 1;
        }
        return includes;
    }

public:
    static std::filesystem::path library_include_path;

    static void prepare_init(const std::string& library_root_path) {
        library_include_path = std::filesystem::path(library_root_path) / "include";
    }

    std::string get_hash_value(const std::string& code, const bool& exclude_code = true) {
        std::stringstream ss;
        for (const auto& i: get_includes(code))
            ss << get_hash_value_by_path(library_include_path / i) << "$";
        if (not exclude_code)
            ss << "#" << get_hex_digest(code);
        return get_hex_digest(ss.str());
    }

    std::string get_hash_value_by_path(const std::filesystem::path& path) {
        // Check whether hit in cache
        // ReSharper disable once CppUseAssociativeContains
        if (cache.count(path) > 0) {
            const auto opt = cache[path];
            if (not opt.has_value())
                DG_HOST_UNREACHABLE(fmt::format("Circular include may occur: {}", path.string()));
            return opt.value();
        }

        // Read file and calculate hash recursively
        std::ifstream in(path);
        if (not in.is_open())
            DG_HOST_UNREACHABLE(fmt::format("Failed to open: {}", path.string()));
        std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        cache[path] = std::nullopt;
        return (cache[path] = get_hash_value(code, false)).value();
    }
};

DG_DECLARE_STATIC_VAR_IN_CLASS(IncludeParser, library_include_path);

static auto include_parser = std::make_shared<IncludeParser>();

}  // namespace deep_gemm
