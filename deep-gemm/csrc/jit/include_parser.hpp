#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "../utils/format.hpp"
#include "../utils/system.hpp"

namespace deep_gemm {

class IncludeParser {
    std::unordered_map<std::string, std::optional<std::string>> cache;

    static bool is_include_space(const char& c) {
        return c == ' ' or c == '\t' or c == '\r' or c == '\f' or c == '\v';
    }

    static void raise_non_standard_include(
        const std::string& include_str, const std::filesystem::path& file_path) {
        std::string error_info = fmt::format("Non-standard include: {}", include_str);
        if (file_path != "")
            error_info += fmt::format(" ({})", file_path.string());
        DG_HOST_UNREACHABLE(error_info);
    }

    static std::vector<std::string> get_includes(const std::string& code, const std::filesystem::path& file_path = "") {
        std::vector<std::string> includes;

        // TODO: parse relative paths as well
        size_t line_begin = 0;
        while (line_begin < code.size()) {
            auto line_end = code.find('\n', line_begin);
            if (line_end == std::string::npos)
                line_end = code.size();

            auto pos = line_begin;
            while (pos < line_end and is_include_space(code[pos]))
                ++ pos;

            const auto directive_begin = pos;
            if (pos < line_end and code[pos] == '#') {
                ++ pos;
                while (pos < line_end and is_include_space(code[pos]))
                    ++ pos;

                constexpr size_t kIncludeLen = 7;
                if (line_end - pos >= kIncludeLen and code.compare(pos, kIncludeLen, "include") == 0) {
                    pos += kIncludeLen;
                    if (pos < line_end and not is_include_space(code[pos]) and code[pos] != '<' and code[pos] != '"') {
                        line_begin = line_end + (line_end < code.size());
                        continue;
                    }

                    while (pos < line_end and is_include_space(code[pos]))
                        ++ pos;

                    if (pos < line_end and code[pos] == '<') {
                        const auto name_begin = pos + 1;
                        const auto name_end = code.find('>', name_begin);
                        if (name_end == std::string::npos or name_end > line_end or
                            name_begin == name_end or code[name_begin] == ' ' or code[name_end - 1] == ' ') {
                            raise_non_standard_include(code.substr(directive_begin, line_end - directive_begin), file_path);
                        }

                        const auto filename = code.substr(name_begin, name_end - name_begin);
                        if (filename.substr(0, 9) == "deep_gemm")  // We only parse `<deep_gemm/*>`
                            includes.push_back(filename);
                    } else if (pos < line_end and code[pos] == '"') {
                        const auto quote_end = code.find('"', pos + 1);
                        const auto include_end = quote_end == std::string::npos or quote_end > line_end ? line_end : quote_end + 1;
                        raise_non_standard_include(code.substr(directive_begin, include_end - directive_begin), file_path);
                    }
                }
            }

            line_begin = line_end + (line_end < code.size());
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
