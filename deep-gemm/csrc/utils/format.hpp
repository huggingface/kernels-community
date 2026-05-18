#pragma once

// Minimal fmt::format shim — supports only "{}" placeholders and "{{" / "}}"
// escapes. This covers all usage in DeepGEMM and avoids depending on libfmt.
//
// Uses std::string concatenation instead of locale-backed iostream formatting.

#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <filesystem>

namespace fmt {

namespace detail {

template<typename>
inline constexpr bool kAlwaysFalse = false;

// Convert value to string — specializations for common types
template<typename T>
inline std::string to_str(const T& v) {
    if constexpr (std::is_same_v<T, std::string>) {
        return v;
    } else if constexpr (std::is_same_v<T, const char*> || std::is_same_v<T, char*>) {
        return std::string(v);
    } else if constexpr (std::is_same_v<T, std::string_view>) {
        return std::string(v);
    } else if constexpr (std::is_same_v<T, char>) {
        return std::string(1, v);
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::filesystem::path>) {
        return v.string();
    } else if constexpr (std::is_integral_v<T>) {
        return std::to_string(v);
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::to_string(v);
    } else {
        static_assert(kAlwaysFalse<T>, "Unsupported fmt::format argument type");
    }
}

template<typename T>
inline std::string to_hex_float_str(const T& v) {
    if constexpr (std::is_floating_point_v<T>) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%a", static_cast<double>(v));
        return buffer;
    } else {
        return to_str(v);
    }
}

// Overload for C string literals (arrays)
template<size_t N>
inline std::string to_str(const char (&s)[N]) {
    return std::string(s, N - 1);
}

inline std::string format_impl(std::string_view fmt) {
    std::string result;
    result.reserve(fmt.size());
    size_t i = 0;
    while (i < fmt.size()) {
        if (fmt[i] == '{' && i + 1 < fmt.size() && fmt[i + 1] == '{') {
            result += '{';
            i += 2;
        } else if (fmt[i] == '}' && i + 1 < fmt.size() && fmt[i + 1] == '}') {
            result += '}';
            i += 2;
        } else {
            result += fmt[i++];
        }
    }
    return result;
}

template<typename T, typename... Args>
std::string format_impl(std::string_view fmt,
                         const T& first, const Args&... rest) {
    std::string result;
    result.reserve(fmt.size());
    size_t i = 0;
    while (i < fmt.size()) {
        if (fmt[i] == '{') {
            if (i + 1 < fmt.size() && fmt[i + 1] == '{') {
                result += '{';
                i += 2;
            } else if (i + 1 < fmt.size() && fmt[i + 1] == '}') {
                result += to_str(first);
                result += format_impl(fmt.substr(i + 2), rest...);
                return result;
            } else if (i + 3 < fmt.size() && fmt[i + 1] == ':' &&
                       fmt[i + 2] == 'a' && fmt[i + 3] == '}') {
                result += to_hex_float_str(first);
                result += format_impl(fmt.substr(i + 4), rest...);
                return result;
            } else {
                result += fmt[i++];
            }
        } else if (fmt[i] == '}' && i + 1 < fmt.size() && fmt[i + 1] == '}') {
            result += '}';
            i += 2;
        } else {
            result += fmt[i++];
        }
    }
    return result;
}

} // namespace detail

template<typename... Args>
std::string format(std::string_view fmt, const Args&... args) {
    return detail::format_impl(fmt, args...);
}

} // namespace fmt
