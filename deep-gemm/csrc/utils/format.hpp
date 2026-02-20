#pragma once

// Minimal fmt::format shim — supports only "{}" placeholders and "{{" / "}}"
// escapes. This covers all usage in DeepGEMM and avoids depending on libfmt.
//
// Uses std::string concatenation instead of std::ostringstream to avoid
// potential locale/ABI issues with ostringstream across different platforms.

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <filesystem>

namespace fmt {

namespace detail {

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
        // Fallback for other types with operator<<
        std::ostringstream os;
        os << v;
        return os.str();
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
