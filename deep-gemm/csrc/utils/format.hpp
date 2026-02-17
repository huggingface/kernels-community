#pragma once

// Minimal fmt::format shim — supports only "{}" placeholders and "{{" / "}}"
// escapes. This covers all usage in DeepGEMM and avoids depending on libfmt.

#include <sstream>
#include <string>
#include <string_view>

namespace fmt {

namespace detail {

inline void format_impl(std::ostringstream& os, std::string_view fmt) {
    size_t i = 0;
    while (i < fmt.size()) {
        if (fmt[i] == '{' && i + 1 < fmt.size() && fmt[i + 1] == '{') {
            os << '{';
            i += 2;
        } else if (fmt[i] == '}' && i + 1 < fmt.size() && fmt[i + 1] == '}') {
            os << '}';
            i += 2;
        } else {
            os << fmt[i++];
        }
    }
}

template<typename T, typename... Args>
void format_impl(std::ostringstream& os, std::string_view fmt,
                  const T& first, const Args&... rest) {
    size_t i = 0;
    while (i < fmt.size()) {
        if (fmt[i] == '{') {
            if (i + 1 < fmt.size() && fmt[i + 1] == '{') {
                os << '{';
                i += 2;
            } else if (i + 1 < fmt.size() && fmt[i + 1] == '}') {
                os << first;
                format_impl(os, fmt.substr(i + 2), rest...);
                return;
            } else {
                os << fmt[i++];
            }
        } else if (fmt[i] == '}' && i + 1 < fmt.size() && fmt[i + 1] == '}') {
            os << '}';
            i += 2;
        } else {
            os << fmt[i++];
        }
    }
}

} // namespace detail

template<typename... Args>
std::string format(std::string_view fmt, const Args&... args) {
    std::ostringstream os;
    detail::format_impl(os, fmt, args...);
    return os.str();
}

} // namespace fmt
