#pragma once

#include <string>

namespace deep_gemm {

static uint64_t fnv1a(const std::vector<char>& data, const uint64_t& seed) {
    uint64_t h = seed;
    const uint64_t prime = 0x100000001b3ull;
    for (const char& c: data) {
        h ^= static_cast<uint8_t>(c);
        h *= prime;
    }
    return h;
}

static std::string get_hex_digest(const std::vector<char>& data) {
    const auto state_0 = fnv1a(data, 0xc6a4a7935bd1e995ull);
    const auto state_1 = fnv1a(data, 0x9e3779b97f4a7c15ull);

    // Split-mix 64
    const auto split_mix = [](uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };

    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(32, '0');
    const uint64_t states[] = {split_mix(state_0), split_mix(state_1)};
    for (size_t state_idx = 0; state_idx < 2; ++ state_idx) {
        auto value = states[state_idx];
        for (int nibble = 15; nibble >= 0; -- nibble) {
            out[state_idx * 16 + static_cast<size_t>(nibble)] = kHex[value & 0x0f];
            value >>= 4;
        }
    }
    return out;
}

static std::string get_hex_digest(const std::string& data) {
    return get_hex_digest(std::vector<char>{data.begin(), data.end()});
}

} // namespace deep_gemm
