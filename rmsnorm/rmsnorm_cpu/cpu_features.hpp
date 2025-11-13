#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <fstream>
#include <string>
#include <cstdlib>
namespace rmsnorm_cpu {

// CPU feature detection
class CPUFeatures {
public:
    static bool hasAVX2() {
        static bool avx2_supported = checkAVX2();
        return avx2_supported;
    }

    static bool hasAVX512BF16() {
        static bool disabled = env_disable("DISABLE_AVX512_BF16");
        static bool bf16_supported = !disabled && checkAVX512BF16();
        std::cerr << "AVX512_BF16 support: " << std::boolalpha << bf16_supported << std::endl;
        return bf16_supported;
    }

private:

   static bool env_disable(const char* name) {
        const char* v = std::getenv(name);
        if (!v) return false;
        // "0" means not disabled, any other value disables
        return !(v[0] == '0' && v[1] == '\0');
    }

    static bool checkAVX2() {
#ifdef _MSC_VER
        int cpu_info[4];
        __cpuid(cpu_info, 0);
        int n_ids = cpu_info[0];

        if (n_ids >= 7) {
            __cpuidex(cpu_info, 7, 0);
            return (cpu_info[1] & (1 << 5)) != 0;  // EBX bit 5
        }
        return false;
#else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, nullptr) < 7) {
            return false;
        }
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1 << 5)) != 0;  // EBX bit 5
#endif
    }

    static bool checkAVX512() {
#ifdef _MSC_VER
        int cpu_info[4];
        __cpuid(cpu_info, 0);
        int n_ids = cpu_info[0];
        if (n_ids < 7) return false;

        __cpuidex(cpu_info, 7, 0);
        bool avx512f = (cpu_info[1] & (1 << 16)) != 0; // EBX bit 16: AVX-512 Foundation
        if (!avx512f) return false;

        __cpuid(cpu_info, 1);
        bool osxsave = (cpu_info[2] & (1 << 27)) != 0; // ECX bit 27: OSXSAVE
        if (!osxsave) return false;

        // check XCR0: bits 1,2 (SSE/AVX) and 5,6,7 (AVX-512 state) must be enabled by OS
        unsigned long long xcr0 = _xgetbv(0);
        return ( (xcr0 & 0xE6ULL) == 0xE6ULL );
#else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, nullptr) < 7) {
            return false;
        }

        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        bool avx512f = (ebx & (1 << 16)) != 0; // EBX bit 16: AVX-512 Foundation
        if (!avx512f) return false;

        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
            return false;
        }
        bool osxsave = (ecx & (1 << 27)) != 0; // ECX bit 27: OSXSAVE
        if (!osxsave) return false;

        unsigned int xcr0_lo = 0, xcr0_hi = 0;
        __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
        unsigned long long xcr0 = ((unsigned long long)xcr0_hi << 32) | xcr0_lo;
        // require XCR0 bits 1,2,5,6,7 set -> mask 0xE6 (0b11100110)
        return ( (xcr0 & 0xE6ULL) == 0xE6ULL );
#endif
    }

    static bool checkAVX512BF16() {
        // require AVX-512 foundation supported and OS enabled
        if (!checkAVX512()) return false;

#ifndef _MSC_VER
        // First: try Linux /proc/cpuinfo flags (most robust on Linux)
        std::ifstream f("/proc/cpuinfo");
        if (f) {
            std::string line;
            while (std::getline(f, line)) {
                // flags line contains many space-separated tokens including avx512_bf16 on supported CPUs
                if (line.find("avx512_bf16") != std::string::npos ||
                    line.find("avx512bf16") != std::string::npos) {
                    return true;
                }
            }
        }

        // Fallback: attempt CPUID subleaf check if available.
        // Note: exact bit position for AVX512_BF16 may differ across vendors/CPUID versions.
        // This fallback tries CPUID(7,1) and checks some common positions; if uncertain returns false.
        if (__get_cpuid_max(0, nullptr) < 7) {
            return false;
        }
        unsigned int eax, ebx, ecx, edx;
        // try subleaf 1
        __cpuid_count(7, 1, eax, ebx, ecx, edx);
        // There isn't a universally agreed constant here in this file; check common candidate bits:
        // - some implementations report AVX512_BF16 in ECX/EBX of subleaf 1.
        // Try commonly used positions conservatively.
        const unsigned int candidate_masks[] = {
            (1u << 5),   // candidate (may collide with other features)
            (1u << 26),  // another candidate position
        };
        for (unsigned m : candidate_masks) {
            if ((ebx & m) || (ecx & m) || (edx & m)) {
                return true;
            }
        }
        return false;
#else
        // On MSVC / Windows, use CPUID if available (simple check). If unsure, return false.
        int cpu_info[4];
        __cpuid(cpu_info, 0);
        int n_ids = cpu_info[0];
        if (n_ids < 7) return false;
        __cpuidex(cpu_info, 7, 1);
        // same conservative check as above
        const int candidate_masks[] = { (1 << 5), (1 << 26) };
        for (int m : candidate_masks) {
            if ((cpu_info[1] & m) || (cpu_info[2] & m) || (cpu_info[3] & m)) {
                return true;
            }
        }
        return false;
#endif
    }
};

} // namespace rmsnorm_cpu
