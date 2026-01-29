/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// CPU Feature Detection for MegaBlocks

#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace megablocks {
namespace cpu {

// CPU feature detection
class CPUFeatures {
public:
    static bool hasAVX512() {
        static bool avx512_supported = checkAVX512();
        return avx512_supported;
    }

    static bool hasAVX512BF16() {
        static bool bf16_supported = checkAVX512BF16();
        return bf16_supported;
    }

    static bool hasAMX() {
        static bool amx_supported = checkAMX();
        return amx_supported;
    }

    // Check if all required features for flash attention are available
    static bool hasAllRequiredFeatures() {
        return hasAVX512() && hasAVX512BF16() && hasAMX();
    }

private:
    static bool checkAVX512() {
#ifdef _MSC_VER
        int cpu_info[4];
        __cpuid(cpu_info, 0);
        int n_ids = cpu_info[0];
        if (n_ids < 7) return false;

        __cpuidex(cpu_info, 7, 0);
        bool avx512f = (cpu_info[1] & (1 << 16)) != 0;  // EBX bit 16: AVX-512 Foundation
        if (!avx512f) return false;

        __cpuid(cpu_info, 1);
        bool osxsave = (cpu_info[2] & (1 << 27)) != 0;  // ECX bit 27: OSXSAVE
        if (!osxsave) return false;

        unsigned long long xcr0 = _xgetbv(0);
        return ((xcr0 & 0xE6ULL) == 0xE6ULL);
#else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, nullptr) < 7) {
            return false;
        }

        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        bool avx512f = (ebx & (1 << 16)) != 0;  // EBX bit 16: AVX-512 Foundation
        if (!avx512f) return false;

        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
            return false;
        }
        bool osxsave = (ecx & (1 << 27)) != 0;  // ECX bit 27: OSXSAVE
        if (!osxsave) return false;

        unsigned int xcr0_lo = 0, xcr0_hi = 0;
        __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
        unsigned long long xcr0 = ((unsigned long long)xcr0_hi << 32) | xcr0_lo;
        return ((xcr0 & 0xE6ULL) == 0xE6ULL);
#endif
    }

    static bool checkAVX512BF16() {
        if (!checkAVX512()) return false;

#ifdef _MSC_VER
        int cpu_info[4];
        __cpuidex(cpu_info, 7, 1);
        return (cpu_info[0] & (1 << 5)) != 0;  // EAX bit 5: AVX512_BF16
#else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, nullptr) < 7) {
            return false;
        }
        __cpuid_count(7, 1, eax, ebx, ecx, edx);
        return (eax & (1 << 5)) != 0;  // EAX bit 5: AVX512_BF16
#endif
    }

    static bool checkAMX() {
#ifdef _MSC_VER
        int cpu_info[4];
        __cpuid(cpu_info, 0);
        if (cpu_info[0] < 7) return false;

        __cpuidex(cpu_info, 7, 0);
        bool amx_bf16 = (cpu_info[3] & (1 << 22)) != 0;  // EDX bit 22: AMX-BF16
        bool amx_tile = (cpu_info[3] & (1 << 24)) != 0;  // EDX bit 24: AMX-TILE
        return amx_bf16 && amx_tile;
#else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, nullptr) < 7) {
            return false;
        }
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        bool amx_bf16 = (edx & (1 << 22)) != 0;  // EDX bit 22: AMX-BF16
        bool amx_tile = (edx & (1 << 24)) != 0;  // EDX bit 24: AMX-TILE
        return amx_bf16 && amx_tile;
#endif
    }
};

}  // namespace cpu
}  // namespace megablocks
