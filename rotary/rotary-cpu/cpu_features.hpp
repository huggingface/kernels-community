#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace rotary_cpu {

// Runtime CPU feature detection. Used to pick the AVX512 path only on hardware
// that actually supports it; otherwise the generic ATen fallback is used.
class CPUFeatures {
public:
  static bool hasAVX512() {
    static bool supported = checkAVX512();
    return supported;
  }

private:
  static bool checkAVX512() {
#ifdef _MSC_VER
    int cpu_info[4];
    __cpuid(cpu_info, 0);
    if (cpu_info[0] < 7)
      return false;
    __cpuidex(cpu_info, 7, 0);
    bool avx512f = (cpu_info[1] & (1 << 16)) != 0; // EBX bit 16
    if (!avx512f)
      return false;
    __cpuid(cpu_info, 1);
    bool osxsave = (cpu_info[2] & (1 << 27)) != 0; // ECX bit 27
    if (!osxsave)
      return false;
    unsigned long long xcr0 = _xgetbv(0);
    return ((xcr0 & 0xE6ULL) == 0xE6ULL);
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, nullptr) < 7)
      return false;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    bool avx512f = (ebx & (1 << 16)) != 0; // EBX bit 16
    if (!avx512f)
      return false;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0)
      return false;
    bool osxsave = (ecx & (1 << 27)) != 0; // ECX bit 27
    if (!osxsave)
      return false;
    unsigned int xcr0_lo = 0, xcr0_hi = 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    unsigned long long xcr0 =
        ((unsigned long long)xcr0_hi << 32) | xcr0_lo;
    return ((xcr0 & 0xE6ULL) == 0xE6ULL);
#endif
  }
};

} // namespace rotary_cpu
