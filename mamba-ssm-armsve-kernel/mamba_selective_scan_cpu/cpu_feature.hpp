#pragma once

#include <fstream>
#include <string>

#if defined(__linux__) || defined(__ANDROID__)
  #include <sys/auxv.h>
  #include <asm/hwcap.h>
#endif

namespace cpuinfo {

class CPUFeaturesARM {
public:
    static bool hasSVE() {
        static bool supported = checkSVE();
        return supported;
    }

private:
    static bool checkSVE() {
#if (defined(__aarch64__) || defined(_M_ARM64))
  #if defined(__linux__) || defined(__ANDROID__)
        // Best: auxv HWCAP bit (kernel-validated for user-space)
        const unsigned long hwcap = getauxval(AT_HWCAP);
        #ifdef HWCAP_SVE
        if (hwcap & HWCAP_SVE) return true;
        #endif
        return cpuinfoHasFlag("sve");
  #else
        return false;
  #endif
#else
        return false;
#endif
    }

    static bool cpuinfoHasFlag(const char* flag) {
#if defined(__linux__) || defined(__ANDROID__)
        std::ifstream f("/proc/cpuinfo");
        if (!f) return false;

        std::string line;
        while (std::getline(f, line)) {
            // Typical key is "Features" on ARM. We also accept "flags".
            if ((line.find("Features") != std::string::npos ||
                 line.find("flags")    != std::string::npos ||
                 line.find("Flags")    != std::string::npos) &&
                line.find(flag) != std::string::npos) {
                return true;
            }
        }
        return false;
#else
        (void)flag;
        return false;
#endif
    }
};

} // namespace rmsnorm_cpu