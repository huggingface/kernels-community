#pragma once

#include <optional>

#ifndef DEEP_EP_KERNELS
#include <pybind11/pybind11.h>
#endif

namespace deep_jit {

class GilScopedRelease {
public:
#ifndef DEEP_EP_KERNELS
    std::optional<pybind11::gil_scoped_release> release;

    GilScopedRelease() {
        if (Py_IsInitialized() and PyGILState_Check())
            release.emplace();
    }
#else
    // kernels-community: Torch already releases the GIL before running custom ops and custom class methods
    GilScopedRelease() = default;
#endif

    GilScopedRelease(const GilScopedRelease&) = delete;
    GilScopedRelease& operator=(const GilScopedRelease&) = delete;
    GilScopedRelease(GilScopedRelease&&) = delete;
    GilScopedRelease& operator=(GilScopedRelease&&) = delete;
};

}  // namespace deep_jit
