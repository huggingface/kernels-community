#pragma once

// Define namespace based on CUTLASS_SYCL_REVISION
#if defined(OLD_API)
    #define COMPAT syclcompat
    #include <syclcompat.hpp>
#else
    #define COMPAT compat
    #include <cute/util/compat/device.hpp>
#endif
