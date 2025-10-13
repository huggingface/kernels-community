#pragma once

// Define namespace based on CUTLASS_SYCL_REVISION
#if defined(OLD_API)
    #define COMPAT syclcompat
#else
    #define COMPAT compat
#endif
