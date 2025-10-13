#pragma once

// Define namespace based on CUTLASS_SYCL_REVISION
#if defined(CUTLASS_SYCL_REVISION) && (CUTLASS_SYCL_REVISION == "v0.5")
    #define COMPAT compat
#else
    #define COMPAT syclcompat
#endif
