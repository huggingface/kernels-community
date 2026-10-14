// Upstream's flash-attention kernels, compiled the way llama.cpp compiles them on Apple Silicon:
// with GGML_METAL_HAS_BF16, which upstream defines when it builds its shaders on a device with
// bfloat support and kernel-builder has no way to pass. Upstream's common.h undefines it again
// below Metal 3.1. The vendored file is shipped as fa.metal.h so it is not also compiled on its own.
#define GGML_METAL_HAS_BF16 1

#include "fa.metal.h"
