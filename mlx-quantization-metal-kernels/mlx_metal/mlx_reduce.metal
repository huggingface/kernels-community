// Upstream's reduction kernels, compiled the way MLX compiles them; see mlx_quantized.metal.
// Split-K matmuls sum their partial products with these, as MLX does.
#pragma METAL fp math_mode(safe)

#include "mlx/backend/metal/kernels/reduce.metal.h"
