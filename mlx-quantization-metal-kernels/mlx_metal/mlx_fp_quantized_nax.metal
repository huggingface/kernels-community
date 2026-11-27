// Upstream's fp_quantized_nax.metal, compiled the way MLX compiles it; see mlx_quantized.metal.
#pragma METAL fp math_mode(safe)

#include "nax_deployment_target.h"

#include "mlx/backend/metal/kernels/fp_quantized_nax.metal.h"
