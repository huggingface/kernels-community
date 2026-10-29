// Upstream's kernels, compiled the way MLX compiles them. MLX builds its shaders with
// `-fno-fast-math` (mlx/backend/metal/kernels/CMakeLists.txt); kernel-builder takes no Metal flags
// and Metal defaults to fast math, which rounds differently from MLX. The pragma is the in-source
// form of that flag, so the vendored file itself stays untouched
// (vendor.py ships it with a .h suffix so it is not also compiled on its own).
#pragma METAL fp math_mode(safe)

#include "mlx/backend/metal/kernels/quantized.metal.h"
