// MLX's NAX code is written against the 26.2 matmul2d. Built against the 26.0 one it returns wrong
// half and bfloat results. kernel-builder builds Metal shaders for macOS 26.0, so raise the target
// the headers see to the one MLX builds its NAX kernels for.
#if __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200
#undef __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__
#define __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ 260200
#endif
