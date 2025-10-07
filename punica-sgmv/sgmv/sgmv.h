#pragma once
#include <cuda_runtime.h>

#include <cstdint>

template <typename DType>
bool sgmv(DType *y, DType *x, DType **w, int32_t *s_start, int32_t *s_end,
          void *tmp_d, int num_problems, int d_in, int d_out, int layer_idx, cudaStream_t stream);

int64_t sgmv_tmp_size(int64_t num_problems);
