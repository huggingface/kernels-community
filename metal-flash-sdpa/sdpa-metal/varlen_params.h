// Parameters shared by the variable-length attention kernels. Must match
// VarlenAttnParams in scaled_dot_product_attention.mm.

#pragma once

struct VarlenAttnParams {
  int64_t q_strides[2]; ///< Query  strides (token, head), head_dim stride is 1
  int64_t k_strides[2]; ///< Key    strides (token, head)
  int64_t v_strides[2]; ///< Value  strides (token, head)
  int64_t o_strides[2]; ///< Output strides (token, head)

  int H; ///< Query heads
  int H_kv; ///< Key/value heads
  float scale; ///< Attention scale
  float softcap; ///< Softcap value, only read when has_softcap
};
