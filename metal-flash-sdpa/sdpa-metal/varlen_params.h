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
  int window_left; ///< Keys visible to the left of the query, -1 for all
  int window_right; ///< Keys visible to the right of the query, -1 for all
};

// Range of keys [lo, hi] that query row `row` may attend to. `row` is the
// query's position in the key sequence (bottom-right alignment), so under
// causal masking it sees keys up to `row`. The range is empty if lo > hi.
METAL_FUNC int2 visible_keys(
    int row,
    int kL,
    bool causal,
    bool windowed,
    const constant VarlenAttnParams* params) {
  int lo = 0;
  int hi = kL - 1;
  if (causal) {
    hi = min(hi, row);
  }
  if (windowed) {
    if (params->window_left >= 0) {
      lo = max(lo, row - params->window_left);
    }
    if (params->window_right >= 0) {
      hi = min(hi, row + params->window_right);
    }
  }
  return int2(lo, hi);
}
