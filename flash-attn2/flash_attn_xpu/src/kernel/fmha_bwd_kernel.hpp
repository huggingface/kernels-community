#pragma once
#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include "../philox.hpp"
using namespace cute;

template <class T_, int kHeadDim_, int kBlockM_, int kBlockN_, int kNSGs_,
          int AtomLayoutMSdP_ = 2, int AtomLayoutNdKV_ = 2, int AtomLayoutMdQ_ = 2,
          bool is_causal_ = false, bool is_local_ = false>
struct FABwdKernel {
    /*
      Q BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
      K BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
      V BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_VO
      P BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
      O BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
    */
    using DType = T_;
    using VType = float; // accumulation type
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kNSGs = kNSGs_;
    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static constexpr int AtomLayoutNdKV = AtomLayoutNdKV_;
    static constexpr int AtomLayoutMdQ = AtomLayoutMdQ_;
    static constexpr bool is_causal = is_causal_;
    static constexpr bool is_local = is_local_;
    using MMA_Atom_ARCH = XE_DPAS_TT<8, VType, DType>;
    using _K = Int<MMA_Atom_ARCH::K>;
    using SubgroupLayoutSdP = Layout<Shape<Int<AtomLayoutMSdP>, Int<kNSGs / AtomLayoutMSdP>, _1>>;
    using SubgroupLayoutdKV = Layout<Shape<Int<AtomLayoutNdKV>, Int<kNSGs / AtomLayoutNdKV>, _1>>;
    using SubgroupLayoutdQ = Layout<Shape<Int<AtomLayoutMdQ>, Int<kNSGs / AtomLayoutMdQ>, _1>>;
    using TileShapeSdP = Layout<Shape<Int<kBlockN>, Int<kBlockM>, _K>>;
    using TileShapedKV = Layout<Shape<Int<kBlockN>, Int<kHeadDim>, _K>>;
    using TileShapedQ = Layout<Shape<Int<kBlockM>, Int<kHeadDim>, _K>>;

    using TiledMmaSdP = typename TiledMMAHelper<MMA_Atom<MMA_Atom_ARCH>,
                                                TileShapeSdP,
                                                SubgroupLayoutSdP>::TiledMMA;

    using TiledMmadKV = typename TiledMMAHelper<MMA_Atom<MMA_Atom_ARCH>,
                                                TileShapedKV,
                                                SubgroupLayoutdKV>::TiledMMA;

    using TiledMmadQ = typename TiledMMAHelper<MMA_Atom<MMA_Atom_ARCH>,
                                               TileShapedQ,
                                               SubgroupLayoutdQ>::TiledMMA;
    static constexpr auto bP = Int<2>{}; // Pipeline
    static constexpr int SubgroupSize = 16;
    static constexpr int smem_size = 0;

    FABwdKernel() {}
};

using index_t = uint64_t;

template<typename T>
struct BwdParam {
    BwdParam(const T *dO,
             const T *o,
             const T *q,
             const T *k,
             const T *v,
             const float *lse,
             float *odo,
             float *dqaccum,
             T *dq,
             T *dk,
             T *dv,
             T *pb,
             const float softmax_scale)
        : do_ptr(dO),
          o_ptr(o),
          q_ptr(q),
          k_ptr(k),
          v_ptr(v),
          lse_ptr(lse),
          odo_ptr(odo),
          dqaccum_ptr(dqaccum),
          dq_ptr(dq),
          dk_ptr(dk),
          dv_ptr(dv),
          pb_ptr(pb),
          scale_softmax(softmax_scale),
          scale_softmax_log2(softmax_scale * M_LOG2E),
          is_bhsd(true) {}
    
    // read only
    const T *do_ptr;
    const T *o_ptr;
    const T *q_ptr;
    const T *k_ptr;
    const T *v_ptr;
    const float *lse_ptr;
    const float scale_softmax;
    const float scale_softmax_log2;
    
    // write
    float *odo_ptr;
    float *dqaccum_ptr;
    T *dq_ptr;
    T *dk_ptr;
    T *dv_ptr;
    T *pb_ptr;

    // const dimension
    int batch;
    int num_head_q;
    int num_head_kv;
    int seq_len_q;
    int seq_len_q_pad;
    int seq_len_kv;
    int seq_len_kv_pad;
    int head_dim;
    int n_block;
    int tail_n;
    int m_block;
    int tail_m;
    int num_qh_per_kvh;
    int num_nb_per_blk;
    
    // Strides for Q
    int q_r_stride;
    int q_h_stride;
    int q_b_stride;

    // Strides for K
    int k_r_stride;
    int k_h_stride;
    int k_b_stride;

    // Strides for dK
    int dk_r_stride;
    int dk_h_stride;
    int dk_b_stride;

    // Strides for V
    int v_r_stride;
    int v_h_stride;
    int v_b_stride;

    // Strides for dV
    int dv_r_stride;
    int dv_h_stride;
    int dv_b_stride;

    // Strides for O
    int o_r_stride;
    int o_h_stride;
    int o_b_stride;

    // Strides for S (softmax)
    int s_r_stride;
    int s_s_stride;
    int s_b_stride;

    // Strides for dQ
    int dq_r_stride;
    int dq_h_stride;
    int dq_b_stride;

    // Window size for local attention
    int window_size_left;
    int window_size_right;

    bool is_bhsd;
    bool is_local;
    
    // Dropout
    cutlass::fmha::Dropout dropout;
};

template<typename T>
struct BwdOffset {
    BwdOffset(BwdParam<T> &param_) : param(param_) {}
    
    index_t q_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.q_b_stride + h_id * param.q_h_stride + s_id * param.q_r_stride;
    }
    index_t k_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.k_b_stride + h_id * param.k_h_stride + s_id * param.k_r_stride;
    }
    index_t v_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.v_b_stride + h_id * param.v_h_stride + s_id * param.v_r_stride;
    }
    index_t dk_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.dk_b_stride + h_id * param.dk_h_stride + s_id * param.dk_r_stride;
    }
    index_t dv_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.dv_b_stride + h_id * param.dv_h_stride + s_id * param.dv_r_stride;
    }
    index_t lse_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.seq_len_q * param.num_head_q + h_id * param.seq_len_q + s_id;
    }
    index_t o_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.o_b_stride + h_id * param.o_h_stride + s_id * param.o_r_stride;
    }
    index_t dq_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
        return b_id * param.dq_b_stride + h_id * param.dq_h_stride + s_id * param.dq_r_stride;
    }
    
    BwdParam<T> &param;
};

// Setup strides for BHSD layout (batch, heads, seq, dim)
template<typename T>
void setup_bhsd_stride_bwd(BwdParam<T> &param) {
    param.q_r_stride = param.head_dim;
    param.q_h_stride = param.seq_len_q * param.head_dim;
    param.q_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    param.k_r_stride = param.head_dim;
    param.k_h_stride = param.seq_len_kv * param.head_dim;
    param.k_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    param.dk_r_stride = param.head_dim;
    param.dk_h_stride = param.seq_len_kv * param.head_dim;
    param.dk_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

    param.v_r_stride = param.head_dim;
    param.v_h_stride = param.seq_len_kv * param.head_dim;
    param.v_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    param.dv_r_stride = param.head_dim;
    param.dv_h_stride = param.seq_len_kv * param.head_dim;
    param.dv_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

    param.o_r_stride = param.head_dim;
    param.o_h_stride = param.seq_len_q * param.head_dim;
    param.o_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    param.dq_r_stride = param.head_dim;
    param.dq_h_stride = param.seq_len_q_pad * param.head_dim;
    param.dq_b_stride = param.num_head_q * param.seq_len_q_pad * param.head_dim;
}

// Setup strides for BSHD layout (batch, seq, heads, dim)
template<typename T>
void setup_bshd_stride_bwd(BwdParam<T> &param) {
    param.q_r_stride = param.num_head_q * param.head_dim;
    param.q_h_stride = param.head_dim;
    param.q_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    param.k_r_stride = param.num_head_kv * param.head_dim;
    param.k_h_stride = param.head_dim;
    param.k_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    param.dk_r_stride = param.num_head_q * param.head_dim;
    param.dk_h_stride = param.head_dim;
    param.dk_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

    param.v_r_stride = param.num_head_kv * param.head_dim;
    param.v_h_stride = param.head_dim;
    param.v_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

    param.dv_r_stride = param.num_head_q * param.head_dim;
    param.dv_h_stride = param.head_dim;
    param.dv_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

    param.o_r_stride = param.num_head_q * param.head_dim;
    param.o_h_stride = param.head_dim;
    param.o_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

    param.dq_r_stride = param.num_head_q * param.head_dim;
    param.dq_h_stride = param.head_dim;
    param.dq_b_stride = param.num_head_q * param.seq_len_q_pad * param.head_dim;
}
