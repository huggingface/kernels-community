#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

namespace cutlass::fmha {

// Philox constants
constexpr uint32_t kPhiloxSA = 0xD2511F53u;
constexpr uint32_t kPhiloxSB = 0xCD9E8D57u;
constexpr uint32_t kPhiloxM0 = 0x9E3779B9u;
constexpr uint32_t kPhiloxM1 = 0xBB67AE85u;

// Multiply high and low 32 bits
CUTLASS_DEVICE
void mulhilo32(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
    uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    hi = static_cast<uint32_t>(product >> 32);
    lo = static_cast<uint32_t>(product);
}

// Single round of Philox
CUTLASS_DEVICE
void philox_single_round(uint32_t (&ctr)[4], const uint32_t (&key)[2]) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(kPhiloxSA, ctr[0], hi0, lo0);
    mulhilo32(kPhiloxSB, ctr[2], hi1, lo1);
    
    uint32_t tmp0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t tmp1 = lo1;
    uint32_t tmp2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t tmp3 = lo0;
    
    ctr[0] = tmp0;
    ctr[1] = tmp1;
    ctr[2] = tmp2;
    ctr[3] = tmp3;
}

// Philox 4x32_10 - generate 4 random uint32 values
// Input: seed (64-bit), offset (64-bit), subsequence determined by batch/head/position
CUTLASS_DEVICE
void philox_4x32_10(
    uint64_t seed,
    uint64_t offset,
    uint32_t batch_head,
    uint32_t row,
    uint32_t col,
    uint32_t (&output)[4]) {
    
    // Initialize counter with position information
    uint32_t ctr[4] = {
        static_cast<uint32_t>(offset),
        static_cast<uint32_t>(offset >> 32),
        row * 32 + col,  // Encodes position in attention matrix
        batch_head
    };
    
    // Initialize key from seed
    uint32_t key[2] = {
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32)
    };
    
    // Run 10 rounds of Philox
    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        philox_single_round(ctr, key);
        // Bump the key
        key[0] += kPhiloxM0;
        key[1] += kPhiloxM1;
    }
    
    output[0] = ctr[0];
    output[1] = ctr[1];
    output[2] = ctr[2];
    output[3] = ctr[3];
}

// Dropout class for Flash Attention
struct Dropout {
    uint64_t seed;
    uint64_t offset;
    float p_dropout;           // Probability of dropping (NOT keeping)
    float rp_dropout;          // Reciprocal of (1 - p_dropout) for scaling
    uint8_t p_dropout_uint8;   // p_dropout scaled to uint8 for comparison
    bool is_enabled;
    
    // Default constructor (no dropout)
    CUTLASS_HOST_DEVICE
    Dropout() : seed(0), offset(0), p_dropout(0.0f), rp_dropout(1.0f), 
                p_dropout_uint8(0), is_enabled(false) {}
    
    // Constructor with dropout parameters
    CUTLASS_HOST_DEVICE
    Dropout(uint64_t seed_, uint64_t offset_, float p_dropout_)
        : seed(seed_), offset(offset_), p_dropout(p_dropout_) {
        is_enabled = (p_dropout > 0.0f);
        if (is_enabled) {
            rp_dropout = 1.0f / (1.0f - p_dropout);
            // Scale p_dropout to uint8 range (0-255) for comparison
            // We keep if random < (1 - p_dropout), so threshold = (1 - p_dropout) * 256
            p_dropout_uint8 = static_cast<uint8_t>((1.0f - p_dropout) * 256.0f);
        } else {
            rp_dropout = 1.0f;
            p_dropout_uint8 = 255;  // Always keep
        }
    }
    
    // Apply dropout to a tensor element
    // Returns: (value after dropout, keep_mask as 0 or 1)
    template<typename T>
    CUTLASS_DEVICE
    T apply(T value, uint32_t batch_head, uint32_t row, uint32_t col) const {
        if (!is_enabled) {
            return value;
        }
        
        uint32_t rng_output[4];
        philox_4x32_10(seed, offset, batch_head, row, col, rng_output);
        
        // Use first byte of first random number for comparison
        uint8_t rand_val = static_cast<uint8_t>(rng_output[0] & 0xFF);
        
        // Keep if rand_val < threshold (i.e., rand_val < (1-p) * 256)
        bool keep = (rand_val < p_dropout_uint8);
        
        return keep ? static_cast<T>(static_cast<float>(value) * rp_dropout) : T(0);
    }
    
    // Check if a position should be kept (for mask regeneration in backward)
    CUTLASS_DEVICE
    bool should_keep(uint32_t batch_head, uint32_t row, uint32_t col) const {
        if (!is_enabled) {
            return true;
        }
        
        uint32_t rng_output[4];
        philox_4x32_10(seed, offset, batch_head, row, col, rng_output);
        
        uint8_t rand_val = static_cast<uint8_t>(rng_output[0] & 0xFF);
        return (rand_val < p_dropout_uint8);
    }
    
    // Get the scale factor for kept elements
    CUTLASS_HOST_DEVICE
    float get_scale() const {
        return rp_dropout;
    }
};

// Helper to apply dropout to a fragment (tensor in registers)
template<typename Engine, typename Layout, typename DropoutOp>
CUTLASS_DEVICE
void apply_dropout_to_fragment(
    cute::Tensor<Engine, Layout>& tensor,
    DropoutOp const& dropout,
    uint32_t batch_head,
    uint32_t m_offset,
    uint32_t n_offset,
    cute::Tensor<Engine, Layout> const& coord_tensor) {
    
    using namespace cute;
    
    if (!dropout.is_enabled) {
        return;
    }
    
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int sg_local_id = sg.get_local_id();
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tensor); ++i) {
        // Get the global row and column indices
        auto coord = coord_tensor(i);
        int row = m_offset + get<0>(coord);
        int col = n_offset + get<1>(coord) + sg_local_id;
        
        tensor(i) = dropout.apply(tensor(i), batch_head, row, col);
    }
}

// Helper to regenerate dropout mask and scale in backward pass
template<typename Engine, typename Layout, typename CoordEngine, typename CoordLayout>
CUTLASS_DEVICE
void apply_dropout_mask_backward(
    cute::Tensor<Engine, Layout>& P_tensor,      // Attention probabilities
    cute::Tensor<Engine, Layout>& dP_tensor,     // Gradient w.r.t. attention probabilities
    Dropout const& dropout,
    uint32_t batch_head,
    uint32_t m_offset,
    uint32_t n_offset,
    cute::Tensor<CoordEngine, CoordLayout> const& coord_tensor) {
    
    using namespace cute;
    
    if (!dropout.is_enabled) {
        return;
    }
    
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int sg_local_id = sg.get_local_id();
    
    float scale = dropout.get_scale();
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(P_tensor); ++i) {
        auto coord = coord_tensor(i);
        int row = m_offset + get<0>(coord);
        int col = n_offset + get<1>(coord) + sg_local_id;
        
        bool keep = dropout.should_keep(batch_head, row, col);
        
        if (keep) {
            P_tensor(i) = static_cast<typename Engine::value_type>(
                static_cast<float>(P_tensor(i)) * scale);
            dP_tensor(i) = static_cast<typename Engine::value_type>(
                static_cast<float>(dP_tensor(i)) * scale);
        } else {
            P_tensor(i) = typename Engine::value_type(0);
            dP_tensor(i) = typename Engine::value_type(0);
        }
    }
}

}  // namespace cutlass::fmha
