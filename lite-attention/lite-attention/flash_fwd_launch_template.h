/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp" // CuTe (C++ CUDA Template Extensions) for tensor operations

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h" // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/kernel_launch.h"

#include "static_switch.h"
#include "flash.h"
#include "tile_size.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel_sm90.h"
#include "flash_fwd_kernel_sm80.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "mainloop_fwd_sm80.hpp"
#include "epilogue_fwd.hpp"

using namespace cute; // Import CuTe namespace for tensor operations and layout management

/**
 * Flash Attention Forward Kernel Launch Template
 *
 * This template function orchestrates the launch of Flash Attention forward kernels using CuTe API.
 * CuTe provides powerful compile-time tensor layout and operation abstractions that enable
 * efficient CUDA kernel implementations with clear mathematical notation.
 *
 * Template Parameters:
 * @param Arch - Target GPU architecture (80, 86, 89, 90)
 * @param kHeadDim - Dimension of attention heads for Q/K
 * @param kHeadDimV - Dimension of attention heads for V (can differ from kHeadDim)
 * @param ClusterM - Cluster size in M dimension for SM90+ cooperative groups
 * @param Element - Input data type (half, bfloat16, FP8)
 * @param ElementOut - Output data type
 * @param Is_causal - Whether to apply causal masking
 * @param Is_local - Whether to apply local window attention
 * @param Has_softcap - Whether to apply softmax capping
 * @param Varlen - Whether sequences have variable lengths
 * @param PagedKVNonTMA - Whether to use paged KV cache without TMA
 * @param AppendKV - Whether to append new KV to existing cache
 * @param HasQv - Whether Q and V projections are fused
 * @param PackGQA - Whether to pack grouped query attention
 * @param Split - Whether to use split-K optimization
 * @param V_colmajor - Whether V tensor is column-major layout
 */
template <int Arch, int kHeadDim, int kHeadDimV, int ClusterM, typename Element, typename ElementOut,
          bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool PagedKVNonTMA, bool AppendKV, bool HasQv,
          bool PackGQA, bool Split, bool V_colmajor, bool Is_skipable, bool ReverseSkipList=false, bool Phase=true, bool HasMustDoList=false>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream)
{
    // Compile-time validation of template parameter combinations
    static_assert(!(Is_causal && Is_local), "Causal and Local cannot be enabled at the same time");
    static_assert(!(AppendKV && V_colmajor), "AppendKV and V_colmajor cannot be enabled at the same time");
    static_assert(!(AppendKV && !Varlen), "AppendKV requires Varlen");

    // Type traits using CuTe's type system for FP8/INT8 detection
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;
    static constexpr bool Is_INT8 = cute::is_same_v<Element, int8_t>;

    // For INT8, V uses bfloat16 (not int8) to maintain precision
    using ElementV = std::conditional_t<Is_INT8, cute::bfloat16_t, Element>;

    // For INT8, V is bf16, so no transpose needed (only FP8 has 8-bit V)
    static constexpr bool FP8_TransposeV = Is_FP8 && !V_colmajor;
    using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;

    // Architecture-specific tile size computation for optimal memory access patterns
    // Can't use structured binding since it's not compatible with constexpr

    // SM90+ tile configuration: returns (BlockM, BlockN, MmaPV_is_RS, IntraWGOverlap)
    static constexpr std::tuple<int, int, bool, bool> kBlockMN_RS_IntraWGOverlap =
        tile_size_fwd_sm90(kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(Element) /*element_size*/, V_colmajor, PagedKVNonTMA, Has_softcap, Is_skipable, Is_INT8);

    // SM80-89 tile configuration: returns (BlockM, BlockN, NWarps, Stages, Q_in_regs)
    static constexpr std::tuple<int, int, int, int, bool> kBlockMN_kNWarps_Stages_RS =
        tile_size_fwd_sm8x(Arch == 86 || Arch == 89, kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(Element) /*element_size*/, PagedKVNonTMA, Varlen && Split, Has_softcap, AppendKV);

    // Extract tile dimensions - these define the basic computation granularity
    static constexpr int kBlockM = Arch >= 90 ? std::get<0>(kBlockMN_RS_IntraWGOverlap) : std::get<0>(kBlockMN_kNWarps_Stages_RS); // Rows of Q processed per thread block
    static constexpr int kBlockN = Arch >= 90 ? std::get<1>(kBlockMN_RS_IntraWGOverlap) : std::get<1>(kBlockMN_kNWarps_Stages_RS); // Columns of K/V processed per thread block

    // SM90+ specific configuration flags
    static constexpr bool MmaPV_is_RS = std::get<2>(kBlockMN_RS_IntraWGOverlap);    // Whether P@V matrix multiplication uses register spilling
    static constexpr bool IntraWGOverlap = std::get<3>(kBlockMN_RS_IntraWGOverlap); // Whether intra-warpgroup overlap optimization is enabled

    // SM80-89 specific configuration parameters
    static constexpr int kNWarps = std::get<2>(kBlockMN_kNWarps_Stages_RS);                         // Number of warps per thread block
    static constexpr int kStages = Arch >= 90 ? 2 : std::get<3>(kBlockMN_kNWarps_Stages_RS);        // Pipeline stages for double buffering
    static constexpr bool Q_in_regs = Arch >= 90 ? false : std::get<4>(kBlockMN_kNWarps_Stages_RS); // Whether Q tensor is kept in registers

    // CuTe Shape definitions - these define the logical tensor dimensions for computation tiles
    // CuTe's Shape abstracts multi-dimensional tensor layouts with compile-time dimensions
    // DOR: in our case this is (128, 176, 128)
    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>; // QK^T computation: (M,N,K) = (seqlen_q, seqlen_k, head_dim)
    // DOR: in our case this is (128, 128, 176)
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>; // PV computation: (M,N,K) = (seqlen_q, head_dim_v, seqlen_k)
    using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;                          // Cluster dimensions for cooperative thread blocks (SM90+)

    // Collective operation types - these encapsulate the main computation and output phases
    // CuTe's collective operations coordinate work across thread blocks and manage shared memory
    using CollectiveMainloop = std::conditional_t<
        Arch >= 90,
        // SM90+ mainloop: Advanced features like cluster cooperation, TMA, and warpgroup specialization
        // CollectiveMainloopFwdSm90<2, tuple<C<1>, C<1>, C<1>>, tuple<C<128>, C<176>, C<128>>, 128, bfloat16_t, float, Sm90, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0>
        flash::CollectiveMainloopFwdSm90<
            kStages,             // 2
            ClusterShape,        // tuple<C<1>, C<1>, C<1>>
            TileShape_MNK,       // tuple<C<128>, C<176>, C<128>>
            kHeadDimV,           // 128
            Element,             // bfloat16_t
            float,               // float
            cutlass::arch::Sm90, // Sm90
            Is_causal,           // 0
            Is_local,            // 0
            Has_softcap,         // 0
            Varlen,              // 0
            PagedKVNonTMA,       // 0
            AppendKV,            // 0
            HasQv,               // 0
            MmaPV_is_RS,         // 1
            IntraWGOverlap,      // 1
            PackGQA,             // 0
            Split,               // 0
            V_colmajor,          // 0
            Is_skipable,         // 0
            ReverseSkipList,     // 0
            Phase,               // 0
            HasMustDoList>,      // 0
        // SM80-89 mainloop: Traditional warp-level cooperation with manual shared memory management
        flash::CollectiveMainloopFwdSm80<kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm80,
                                         Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, PackGQA, Split>>;

    // Collective epilogue handles output writing and accumulation using CuTe's tensor operations
    // CollectiveEpilogueFwd<tuple<C<128>, C<128>, C<176>>, tuple<C<1>, C<1>, C<1>>, bfloat16_t, Sm90, 256, 0, 0, 0, 0>,
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
        TileShape_MNK_PV,                  // tuple<C<128>, C<128>, C<176>>
        ClusterShape,                      // tuple<C<1>, C<1>, C<1>>
        ElementOut,                        // bfloat16_t
        ArchTag,                           // Sm90
        CollectiveMainloop::NumMmaThreads, // 256
        Varlen,                            // 0
        PackGQA,                           // 0
        Split,                             // 0
        FP8_TransposeV                     // 0
        >;

    // Thread organization and scheduler selection logic
    static constexpr int NumProducerThreads = Arch >= 90 ? CollectiveMainloop::NumProducerThreads : CollectiveMainloop::NumMmaThreads;
    static constexpr bool LPT = Is_causal || Is_local; // Lower-triangular pattern for causal/local attention
    // DOR: how does it work?
    static constexpr bool Sort = !Is_local; // Whether to sort tiles for better load balancing

    // Persistent scheduler types - these manage work distribution across streaming multiprocessors
    // CuTe-based schedulers coordinate tile processing order for optimal memory access patterns
    using SchedulerPersistent = std::conditional_t<Varlen,
                                                   // Variable length sequences require dynamic scheduling with prepared metadata
                                                   flash::VarlenDynamicPersistentTileScheduler<kBlockM, kBlockN, CollectiveMainloop::NumMmaThreads, NumProducerThreads,
                                                                                               Split, PackGQA, Arch >= 90 /*WarpSpecialized*/, LPT, Sort, true /*Prepared*/>,
                                                   std::conditional_t<!Is_causal && !Is_local,
                                                                      // Non-causal, non-local attention can use static scheduling for maximum throughput
                                                                      flash::StaticPersistentTileScheduler<Split>,
                                                                      // Causal/local attention requires dynamic scheduling to handle irregular work patterns
                                                                      flash::DynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads, NumProducerThreads, Split, PackGQA, Arch >= 90 /*WarpSpecialized*/>>>;

    // Single tile scheduler for cases where persistent scheduling isn't beneficial
    using SchedulerSingleTile = flash::SingleTileScheduler<Varlen, Split, PackGQA, kBlockM>;
    // Scheduler selection heuristics based on workload characteristics
    // If Split then we probably don't have enough work for PersistentScheduler to be useful.
    // However, if Varlen (e.g., during decode where we have max_seqlens), using PersistentScheduler is better
    // since we'll avoid launching a bunch of thread blocks that immediately exit.
    // On Sm80, noncausal persistent seems a bit slower.
    // static constexpr bool UsePersistentScheduler = Arch >= 90 ? !(Split && !Varlen) : ((Is_causal && !Varlen) || (Varlen && Split));
    // static constexpr bool UsePersistentScheduler = ((Arch >= 90) && !Is_skipable) ? !(Split && !Varlen) : ((Is_causal && !Varlen) || (Varlen && Split));
    static constexpr bool UsePersistentScheduler = !Is_skipable && ((Arch >= 90) ? !(Split && !Varlen) : ((Is_causal && !Varlen) || (Varlen && Split)));
    using Scheduler = std::conditional_t<!UsePersistentScheduler, SchedulerSingleTile, SchedulerPersistent>;

    // Final attention kernel type combining all components with CuTe-based tensor operations
    // The kernel orchestrates mainloop, epilogue, and scheduler to implement Flash Attention
    using AttnKernel = std::conditional_t<
        Arch >= 90,
        // SM90+ kernel with advanced CuTe features: TMA, cluster cooperation, warpgroup specialization
        flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>,
        // SM80-89 kernel with traditional CuTe tensor operations and manual memory management
        flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>>;

    // Extract sequence length parameters for variable length handling
    bool const is_varlen_q = params.cu_seqlens_q;        // Query sequences have variable lengths
    bool const is_varlen_k = params.cu_seqlens_k;        // Key sequences have variable lengths
    bool const is_varlen_k_new = params.cu_seqlens_knew; // New key sequences (for KV cache append) have variable lengths

    // Compute effective sequence lengths and batch sizes for tensor shape construction
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;                 // Use total tokens for varlen, max_seqlen otherwise
    int batch_q = !is_varlen_q ? params.b : 1;                                      // Batch dimension becomes 1 for varlen (tokens are flattened)
    int batch_k = !is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1; // Handle GQA batch indexing
    // CuTe stride construction for V tensor - handles both row-major and column-major layouts
    // CuTe's make_stride creates compile-time stride patterns for efficient memory access
    // _1{} represents a unit stride (contiguous dimension)
    typename CollectiveMainloop::StrideV v_strides =
        cute::conditional_return<!V_colmajor>(
            // Row-major V layout: (seqlen, head_dim, num_heads, batch) strides
            make_stride(params.v_row_stride, _1{}, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0),
            // Column-major V layout: (head_dim, seqlen, num_heads, batch) strides
            make_stride(_1{}, params.v_dim_stride, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0));
    // Construct mainloop arguments using CuTe's tensor abstraction
    // CuTe tensor arguments combine data pointers with shape and stride information
    typename CollectiveMainloop::Arguments mainloop_args{
        // ptr_Q
        static_cast<Element const *>(params.q_ptr),
        // shape_Q
        {seqlen_q, params.d, params.h, batch_q}, // shape_Q: CuTe shape for Q tensor (seqlen, head_dim, num_heads, batch)
        // stride_Q
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0}, // stride_Q: CuTe stride pattern
        // ptr_K
        static_cast<Element *>(params.k_ptr),
        // shape_K
        // shape_K: Handle paged KV cache vs. regular tensor layout
        {!params.page_table ? (!is_varlen_k ? params.seqlen_k : params.total_k) : params.page_size,
         params.d, params.h_k, !params.page_table ? batch_k : params.num_pages}, // CuTe shape for K tensor
        // stride_K
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0}, // stride_K: CuTe stride pattern
        // ptr_V
        static_cast<ElementV *>(params.v_ptr),
        // headdim_v
        params.dv, // headdim_v: V tensor head dimension (can differ from Q/K)
        // stride_V
        v_strides, // stride_V: CuTe stride pattern constructed above
        // ptr_K_new
        static_cast<Element const *>(params.knew_ptr),
        // shape_K_new
        {!is_varlen_k_new ? params.seqlen_knew : params.total_knew, params.d, params.h_k, !is_varlen_k_new ? params.b : 1}, // shape_K_new: CuTe shape for new K tensor (KV cache append)
        // stride_K_new
        {params.knew_row_stride, _1{}, params.knew_head_stride, !is_varlen_k_new ? params.knew_batch_stride : 0}, // stride_K_new: CuTe stride pattern
        // ptr_V_new
        static_cast<ElementV const *>(params.vnew_ptr),
        // stride_V_new
        {params.vnew_row_stride, _1{}, params.vnew_head_stride, !is_varlen_k_new ? params.vnew_batch_stride : 0}, // stride_V_new: CuTe stride pattern for new V tensor
        // ptr_Qv
        static_cast<Element const *>(params.qv_ptr),
        // stride_Qv
        {params.qv_row_stride, _1{}, params.qv_head_stride, !is_varlen_q ? params.qv_batch_stride : 0}, // stride_Qv: CuTe stride for fused QV tensor
        static_cast<Element const *>(params.rotary_cos_ptr),
        {params.seqlen_k, params.rotary_dim / 2}, // shape_rotary: CuTe shape for rotary embeddings, the seqlen shape doesn't matter
        {params.rotary_dim / 2, _1{}},            // stride_rotary_cos: CuTe stride pattern for cosine values
        static_cast<Element const *>(params.rotary_sin_ptr),
        {params.rotary_dim / 2, _1{}}, // stride_rotary_sin: CuTe stride pattern for sine values
        params.is_rotary_interleaved,  // RoPE interleaving pattern flag
        params.page_table,             // Paged KV cache page table pointer
        // shape_page_table: CuTe shape for page table (batch, num_pages)
        // if page_size is not set, avoid dividing by zero
        {params.kv_batch_idx ? params.b_k : params.b, !params.page_table ? 0 : params.seqlen_k / params.page_size},
        {params.page_table_batch_stride, _1{}}, // stride_page_table: CuTe stride pattern for page table
        params.scale_softmax,                   // Softmax scaling factor
        // FP8 descaling pointers for quantized tensors
        params.q_descale_ptr,
        params.k_descale_ptr,
        params.v_descale_ptr,
        // CuTe stride patterns for FP8 descaling tensors
        {params.q_descale_batch_stride, params.q_descale_head_stride}, // Q descale strides (FP8)
        {params.k_descale_batch_stride, params.k_descale_head_stride}, // K descale strides (FP8)
        {params.v_descale_batch_stride, params.v_descale_head_stride}, // V descale strides
        // INT8 descale strides (batch, head, block)
        {params.q_descale_batch_stride, params.q_descale_head_stride, params.q_descale_block_stride}, // Q descale strides (INT8)
        {params.k_descale_batch_stride, params.k_descale_head_stride, params.k_descale_block_stride}, // K descale strides (INT8)
        params.window_size_left,
        params.window_size_right,
        params.attention_chunk, // Local attention window parameters
        params.softcap,         // Softmax capping value to prevent attention saturation
        params.num_splits,      // Split-K parallelization factor
        params.kv_batch_idx,    // GQA batch indexing array
        // Variable length sequence metadata arrays
        params.cu_seqlens_q,
        params.cu_seqlens_k,
        params.cu_seqlens_knew, // Cumulative sequence lengths
        params.seqused_q,
        params.seqused_k, // Actual sequence lengths used (for padding)
        params.leftpad_k,
        params.seqlens_rotary, // Left padding for K and rotary position lengths
        params.qk_skip_mask_args,
    };
    // Construct epilogue arguments for output tensor handling using CuTe abstractions
    typename CollectiveEpilogue::Arguments epilogue_args{
        static_cast<ElementOut *>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q, params.num_splits},                                                                            // shape_O: CuTe shape for output tensor (seqlen, head_dim_v, num_heads, batch, splits)
        {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen_q ? params.o_batch_stride : 0, 0},                                         // stride_O: CuTe stride pattern for output
        static_cast<float *>(params.oaccum_ptr),                                                                                                // Partial output accumulation buffer for split-K
        {params.oaccum_row_stride, _1{}, params.oaccum_head_stride, !is_varlen_q ? params.oaccum_batch_stride : 0, params.oaccum_split_stride}, // stride_O_partial: CuTe stride for partial outputs
        static_cast<float *>(params.softmax_lse_ptr),                                                                                           // Log-sum-exp values for stable softmax
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, 0},                                                                            // stride_LSE: CuTe stride pattern for LSE values
        static_cast<float *>(params.softmax_lseaccum_ptr),                                                                                      // Partial LSE accumulation for split-K
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, params.h * seqlen_q * batch_q},                                                // stride_LSE_partial: CuTe stride for partial LSE
        params.h_k,                                                                                                                             // Number of key/value heads (for GQA)
        params.cu_seqlens_q,
        params.seqused_q // Variable length sequence metadata
    };

    // Calculate grid dimensions using CuTe shape introspection
    int qhead_per_khead = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k); // Query heads per key head for GQA
    // Use CuTe's get<> to extract tile dimensions at compile time
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
    num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{})); // Round up for cluster alignment
    // DOR: the total number of Q blocks is ceil_div(len(Q), 128) and

    // printf("Device pointer: %p\n", mainloop_args.qk_skip_mask_args.attn_read_list);

    // Scheduler arguments for coordinating work distribution across SMs
    typename flash::TileSchedulerArguments scheduler_args{
        // grid shape: ceil_div(len(Q), 128), 32, 3, 1
        num_blocks_m, !PackGQA ? params.h : params.h_k, params.b, params.num_splits, // Grid dimensions
        params.h / params.h_k,                                                       // GQA ratio for load balancing
        params.seqlen_q,                                                             // Query sequence length
        params.seqlen_k, params.d, params.dv, sizeof(Element),                       // Tensor dimensions and element size
        params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q,          // Synchronization and varlen metadata
        params.num_splits_dynamic_ptr,                                               // Dynamic split-K configuration
        params.num_m_blocks_ptr,                                                     // Dynamic M-block count
        params.varlen_batch_idx_ptr,                                                 // Variable length batch indexing
        params.num_nheads_in_l2_ptr                                                  // L2 cache optimization hint
    };

    // Pre-compute variable length sequence metadata for efficient scheduling
    if (Varlen && !params.skip_scheduler_metadata_computation)
    {
        prepare_varlen_num_blocks(params, stream, PackGQA, kBlockM, kBlockN, Arch >= 90 && params.prepare_varlen_pdl /*enable_pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
    }

    // Convert CuTe-based arguments to kernel parameters
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args});

    // Extract launch configuration from CuTe kernel introspection
    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params); // Grid dimensions computed by CuTe
    dim3 block_dims = AttnKernel::get_block_shape();            // Thread block dimensions from CuTe collective operations
    int smem_size = AttnKernel::SharedStorageSize;              // Shared memory requirements calculated by CuTe
    // Debugging: CuTe tensor storage size introspection (commented out for performance)
    // int smem_size_q = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_q));
    // int smem_size_k = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v));
    // printf("smem_size = %d, q = %d, k = %d, v = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v);

    // Launch kernel with appropriate method based on cluster configuration
    // CuTe's size<> introspects cluster dimensions at compile time
    if constexpr (size(ClusterShape{}) > 1)
    {
        // Cluster launch path for SM90+ with cooperative thread blocks
        // CuTe enables cluster-level coordination for improved performance
        void const *kernel = (void const *)cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024)
        {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        // Extract cluster dimensions using CuTe's compile-time size introspection
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
        cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
    }
    else
    {
        // Standard launch path for single thread block per cluster
        auto kernel = cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024)
        {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        // kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);  // Direct launch alternative
        cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params,
                                           Arch >= 90 && Varlen && !params.skip_scheduler_metadata_computation && params.prepare_varlen_pdl /*launch_with_pdl*/);
    }
    CHECK_CUDA_KERNEL_LAUNCH(); // Verify successful kernel launch
}

template <int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream)
{
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "Only 16bit and 8bit are supported");
    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
    static constexpr bool Is_INT8 = cute::is_same_v<T, int8_t>;
    static constexpr bool Is_8Bit = Is_FP8 || Is_INT8;
    using T_out = std::conditional_t<!Is_8Bit, T, cutlass::bfloat16_t>;
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&]
                        { VCOLMAJOR_SWITCH(params.v_dim_stride != 1, V_colmajor_, [&]
                                           {
            static constexpr bool V_colmajor = V_colmajor_ && sizeof(T) == 1;
            VARLEN_SWITCH(params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k, Varlen, [&] {
                // Reorder switches: HasQV_ -> AppendKV -> Is_skipable -> HasMustDoList -> ReverseSkipList -> Phase
                // This ensures invalid combinations (HasMustDoList or ReverseSkipList true when Is_skipable false) are never generated
                BOOL_SWITCH(params.qv_ptr, HasQV_, [&] {
                    static constexpr bool HasQv = HasQV_ && Arch == 90 && !Is_8Bit && kHeadDim == 64 && kHeadDimV >= 256;
                    APPENDKV_SWITCH(params.knew_ptr, AppendKV, [&] {
                        BOOL_SWITCH(params.is_skipable, Is_skipable, [&] {
                            // Only needed here to decide if we should use cluster
                            static constexpr int kBlockM = Arch >= 90 ? std::get<0>(tile_size_fwd_sm90(kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(T) /*element_size*/, V_colmajor, PagedKVNonTMA, Has_softcap, Is_skipable, Is_INT8)) : 128;
                            // DOR: in our case this is always true since kHeadDim == 128, Arch == 90 ...
                            static constexpr bool Enable_cluster = Arch == 90 && (sizeof(T) == 2 ? (kHeadDim >= 128) : (kHeadDim == 192)) && !Is_causal && !Is_local && !Split && !PagedKVNonTMA && !Varlen;
                            // Only use Cluster if number of tiles along seqlen_q is even and not varlen
                            CLUSTER_SWITCH(cutlass::ceil_div(params.seqlen_q * (!PackGQA ? 1 : params.h / params.h_k), kBlockM) % 2 == 0, Use_cluster, [&] {
                                static constexpr int ClusterM = Enable_cluster && Use_cluster ? 2 : 1;
                                if constexpr (!Is_skipable) {
                                    // When Is_skipable is disabled, use default values
                                    static constexpr bool HasMustDoList = false;
                                    static constexpr bool ReverseSkipList = false;
                                    static constexpr bool Phase = true;
                                    run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, PackGQA, Split, V_colmajor, Is_skipable, ReverseSkipList, Phase, HasMustDoList>(params, stream);
                                } else {
                                    // When Is_skipable is enabled, check HasMustDoList, ReverseSkipList, and Phase
                                    BOOL_SWITCH(params.has_must_do_list, HasMustDoList, [&] {
                                        BOOL_SWITCH(params.reverse_skip_list, ReverseSkipList, [&] {
                                            BOOL_SWITCH(params.phase, Phase, [&] {
                                                run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, PackGQA, Split, V_colmajor, Is_skipable, ReverseSkipList, Phase, HasMustDoList>(params, stream);
                                            });
                                        });
                                    });
                                }
                            });
                        });
                    });
                });
            }); }); });
} // End run_mha_fwd_ - CuTe-powered Flash Attention dispatch wrapper
