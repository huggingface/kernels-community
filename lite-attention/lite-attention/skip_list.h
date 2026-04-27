#pragma once

#include "cute/tensor.hpp"

namespace flash
{

    using namespace cute;

    // ============================================================================
    // Unified helper struct for reading skip lists and must-do lists
    // Encapsulates all the logic for iterating through list ranges
    // Template parameters:
    //   - IsSkipList: true for SkipList, false for MustDoList
    //   - Reverse: whether to read the list in reverse
    //   - Phase: only used when IsSkipList=true, controls step direction
    // ============================================================================
    template <bool IsSkipList, bool Reverse, bool Phase = true>
    struct ListReader
    {
        const int16_t *list_ptr;
        int list_len;
        int read_idx;
        int start_idx;
        int end_idx;

        static constexpr int step = Phase ? 1 : -1;
        /*
        For SkipList reverse with phase=1:
        [2, 30, -1] -> [2, 0, 31]
        For SkipList reverse with phase=0:
        [2, 0, 31] -> [2, 30, -1]
        For MustDoList reverse:
        Uses -1 offset
        */

        // Initialize the reader with calculated offset
        template <typename TileShape_MNK, typename ParamsType>
        __device__
        void init(const ParamsType &params, int bidb, int bidh, int m_block)
        {
            if constexpr (IsSkipList) {
                // SkipList initialization: calculate per-query-block offset
                static constexpr int kBlockM = get<0>(TileShape_MNK{});
                static constexpr int kBlockN = get<1>(TileShape_MNK{});
                
                int const num_heads = get<2>(params.shape_Q);
                uint64_t const num_q_blocks = cute::ceil_div(get<0>(params.shape_Q), kBlockM);
                uint64_t const num_k_blocks = cute::ceil_div(get<0>(params.shape_K), kBlockN) + 2;
                const uint32_t q_i = ((uint32_t)m_block);
                uint64_t mask_offset = (static_cast<uint64_t>(bidb) * num_heads * num_q_blocks * num_k_blocks) + 
                                       (static_cast<uint64_t>(bidh) * num_q_blocks * num_k_blocks) + 
                                       (static_cast<uint64_t>(q_i) * num_k_blocks);
                
                list_ptr = &params.qk_skip_mask_args.attn_read_list[mask_offset];
            } else {
                // MustDoList initialization: single global list
                list_ptr = &params.qk_skip_mask_args.attn_must_do_list[0];
            }

            list_len = list_ptr[0];
            read_idx = Reverse ? list_len : 1;

            // we ignore the edge case which list_len == 0 because even in this case
            // we will be better off loading the first range because it's like to use the first range 2 timesteps ago
            load_range();
            advance();
        }

        __device__
        void load_range()
        {
            if constexpr (!Reverse) {
                start_idx = flash::warp_uniform(list_ptr[read_idx]);
                end_idx = flash::warp_uniform(list_ptr[read_idx + 1]);
            } else {
                start_idx = flash::warp_uniform(list_ptr[read_idx] + step);
                end_idx = flash::warp_uniform(list_ptr[read_idx - 1] + step);
            }
        }

        // Advance to the next list range
        __device__
        void advance()
        {
            if constexpr (!Reverse) {
                read_idx += 2;
            } else {
                read_idx -= 2;
            }
        }

        // Check if we have more ranges to process
        __device__
        bool has_more()
        {
            if constexpr (!Reverse) {
                return flash::warp_uniform(read_idx <= list_len);
            } else {
                return flash::warp_uniform(read_idx >= 1);
            }
        }

        __device__
        int last_n_block() const
        {
            if constexpr (!Reverse) {
                return flash::warp_uniform(list_ptr[list_len] + 1);
            } else {
                return flash::warp_uniform(list_ptr[1]);
            }
        }

        __device__
        bool is_n_block_in_range(int n_block) const
        {
            if constexpr (!Reverse) {
                return flash::warp_uniform(n_block >= start_idx && n_block < end_idx);
            } else {
                return flash::warp_uniform(n_block <= start_idx && n_block > end_idx);
            }
        }

        __device__
        bool has_passed_current_range(int n_block) const
        {
            if constexpr (!Reverse) {
                return flash::warp_uniform(n_block >= end_idx);
            } else {
                return flash::warp_uniform(n_block <= end_idx);
            }
        }

        __device__
        bool find_range(int n_block)
        {
            bool found = is_n_block_in_range(n_block);
            while (has_more() && !found && has_passed_current_range(n_block)) {
                load_range();
                advance();
                found = is_n_block_in_range(n_block);
            }
            return found;
        }
    };

    // ============================================================================
    // Type aliases for backward compatibility
    // ============================================================================
    template <bool ReverseMustDoList>
    using MustDoListReader = ListReader<false, ReverseMustDoList, !ReverseMustDoList>;

    template <bool ReverseSkipList, bool Phase = true>
    using SkipListReader = ListReader<true, ReverseSkipList, Phase>;

    // ============================================================================
    // Helper struct for writing skip lists
    // Encapsulates all the logic for updating skip lists based on skip detection
    // ============================================================================
    template <bool ReverseMustDoList, bool HasMustDoList>
    struct SkipListWriter
    {
        // int *list_ptr;
        int16_t *list_ptr;
        int write_idx = 1;
        bool is_skipping = true;
        
        static constexpr bool Phase = !ReverseMustDoList;

        // Initialize the writer with calculated offset
        template <typename TileShape_MNK, typename ParamsType>
        __device__
        void init(const ParamsType &params, int bidb, int bidh, int m_block)
        {
            static constexpr int kBlockM = get<0>(TileShape_MNK{});
            static constexpr int kBlockN = get<1>(TileShape_MNK{});
            
            int const num_heads = get<2>(params.shape_Q);
            uint64_t const num_q_blocks = cute::ceil_div(get<0>(params.shape_Q), kBlockM);
            uint64_t const num_k_blocks = cute::ceil_div(get<0>(params.shape_K), kBlockN) + 2;
            const uint32_t q_i = ((uint32_t)m_block);
            uint64_t mask_offset = (static_cast<uint64_t>(bidb) * num_heads * num_q_blocks * num_k_blocks) + 
                                   (static_cast<uint64_t>(bidh) * num_q_blocks * num_k_blocks) + 
                                   (static_cast<uint64_t>(q_i) * num_k_blocks);
            
            list_ptr = &params.qk_skip_mask_args.attn_write_list[mask_offset];
        }

        // Record a transition in skip state
        __device__
        void record_transition(bool skip, int n_block)
        {
            if (skip != is_skipping)
            {
                list_ptr[write_idx] = n_block;
                write_idx++;
                is_skipping = skip;
            }
        }

        // Record the end of a range (force transition to skipping)
        __device__
        void record_range_end(bool skip, int end_idx)
        {
            is_skipping = true;
            if (skip != is_skipping)
            {
                list_ptr[write_idx] = end_idx;
                write_idx++;
            }
        }

        // Finalize the skip list by writing the count
        __device__
        void finalize()
        {
            list_ptr[0] = write_idx - 1;
        }
    };

    // ============================================================================
    // Delayed wrapper for SkipListReader using circular buffer
    // Buffers operations and replays them after a specified delay
    // This allows the reader to lag behind the writer by DelayAmount iterations
    // ============================================================================
    template <int DelayAmount, int NumMmaWarpGroups>
    struct DelayedSkipListReader
    {
        static constexpr int BufferSize = DelayAmount * 2;
        
        // Pointers to shared memory buffers
        int* n_blocks_buffer;
        int (*skip_tests)[4];
        int* last_n_block;

        // we start with -1 because the first call to next_n_block will increment it to 0.
        int index = -1;

        // Constructor to initialize with shared memory pointers
        __device__
        DelayedSkipListReader(int* n_blocks, int (*skip)[4], int* final_n_block)
            : n_blocks_buffer(n_blocks), 
              skip_tests(skip), last_n_block(final_n_block) {}


        __device__
        int next_n_block()
        {
            // issue: many uniform instructions!
            index = (index + 1) % BufferSize;
            return flash::warp_uniform(n_blocks_buffer[index]);
            // index = (index + 1) % BufferSize;
            // return n_blocks_buffer[index];
        }

        __device__
        void update_skip(bool skip, int warp_idx_in_warpgroup){
            if constexpr (NumMmaWarpGroups > 2) {
                atomicAnd(&(skip_tests[index][warp_idx_in_warpgroup]), static_cast<int>(skip));
            }else{
                skip_tests[index][warp_idx_in_warpgroup] &= static_cast<int>(skip);
            }
        }

        __device__
        bool has_more(int n_block)
        {
            return flash::warp_uniform(*last_n_block != n_block);
            // return *last_n_block != n_block;
        }

    };

    // ============================================================================
    // Delayed wrapper for SkipListWriter using circular buffer
    // Buffers operations and replays them after a specified delay
    // This allows the writer to lag behind the reader by DelayAmount iterations
    // ============================================================================
    template <int DelayAmount, bool ReverseSkipList, bool Phase, bool HasMustDoList>
    struct DelayedSkipListWriter
    {
        static constexpr int BufferSize = DelayAmount * 2;
        
        // Pointers to shared memory buffers
        int* n_blocks_buffer;
        int* end_range_buffer;
        int (*skip_tests)[4];

        //should reside in thread registers.
        // SkipListWriter<ReverseSkipList && !Phase, HasMustDoList> writer;
        SkipListWriter<(Phase == false), HasMustDoList> writer;
        bool replayed_skip;
        int record_idx = -1;
        int replay_idx = DelayAmount - 1;

        // Constructor to initialize with shared memory pointers
        __device__
        DelayedSkipListWriter(int* n_blocks, int* end_range, int (*skip)[4])
            : n_blocks_buffer(n_blocks), end_range_buffer(end_range), 
              skip_tests(skip) {}

        /*
        DelayAmount = 4, and we iterate over the range [5, 0] example:
        Producer - record n_block=5 with record_idx = 0 -> load K0
        Consumer - waits for K0 to load -> load n_block=5 with index = 0 -> QK0 -> release K0 -> update skip with index 0
        Producer - record n_block=4 with record_idx = 1 -> load K1
        Consumer - waits for K1 to load -> load n_block=4 with index = 1 -> QK1 -> release K1 -> update skip with index 1
        Producer - replay n_block=nothing with replay_idx=2 -> load V0
        Consumer - waits for V0 to load -> PV0 -> release V0
        Producer - waits for K0 release -> record n_block=3 with record_idx = 2 -> load K2
        Consumer - waits for K2 to load -> load n_block=3 with index = 2 -> QK2 -> release K2 -> update skip with index 2
        Producer - replay n_block=nothing with replay_idx=3 -> load V1
        Consumer - waits for V1 to load -> PV1 -> release V1
        Producer - waits for K1 release -> record n_block=2 with record_idx = 3 -> load K3
        Consumer - waits for K3 to load -> load n_block=2 with index = 3 -> QK3 -> release K3 -> update skip with index 3
        Producer - waits for V0 release -> replay n_block=5 with replay_idx=0 -> load V2
        Consumer - waits for V2 to load -> PV2 -> release V2
        Producer - waits for K2 release -> record n_block=1 with record_idx = 0 -> load K4
        Consumer - waits for K4 to load -> load n_block=1 with index = 0 -> QK4 -> release K4 -> update skip with index 0
        */
        
        // Initialize the underlying writer
        template <typename TileShape_MNK, typename ParamsType>
        __device__
        void init(const ParamsType &params, int bidb, int bidh, int m_block)
        {
            writer.template init<TileShape_MNK>(params, bidb, bidh, m_block);
            for (int i = 0; i < BufferSize; ++i) {
                // init everything to true because we would write something only when we
                // encounter a false skip result. and now it could happen only after the buffers are full.
                skip_tests[i][0] = 1;
                skip_tests[i][1] = 1;
                skip_tests[i][2] = 1;
                skip_tests[i][3] = 1;
                end_range_buffer[i] = -2;
            }
        }

        // consider: calling this when acquiring K for loading.
        // is_must_do: if true, this n_block is in the MustDoList and should not be skipped
        __device__
        void record_n_block(int n_block, bool is_must_do = false)
        {
            record_idx = (record_idx + 1) % BufferSize;
            // record the current n_block for replay in DelayAmount iterations from now.
            n_blocks_buffer[record_idx] = n_block;
            
            // If this n_block is in the MustDoList, initialize skip_tests to 0 (force computation)
            // Otherwise, initialize to 1 (allow skipping)
            int init_value = is_must_do ? 0 : 1;
            skip_tests[record_idx][0] = init_value;
            skip_tests[record_idx][1] = init_value;
            skip_tests[record_idx][2] = init_value;
            skip_tests[record_idx][3] = init_value;
        }

        __device__
        void record_range_end(int end_idx)
        {
            // we save into previous index!
            // end_range_buffer[(record_idx - 1) % DelayAmount] = end_idx;
            // end_range_buffer[(record_idx + BufferSize - 1) % BufferSize] = end_idx;
            end_range_buffer[record_idx] = end_idx;
        }

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // consider: making these functions private and making replay() public.

        __device__
        void replay_transition()
        {
            // calculate the replayed skip result from DelayAmount ago.
            replayed_skip = 
                skip_tests[replay_idx][0] &
                skip_tests[replay_idx][1] &
                skip_tests[replay_idx][2] &
                skip_tests[replay_idx][3];
            
            // replay the n_block from DelayAmount ago.
            int replayed_n_block = n_blocks_buffer[replay_idx];

            // replay record_transition with the values from DelayAmount ago.
            writer.record_transition(replayed_skip, replayed_n_block);

            // Note: skip_tests reset is now done in record_n_block() when recording the next value
        }

        __device__
        void replay_end_range()
        {
            int replayed_end_idx = end_range_buffer[replay_idx];
            if (replayed_end_idx != -2) {
                writer.record_range_end(replayed_skip, replayed_end_idx);
            }
            end_range_buffer[replay_idx] = -2;
        }

        __device__
        void replay()
        {
            replay_idx = (replay_idx + 1) % BufferSize;
            replay_transition();
            replay_end_range();
            // replay_idx = (replay_idx + 1) % BufferSize;
        }

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        // Finalize by flushing all remaining queue entries
        __device__
        void finalize()
        {
            // replay all of the buffer.
            // we don't need to warry about the buffer not being full becuase we init skip_tests
            // in such a way that it woudn't effect the resulting write skip list.
            for (int i = 0; i < DelayAmount; ++i) {
                replay();
            }

            writer.finalize();
        }
    };

    template <const int BufferSize, bool ReverseSkipList, bool Phase, bool HasMustDoList>
    struct SkipListStorage
    {
        alignas(16) int n_blocks_buffer[BufferSize]; // 4
        alignas(16) int end_range_buffer[BufferSize]; // 4
        alignas(16) int skip_tests[BufferSize][4]; // 16
        int last_n_block[1]; // 4
        SkipListReader<ReverseSkipList, Phase> reader;
        // DelayedSkipListWriter<BufferSize / 2, ReverseSkipList, Phase, HasMustDoList> writer;  // BufferSize = DelayAmount * 2, so DelayAmount = BufferSize / 2
        MustDoListReader<!Phase> must_do_reader;  // Lives in shared memory similar to reader
    };

} // namespace flash
