// kernels-community: Torch registrations replacing upstream's pybind11 module
// (`csrc/python_api.cpp` and the `register_apis` functions, which are compiled
// out with `DEEP_EP_KERNELS`). Free functions become Torch ops; DeepEP's
// stateful objects become Torch custom classes. `torch-ext/deep_ep/_C.py`
// puts the upstream `deep_ep._C` Python API back on top of these.

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <torch/custom_class.h>
#include <torch/library.h>

#include <nccl.h>

#include <deep_ep/common/compiled.cuh>

// Order matters: the holders must be declared before the buffer headers
#include "utils/event.hpp"
#include "deep_ep_holders.hpp"

#include "buffers/base.hpp"
#include "buffers/bucket.hpp"
#include "buffers/engram.hpp"
#include "buffers/ep.hpp"
#include "buffers/pp.hpp"
#include "comm/api.hpp"
#include "runtime/jit.hpp"

#include "registration.h"

namespace deep_ep::kernels {

using IntList = std::vector<int64_t>;

inline std::vector<int> to_int_vector(const IntList& values) {
    return {values.begin(), values.end()};
}

inline std::optional<int> to_optional_int(const std::optional<int64_t>& value) {
    return value.has_value() ? std::optional<int>(static_cast<int>(*value)) : std::nullopt;
}

inline std::optional<std::vector<int>> to_optional_int_vector(const std::optional<IntList>& values) {
    return values.has_value() ? std::optional(to_int_vector(*values)) : std::nullopt;
}

template <typename buffer_t>
struct BufferHolder: torch::CustomClassHolder {
    std::unique_ptr<buffer_t> buffer;

    explicit BufferHolder(std::unique_ptr<buffer_t> buffer): buffer(std::move(buffer)) {}

    c10::IValue main_context() const { return wrap_context(buffer->main_context); }
    void print_memory_usage() const { buffer->print_memory_usage(); }
    void destroy() const { buffer->destroy(); }
};

using EPBufferHolder = BufferHolder<ep::EPBuffer>;
using BucketBufferHolder = BufferHolder<bucket::BucketBuffer>;
using EngramBufferHolder = BufferHolder<engram::EngramBuffer>;
using PPBufferHolder = BufferHolder<pp::PPBuffer>;

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

int64_t get_num_allocation_alignment() { return kNumAllocationAlignmentBytes; }
int64_t get_num_tma_alignment() { return kNumTMAAlignmentBytes; }
int64_t get_num_rdma_alignment() { return kNumRDMAAlignmentBytes; }
int64_t get_num_topk_idx_bits() { return EP_NUM_TOPK_IDX_BITS; }

int64_t get_nccl_runtime_version() {
    int version = 0;
    NCCL_CHECK(ncclGetVersion(&version));
    return version;
}

int64_t get_nccl_compile_version() { return NCCL_VERSION_CODE; }

void init_jit(const std::string& library_root_path, const std::string& nccl_root_path) {
    deep_ep::init_jit(library_root_path, nccl_root_path);
}

at::Tensor get_local_nccl_unique_id() {
    const auto unique_id = comm::get_local_unique_id().cast<std::string>();
    auto result = at::empty({static_cast<int64_t>(unique_id.size())}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(result.data_ptr(), unique_id.data(), unique_id.size());
    return result;
}

int64_t create_nccl_comm(const at::Tensor& root_unique_id, int64_t num_ranks, int64_t rank_idx) {
    const auto unique_id = root_unique_id.contiguous().cpu();
    return comm::create_nccl_comm(
        pybind11::bytearray(static_cast<const char*>(unique_id.data_ptr()), unique_id.nbytes()),
        static_cast<int>(num_ranks), static_cast<int>(rank_idx));
}

void destroy_nccl_comm(int64_t nccl_comm) { comm::destroy_nccl_comm(nccl_comm); }

std::tuple<int64_t, int64_t> get_physical_domain_size(int64_t nccl_comm) {
    return comm::get_physical_domain_size(nccl_comm);
}

std::tuple<int64_t, int64_t> get_logical_domain_size(int64_t nccl_comm, bool allow_hybrid_mode) {
    return comm::get_logical_domain_size(nccl_comm, allow_hybrid_mode);
}

// `torch::Stream` is not a schema type: return what `torch.cuda.Stream` needs.
std::tuple<int64_t, int64_t, int64_t> get_comm_stream() {
    const auto stream = comm::get_comm_stream().unwrap();
    return {stream.id(), stream.device_index(), static_cast<int64_t>(stream.device_type())};
}

int64_t calculate_ep_buffer_size(int64_t nccl_comm, int64_t num_max_tokens_per_rank, int64_t hidden,
                                 int64_t num_topk, bool use_fp8_dispatch, bool allow_hybrid_mode,
                                 bool allow_multiple_reduction) {
    return ep::EPBuffer::calculate_buffer_size(
        nccl_comm, static_cast<int>(num_max_tokens_per_rank), static_cast<int>(hidden),
        static_cast<int>(num_topk), use_fp8_dispatch, allow_hybrid_mode, allow_multiple_reduction);
}

int64_t get_ep_buffer_alignment() { return kNumAllocationAlignmentBytes; }

std::tuple<int64_t, int64_t> engram_create_shared_handle(int64_t num_rdma_storage_bytes, bool use_cpu_rdma_storage) {
    return engram::EngramBuffer::create_shared_handle(num_rdma_storage_bytes, use_cpu_rdma_storage);
}

std::tuple<int64_t, int64_t> engram_calculate_storage_size(int64_t nccl_comm, const IntList& num_entries_per_layer,
                                                           int64_t hidden, int64_t elem_size, int64_t num_max_tokens,
                                                           int64_t num_entries_per_token, int64_t num_sf_packs) {
    return engram::EngramBuffer::calculate_storage_size(
        nccl_comm, to_int_vector(num_entries_per_layer), static_cast<int>(hidden), static_cast<int>(elem_size),
        static_cast<int>(num_max_tokens), static_cast<int>(num_entries_per_token), static_cast<int>(num_sf_packs));
}

std::tuple<int64_t, int64_t, bool> engram_get_theoretical_config(int64_t nccl_comm, int64_t num_layers,
                                                                 int64_t num_max_tokens, int64_t num_entries_per_token,
                                                                 double nic_read_mpps,
                                                                 int64_t num_max_read_requests_per_qp) {
    return engram::EngramBuffer::get_theoretical_config(
        nccl_comm, static_cast<int>(num_layers), static_cast<int>(num_max_tokens),
        static_cast<int>(num_entries_per_token), static_cast<float>(nic_read_mpps),
        static_cast<int>(num_max_read_requests_per_qp));
}

// ---------------------------------------------------------------------------
// Class methods
// ---------------------------------------------------------------------------

using ContextPtr = c10::intrusive_ptr<ContextHolder>;
using EventPtr = c10::intrusive_ptr<EventHandleHolder>;
using OptionalEvent = std::optional<EventPtr>;

c10::IValue wrap_event(const std::optional<EventHandle>& event) {
    return pybind11::cast(event);
}

template <typename holder_t>
void register_buffer_base_methods(torch::class_<holder_t>& cls) {
    cls.def("main_context", &holder_t::main_context)
       .def("print_memory_usage", &holder_t::print_memory_usage)
       .def("destroy", &holder_t::destroy);
}

}  // namespace deep_ep::kernels

using namespace deep_ep;
using namespace deep_ep::kernels;

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // Constants and runtime
    ops.def("get_num_allocation_alignment() -> int", &get_num_allocation_alignment);
    ops.def("get_num_tma_alignment() -> int", &get_num_tma_alignment);
    ops.def("get_num_rdma_alignment() -> int", &get_num_rdma_alignment);
    ops.def("get_num_topk_idx_bits() -> int", &get_num_topk_idx_bits);
    ops.def("get_nccl_runtime_version() -> int", &get_nccl_runtime_version);
    ops.def("get_nccl_compile_version() -> int", &get_nccl_compile_version);
    ops.def("init_jit(str library_root_path, str nccl_root_path) -> ()", &deep_ep::kernels::init_jit);

    // Communication
    ops.def("get_local_nccl_unique_id() -> Tensor", &get_local_nccl_unique_id);
    ops.def("create_nccl_comm(Tensor root_unique_id, int num_ranks, int rank_idx) -> int", &create_nccl_comm);
    ops.def("destroy_nccl_comm(int nccl_comm) -> ()", &destroy_nccl_comm);
    ops.def("get_physical_domain_size(int nccl_comm) -> (int, int)", &get_physical_domain_size);
    ops.def("get_logical_domain_size(int nccl_comm, bool allow_hybrid_mode) -> (int, int)", &get_logical_domain_size);
    ops.def("get_comm_stream() -> (int, int, int)", &get_comm_stream);

    // EP
    ops.def("calculate_ep_buffer_size(int nccl_comm, int num_max_tokens_per_rank, int hidden, int num_topk, "
            "bool use_fp8_dispatch, bool allow_hybrid_mode, bool allow_multiple_reduction) -> int",
            &calculate_ep_buffer_size);
    ops.def("get_ep_buffer_alignment() -> int", &get_ep_buffer_alignment);

    // Engram static helpers
    ops.def("engram_create_shared_handle(int num_rdma_storage_bytes, bool use_cpu_rdma_storage) -> (int, int)",
            &engram_create_shared_handle);
    ops.def("engram_calculate_storage_size(int nccl_comm, int[] num_entries_per_layer, int hidden, int elem_size, "
            "int num_max_tokens, int num_entries_per_token, int num_sf_packs) -> (int, int)",
            &engram_calculate_storage_size);
    ops.def("engram_get_theoretical_config(int nccl_comm, int num_layers, int num_max_tokens, "
            "int num_entries_per_token, float nic_read_mpps, int num_max_read_requests_per_qp) -> (int, int, bool)",
            &engram_get_theoretical_config);

    // Events and epilogues
    ops.class_<EventHandleHolder>("EventHandle")
        .def(torch::init<>())
        .def("current_stream_wait", &EventHandleHolder::current_stream_wait);

    ops.class_<EpilogueHolder>("Epilogue")
        .def("run", &EpilogueHolder::run);

    // Communication context
    ops.class_<ContextHolder>("Context")
        .def("get_physical_domain_size", [](const ContextPtr& self) -> std::tuple<int64_t, int64_t> {
            return self->context->get_physical_domain_size();
        })
        .def("get_logical_domain_size", [](const ContextPtr& self) -> std::tuple<int64_t, int64_t> {
            return self->context->get_logical_domain_size();
        })
        .def("num_workspace_bytes", [](const ContextPtr& self) { return self->context->num_workspace_bytes; })
        .def("num_gpu_buffer_bytes", [](const ContextPtr& self) { return self->context->num_gpu_buffer_bytes; })
        .def("num_rdma_storage_bytes", [](const ContextPtr& self) { return self->context->num_rdma_storage_bytes; })
        .def("barrier", [](const ContextPtr& self, bool with_cpu_sync, bool sequential, bool wait_comm_stream) {
            // Same as upstream's `deep_ep._C.barrier`
            const auto& context = self->context;
            EP_HOST_ASSERT(context != nullptr);
            EP_HOST_ASSERT(context->barrier_signals != nullptr);
            comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                          context->num_gpu_timeout_cycles, with_cpu_sync, sequential, true, wait_comm_stream);
        });

    // EP buffer
    auto ep_buffer = ops.class_<EPBufferHolder>("EPBuffer");
    ep_buffer
        .def(torch::init([](int64_t rank_idx, int64_t num_ranks, int64_t nccl_comm,
                            int64_t num_buffer_bytes, int64_t num_lb_buffer_bytes,
                            bool allow_hybrid_mode, bool allow_multiple_reduction, bool prefer_overlap_with_compute,
                            std::optional<int64_t> sl_idx, int64_t num_allocated_qps,
                            int64_t num_cpu_timeout_secs, int64_t num_gpu_timeout_secs,
                            bool explicitly_destroy) {
            return c10::make_intrusive<EPBufferHolder>(std::make_unique<ep::EPBuffer>(
                rank_idx, num_ranks, nccl_comm, num_buffer_bytes, num_lb_buffer_bytes,
                allow_hybrid_mode, allow_multiple_reduction, prefer_overlap_with_compute,
                to_optional_int(sl_idx), num_allocated_qps, num_cpu_timeout_secs, num_gpu_timeout_secs,
                explicitly_destroy));
        }))
        .def("context", [](const c10::intrusive_ptr<EPBufferHolder>& self) {
            return wrap_context(self->buffer->context);
        })
        .def("lb_storage", [](const c10::intrusive_ptr<EPBufferHolder>& self) { return self->buffer->lb_storage; })
        .def("dispatch", [](const c10::intrusive_ptr<EPBufferHolder>& self,
                            const at::Tensor& x,
                            const std::optional<at::Tensor>& sf,
                            const at::Tensor& topk_idx,
                            const std::optional<at::Tensor>& topk_weights,
                            const std::optional<at::Tensor>& cumulative_local_expert_recv_stats,
                            std::optional<int64_t> cached_num_recv_tokens,
                            std::optional<int64_t> cached_num_expanded_tokens,
                            const std::optional<IntList>& cached_num_recv_tokens_per_expert_list,
                            const std::optional<at::Tensor>& cached_psum_num_recv_tokens_per_scaleup_rank,
                            const std::optional<at::Tensor>& cached_psum_num_recv_tokens_per_expert,
                            const std::optional<at::Tensor>& cached_num_unaligned_recv_tokens_per_expert,
                            const std::optional<at::Tensor>& cached_dst_buffer_slot_idx,
                            const std::optional<at::Tensor>& cached_token_metadata_at_forward,
                            const std::optional<at::Tensor>& cached_recv_src_metadata,
                            const std::optional<at::Tensor>& cached_channel_linked_list,
                            int64_t num_max_tokens_per_rank,
                            int64_t num_experts, int64_t expert_alignment,
                            int64_t num_sms, int64_t num_qps,
                            const OptionalEvent& previous_event,
                            bool async_with_compute_stream,
                            bool allocate_on_comm_stream,
                            bool do_cpu_sync,
                            bool do_expand, bool do_zero_padding,
                            bool use_tma_aligned_col_major_sf,
                            bool defer_epilogue) -> c10::IValue {
            return self->buffer->dispatch(
                x, sf, topk_idx, topk_weights, cumulative_local_expert_recv_stats,
                to_optional_int(cached_num_recv_tokens), to_optional_int(cached_num_expanded_tokens),
                to_optional_int_vector(cached_num_recv_tokens_per_expert_list),
                cached_psum_num_recv_tokens_per_scaleup_rank, cached_psum_num_recv_tokens_per_expert,
                cached_num_unaligned_recv_tokens_per_expert, cached_dst_buffer_slot_idx,
                cached_token_metadata_at_forward, cached_recv_src_metadata, cached_channel_linked_list,
                num_max_tokens_per_rank, num_experts, expert_alignment, num_sms, num_qps,
                unwrap_event(previous_event),
                async_with_compute_stream, allocate_on_comm_stream, do_cpu_sync,
                do_expand, do_zero_padding, use_tma_aligned_col_major_sf, defer_epilogue);
        })
        .def("combine", [](const c10::intrusive_ptr<EPBufferHolder>& self,
                           const at::Tensor& x,
                           const std::optional<at::Tensor>& topk_weights,
                           const std::optional<at::Tensor>& bias_0,
                           const std::optional<at::Tensor>& bias_1,
                           const at::Tensor& src_metadata,
                           const at::Tensor& combined_topk_idx,
                           const at::Tensor& psum_num_recv_tokens_per_scaleup_rank,
                           const std::optional<at::Tensor>& token_metadata_at_forward,
                           const std::optional<at::Tensor>& channel_linked_list,
                           int64_t num_experts,
                           int64_t num_max_tokens_per_rank,
                           int64_t num_sms, int64_t num_qps,
                           const OptionalEvent& previous_event,
                           bool async_with_compute_stream,
                           bool allocate_on_comm_stream,
                           bool use_expanded_layout,
                           bool defer_epilogue) -> c10::IValue {
            return self->buffer->combine(
                x, topk_weights, bias_0, bias_1, src_metadata, combined_topk_idx,
                psum_num_recv_tokens_per_scaleup_rank, token_metadata_at_forward, channel_linked_list,
                num_experts, num_max_tokens_per_rank, num_sms, num_qps,
                unwrap_event(previous_event),
                async_with_compute_stream, allocate_on_comm_stream, use_expanded_layout, defer_epilogue);
        })
        .def("lb_prefetch_weights", [](const c10::intrusive_ptr<EPBufferHolder>& self,
                                       const std::vector<at::Tensor>& redundant_expert_weights,
                                       const std::vector<at::Tensor>& expert_weights,
                                       const at::Tensor& redundancy_mapping,
                                       int64_t num_sms,
                                       const OptionalEvent& previous_event) -> c10::IValue {
            return wrap_event(self->buffer->lb_prefetch_weights(
                redundant_expert_weights, expert_weights, redundancy_mapping, num_sms, unwrap_event(previous_event)));
        })
        .def("lb_reduce_grads", [](const c10::intrusive_ptr<EPBufferHolder>& self,
                                   const at::Tensor& redundant_expert_grads,
                                   const at::Tensor& expert_grads,
                                   const at::Tensor& redundancy_mapping,
                                   int64_t num_sms,
                                   const OptionalEvent& previous_event) -> c10::IValue {
            return wrap_event(self->buffer->lb_reduce_grads(
                redundant_expert_grads, expert_grads, redundancy_mapping, num_sms, unwrap_event(previous_event)));
        });
    register_buffer_base_methods(ep_buffer);

    // Bucket buffer (experimental upstream)
    auto bucket_buffer = ops.class_<BucketBufferHolder>("BucketBuffer");
    bucket_buffer
        .def(torch::init([](const IntList& rank_indices, const IntList& num_ranks, const IntList& nccl_comms,
                            int64_t num_buffer_bytes, std::optional<int64_t> sl_idx, int64_t num_gpu_timeout_secs,
                            double nvl_gbs, double rdma_gbs, bool explicitly_destroy) {
            return c10::make_intrusive<BucketBufferHolder>(std::make_unique<bucket::BucketBuffer>(
                to_int_vector(rank_indices), to_int_vector(num_ranks), nccl_comms, num_buffer_bytes,
                to_optional_int(sl_idx), num_gpu_timeout_secs,
                static_cast<float>(nvl_gbs), static_cast<float>(rdma_gbs), explicitly_destroy));
        }))
        .def("contexts", [](const c10::intrusive_ptr<BucketBufferHolder>& self) {
            c10::impl::GenericList contexts(c10::AnyType::get());
            for (const auto& context: self->buffer->contexts)
                contexts.push_back(wrap_context(context));
            return c10::IValue(std::move(contexts));
        })
        .def("storage", [](const c10::intrusive_ptr<BucketBufferHolder>& self) { return self->buffer->storage; })
        .def("reduce_scatter", [](const c10::intrusive_ptr<BucketBufferHolder>& self,
                                  const std::vector<at::Tensor>& srcs, int64_t context_idx, int64_t num_sms,
                                  const std::string& comm_precision, double scale) -> c10::IValue {
            return pybind11::cast(self->buffer->reduce_scatter(
                srcs, context_idx, num_sms, comm_precision, static_cast<float>(scale)));
        })
        .def("all_reduce", [](const c10::intrusive_ptr<BucketBufferHolder>& self,
                              const std::vector<at::Tensor>& srcs, int64_t context_idx, int64_t num_sms,
                              double scale) -> c10::IValue {
            return pybind11::cast(self->buffer->all_reduce(srcs, context_idx, num_sms, static_cast<float>(scale)));
        })
        .def("all_gather", [](const c10::intrusive_ptr<BucketBufferHolder>& self,
                              const std::vector<at::Tensor>& srcs, const std::vector<at::Tensor>& dsts,
                              int64_t context_idx, int64_t num_sms) -> c10::IValue {
            return self->buffer->all_gather(srcs, dsts, context_idx, num_sms);
        });
    register_buffer_base_methods(bucket_buffer);

    // Engram buffer (experimental upstream)
    auto engram_buffer = ops.class_<EngramBufferHolder>("EngramBuffer");
    engram_buffer
        .def(torch::init([](int64_t rank_idx, int64_t num_ranks, int64_t nccl_comm,
                            const IntList& shared_comm_flat,
                            int64_t num_gpu_buffer_bytes, int64_t num_rdma_storage_bytes,
                            bool use_cpu_rdma_storage, bool allow_hybrid_mode,
                            std::optional<int64_t> sl_idx, int64_t num_allocated_qps, int64_t qp_depth,
                            int64_t num_gpu_timeout_secs, bool explicitly_destroy) {
            // `(pid, fd)` pairs, flattened
            EP_HOST_ASSERT(shared_comm_flat.size() % 2 == 0);
            symmetric::shared_comm_t shared_comm;
            for (size_t i = 0; i < shared_comm_flat.size(); i += 2)
                shared_comm.emplace_back(static_cast<int>(shared_comm_flat[i]), static_cast<int>(shared_comm_flat[i + 1]));
            return c10::make_intrusive<EngramBufferHolder>(std::make_unique<engram::EngramBuffer>(
                rank_idx, num_ranks, nccl_comm, shared_comm, num_gpu_buffer_bytes, num_rdma_storage_bytes,
                use_cpu_rdma_storage, allow_hybrid_mode, to_optional_int(sl_idx), num_allocated_qps, qp_depth,
                num_gpu_timeout_secs, explicitly_destroy));
        }))
        .def("context", [](const c10::intrusive_ptr<EngramBufferHolder>& self) {
            return wrap_context(self->buffer->context);
        })
        .def("set_config", [](const c10::intrusive_ptr<EngramBufferHolder>& self,
                              const IntList& num_entries_per_layer, int64_t hidden, int64_t elem_size,
                              int64_t num_max_tokens, int64_t num_entries_per_token, int64_t num_sf_packs) {
            self->buffer->set_config(to_int_vector(num_entries_per_layer), hidden, elem_size,
                                     num_max_tokens, num_entries_per_token, num_sf_packs);
        })
        .def("write", [](const c10::intrusive_ptr<EngramBufferHolder>& self,
                         const std::vector<at::Tensor>& storages,
                         const std::optional<std::vector<at::Tensor>>& sfs) {
            self->buffer->write(storages, sfs);
        })
        .def("fetch", [](const c10::intrusive_ptr<EngramBufferHolder>& self,
                         const at::Tensor& indices, int64_t num_qps, bool use_tma_aligned_col_major_sf) -> c10::IValue {
            return pybind11::cast(self->buffer->fetch(indices, num_qps, use_tma_aligned_col_major_sf));
        });
    register_buffer_base_methods(engram_buffer);

    // PP buffer (experimental upstream)
    auto pp_buffer = ops.class_<PPBufferHolder>("PPBuffer");
    pp_buffer
        .def(torch::init([](int64_t rank_idx, int64_t num_ranks, int64_t nccl_comm,
                            int64_t num_max_tensor_bytes, int64_t num_max_inflight_tensors,
                            std::optional<int64_t> sl_idx, int64_t num_gpu_timeout_secs, bool explicitly_destroy) {
            return c10::make_intrusive<PPBufferHolder>(std::make_unique<pp::PPBuffer>(
                rank_idx, num_ranks, nccl_comm, num_max_tensor_bytes, num_max_inflight_tensors,
                to_optional_int(sl_idx), num_gpu_timeout_secs, explicitly_destroy));
        }))
        .def("storage", [](const c10::intrusive_ptr<PPBufferHolder>& self) { return self->buffer->storage; })
        .def("send", [](const c10::intrusive_ptr<PPBufferHolder>& self,
                        const at::Tensor& x, int64_t dst_rank_idx, int64_t num_sms) {
            self->buffer->send(x, dst_rank_idx, num_sms);
        })
        .def("recv", [](const c10::intrusive_ptr<PPBufferHolder>& self,
                        const at::Tensor& x, int64_t src_rank_idx, int64_t num_sms) {
            self->buffer->recv(x, src_rank_idx, num_sms);
        });
    register_buffer_base_methods(pp_buffer);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
