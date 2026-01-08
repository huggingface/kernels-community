#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/mps/MPSStream.h>

#include <internal/metal-kernels.h>
#include <internal/metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace {

class MetalBuffer {
public:
    MetalBuffer() = default;
    MetalBuffer(const MetalBuffer&) = delete;
    MetalBuffer& operator=(const MetalBuffer&) = delete;

    MetalBuffer(MetalBuffer&& other) noexcept
        : buffer_(other.buffer_), has_value_(other.has_value_) {
        other.buffer_ = {};
        other.has_value_ = false;
    }

    MetalBuffer& operator=(MetalBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            buffer_ = other.buffer_;
            has_value_ = other.has_value_;
            other.buffer_ = {};
            other.has_value_ = false;
        }
        return *this;
    }

    ~MetalBuffer() {
        reset();
    }

    gptoss_metal_buffer* get() {
        return &buffer_;
    }

    const gptoss_metal_buffer* get() const {
        return &buffer_;
    }

    void* ptr() const {
        return buffer_.ptr;
    }

    size_t size_bytes() const {
        return buffer_.size;
    }

    bool valid() const {
        return has_value_;
    }

    void wrap(const gptoss_metal_device* device, size_t size, const void* data) {
        reset();
        TORCH_CHECK(gptoss_metal_buffer_wrap(device, size, data, &buffer_) == gptoss_status_success,
            "metal_buffer_wrap failed");
        has_value_ = true;
    }

    void wrap_mps_tensor(const gptoss_metal_device* device, const at::Tensor& tensor) {
        reset();
        TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous for MPS wrap");
        const size_t size_bytes = static_cast<size_t>(tensor.numel()) * tensor.element_size();
        const size_t byte_offset = static_cast<size_t>(tensor.storage_offset()) * tensor.element_size();
        const void* buffer_handle = tensor.storage().data();
        TORCH_CHECK(
            gptoss_metal_buffer_wrap_mtlbuffer(device, buffer_handle, size_bytes, byte_offset, &buffer_)
                == gptoss_status_success,
            "metal_buffer_wrap_mtlbuffer failed");
        has_value_ = true;
    }

    void create(const gptoss_metal_device* device, size_t size, const void* data = nullptr) {
        reset();
        TORCH_CHECK(gptoss_metal_buffer_create(device, size, data, &buffer_) == gptoss_status_success,
            "metal_buffer_create failed");
        has_value_ = true;
    }

    void reset() {
        if (has_value_) {
            (void) gptoss_metal_buffer_release(&buffer_);
            buffer_ = {};
            has_value_ = false;
        }
    }

private:
    gptoss_metal_buffer buffer_{};
    bool has_value_ = false;
};

template <typename EncodeFn>
void run_metal_kernel(const char* kernel_symbol, EncodeFn&& encode_fn) {
    gptoss_metal_device device{};
    gptoss_metal_library library{};
    gptoss_metal_function fn{};
    gptoss_metal_command_queue cq{};
    gptoss_metal_command_buffer cb{};

    bool borrowed_queue = false;
    bool borrowed_buffer = false;
    at::mps::MPSStream* stream = nullptr;

    auto cleanup = [&]() {
        if (!borrowed_buffer) {
            (void) gptoss_metal_command_buffer_release(&cb);
        } else {
            cb = {};
        }
        if (!borrowed_queue) {
            (void) gptoss_metal_command_queue_release(&cq);
        } else {
            cq = {};
        }
        (void) gptoss_metal_function_release(&fn);
        (void) gptoss_metal_library_release(&library);
        (void) gptoss_metal_device_release(&device);
    };

    TORCH_CHECK(gptoss_metal_device_create_system_default(&device) == gptoss_status_success,
        "device_create failed");
    try {
        TORCH_CHECK(gptoss_metal_library_create_default(&device, &library) == gptoss_status_success,
            "library_create failed");
        TORCH_CHECK(gptoss_metal_function_create(&library, kernel_symbol, &fn) == gptoss_status_success,
            "function_create failed");
        stream = at::mps::getCurrentMPSStream();
        if (stream != nullptr) {
            // stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
            stream->endKernelCoalescing();
            void* command_queue_obj = stream->commandQueue();
            if (command_queue_obj != nullptr) {
                TORCH_CHECK(
                    gptoss_metal_command_queue_wrap_existing(command_queue_obj, &cq) == gptoss_status_success,
                    "wrap_command_queue failed");
                borrowed_queue = true;
                void* command_buffer_obj = stream->commandBuffer();
                if (command_buffer_obj != nullptr) {
                    TORCH_CHECK(
                        gptoss_metal_command_buffer_wrap_existing(command_buffer_obj, &cb) == gptoss_status_success,
                        "wrap_command_buffer failed");
                    borrowed_buffer = true;
                }
            }
        }

        if (!borrowed_queue) {
            TORCH_CHECK(gptoss_metal_command_queue_create(&device, &cq) == gptoss_status_success,
                "cq_create failed");
        }
        if (!borrowed_buffer) {
            TORCH_CHECK(gptoss_metal_command_buffer_create(&cq, &cb) == gptoss_status_success,
                "cb_create failed");
        }

        encode_fn(device, fn, cb);

        if (borrowed_buffer) {
            TORCH_CHECK(stream != nullptr, "stream must be valid when using borrowed command buffer");
            stream->synchronize(at::mps::SyncType::COMMIT);
        } else {
            TORCH_CHECK(gptoss_metal_command_buffer_commit(&cb) == gptoss_status_success,
                "commit failed");
            TORCH_CHECK(gptoss_metal_command_buffer_wait_completion(&cb, nullptr) == gptoss_status_success,
                "wait failed");
        }
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
}

at::Tensor to_cpu_contiguous(const at::Tensor& tensor) {
    if (tensor.device().is_cpu() && tensor.is_contiguous()) {
        return tensor;
    }
    return tensor.contiguous().to(at::kCPU);
}

at::Tensor empty_cpu_like(const at::Tensor& tensor) {
    return at::empty_like(tensor, tensor.options().device(at::kCPU)).contiguous();
}

void copy_back(at::Tensor& dst, const at::Tensor& src_cpu) {
    dst.copy_(src_cpu.to(dst.device(), /*non_blocking=*/false, /*copy=*/true));
}

void create_control_buffer(const gptoss_metal_device* device, MetalBuffer& buffer) {
    struct gptoss_control ctrl {0};
    buffer.create(device, sizeof(ctrl), &ctrl);
}

template <typename LaunchFn>
void run_dense_matmul_bf16(const char* kernel_symbol,
    LaunchFn&& launch_fn,
    const at::Tensor& input,
    const at::Tensor& weight_bf16,
    const at::Tensor& bias_bf16,
    at::Tensor& output)
{
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight_bf16.dtype() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(bias_bf16.dtype() == at::kBFloat16, "bias must be bfloat16");
    TORCH_CHECK(output.dtype() == at::kFloat, "output must be float32");

    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight_bf16.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias_bf16.dim() == 1, "bias must be 1D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");

    const int64_t num_tokens = input.size(0);
    const int64_t num_cols = input.size(1);
    const int64_t num_rows = output.size(1);

    TORCH_CHECK(output.size(0) == num_tokens,
        "output first dimension must match number of tokens");
    TORCH_CHECK(weight_bf16.size(0) == num_cols && weight_bf16.size(1) == num_rows,
        "weight shape must be [num_cols, num_rows]");
    TORCH_CHECK(bias_bf16.size(0) == num_rows,
        "bias length must equal number of rows");
    const bool use_mps = input.device().is_mps() && weight_bf16.device().is_mps() &&
        bias_bf16.device().is_mps() && output.device().is_mps();

    at::Tensor input_cpu;
    at::Tensor weight_cpu;
    at::Tensor bias_cpu;
    at::Tensor out_cpu;
    size_t in_bytes = 0;
    size_t weight_bytes = 0;
    size_t bias_bytes = 0;
    size_t out_bytes = 0;

    at::Tensor input_mps;
    at::Tensor weight_mps;
    at::Tensor bias_mps;
    at::Tensor output_mps;

    if (use_mps) {
        input_mps = input.contiguous();
        weight_mps = weight_bf16.transpose(0, 1).contiguous();
        bias_mps = bias_bf16.contiguous();
        output_mps = output.contiguous();
    } else {
        input_cpu = to_cpu_contiguous(input);
        weight_cpu = weight_bf16.transpose(0, 1).contiguous().to(at::kCPU);
        bias_cpu = to_cpu_contiguous(bias_bf16);
        out_cpu = empty_cpu_like(output);

        in_bytes = static_cast<size_t>(input_cpu.numel()) * input_cpu.element_size();
        weight_bytes = static_cast<size_t>(weight_cpu.numel()) * weight_cpu.element_size();
        bias_bytes = static_cast<size_t>(bias_cpu.numel()) * bias_cpu.element_size();
        out_bytes = static_cast<size_t>(out_cpu.numel()) * out_cpu.element_size();
    }

    MetalBuffer input_buf;
    MetalBuffer weight_buf;
    MetalBuffer bias_buf;
    MetalBuffer out_buf;
    MetalBuffer control_buf;

    run_metal_kernel(kernel_symbol, [&](const gptoss_metal_device& device,
                                        const gptoss_metal_function& fn,
                                        gptoss_metal_command_buffer& cb) {
        if (use_mps) {
            input_buf.wrap_mps_tensor(&device, input_mps);
            weight_buf.wrap_mps_tensor(&device, weight_mps);
            bias_buf.wrap_mps_tensor(&device, bias_mps);
            out_buf.wrap_mps_tensor(&device, output_mps);
        } else {
            input_buf.wrap(&device, in_bytes, input_cpu.data_ptr());
            weight_buf.wrap(&device, weight_bytes, weight_cpu.data_ptr());
            bias_buf.wrap(&device, bias_bytes, bias_cpu.data_ptr());
            out_buf.create(&device, out_bytes, nullptr);
        }
        create_control_buffer(&device, control_buf);

        TORCH_CHECK(
            launch_fn(
                &cb, &fn,
                input_buf.get(), 0,
                weight_buf.get(), 0,
                bias_buf.get(), 0,
                out_buf.get(), 0,
                control_buf.get(), 0,
                static_cast<uint32_t>(num_tokens),
                static_cast<uint32_t>(num_cols),
                static_cast<uint32_t>(num_rows)) == gptoss_status_success,
            "encode dense matmul failed");
    });

    if (use_mps) {
        if (!output_mps.is_alias_of(output)) {
            output.copy_(output_mps);
        }
    } else {
        std::memcpy(out_cpu.data_ptr(), out_buf.ptr(), out_bytes);
        copy_back(output, out_cpu);
    }
}

}  // namespace

void f32_bf16w_matmul_torch(const at::Tensor &input,
    const at::Tensor &weight_bf16,
    const at::Tensor &bias_bf16,
    at::Tensor &output,
    int64_t num_tokens, int64_t num_cols, int64_t num_rows, int64_t threadgroup_size)
{
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight_bf16.dtype() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(bias_bf16.dtype() == at::kBFloat16, "bias must be bfloat16");
    TORCH_CHECK(output.dtype() == at::kFloat, "output must be float32");

    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight_bf16.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias_bf16.dim() == 1, "bias must be 1D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(input.size(0) == num_tokens && input.size(1) == num_cols,
                "input shape must be [num_tokens, num_cols]");
    TORCH_CHECK(weight_bf16.size(0) == num_cols && weight_bf16.size(1) == num_rows,
                "weight shape must be [num_cols, num_rows]");
    TORCH_CHECK(bias_bf16.size(0) == num_rows, "bias length must be num_rows");
    TORCH_CHECK(output.size(0) == num_tokens && output.size(1) == num_rows,
                "output shape must be [num_tokens, num_rows]");
    const bool use_mps = input.device().is_mps() && weight_bf16.device().is_mps() &&
        bias_bf16.device().is_mps() && output.device().is_mps();

    at::Tensor input_cpu;
    at::Tensor weight_cpu;
    at::Tensor bias_cpu;
    at::Tensor out_cpu;
    size_t in_bytes = 0;
    size_t wt_bytes = 0;
    size_t bs_bytes = 0;
    size_t out_bytes = 0;

    at::Tensor input_mps;
    at::Tensor weight_mps;
    at::Tensor bias_mps;
    at::Tensor output_mps;

    if (use_mps) {
        input_mps = input.contiguous();
        weight_mps = weight_bf16.transpose(0, 1).contiguous();
        bias_mps = bias_bf16.contiguous();
        output_mps = output.contiguous();
    } else {
        input_cpu = to_cpu_contiguous(input);
        weight_cpu = weight_bf16.transpose(0, 1).contiguous().to(at::kCPU);
        bias_cpu = to_cpu_contiguous(bias_bf16);
        out_cpu = empty_cpu_like(output);

        in_bytes = static_cast<size_t>(num_tokens) * static_cast<size_t>(num_cols) * sizeof(float);
        wt_bytes = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols) * sizeof(uint16_t);
        bs_bytes = static_cast<size_t>(num_rows) * sizeof(uint16_t);
        out_bytes = static_cast<size_t>(num_tokens) * static_cast<size_t>(num_rows) * sizeof(float);
    }

    MetalBuffer in_buf;
    MetalBuffer wt_buf;
    MetalBuffer bs_buf;
    MetalBuffer out_buf;
    MetalBuffer ctrl_buf;

    run_metal_kernel("gptoss_f32_bf16w_matmul", [&](const gptoss_metal_device& device,
                                                    const gptoss_metal_function& fn,
                                                    gptoss_metal_command_buffer& cb) {
        if (use_mps) {
            in_buf.wrap_mps_tensor(&device, input_mps);
            wt_buf.wrap_mps_tensor(&device, weight_mps);
            bs_buf.wrap_mps_tensor(&device, bias_mps);
            out_buf.wrap_mps_tensor(&device, output_mps);
        } else {
            in_buf.wrap(&device, in_bytes, input_cpu.data_ptr());
            wt_buf.wrap(&device, wt_bytes, weight_cpu.data_ptr());
            bs_buf.wrap(&device, bs_bytes, bias_cpu.data_ptr());
            out_buf.create(&device, out_bytes, nullptr);
        }
        create_control_buffer(&device, ctrl_buf);

        TORCH_CHECK(
            gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul(
                &cb, &fn, static_cast<size_t>(threadgroup_size),
                in_buf.get(), 0,
                wt_buf.get(), 0,
                bs_buf.get(), 0,
                out_buf.get(), 0,
                ctrl_buf.get(), 0,
                static_cast<uint32_t>(num_tokens),
                static_cast<uint32_t>(num_cols),
                static_cast<uint32_t>(num_rows)) == gptoss_status_success,
            "encode failed");
    });

    if (use_mps) {
        if (!output_mps.is_alias_of(output)) {
            output.copy_(output_mps);
        }
    } else {
        std::memcpy(out_cpu.data_ptr(), out_buf.ptr(), out_bytes);
        copy_back(output, out_cpu);
    }
}

void bf16_f32_embeddings_torch(const at::Tensor& token_ids,
    const at::Tensor& weight_bf16,
    at::Tensor& output,
    int64_t threadgroup_size)
{
    TORCH_CHECK(token_ids.dtype() == at::kInt || token_ids.dtype() == at::kLong,
        "token_ids must be int32 or int64");
    TORCH_CHECK(weight_bf16.dtype() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(output.dtype() == at::kFloat, "output must be float32");

    TORCH_CHECK(token_ids.dim() == 1, "token_ids must be 1D");
    TORCH_CHECK(weight_bf16.dim() == 2, "weight must be 2D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");

    const int64_t num_tokens = token_ids.size(0);
    TORCH_CHECK(output.size(0) == num_tokens, "output first dimension must match num_tokens");
    const int64_t num_channels = output.size(1);
    TORCH_CHECK(num_channels % 4 == 0, "num_channels must be divisible by 4");
    TORCH_CHECK(weight_bf16.size(1) == num_channels,
        "weight second dimension must equal embedding dimension (num_channels)");

    TORCH_CHECK(threadgroup_size >= 0, "threadgroup_size must be non-negative");
    const bool use_mps = token_ids.device().is_mps() && weight_bf16.device().is_mps() &&
        output.device().is_mps();

    at::Tensor tokens_cpu;
    at::Tensor weight_cpu;
    at::Tensor out_cpu;
    size_t token_bytes = 0;
    size_t weight_bytes = 0;
    size_t out_bytes = 0;

    at::Tensor tokens_mps;
    at::Tensor weight_mps;
    at::Tensor output_mps;

    if (use_mps) {
        tokens_mps = (token_ids.dtype() == at::kInt ? token_ids : token_ids.to(at::kInt)).contiguous();
        weight_mps = weight_bf16.contiguous();
        output_mps = output.contiguous();
    } else {
        tokens_cpu = token_ids.dtype() == at::kInt
            ? to_cpu_contiguous(token_ids)
            : token_ids.to(at::kInt).contiguous().to(at::kCPU);
        weight_cpu = to_cpu_contiguous(weight_bf16);
        out_cpu = empty_cpu_like(output);

        token_bytes = static_cast<size_t>(num_tokens) * sizeof(uint32_t);
        weight_bytes = static_cast<size_t>(weight_cpu.numel()) * weight_cpu.element_size();
        out_bytes = static_cast<size_t>(out_cpu.numel()) * out_cpu.element_size();
    }

    MetalBuffer tokens_buf;
    MetalBuffer weight_buf;
    MetalBuffer out_buf;
    MetalBuffer control_buf;

    run_metal_kernel("gptoss_bf16_f32_embeddings", [&](const gptoss_metal_device& device,
                                                         const gptoss_metal_function& fn,
                                                         gptoss_metal_command_buffer& cb) {
        if (use_mps) {
            tokens_buf.wrap_mps_tensor(&device, tokens_mps);
            weight_buf.wrap_mps_tensor(&device, weight_mps);
            out_buf.wrap_mps_tensor(&device, output_mps);
        } else {
            tokens_buf.wrap(&device, token_bytes, tokens_cpu.data_ptr());
            weight_buf.wrap(&device, weight_bytes, weight_cpu.data_ptr());
            out_buf.create(&device, out_bytes, nullptr);
        }
        create_control_buffer(&device, control_buf);

        TORCH_CHECK(
            gptoss_metal_command_buffer_encode_launch_bf16_f32_embeddings(
                &cb, &fn, static_cast<size_t>(threadgroup_size),
                tokens_buf.get(), 0,
                weight_buf.get(), 0,
                out_buf.get(), 0,
                control_buf.get(), 0,
                static_cast<uint32_t>(num_tokens),
                static_cast<uint32_t>(num_channels)) == gptoss_status_success,
            "encode embeddings failed");
    });

    if (use_mps) {
        if (!output_mps.is_alias_of(output)) {
            output.copy_(output_mps);
        }
    } else {
        std::memcpy(out_cpu.data_ptr(), out_buf.ptr(), out_bytes);
        copy_back(output, out_cpu);
    }
}

void f32_bf16w_rmsnorm_torch(const at::Tensor& input,
    const at::Tensor& weight_bf16,
    at::Tensor& output,
    double epsilon)
{
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight_bf16.dtype() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(output.dtype() == at::kFloat, "output must be float32");

    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight_bf16.dim() == 1, "weight must be 1D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");

    const int64_t num_tokens = input.size(0);
    const int64_t num_channels = input.size(1);
    TORCH_CHECK(output.size(0) == num_tokens && output.size(1) == num_channels,
        "output shape must match input shape");
    TORCH_CHECK(weight_bf16.size(0) == num_channels,
        "weight length must equal number of channels");
    TORCH_CHECK(num_channels % 4 == 0, "num_channels must be divisible by 4");

    const bool use_mps = input.device().is_mps();

    at::Tensor input_cpu;
    at::Tensor weight_cpu;
    at::Tensor out_cpu;
    size_t in_bytes = 0;
    size_t weight_bytes = 0;
    size_t out_bytes = 0;

    at::Tensor input_mps;
    at::Tensor weight_mps;
    at::Tensor output_mps;

    if (use_mps) {
        input_mps = input.contiguous();
        weight_mps = weight_bf16.contiguous();
        output_mps = output.contiguous();
    } else {
        input_cpu = to_cpu_contiguous(input);
        weight_cpu = to_cpu_contiguous(weight_bf16);
        out_cpu = empty_cpu_like(output);

        in_bytes = static_cast<size_t>(input_cpu.numel()) * input_cpu.element_size();
        weight_bytes = static_cast<size_t>(weight_cpu.numel()) * weight_cpu.element_size();
        out_bytes = static_cast<size_t>(out_cpu.numel()) * out_cpu.element_size();
    }

    MetalBuffer input_buf;
    MetalBuffer weight_buf;
    MetalBuffer out_buf;
    MetalBuffer control_buf;

    run_metal_kernel("gptoss_f32_bf16w_rmsnorm", [&](const gptoss_metal_device& device,
                                                       const gptoss_metal_function& fn,
                                                       gptoss_metal_command_buffer& cb) {
        if (use_mps) {
            input_buf.wrap_mps_tensor(&device, input_mps);
            weight_buf.wrap_mps_tensor(&device, weight_mps);
            out_buf.wrap_mps_tensor(&device, output_mps);
        } else {
            input_buf.wrap(&device, in_bytes, input_cpu.data_ptr());
            weight_buf.wrap(&device, weight_bytes, weight_cpu.data_ptr());
            out_buf.create(&device, out_bytes, nullptr);
        }
        create_control_buffer(&device, control_buf);

        TORCH_CHECK(
            gptoss_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
                &cb, &fn,
                input_buf.get(), 0,
                weight_buf.get(), 0,
                out_buf.get(), 0,
                control_buf.get(), 0,
                static_cast<uint32_t>(num_tokens),
                static_cast<uint32_t>(num_channels),
                static_cast<float>(epsilon)) == gptoss_status_success,
            "encode rmsnorm failed");
    });

    if (use_mps) {
        if (!output_mps.is_alias_of(output)) {
            output.copy_(output_mps);
        }
    } else {
        std::memcpy(out_cpu.data_ptr(), out_buf.ptr(), out_bytes);
        copy_back(output, out_cpu);
    }
}

void f32_bf16w_dense_matmul_qkv_torch(const at::Tensor& input,
    const at::Tensor& weight_bf16,
    const at::Tensor& bias_bf16,
    at::Tensor& output)
{
    run_dense_matmul_bf16(
        "gptoss_f32_bf16w_dense_matmul_qkv",
        gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_qkv,
        input, weight_bf16, bias_bf16, output);
}

void f32_bf16w_dense_matmul_attn_output_torch(const at::Tensor& input,
    const at::Tensor& weight_bf16,
    const at::Tensor& bias_bf16,
    at::Tensor& output)
{
    run_dense_matmul_bf16(
        "gptoss_f32_bf16w_dense_matmul_attn_output",
        gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_attn_output,
        input, weight_bf16, bias_bf16, output);
}

void f32_bf16w_dense_matmul_mlp_gate_torch(const at::Tensor& input,
    const at::Tensor& weight_bf16,
    const at::Tensor& bias_bf16,
    at::Tensor& output)
{
    run_dense_matmul_bf16(
        "gptoss_f32_bf16w_dense_matmul_mlp_gate",
        gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_mlp_gate,
        input, weight_bf16, bias_bf16, output);
}

void f32_rope_torch(at::Tensor& activations,
    double rope_base,
    double interpolation_scale,
    double yarn_offset,
    double yarn_scale,
    double yarn_multiplier,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t attn_head_dim,
    int64_t token_offset,
    int64_t threadgroup_size)
{
    TORCH_CHECK(activations.dtype() == at::kFloat, "activations must be float32");
    TORCH_CHECK(num_tokens >= 0 && num_q_heads >= 0 && num_kv_heads >= 0 && attn_head_dim >= 0,
        "shape parameters must be non-negative");
    TORCH_CHECK(threadgroup_size >= 0, "threadgroup_size must be non-negative");
    const bool use_mps = activations.device().is_mps();

    at::Tensor activations_cpu;
    at::Tensor activations_mps;
    size_t activations_bytes = 0;

    if (use_mps) {
        activations_mps = activations.contiguous();
        activations_bytes = static_cast<size_t>(activations_mps.numel()) * activations_mps.element_size();
    } else {
        activations_cpu = to_cpu_contiguous(activations);
        activations_bytes = static_cast<size_t>(activations_cpu.numel()) * activations_cpu.element_size();
    }

    MetalBuffer activations_buf;
    MetalBuffer control_buf;

    run_metal_kernel("gptoss_f32_rope", [&](const gptoss_metal_device& device,
                                             const gptoss_metal_function& fn,
                                             gptoss_metal_command_buffer& cb) {
        if (use_mps) {
            activations_buf.wrap_mps_tensor(&device, activations_mps);
        } else {
            activations_buf.wrap(&device, activations_bytes, activations_cpu.data_ptr());
        }
        create_control_buffer(&device, control_buf);

        TORCH_CHECK(
            gptoss_metal_command_buffer_encode_launch_f32_rope(
                &cb, &fn,
                static_cast<size_t>(threadgroup_size),
                activations_buf.get(), 0,
                control_buf.get(), 0,
                static_cast<float>(rope_base),
                static_cast<float>(interpolation_scale),
                static_cast<float>(yarn_offset),
                static_cast<float>(yarn_scale),
                static_cast<float>(yarn_multiplier),
                static_cast<uint32_t>(num_tokens),
                static_cast<uint32_t>(num_q_heads),
                static_cast<uint32_t>(num_kv_heads),
                static_cast<uint32_t>(attn_head_dim),
                static_cast<uint32_t>(token_offset)) == gptoss_status_success,
            "encode rope failed");
    });

    if (use_mps) {
        if (!activations_mps.is_alias_of(activations)) {
            activations.copy_(activations_mps);
        }
    } else {
        copy_back(activations, activations_cpu);
    }
}

void f32_bf16w_matmul_qkv_torch(const at::Tensor& input,
    const at::Tensor& weight_bf16,
    const at::Tensor& bias_bf16,
    at::Tensor& output,
    at::Tensor& kv_cache,
    int64_t kv_cache_offset_bytes,
    int64_t num_tokens,
    int64_t num_cols,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t attn_head_dim,
    int64_t token_offset,
    int64_t max_tokens,
    double rope_base,
    double interpolation_scale,
    double yarn_offset,
    double yarn_scale,
    double yarn_multiplier,
    int64_t threadgroup_size)
{
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight_bf16.dtype() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(bias_bf16.dtype() == at::kBFloat16, "bias must be bfloat16");
    TORCH_CHECK(output.dtype() == at::kFloat, "output must be float32");
    TORCH_CHECK(kv_cache.dtype() == at::kFloat, "kv_cache must be float32");

    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight_bf16.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias_bf16.dim() == 1, "bias must be 1D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");

    TORCH_CHECK(num_tokens >= 0 && num_cols >= 0 && num_q_heads >= 0 && num_kv_heads >= 0 && attn_head_dim >= 0 && max_tokens >= 0,
        "shape parameters must be non-negative");
    TORCH_CHECK(threadgroup_size >= 0, "threadgroup_size must be non-negative");
    TORCH_CHECK(kv_cache_offset_bytes >= 0, "kv_cache_offset_bytes must be non-negative");

    TORCH_CHECK(input.size(0) == num_tokens && input.size(1) == num_cols,
        "input shape must be [num_tokens, num_cols]");
    const int64_t num_rows = (num_q_heads + 2 * num_kv_heads) * attn_head_dim;
    TORCH_CHECK(output.size(0) == num_tokens && output.size(1) == num_rows,
        "output shape must be [num_tokens, (num_q_heads + 2 * num_kv_heads) * attn_head_dim]");
    TORCH_CHECK(weight_bf16.size(0) == num_cols && weight_bf16.size(1) == num_rows,
        "weight shape must be [num_cols, (num_q_heads + 2 * num_kv_heads) * attn_head_dim]");
    TORCH_CHECK(bias_bf16.size(0) == num_rows,
        "bias length must equal output feature dimension");

    const bool use_mps = input.device().is_mps() && weight_bf16.device().is_mps() &&
        bias_bf16.device().is_mps() && output.device().is_mps() && kv_cache.device().is_mps();

    at::Tensor input_cpu;
    at::Tensor weight_cpu;
    at::Tensor bias_cpu;
    at::Tensor out_cpu;
    at::Tensor kv_cpu;
    size_t in_bytes = 0;
    size_t weight_bytes = 0;
    size_t bias_bytes = 0;
    size_t out_bytes = 0;
    size_t kv_bytes = 0;

    at::Tensor input_mps;
    at::Tensor weight_mps;
    at::Tensor bias_mps;
    at::Tensor out_mps;
    at::Tensor kv_mps;

    if (use_mps) {
        input_mps = input.contiguous();
        weight_mps = weight_bf16.transpose(0, 1).contiguous();
        bias_mps = bias_bf16.contiguous();
        out_mps = output.contiguous();
        kv_mps = kv_cache.contiguous();
    } else {
        input_cpu = to_cpu_contiguous(input);
        weight_cpu = weight_bf16.transpose(0, 1).contiguous().to(at::kCPU);
        bias_cpu = to_cpu_contiguous(bias_bf16);
        out_cpu = empty_cpu_like(output);
        kv_cpu = to_cpu_contiguous(kv_cache);

        in_bytes = static_cast<size_t>(input_cpu.numel()) * input_cpu.element_size();
        weight_bytes = static_cast<size_t>(weight_cpu.numel()) * weight_cpu.element_size();
        bias_bytes = static_cast<size_t>(bias_cpu.numel()) * bias_cpu.element_size();
        out_bytes = static_cast<size_t>(out_cpu.numel()) * out_cpu.element_size();
        kv_bytes = static_cast<size_t>(kv_cpu.numel()) * kv_cpu.element_size();
    }

    MetalBuffer input_buf;
    MetalBuffer weight_buf;
    MetalBuffer bias_buf;
    MetalBuffer out_buf;
    MetalBuffer kv_buf;
    MetalBuffer control_buf;

    run_metal_kernel("gptoss_f32_bf16w_matmul_qkv", [&](const gptoss_metal_device& device,
                                                          const gptoss_metal_function& fn,
                                                          gptoss_metal_command_buffer& cb) {
        if (use_mps) {
            input_buf.wrap_mps_tensor(&device, input_mps);
            weight_buf.wrap_mps_tensor(&device, weight_mps);
            bias_buf.wrap_mps_tensor(&device, bias_mps);
            out_buf.wrap_mps_tensor(&device, out_mps);
            kv_buf.wrap_mps_tensor(&device, kv_mps);
        } else {
            input_buf.wrap(&device, in_bytes, input_cpu.data_ptr());
            weight_buf.wrap(&device, weight_bytes, weight_cpu.data_ptr());
            bias_buf.wrap(&device, bias_bytes, bias_cpu.data_ptr());
            out_buf.create(&device, out_bytes, nullptr);
            kv_buf.wrap(&device, kv_bytes, kv_cpu.data_ptr());
        }
        create_control_buffer(&device, control_buf);

        TORCH_CHECK(
            gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul_qkv(
                &cb, &fn,
                static_cast<size_t>(threadgroup_size),
                input_buf.get(), 0,
                weight_buf.get(), 0,
                bias_buf.get(), 0,
                out_buf.get(), 0,
                kv_buf.get(), static_cast<size_t>(kv_cache_offset_bytes),
                control_buf.get(), 0,
                static_cast<uint32_t>(num_tokens),
                static_cast<uint32_t>(num_cols),
                static_cast<uint32_t>(num_q_heads),
                static_cast<uint32_t>(num_kv_heads),
                static_cast<uint32_t>(attn_head_dim),
                static_cast<uint32_t>(token_offset),
                static_cast<uint32_t>(max_tokens),
                static_cast<float>(rope_base),
                static_cast<float>(interpolation_scale),
                static_cast<float>(yarn_offset),
                static_cast<float>(yarn_scale),
                static_cast<float>(yarn_multiplier)) == gptoss_status_success,
            "encode matmul_qkv failed");
    });

    if (use_mps) {
        if (!out_mps.is_alias_of(output)) {
            output.copy_(out_mps);
        }
        if (!kv_mps.is_alias_of(kv_cache)) {
            kv_cache.copy_(kv_mps);
        }
    } else {
        std::memcpy(out_cpu.data_ptr(), out_buf.ptr(), out_bytes);
        copy_back(output, out_cpu);
        copy_back(kv_cache, kv_cpu);
    }
}

void f32_sdpa_torch(const at::Tensor& q,
    int64_t q_offset_bytes,
    const at::Tensor& kv,
    int64_t kv_offset_bytes,
    const at::Tensor& s_bf16,
    int64_t s_offset_bytes,
    at::Tensor& output,
    int64_t output_offset_bytes,
    int64_t window,
    int64_t kv_stride,
    int64_t num_q_tokens,
    int64_t num_kv_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim)
{
    TORCH_CHECK(q.dtype() == at::kFloat, "q must be float32");
    TORCH_CHECK(kv.dtype() == at::kFloat, "kv must be float32");
    TORCH_CHECK(s_bf16.dtype() == at::kBFloat16, "s must be bfloat16");
    TORCH_CHECK(output.dtype() == at::kFloat, "output must be float32");

    TORCH_CHECK(q_offset_bytes >= 0 && kv_offset_bytes >= 0 && s_offset_bytes >= 0 && output_offset_bytes >= 0,
        "offsets must be non-negative");
    TORCH_CHECK(window >= 0 && kv_stride >= 0 && num_q_tokens >= 0 && num_kv_tokens >= 0 && num_q_heads >= 0 && num_kv_heads >= 0 && head_dim >= 0,
        "shape parameters must be non-negative");

    const bool use_mps = q.device().is_mps() && kv.device().is_mps() &&
        s_bf16.device().is_mps() && output.device().is_mps();

    at::Tensor q_cpu;
    at::Tensor kv_cpu;
    at::Tensor s_cpu;
    at::Tensor out_cpu;
    size_t q_bytes = 0;
    size_t kv_bytes = 0;
    size_t s_bytes = 0;
    size_t out_bytes = 0;

    at::Tensor q_mps;
    at::Tensor kv_mps;
    at::Tensor s_mps;
    at::Tensor out_mps;

    if (use_mps) {
        q_mps = q.contiguous();
        kv_mps = kv.contiguous();
        s_mps = s_bf16.contiguous();
        out_mps = output.contiguous();
    } else {
        q_cpu = to_cpu_contiguous(q);
        kv_cpu = to_cpu_contiguous(kv);
        s_cpu = to_cpu_contiguous(s_bf16);
        out_cpu = empty_cpu_like(output);

        q_bytes = static_cast<size_t>(q_cpu.numel()) * q_cpu.element_size();
        kv_bytes = static_cast<size_t>(kv_cpu.numel()) * kv_cpu.element_size();
        s_bytes = static_cast<size_t>(s_cpu.numel()) * s_cpu.element_size();
        out_bytes = static_cast<size_t>(out_cpu.numel()) * out_cpu.element_size();
    }

    MetalBuffer q_buf;
    MetalBuffer kv_buf;
    MetalBuffer s_buf;
    MetalBuffer out_buf;
    MetalBuffer control_buf;

    run_metal_kernel("gptoss_f32_sdpa_q8_d64", [&](const gptoss_metal_device& device,
                                                     const gptoss_metal_function& fn,
                                                     gptoss_metal_command_buffer& cb) {
        if (use_mps) {
            q_buf.wrap_mps_tensor(&device, q_mps);
            kv_buf.wrap_mps_tensor(&device, kv_mps);
            s_buf.wrap_mps_tensor(&device, s_mps);
            out_buf.wrap_mps_tensor(&device, out_mps);
        } else {
            q_buf.wrap(&device, q_bytes, q_cpu.data_ptr());
            kv_buf.wrap(&device, kv_bytes, kv_cpu.data_ptr());
            s_buf.wrap(&device, s_bytes, s_cpu.data_ptr());
            out_buf.create(&device, out_bytes, nullptr);
        }
        create_control_buffer(&device, control_buf);

        TORCH_CHECK(
            gptoss_metal_command_buffer_encode_launch_f32_sdpa(
                &cb, &fn,
                q_buf.get(), static_cast<size_t>(q_offset_bytes),
                kv_buf.get(), static_cast<size_t>(kv_offset_bytes),
                s_buf.get(), static_cast<size_t>(s_offset_bytes),
                out_buf.get(), static_cast<size_t>(output_offset_bytes),
                control_buf.get(), 0,
                static_cast<uint32_t>(window),
                static_cast<uint32_t>(kv_stride),
                static_cast<uint32_t>(num_q_tokens),
                static_cast<uint32_t>(num_kv_tokens),
                static_cast<uint32_t>(num_q_heads),
                static_cast<uint32_t>(num_kv_heads),
                static_cast<uint32_t>(head_dim)) == gptoss_status_success,
            "encode sdpa failed");
    });

    if (use_mps) {
        if (!out_mps.is_alias_of(output)) {
            output.copy_(out_mps);
        }
    } else {
        std::memcpy(out_cpu.data_ptr(), out_buf.ptr(), out_bytes);
        copy_back(output, out_cpu);
    }
}

void f32_topk_torch(const at::Tensor& scores,
    at::Tensor& expert_ids,
    at::Tensor& expert_scores,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t num_active_experts)
{
    TORCH_CHECK(scores.dtype() == at::kFloat, "scores must be float32");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "expert_ids must be int32");
    TORCH_CHECK(expert_scores.dtype() == at::kFloat, "expert_scores must be float32");

    TORCH_CHECK(num_tokens >= 0 && num_experts >= 0 && num_active_experts >= 0,
        "shape parameters must be non-negative");

    TORCH_CHECK(scores.size(0) == num_tokens,
        "scores first dimension must match num_tokens");
    TORCH_CHECK(scores.numel() == num_tokens * num_experts,
        "scores must have num_tokens * num_experts elements");
    TORCH_CHECK(expert_ids.numel() == num_tokens * num_active_experts,
        "expert_ids must have num_tokens * num_active_experts elements");
    TORCH_CHECK(expert_scores.numel() == num_tokens * num_active_experts,
        "expert_scores must have num_tokens * num_active_experts elements");

    const bool scores_use_mps = scores.device().is_mps();
    at::Tensor scores_cpu;
    at::Tensor scores_mps;
    size_t score_bytes = 0;

    if (scores_use_mps) {
        scores_mps = scores.contiguous();
    } else {
        scores_cpu = to_cpu_contiguous(scores);
        score_bytes = static_cast<size_t>(scores_cpu.numel()) * scores_cpu.element_size();
    }
    std::vector<gptoss_expert_prediction> predictions(static_cast<size_t>(num_tokens) * static_cast<size_t>(num_active_experts));

    const size_t pred_bytes = predictions.size() * sizeof(gptoss_expert_prediction);

    MetalBuffer score_buf;
    MetalBuffer pred_buf;
    MetalBuffer control_buf;

    run_metal_kernel("gptoss_f32_topk_softmax_e128_k4", [&](const gptoss_metal_device& device,
                                                             const gptoss_metal_function& fn,
                                                             gptoss_metal_command_buffer& cb) {
        if (scores_use_mps) {
            score_buf.wrap_mps_tensor(&device, scores_mps);
        } else {
            score_buf.wrap(&device, score_bytes, scores_cpu.data_ptr());
        }
        pred_buf.wrap(&device, pred_bytes, predictions.data());
        create_control_buffer(&device, control_buf);

        TORCH_CHECK(
            gptoss_metal_command_buffer_encode_launch_f32_topk(
                &cb, &fn,
                score_buf.get(), 0,
                pred_buf.get(), 0,
                control_buf.get(), 0,
                static_cast<uint32_t>(num_tokens),
                static_cast<uint32_t>(num_experts),
                static_cast<uint32_t>(num_active_experts)) == gptoss_status_success,
            "encode topk failed");
    });

    auto ids_cpu = expert_ids.to(at::kCPU).contiguous();
    auto scores_out_cpu = expert_scores.to(at::kCPU).contiguous();
    auto* ids_ptr = ids_cpu.data_ptr<int32_t>();
    auto* scores_ptr = scores_out_cpu.data_ptr<float>();
    const size_t total = predictions.size();
    for (size_t i = 0; i < total; ++i) {
        ids_ptr[i] = static_cast<int32_t>(predictions[i].expert_id);
        scores_ptr[i] = predictions[i].score;
    }
    copy_back(expert_ids, ids_cpu);
    copy_back(expert_scores, scores_out_cpu);
}

void expert_routing_metadata_torch(const at::Tensor& expert_ids,
    const at::Tensor& expert_scores,
    at::Tensor& expert_offsets,
    at::Tensor& intra_expert_offsets,
    int64_t num_tokens,
    int64_t num_experts)
{
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "expert_ids must be int32");
    TORCH_CHECK(expert_scores.dtype() == at::kFloat, "expert_scores must be float32");
    TORCH_CHECK(expert_offsets.dtype() == at::kInt, "expert_offsets must be int32");
    TORCH_CHECK(intra_expert_offsets.dtype() == at::kInt, "intra_expert_offsets must be int32");

    TORCH_CHECK(num_tokens >= 0 && num_experts >= 0, "shape parameters must be non-negative");
    TORCH_CHECK(expert_ids.numel() == num_tokens,
        "expert_ids must have num_tokens elements");
    TORCH_CHECK(expert_scores.numel() == num_tokens,
        "expert_scores must have num_tokens elements");
    TORCH_CHECK(intra_expert_offsets.numel() == num_tokens,
        "intra_expert_offsets must have num_tokens elements");
    TORCH_CHECK(expert_offsets.numel() == num_experts + 1,
        "expert_offsets must have num_experts + 1 elements");

    auto ids_cpu = to_cpu_contiguous(expert_ids);
    auto scores_cpu = to_cpu_contiguous(expert_scores);

    const bool use_mps = expert_offsets.device().is_mps() && intra_expert_offsets.device().is_mps();
    at::Tensor offsets_cpu;
    at::Tensor intra_offsets_cpu;
    at::Tensor offsets_mps;
    at::Tensor intra_offsets_mps;
    size_t offsets_bytes = 0;
    size_t intra_bytes = 0;

    if (use_mps) {
        offsets_mps = expert_offsets.contiguous();
        intra_offsets_mps = intra_expert_offsets.contiguous();
    } else {
        offsets_cpu = to_cpu_contiguous(expert_offsets);
        intra_offsets_cpu = to_cpu_contiguous(intra_expert_offsets);
        offsets_bytes = static_cast<size_t>(offsets_cpu.numel()) * offsets_cpu.element_size();
        intra_bytes = static_cast<size_t>(intra_offsets_cpu.numel()) * intra_offsets_cpu.element_size();
    }

    std::vector<gptoss_expert_prediction> predictions(static_cast<size_t>(num_tokens));
    const auto* ids_ptr = ids_cpu.data_ptr<int32_t>();
    const auto* scores_ptr = scores_cpu.data_ptr<float>();
    for (int64_t i = 0; i < num_tokens; ++i) {
        predictions[static_cast<size_t>(i)] = gptoss_expert_prediction {
            .expert_id = static_cast<uint32_t>(ids_ptr[i]),
            .score = scores_ptr[i],
        };
    }

    const size_t pred_bytes = predictions.size() * sizeof(gptoss_expert_prediction);

    MetalBuffer pred_buf;
    MetalBuffer offsets_buf;
    MetalBuffer intra_offsets_buf;

    run_metal_kernel("gptoss_f32_expert_routing_metadata", [&](const gptoss_metal_device& device,
                                                                  const gptoss_metal_function& fn,
                                                                  gptoss_metal_command_buffer& cb) {
        pred_buf.wrap(&device, pred_bytes, predictions.data());
        if (use_mps) {
            offsets_buf.wrap_mps_tensor(&device, offsets_mps);
            intra_offsets_buf.wrap_mps_tensor(&device, intra_offsets_mps);
        } else {
            offsets_buf.wrap(&device, offsets_bytes, offsets_cpu.data_ptr());
            intra_offsets_buf.wrap(&device, intra_bytes, intra_offsets_cpu.data_ptr());
        }

        TORCH_CHECK(
            gptoss_metal_command_buffer_encode_launch_expert_routing_metadata(
                &cb, &fn,
                pred_buf.get(), 0,
                offsets_buf.get(), 0,
                intra_offsets_buf.get(), 0,
                static_cast<uint32_t>(num_tokens),
                static_cast<uint32_t>(num_experts)) == gptoss_status_success,
            "encode expert_routing_metadata failed");
    });

    if (use_mps) {
        if (!offsets_mps.is_alias_of(expert_offsets)) {
            expert_offsets.copy_(offsets_mps);
        }
        if (!intra_offsets_mps.is_alias_of(intra_expert_offsets)) {
            intra_expert_offsets.copy_(intra_offsets_mps);
        }
    } else {
        copy_back(expert_offsets, offsets_cpu);
        copy_back(intra_expert_offsets, intra_offsets_cpu);
    }
}

void f32_scatter_torch(const at::Tensor& input,
    const at::Tensor& expert_ids,
    const at::Tensor& expert_scores,
    const at::Tensor& expert_offsets,
    const at::Tensor& intra_expert_offsets,
    at::Tensor& output,
    int64_t num_channels,
    int64_t num_tokens,
    int64_t num_active_experts)
{
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "expert_ids must be int32");
    TORCH_CHECK(expert_scores.dtype() == at::kFloat, "expert_scores must be float32");
    TORCH_CHECK(expert_offsets.dtype() == at::kInt, "expert_offsets must be int32");
    TORCH_CHECK(intra_expert_offsets.dtype() == at::kInt, "intra_expert_offsets must be int32");
    TORCH_CHECK(output.dtype() == at::kFloat, "output must be float32");

    TORCH_CHECK(num_channels >= 0 && num_tokens >= 0 && num_active_experts >= 0,
        "shape parameters must be non-negative");

    TORCH_CHECK(input.numel() == static_cast<int64_t>(num_tokens / num_active_experts) * num_channels,
        "input size mismatch");
    TORCH_CHECK(expert_ids.numel() == num_tokens,
        "expert_ids must have num_tokens elements");
    TORCH_CHECK(expert_scores.numel() == num_tokens,
        "expert_scores must have num_tokens elements");
    TORCH_CHECK(intra_expert_offsets.numel() == num_tokens,
        "intra_expert_offsets must have num_tokens elements");
    TORCH_CHECK(output.numel() == num_tokens * num_channels / num_active_experts,
        "output size mismatch");

    const bool use_mps = input.device().is_mps() && expert_offsets.device().is_mps() &&
        intra_expert_offsets.device().is_mps() && output.device().is_mps();

    at::Tensor input_cpu;
    at::Tensor expert_offsets_cpu;
    at::Tensor intra_offsets_cpu;
    at::Tensor output_cpu;
    size_t input_bytes = 0;
    size_t offsets_bytes = 0;
    size_t intra_bytes = 0;
    size_t output_bytes = 0;

    at::Tensor input_mps;
    at::Tensor expert_offsets_mps;
    at::Tensor intra_offsets_mps;
    at::Tensor output_mps;

    if (use_mps) {
        input_mps = input.contiguous();
        expert_offsets_mps = expert_offsets.contiguous();
        intra_offsets_mps = intra_expert_offsets.contiguous();
        output_mps = output.contiguous();
    } else {
        input_cpu = to_cpu_contiguous(input);
        expert_offsets_cpu = to_cpu_contiguous(expert_offsets);
        intra_offsets_cpu = to_cpu_contiguous(intra_expert_offsets);
        output_cpu = empty_cpu_like(output);

        input_bytes = static_cast<size_t>(input_cpu.numel()) * input_cpu.element_size();
        offsets_bytes = static_cast<size_t>(expert_offsets_cpu.numel()) * expert_offsets_cpu.element_size();
        intra_bytes = static_cast<size_t>(intra_offsets_cpu.numel()) * intra_offsets_cpu.element_size();
        output_bytes = static_cast<size_t>(output_cpu.numel()) * output_cpu.element_size();
    }

    std::vector<gptoss_expert_prediction> predictions(static_cast<size_t>(num_tokens));
    const auto* ids_ptr = expert_ids.to(at::kCPU).contiguous().data_ptr<int32_t>();
    const auto* scores_ptr = expert_scores.to(at::kCPU).contiguous().data_ptr<float>();
    for (int64_t i = 0; i < num_tokens; ++i) {
        predictions[static_cast<size_t>(i)] = gptoss_expert_prediction {
            .expert_id = static_cast<uint32_t>(ids_ptr[i]),
            .score = scores_ptr[i],
        };
    }

    const size_t pred_bytes = predictions.size() * sizeof(gptoss_expert_prediction);

    MetalBuffer input_buf;
    MetalBuffer pred_buf;
    MetalBuffer offsets_buf;
    MetalBuffer intra_offsets_buf;
    MetalBuffer output_buf;

    run_metal_kernel("gptoss_f32_scatter_e4", [&](const gptoss_metal_device& device,
                                                    const gptoss_metal_function& fn,
                                                    gptoss_metal_command_buffer& cb) {
        if (use_mps) {
            input_buf.wrap_mps_tensor(&device, input_mps);
            offsets_buf.wrap_mps_tensor(&device, expert_offsets_mps);
            intra_offsets_buf.wrap_mps_tensor(&device, intra_offsets_mps);
            output_buf.wrap_mps_tensor(&device, output_mps);
        } else {
            input_buf.wrap(&device, input_bytes, input_cpu.data_ptr());
            offsets_buf.wrap(&device, offsets_bytes, expert_offsets_cpu.data_ptr());
            intra_offsets_buf.wrap(&device, intra_bytes, intra_offsets_cpu.data_ptr());
            output_buf.create(&device, output_bytes, nullptr);
        }
        pred_buf.wrap(&device, pred_bytes, predictions.data());

        TORCH_CHECK(
            gptoss_metal_command_buffer_encode_launch_f32_scatter(
                &cb, &fn,
                input_buf.get(), 0,
                pred_buf.get(), 0,
                offsets_buf.get(), 0,
                intra_offsets_buf.get(), 0,
                output_buf.get(), 0,
                static_cast<uint32_t>(num_channels),
                static_cast<uint32_t>(num_tokens / num_active_experts),
                static_cast<uint32_t>(num_active_experts)) == gptoss_status_success,
            "encode scatter failed");
    });

    if (use_mps) {
        if (!output_mps.is_alias_of(output)) {
            output.copy_(output_mps);
        }
    } else {
        std::memcpy(output_cpu.data_ptr(), output_buf.ptr(), output_bytes);
        copy_back(output, output_cpu);
    }
}

void f32_bf16w_matmul_add_torch(const at::Tensor& input,
    const at::Tensor& weight_bf16,
    const at::Tensor& bias_bf16,
    at::Tensor& output,
    int64_t num_tokens,
    int64_t num_cols,
    int64_t num_rows,
    int64_t threadgroup_size)
{
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight_bf16.dtype() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(bias_bf16.dtype() == at::kBFloat16, "bias must be bfloat16");
    TORCH_CHECK(output.dtype() == at::kFloat, "output must be float32");

    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight_bf16.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias_bf16.dim() == 1, "bias must be 1D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");

    TORCH_CHECK(input.size(0) == num_tokens && input.size(1) == num_cols,
        "input shape must be [num_tokens, num_cols]");
    TORCH_CHECK(weight_bf16.size(0) == num_cols && weight_bf16.size(1) == num_rows,
        "weight shape must be [num_cols, num_rows]");
    TORCH_CHECK(bias_bf16.size(0) == num_rows,
        "bias length must equal num_rows");
    TORCH_CHECK(output.size(0) == num_tokens && output.size(1) == num_rows,
        "output shape must be [num_tokens, num_rows]");

    const bool use_mps = input.device().is_mps() && weight_bf16.device().is_mps() &&
        bias_bf16.device().is_mps() && output.device().is_mps();

    at::Tensor input_cpu;
    at::Tensor weight_cpu;
    at::Tensor bias_cpu;
    at::Tensor out_cpu;
    size_t in_bytes = 0;
    size_t weight_bytes = 0;
    size_t bias_bytes = 0;
    size_t out_bytes = 0;

    at::Tensor input_mps;
    at::Tensor weight_mps;
    at::Tensor bias_mps;
    at::Tensor out_mps;

    if (use_mps) {
        input_mps = input.contiguous();
        weight_mps = weight_bf16.transpose(0, 1).contiguous();
        bias_mps = bias_bf16.contiguous();
        out_mps = output.contiguous();
    } else {
        input_cpu = to_cpu_contiguous(input);
        weight_cpu = weight_bf16.transpose(0, 1).contiguous().to(at::kCPU);
        bias_cpu = to_cpu_contiguous(bias_bf16);
        out_cpu = to_cpu_contiguous(output);

        in_bytes = static_cast<size_t>(input_cpu.numel()) * input_cpu.element_size();
        weight_bytes = static_cast<size_t>(weight_cpu.numel()) * weight_cpu.element_size();
        bias_bytes = static_cast<size_t>(bias_cpu.numel()) * bias_cpu.element_size();
        out_bytes = static_cast<size_t>(out_cpu.numel()) * out_cpu.element_size();
    }

    MetalBuffer input_buf;
    MetalBuffer weight_buf;
    MetalBuffer bias_buf;
    MetalBuffer out_buf;
    MetalBuffer control_buf;

    run_metal_kernel("gptoss_f32_bf16w_matmul", [&](const gptoss_metal_device& device,
                                                     const gptoss_metal_function& fn,
                                                     gptoss_metal_command_buffer& cb) {
        if (use_mps) {
            input_buf.wrap_mps_tensor(&device, input_mps);
            weight_buf.wrap_mps_tensor(&device, weight_mps);
            bias_buf.wrap_mps_tensor(&device, bias_mps);
            out_buf.wrap_mps_tensor(&device, out_mps);
        } else {
            input_buf.wrap(&device, in_bytes, input_cpu.data_ptr());
            weight_buf.wrap(&device, weight_bytes, weight_cpu.data_ptr());
            bias_buf.wrap(&device, bias_bytes, bias_cpu.data_ptr());
            out_buf.create(&device, out_bytes, nullptr);
            std::memcpy(out_buf.ptr(), out_cpu.data_ptr(), out_bytes);
        }
        create_control_buffer(&device, control_buf);

        TORCH_CHECK(
            gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul_add(
                &cb, &fn,
                static_cast<size_t>(threadgroup_size),
                input_buf.get(), 0,
                weight_buf.get(), 0,
                bias_buf.get(), 0,
                out_buf.get(), 0,
                control_buf.get(), 0,
                static_cast<uint32_t>(num_tokens),
                static_cast<uint32_t>(num_cols),
                static_cast<uint32_t>(num_rows)) == gptoss_status_success,
            "encode matmul_add failed");
    });

    if (use_mps) {
        if (!out_mps.is_alias_of(output)) {
            output.copy_(out_mps);
        }
    } else {
        std::memcpy(out_cpu.data_ptr(), out_buf.ptr(), out_bytes);
        copy_back(output, out_cpu);
    }
}