#include <torch/library.h>
#include <torch/torch.h>
#include <vector>

#include "registration.h"
#include "torch_binding.h"

extern "C" void launch_givens_fwd(
    const void* x, const float* angles, const float* alpha,
    const float* scales, void* out, const int* shifts,
    int N, int H, int D, int K, int use_half);

extern "C" void launch_givens_bwd(
    const void* grad_out, const void* x_in,
    const float* angles, const float* alpha, const float* scales,
    void* grad_x, float* grad_angles, float* grad_alpha, float* grad_scales,
    const int* shifts, float* save_buf,
    int N, int H, int D, int K, int use_half);

extern "C" void launch_givens_mor_fwd(
    const void* x, const float* expert_angles, const float* expert_scales,
    const float* router_weights, void* out, const int* shifts,
    int N, int H, int D, int K, int E, int use_half);

extern "C" void launch_givens_mor_bwd(
    const void* grad_out, const void* x_in,
    const float* expert_angles, const float* expert_scales,
    const float* router_weights,
    void* grad_x, float* grad_expert_angles, float* grad_expert_scales,
    float* grad_router, const int* shifts, float* save_buf,
    int N, int H, int D, int K, int E, int use_half);

extern "C" void launch_givens_coupling_fwd(
    const float* x, const float* angles,
    const float* coupling, const float* scales, const int* shifts, float* out,
    int N, int H, int D, int K, int B);

extern "C" void launch_givens_coupling_bwd(
    const float* grad_out, const float* x_in,
    const float* angles,
    const float* coupling, const float* scales,
    const int* shifts,
    float* grad_x, float* grad_angles, float* grad_scales,
    float* feat_buf, float* grad_mod_buf,
    float* save_buf,
    int N, int H, int D, int K, int B);

extern "C" void clifford_fwd_launcher(
    const float* x_pairs,
    float* features,
    int batch,
    int P);

extern "C" void clifford_bwd_launcher(
    const float* x_pairs,
    const float* grad_features,
    float* grad_x_pairs,
    int batch,
    int P);

extern "C" void fused_embedding_fwd_launcher(
    const float* phase_angles,
    const float* log_magnitude,
    const int* token_ids,
    float* output,
    int N, int P);

extern "C" void fused_embedding_bwd_launcher(
    const float* phase_angles,
    const float* log_magnitude,
    const int* token_ids,
    const float* grad_output,
    float* grad_phase_angles,
    float* grad_log_magnitude,
    int N, int P);

extern "C" void fused_norm_fwd_launcher(
    const float* x, const float* scale, float* out, float* rms_inv_out,
    int N, int D, float eps);

extern "C" void fused_norm_bwd_launcher(
    const float* grad_out, const float* x, const float* scale,
    const float* rms_inv, float* grad_x,
    int N, int D, float eps);

extern "C" void fused_gelu_fwd_launcher(
    const float* x, float* out,
    int total_pairs, int half_D, float eps);

extern "C" void fused_gelu_bwd_launcher(
    const float* grad_out, const float* x, float* grad_x,
    int total_pairs, int half_D, float eps);

extern "C" void fused_mixer_fwd_launcher(
    const float* x_in,
    const float* angles, const float* coupling, const float* raw_scales,
    float* x_out,
    float* save_feat_norm, float* save_inv_norm, float* save_theta,
    int N, int D, int n_groups, int G, float eps);

extern "C" void fused_mixer_bwd_launcher(
    const float* grad_out, const float* x_in,
    const float* angles, const float* coupling, const float* raw_scales,
    const float* save_feat_norm, const float* save_inv_norm, const float* save_theta,
    float* grad_x, float* grad_angles, float* grad_coupling, float* grad_raw_scales,
    int N, int D, int n_groups, int G, float eps);

extern "C" void fused_zloss_fwd_launcher(
    const float* logits, const int* mask, float* lse_out,
    int N, int V);

extern "C" void fused_zloss_bwd_launcher(
    const float* logits, const float* lse, const int* mask,
    float grad_scale, float* grad_logits,
    int N, int V);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor givens_forward(
    torch::Tensor x, torch::Tensor angles, torch::Tensor alpha,
    torch::Tensor scales, torch::Tensor shifts, int64_t K
) {
    CHECK_INPUT(x);
    CHECK_INPUT(angles);
    CHECK_INPUT(alpha);
    CHECK_INPUT(scales);
    CHECK_CUDA(shifts);

    TORCH_CHECK(x.dim() == 3, "x must be [N, H, D]");
    TORCH_CHECK(x.size(2) % 2 == 0, "D must be even");
    TORCH_CHECK((x.size(2) & (x.size(2) - 1)) == 0, "D must be a power of 2");

    int N = x.size(0);
    int H = x.size(1);
    int D = x.size(2);
    int use_half = (x.scalar_type() == at::ScalarType::Half) ? 1 : 0;

    auto out = torch::empty_like(x);

    launch_givens_fwd(
        x.data_ptr(),
        angles.data_ptr<float>(),
        alpha.data_ptr<float>(),
        scales.data_ptr<float>(),
        out.data_ptr(),
        shifts.data_ptr<int>(),
        N, H, D, static_cast<int>(K), use_half
    );

    return out;
}

std::vector<torch::Tensor> givens_backward(
    torch::Tensor grad_out, torch::Tensor x_in,
    torch::Tensor angles, torch::Tensor alpha,
    torch::Tensor scales, torch::Tensor shifts, int64_t K
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(x_in);
    CHECK_INPUT(angles);
    CHECK_INPUT(alpha);
    CHECK_INPUT(scales);
    CHECK_CUDA(shifts);

    int N = x_in.size(0);
    int H = x_in.size(1);
    int D = x_in.size(2);
    int half_D = D / 2;
    int use_half = (x_in.scalar_type() == at::ScalarType::Half) ? 1 : 0;

    auto float_opts = x_in.options().dtype(torch::kFloat32);
    auto grad_x      = torch::empty_like(x_in);
    auto grad_angles = torch::zeros({K, H, half_D}, float_opts);
    auto grad_alpha  = torch::zeros({K, H, half_D}, float_opts);
    auto grad_scales = torch::zeros({K, H, D},      float_opts);
    auto save_buf    = torch::empty({N * H, K, D},  float_opts);

    launch_givens_bwd(
        grad_out.data_ptr(),
        x_in.data_ptr(),
        angles.data_ptr<float>(),
        alpha.data_ptr<float>(),
        scales.data_ptr<float>(),
        grad_x.data_ptr(),
        grad_angles.data_ptr<float>(),
        grad_alpha.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        shifts.data_ptr<int>(),
        save_buf.data_ptr<float>(),
        N, H, D, static_cast<int>(K), use_half
    );

    return {grad_x, grad_angles, grad_alpha, grad_scales};
}

torch::Tensor givens_mor_forward(
    torch::Tensor x, torch::Tensor expert_angles, torch::Tensor expert_scales,
    torch::Tensor router_weights, torch::Tensor shifts, int64_t K, int64_t E
) {
    CHECK_INPUT(x);
    CHECK_INPUT(expert_angles);
    CHECK_INPUT(expert_scales);
    CHECK_INPUT(router_weights);
    CHECK_CUDA(shifts);

    TORCH_CHECK(x.dim() == 3, "x must be [N, H, D]");
    TORCH_CHECK(x.size(2) % 2 == 0, "D must be even");
    TORCH_CHECK((x.size(2) & (x.size(2) - 1)) == 0, "D must be a power of 2");

    int N = x.size(0);
    int H = x.size(1);
    int D = x.size(2);
    int use_half = (x.scalar_type() == at::ScalarType::Half) ? 1 : 0;

    auto out = torch::empty_like(x);

    launch_givens_mor_fwd(
        x.data_ptr(),
        expert_angles.data_ptr<float>(),
        expert_scales.data_ptr<float>(),
        router_weights.data_ptr<float>(),
        out.data_ptr(),
        shifts.data_ptr<int>(),
        N, H, D, static_cast<int>(K), static_cast<int>(E), use_half
    );

    return out;
}

std::vector<torch::Tensor> givens_mor_backward(
    torch::Tensor grad_out, torch::Tensor x_in,
    torch::Tensor expert_angles, torch::Tensor expert_scales,
    torch::Tensor router_weights, torch::Tensor shifts, int64_t K, int64_t E
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(x_in);
    CHECK_INPUT(expert_angles);
    CHECK_INPUT(expert_scales);
    CHECK_INPUT(router_weights);
    CHECK_CUDA(shifts);

    int N = x_in.size(0);
    int H = x_in.size(1);
    int D = x_in.size(2);
    int half_D = D / 2;
    int use_half = (x_in.scalar_type() == at::ScalarType::Half) ? 1 : 0;

    auto float_opts = x_in.options().dtype(torch::kFloat32);
    auto grad_x             = torch::empty_like(x_in);
    auto grad_expert_angles = torch::zeros({E, K, H, half_D}, float_opts);
    auto grad_expert_scales = torch::zeros({E, K, H, D},      float_opts);
    auto grad_router        = torch::zeros({N, E},             float_opts);
    auto save_buf           = torch::empty({N * H, K, D},      float_opts);

    launch_givens_mor_bwd(
        grad_out.data_ptr(),
        x_in.data_ptr(),
        expert_angles.data_ptr<float>(),
        expert_scales.data_ptr<float>(),
        router_weights.data_ptr<float>(),
        grad_x.data_ptr(),
        grad_expert_angles.data_ptr<float>(),
        grad_expert_scales.data_ptr<float>(),
        grad_router.data_ptr<float>(),
        shifts.data_ptr<int>(),
        save_buf.data_ptr<float>(),
        N, H, D, static_cast<int>(K), static_cast<int>(E), use_half
    );

    return {grad_x, grad_expert_angles, grad_expert_scales, grad_router};
}

torch::Tensor givens_coupling_forward(
    torch::Tensor x, torch::Tensor angles,
    torch::Tensor coupling, torch::Tensor scales, torch::Tensor shifts,
    int64_t K, int64_t B
) {
    CHECK_INPUT(x);
    CHECK_INPUT(angles);
    CHECK_INPUT(coupling);
    CHECK_INPUT(scales);
    CHECK_CUDA(shifts);

    TORCH_CHECK(x.dim() == 3, "x must be [N, H, D]");
    TORCH_CHECK(x.size(2) % 2 == 0, "D must be even");
    TORCH_CHECK((x.size(2) & (x.size(2) - 1)) == 0, "D must be a power of 2");
    TORCH_CHECK(angles.dim() == 4, "angles must be [B, K, H, D//2]");
    TORCH_CHECK(angles.size(0) == B, "angles batch dimension must match B");

    int N = x.size(0);
    int H = x.size(1);
    int D = x.size(2);

    auto out = torch::empty({B, N, H, D}, x.options());

    launch_givens_coupling_fwd(
        x.data_ptr<float>(),
        angles.data_ptr<float>(),
        coupling.data_ptr<float>(),
        scales.data_ptr<float>(),
        shifts.data_ptr<int>(),
        out.data_ptr<float>(),
        N, H, D, static_cast<int>(K), static_cast<int>(B)
    );

    return out;
}

std::vector<torch::Tensor> givens_coupling_backward(
    torch::Tensor grad_out, torch::Tensor x_in,
    torch::Tensor angles,
    torch::Tensor coupling, torch::Tensor scales,
    torch::Tensor shifts, int64_t K, int64_t B
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(x_in);
    CHECK_INPUT(angles);
    CHECK_INPUT(coupling);
    CHECK_INPUT(scales);
    CHECK_CUDA(shifts);

    TORCH_CHECK(x_in.dim() == 3, "x_in must be [N, H, D]");

    int N = x_in.size(0);
    int H = x_in.size(1);
    int D = x_in.size(2);
    int half_D = D / 2;
    int f_feat = 4;
    int64_t BKH = B * K * H;

    auto grad_x = torch::zeros_like(x_in);
    auto grad_angles = torch::zeros({B, K, H, half_D}, x_in.options());
    auto grad_scales = torch::zeros_like(scales);

    auto feat_buf = torch::empty({BKH, N, f_feat * half_D}, x_in.options());
    auto grad_mod_buf = torch::empty({BKH, N, half_D}, x_in.options());
    auto save_buf = torch::empty({B * N * H, K, D}, x_in.options());

    launch_givens_coupling_bwd(
        grad_out.data_ptr<float>(),
        x_in.data_ptr<float>(),
        angles.data_ptr<float>(),
        coupling.data_ptr<float>(),
        scales.data_ptr<float>(),
        shifts.data_ptr<int>(),
        grad_x.data_ptr<float>(),
        grad_angles.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        feat_buf.data_ptr<float>(),
        grad_mod_buf.data_ptr<float>(),
        save_buf.data_ptr<float>(),
        N, H, D, static_cast<int>(K), static_cast<int>(B)
    );

    auto grad_coupling = torch::bmm(
        feat_buf.transpose(1, 2),
        grad_mod_buf
    );
    grad_coupling = grad_coupling.view({B, K, H, f_feat * half_D, half_D});

    return {grad_x, grad_angles, grad_coupling, grad_scales};
}

torch::Tensor clifford_coupling_forward(torch::Tensor x_pairs) {
    CHECK_CUDA(x_pairs);
    CHECK_CONTIGUOUS(x_pairs);
    TORCH_CHECK(x_pairs.dim() >= 2, "x_pairs must have at least 2 dimensions");
    TORCH_CHECK(x_pairs.size(-1) == 2, "Last dimension must be 2");

    auto sizes = x_pairs.sizes();
    int P = sizes[sizes.size() - 2];

    int batch_size = 1;
    for (size_t i = 0; i < sizes.size() - 2; i++) {
        batch_size *= sizes[i];
    }

    std::vector<int64_t> output_sizes_vec;
    for (size_t i = 0; i < sizes.size() - 2; i++) {
        output_sizes_vec.push_back(sizes[i]);
    }
    output_sizes_vec.push_back(4 * P);

    auto features = torch::empty(output_sizes_vec, x_pairs.options());

    clifford_fwd_launcher(
        x_pairs.data_ptr<float>(),
        features.data_ptr<float>(),
        batch_size,
        P
    );

    return features;
}

torch::Tensor clifford_coupling_backward(
    torch::Tensor x_pairs,
    torch::Tensor grad_features
) {
    CHECK_CUDA(x_pairs);
    CHECK_CONTIGUOUS(x_pairs);
    CHECK_CUDA(grad_features);
    CHECK_CONTIGUOUS(grad_features);

    auto sizes = x_pairs.sizes();
    int P = sizes[sizes.size() - 2];
    int batch_size = 1;
    for (size_t i = 0; i < sizes.size() - 2; i++) {
        batch_size *= sizes[i];
    }

    auto grad_x_pairs = torch::empty_like(x_pairs);

    clifford_bwd_launcher(
        x_pairs.data_ptr<float>(),
        grad_features.data_ptr<float>(),
        grad_x_pairs.data_ptr<float>(),
        batch_size,
        P
    );

    return grad_x_pairs;
}

torch::Tensor fused_embedding_forward(
    torch::Tensor phase_angles,
    torch::Tensor log_magnitude,
    torch::Tensor token_ids
) {
    CHECK_CUDA(phase_angles);
    CHECK_CONTIGUOUS(phase_angles);
    CHECK_CUDA(log_magnitude);
    CHECK_CONTIGUOUS(log_magnitude);
    CHECK_CUDA(token_ids);

    auto orig_shape = token_ids.sizes().vec();
    auto flat_ids = token_ids.reshape({-1}).to(torch::kInt32).contiguous();
    int N = flat_ids.size(0);
    int V = phase_angles.size(0);
    int P = phase_angles.size(1);
    int D = 2 * P;

    TORCH_CHECK(flat_ids.max().item<int>() < V && flat_ids.min().item<int>() >= 0,
                "token_ids out of range [0, ", V, ")");

    auto output = torch::empty({N, D}, phase_angles.options());

    fused_embedding_fwd_launcher(
        phase_angles.data_ptr<float>(),
        log_magnitude.data_ptr<float>(),
        flat_ids.data_ptr<int>(),
        output.data_ptr<float>(),
        N, P
    );

    orig_shape.push_back(D);
    return output.reshape(orig_shape);
}

std::vector<torch::Tensor> fused_embedding_backward(
    torch::Tensor phase_angles,
    torch::Tensor log_magnitude,
    torch::Tensor token_ids,
    torch::Tensor grad_output
) {
    CHECK_CUDA(phase_angles);
    CHECK_CUDA(log_magnitude);
    CHECK_CUDA(token_ids);
    CHECK_CUDA(grad_output);
    CHECK_CONTIGUOUS(grad_output);

    auto flat_ids = token_ids.reshape({-1}).to(torch::kInt32).contiguous();
    auto flat_grad = grad_output.reshape({-1, grad_output.size(-1)}).contiguous();
    int N = flat_ids.size(0);
    int V = phase_angles.size(0);
    int P = phase_angles.size(1);

    TORCH_CHECK(flat_ids.max().item<int>() < V && flat_ids.min().item<int>() >= 0,
                "token_ids out of range [0, ", V, ")");

    auto grad_phase = torch::zeros_like(phase_angles);
    auto grad_log_mag = torch::zeros_like(log_magnitude);

    fused_embedding_bwd_launcher(
        phase_angles.data_ptr<float>(),
        log_magnitude.data_ptr<float>(),
        flat_ids.data_ptr<int>(),
        flat_grad.data_ptr<float>(),
        grad_phase.data_ptr<float>(),
        grad_log_mag.data_ptr<float>(),
        N, P
    );

    return {grad_phase, grad_log_mag};
}

std::vector<torch::Tensor> fused_norm_forward(
    torch::Tensor x,
    torch::Tensor scale,
    double eps
) {
    CHECK_CUDA(x);
    CHECK_CUDA(scale);

    int D = x.size(-1);
    auto flat_x = x.reshape({-1, D}).contiguous();
    int N = flat_x.size(0);

    auto out = torch::empty_like(flat_x);
    auto rms_inv = torch::empty({N}, x.options());

    fused_norm_fwd_launcher(
        flat_x.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        rms_inv.data_ptr<float>(),
        N, D, static_cast<float>(eps)
    );

    return {out.reshape(x.sizes()), rms_inv};
}

std::vector<torch::Tensor> fused_norm_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor rms_inv,
    double eps
) {
    CHECK_CUDA(grad_out);
    CHECK_CUDA(x);
    CHECK_CUDA(scale);
    CHECK_CUDA(rms_inv);

    int D = x.size(-1);
    auto flat_grad = grad_out.reshape({-1, D}).contiguous();
    auto flat_x = x.reshape({-1, D}).contiguous();
    int N = flat_x.size(0);

    auto grad_x = torch::empty_like(flat_x);

    fused_norm_bwd_launcher(
        flat_grad.data_ptr<float>(),
        flat_x.data_ptr<float>(),
        scale.data_ptr<float>(),
        rms_inv.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        N, D, static_cast<float>(eps)
    );

    auto grad_scale = (flat_grad * flat_x * rms_inv.unsqueeze(1)).sum(0);

    return {grad_x.reshape(x.sizes()), grad_scale};
}

torch::Tensor fused_gelu_forward(torch::Tensor x, double eps) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);

    int D = x.size(-1);
    TORCH_CHECK(D % 2 == 0, "D must be even");
    int half_D = D / 2;

    auto flat_x = x.reshape({-1, D}).contiguous();
    int N = flat_x.size(0);
    int total_pairs = N * half_D;

    auto out = torch::empty_like(flat_x);

    fused_gelu_fwd_launcher(
        flat_x.data_ptr<float>(),
        out.data_ptr<float>(),
        total_pairs, half_D, static_cast<float>(eps)
    );

    return out.reshape(x.sizes());
}

torch::Tensor fused_gelu_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    double eps
) {
    CHECK_CUDA(grad_out);
    CHECK_CUDA(x);

    int D = x.size(-1);
    int half_D = D / 2;

    auto flat_grad = grad_out.reshape({-1, D}).contiguous();
    auto flat_x = x.reshape({-1, D}).contiguous();
    int N = flat_x.size(0);
    int total_pairs = N * half_D;

    auto grad_x = torch::empty_like(flat_x);

    fused_gelu_bwd_launcher(
        flat_grad.data_ptr<float>(),
        flat_x.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        total_pairs, half_D, static_cast<float>(eps)
    );

    return grad_x.reshape(x.sizes());
}

std::vector<torch::Tensor> fused_mixer_forward(
    torch::Tensor x,
    torch::Tensor angles,
    torch::Tensor coupling,
    torch::Tensor raw_scales,
    double eps
) {
    CHECK_CUDA(x);
    CHECK_CUDA(angles);
    CHECK_CUDA(coupling);
    CHECK_CUDA(raw_scales);

    int D = x.size(-1);
    auto flat_x = x.reshape({-1, D}).contiguous().to(torch::kFloat32);
    int N = flat_x.size(0);

    int n_groups = angles.size(0);
    int G = angles.size(1);
    TORCH_CHECK(D == 2 * n_groups * G, "D must equal 2 * n_groups * G");
    TORCH_CHECK(coupling.size(0) == n_groups, "coupling first dim mismatch");
    TORCH_CHECK(coupling.size(1) == 4 * G, "coupling second dim must be 4*G");
    TORCH_CHECK(coupling.size(2) == G, "coupling third dim must be G");
    TORCH_CHECK(raw_scales.size(0) == n_groups, "raw_scales first dim mismatch");
    TORCH_CHECK(raw_scales.size(1) == 2 * G, "raw_scales second dim must be 2*G");

    auto flat_angles   = angles.contiguous().to(torch::kFloat32);
    auto flat_coupling = coupling.contiguous().to(torch::kFloat32);
    auto flat_scales   = raw_scales.contiguous().to(torch::kFloat32);

    auto x_out = torch::empty_like(flat_x);
    auto save_feat_norm = torch::empty({N, n_groups, 4 * G}, flat_x.options());
    auto save_inv_norm  = torch::empty({N, n_groups}, flat_x.options());
    auto save_theta     = torch::empty({N, n_groups, G}, flat_x.options());

    fused_mixer_fwd_launcher(
        flat_x.data_ptr<float>(),
        flat_angles.data_ptr<float>(),
        flat_coupling.data_ptr<float>(),
        flat_scales.data_ptr<float>(),
        x_out.data_ptr<float>(),
        save_feat_norm.data_ptr<float>(),
        save_inv_norm.data_ptr<float>(),
        save_theta.data_ptr<float>(),
        N, D, n_groups, G, static_cast<float>(eps)
    );

    return {x_out.reshape(x.sizes()), save_feat_norm, save_inv_norm, save_theta, flat_x};
}

std::vector<torch::Tensor> fused_mixer_backward(
    torch::Tensor grad_out,
    torch::Tensor flat_x,
    torch::Tensor angles,
    torch::Tensor coupling,
    torch::Tensor raw_scales,
    torch::Tensor save_feat_norm,
    torch::Tensor save_inv_norm,
    torch::Tensor save_theta,
    double eps,
    std::vector<int64_t> out_shape
) {
    CHECK_CUDA(grad_out);
    CHECK_CUDA(flat_x);
    CHECK_CUDA(angles);
    CHECK_CUDA(coupling);
    CHECK_CUDA(raw_scales);
    CHECK_CUDA(save_feat_norm);
    CHECK_CUDA(save_inv_norm);
    CHECK_CUDA(save_theta);

    int D = flat_x.size(1);
    auto flat_grad = grad_out.reshape({-1, D}).contiguous().to(torch::kFloat32);
    int N = flat_grad.size(0);

    int n_groups = angles.size(0);
    int G = angles.size(1);

    auto grad_x = torch::zeros_like(flat_x);
    auto grad_angles = torch::zeros({n_groups, G}, flat_x.options());
    auto grad_coupling = torch::zeros({n_groups, 4 * G, G}, flat_x.options());
    auto grad_raw_scales = torch::zeros({n_groups, 2 * G}, flat_x.options());

    auto flat_angles   = angles.contiguous().to(torch::kFloat32);
    auto flat_coupling = coupling.contiguous().to(torch::kFloat32);
    auto flat_scales   = raw_scales.contiguous().to(torch::kFloat32);

    fused_mixer_bwd_launcher(
        flat_grad.data_ptr<float>(),
        flat_x.data_ptr<float>(),
        flat_angles.data_ptr<float>(),
        flat_coupling.data_ptr<float>(),
        flat_scales.data_ptr<float>(),
        save_feat_norm.data_ptr<float>(),
        save_inv_norm.data_ptr<float>(),
        save_theta.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_angles.data_ptr<float>(),
        grad_coupling.data_ptr<float>(),
        grad_raw_scales.data_ptr<float>(),
        N, D, n_groups, G, static_cast<float>(eps)
    );

    return {grad_x.reshape(out_shape), grad_angles, grad_coupling, grad_raw_scales};
}

std::vector<torch::Tensor> fused_zloss_forward(
    torch::Tensor logits,
    torch::Tensor mask
) {
    CHECK_CUDA(logits);
    CHECK_CUDA(mask);

    int V = logits.size(-1);
    auto flat_logits = logits.reshape({-1, V}).contiguous().to(torch::kFloat32);
    auto flat_mask = mask.reshape({-1}).contiguous().to(torch::kInt32);
    int N = flat_logits.size(0);
    TORCH_CHECK(flat_mask.size(0) == N, "mask size must match flattened logits rows");

    auto lse = torch::empty({N}, flat_logits.options());

    fused_zloss_fwd_launcher(
        flat_logits.data_ptr<float>(),
        flat_mask.data_ptr<int>(),
        lse.data_ptr<float>(),
        N, V
    );

    auto mask_f = flat_mask.to(torch::kFloat32);
    auto n_valid = mask_f.sum();
    auto z_loss = (lse * lse * mask_f).sum() / n_valid.clamp_min(1.0);

    return {z_loss, lse, n_valid, flat_logits};
}

torch::Tensor fused_zloss_backward(
    torch::Tensor grad_z_loss,
    torch::Tensor flat_logits,
    torch::Tensor lse,
    torch::Tensor flat_mask,
    torch::Tensor n_valid,
    std::vector<int64_t> out_shape
) {
    CHECK_CUDA(grad_z_loss);
    CHECK_CUDA(flat_logits);
    CHECK_CUDA(lse);
    CHECK_CUDA(flat_mask);

    int N = flat_logits.size(0);
    int V = flat_logits.size(1);
    auto grad_logits = torch::empty_like(flat_logits);

    float g = grad_z_loss.item<float>();
    float nv = n_valid.item<float>();
    float grad_scale = (nv > 0.0f) ? (g * 2.0f / nv) : 0.0f;

    fused_zloss_bwd_launcher(
        flat_logits.data_ptr<float>(),
        lse.data_ptr<float>(),
        flat_mask.data_ptr<int>(),
        grad_scale,
        grad_logits.data_ptr<float>(),
        N, V
    );

    return grad_logits.reshape(out_shape);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("givens_forward(Tensor x, Tensor angles, Tensor alpha, Tensor scales, Tensor shifts, int K) -> Tensor");
    ops.impl("givens_forward", torch::kCUDA, &givens_forward);

    ops.def("givens_backward(Tensor grad_out, Tensor x_in, Tensor angles, Tensor alpha, Tensor scales, Tensor shifts, int K) -> Tensor[]");
    ops.impl("givens_backward", torch::kCUDA, &givens_backward);

    ops.def("givens_coupling_forward(Tensor x, Tensor angles, Tensor coupling, Tensor scales, Tensor shifts, int K, int B) -> Tensor");
    ops.impl("givens_coupling_forward", torch::kCUDA, &givens_coupling_forward);

    ops.def("givens_coupling_backward(Tensor grad_out, Tensor x_in, Tensor angles, Tensor coupling, Tensor scales, Tensor shifts, int K, int B) -> Tensor[]");
    ops.impl("givens_coupling_backward", torch::kCUDA, &givens_coupling_backward);

    ops.def("givens_mor_forward(Tensor x, Tensor expert_angles, Tensor expert_scales, Tensor router_weights, Tensor shifts, int K, int E) -> Tensor");
    ops.impl("givens_mor_forward", torch::kCUDA, &givens_mor_forward);

    ops.def("givens_mor_backward(Tensor grad_out, Tensor x_in, Tensor expert_angles, Tensor expert_scales, Tensor router_weights, Tensor shifts, int K, int E) -> Tensor[]");
    ops.impl("givens_mor_backward", torch::kCUDA, &givens_mor_backward);

    ops.def("clifford_coupling_forward(Tensor x_pairs) -> Tensor");
    ops.impl("clifford_coupling_forward", torch::kCUDA, &clifford_coupling_forward);

    ops.def("clifford_coupling_backward(Tensor x_pairs, Tensor grad_features) -> Tensor");
    ops.impl("clifford_coupling_backward", torch::kCUDA, &clifford_coupling_backward);

    ops.def("fused_embedding_forward(Tensor phase_angles, Tensor log_magnitude, Tensor token_ids) -> Tensor");
    ops.impl("fused_embedding_forward", torch::kCUDA, &fused_embedding_forward);

    ops.def("fused_embedding_backward(Tensor phase_angles, Tensor log_magnitude, Tensor token_ids, Tensor grad_output) -> Tensor[]");
    ops.impl("fused_embedding_backward", torch::kCUDA, &fused_embedding_backward);

    ops.def("fused_norm_forward(Tensor x, Tensor scale, float eps) -> Tensor[]");
    ops.impl("fused_norm_forward", torch::kCUDA, &fused_norm_forward);

    ops.def("fused_norm_backward(Tensor grad_out, Tensor x, Tensor scale, Tensor rms_inv, float eps) -> Tensor[]");
    ops.impl("fused_norm_backward", torch::kCUDA, &fused_norm_backward);

    ops.def("fused_gelu_forward(Tensor x, float eps) -> Tensor");
    ops.impl("fused_gelu_forward", torch::kCUDA, &fused_gelu_forward);

    ops.def("fused_gelu_backward(Tensor grad_out, Tensor x, float eps) -> Tensor");
    ops.impl("fused_gelu_backward", torch::kCUDA, &fused_gelu_backward);

    ops.def("fused_mixer_forward(Tensor x, Tensor angles, Tensor coupling, Tensor raw_scales, float eps) -> Tensor[]");
    ops.impl("fused_mixer_forward", torch::kCUDA, &fused_mixer_forward);

    ops.def("fused_mixer_backward(Tensor grad_out, Tensor flat_x, Tensor angles, Tensor coupling, Tensor raw_scales, Tensor save_feat_norm, Tensor save_inv_norm, Tensor save_theta, float eps, int[] out_shape) -> Tensor[]");
    ops.impl("fused_mixer_backward", torch::kCUDA, &fused_mixer_backward);

    ops.def("fused_zloss_forward(Tensor logits, Tensor mask) -> Tensor[]");
    ops.impl("fused_zloss_forward", torch::kCUDA, &fused_zloss_forward);

    ops.def("fused_zloss_backward(Tensor grad_z_loss, Tensor flat_logits, Tensor lse, Tensor flat_mask, Tensor n_valid, int[] out_shape) -> Tensor");
    ops.impl("fused_zloss_backward", torch::kCUDA, &fused_zloss_backward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
