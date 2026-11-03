import math
import torch
import torch.nn.functional as F

from ._ops import ops
from . import layers

_shifts_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}
_alpha_cache: dict[tuple[tuple[int, ...], torch.device], torch.Tensor] = {}


def _get_shifts(K: int, half_D: int, device: torch.device) -> torch.Tensor:
    key = (K, half_D, device)
    if key not in _shifts_cache:
        n_levels = max(1, int(math.log2(half_D))) if half_D > 1 else 1
        shifts = [-(1 << (k % n_levels)) for k in range(K)]
        _shifts_cache[key] = torch.tensor(shifts, dtype=torch.int32, device=device)
    return _shifts_cache[key]


def _get_alpha_zeros(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    key = (shape, device)
    if key not in _alpha_cache:
        _alpha_cache[key] = torch.zeros(shape, dtype=torch.float32, device=device)
    return _alpha_cache[key]


def clear_kernel_caches() -> None:
    _shifts_cache.clear()
    _alpha_cache.clear()


class _FusedFFNRotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, angles: torch.Tensor, raw_scales: torch.Tensor) -> torch.Tensor:
        scales = F.softplus(raw_scales)
        orig_shape = x.shape
        orig_dtype = x.dtype
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        K, P = angles.shape
        x_3d = x.unsqueeze(1).contiguous()
        scales_3d = scales.unsqueeze(1).contiguous()
        shifts = _get_shifts(K, P, x.device)
        angles_3d = angles.unsqueeze(1).contiguous()
        alpha_3d = _get_alpha_zeros(tuple(angles_3d.shape), angles_3d.device)

        out = ops.givens_forward(x_3d, angles_3d, alpha_3d, scales_3d, shifts, K)

        ctx.save_for_backward(x_3d, angles_3d, alpha_3d, scales_3d, shifts, raw_scales)
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = orig_dtype
        ctx.K = K

        return out.squeeze(1).to(orig_dtype).reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_3d, angles_3d, alpha_3d, scales_3d, shifts, raw_scales = ctx.saved_tensors
        grad_out = grad_output.reshape(-1, grad_output.shape[-1]).unsqueeze(1).contiguous()

        grad_x, grad_angles, _grad_alpha, grad_scales = ops.givens_backward(
            grad_out, x_3d, angles_3d, alpha_3d, scales_3d, shifts, ctx.K
        )

        grad_scales_2d = grad_scales.squeeze(1)
        grad_raw_scales = grad_scales_2d * torch.sigmoid(raw_scales)
        grad_x_out = grad_x.squeeze(1).to(ctx.orig_dtype).reshape(ctx.orig_shape)
        grad_angles_out = grad_angles.squeeze(1).to(raw_scales.dtype)

        return grad_x_out, grad_angles_out, grad_raw_scales.to(raw_scales.dtype)


def apply_fused_ffn(x: torch.Tensor, angles: torch.Tensor, raw_scales: torch.Tensor) -> torch.Tensor:
    return _FusedFFNRotation.apply(x, angles, raw_scales)


class _FusedCouplingRotation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        angles: torch.Tensor,
        coupling: torch.Tensor,
        raw_scales: torch.Tensor,
    ) -> torch.Tensor:
        scales = F.softplus(raw_scales)
        orig_dtype = x.dtype

        if x.dim() != 3:
            raise ValueError("x must be [N, H, D]")
        N, H, D = x.shape

        if angles.dim() != 4:
            raise ValueError("angles must be [B, K, H, D//2]")
        B = angles.shape[0]
        K = angles.shape[1]
        half_D = D // 2

        x_3d = x.float().contiguous()
        angles_4d = angles.float().contiguous()
        coupling_4d = coupling.float().contiguous()
        scales_4d = scales.float().contiguous()
        raw_scales_4d = raw_scales.contiguous()
        shifts = _get_shifts(K, half_D, x.device)

        out = ops.givens_coupling_forward(x_3d, angles_4d, coupling_4d, scales_4d, shifts, K, B)

        ctx.save_for_backward(x_3d, angles_4d, coupling_4d, raw_scales_4d, shifts)
        ctx.orig_dtype = orig_dtype
        ctx.K = K
        ctx.B = B

        return out.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_3d, angles_4d, coupling_4d, raw_scales_4d, shifts = ctx.saved_tensors
        scales_4d = F.softplus(raw_scales_4d).float().contiguous()
        grad_out_4d = grad_output.float().contiguous()

        grad_x_3d, grad_angles_4d, grad_coupling_4d, grad_scales_4d = ops.givens_coupling_backward(
            grad_out_4d, x_3d, angles_4d, coupling_4d, scales_4d, shifts, ctx.K, ctx.B
        )
        grad_scales_raw = grad_scales_4d * torch.sigmoid(raw_scales_4d)

        return (
            grad_x_3d.to(ctx.orig_dtype),
            grad_angles_4d.to(raw_scales_4d.dtype),
            grad_coupling_4d.to(raw_scales_4d.dtype),
            grad_scales_raw.to(raw_scales_4d.dtype),
        )


def apply_fused_coupling(
    x: torch.Tensor,
    angles: torch.Tensor,
    coupling: torch.Tensor,
    raw_scales: torch.Tensor,
) -> torch.Tensor:
    squeeze_out = False
    if angles.dim() == 3:
        angles = angles.unsqueeze(0)
        coupling = coupling.unsqueeze(0)
        raw_scales = raw_scales.unsqueeze(0)
        squeeze_out = True

    out = _FusedCouplingRotation.apply(x, angles, coupling, raw_scales)
    if squeeze_out:
        out = out.squeeze(0)
    return out


class _FusedMoRRotation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        expert_angles: torch.Tensor,
        raw_expert_scales: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_scales = F.softplus(raw_expert_scales)
        orig_shape = x.shape
        orig_dtype = x.dtype
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
            router_weights = router_weights.reshape(-1, router_weights.shape[-1])

        E, K, P = expert_angles.shape
        x_3d = x.unsqueeze(1).contiguous()
        ea_3d = expert_angles.unsqueeze(2).contiguous()
        es_3d = expert_scales.unsqueeze(2).float().contiguous()
        rw = router_weights.float().contiguous()
        shifts = _get_shifts(K, P, x.device)

        out = ops.givens_mor_forward(x_3d, ea_3d, es_3d, rw, shifts, K, E)

        ctx.save_for_backward(x_3d, ea_3d, es_3d, rw, shifts, raw_expert_scales)
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = orig_dtype
        ctx.K = K
        ctx.E = E

        return out.squeeze(1).to(orig_dtype).reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_3d, ea_3d, es_3d, rw, shifts, raw_expert_scales = ctx.saved_tensors
        grad_out = grad_output.reshape(-1, grad_output.shape[-1]).unsqueeze(1).contiguous()

        grad_x, grad_ea, grad_es, grad_rw = ops.givens_mor_backward(
            grad_out, x_3d, ea_3d, es_3d, rw, shifts, ctx.K, ctx.E
        )

        grad_x_out = grad_x.squeeze(1).to(ctx.orig_dtype).reshape(ctx.orig_shape)
        grad_ea_out = grad_ea.squeeze(2).to(ctx.orig_dtype)
        grad_es_raw = grad_es.squeeze(2) * torch.sigmoid(raw_expert_scales)
        grad_es_out = grad_es_raw.to(ctx.orig_dtype)

        if len(ctx.orig_shape) == 3:
            grad_rw = grad_rw.reshape(*ctx.orig_shape[:-1], ctx.E)

        return grad_x_out, grad_ea_out, grad_es_out, grad_rw.to(ctx.orig_dtype)


def apply_fused_mor(
    x: torch.Tensor,
    expert_angles: torch.Tensor,
    expert_scales: torch.Tensor,
    router_weights: torch.Tensor,
) -> torch.Tensor:
    return _FusedMoRRotation.apply(x, expert_angles, expert_scales, router_weights)


class _CliffordCoupling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_pairs: torch.Tensor) -> torch.Tensor:
        x_pairs_c = x_pairs.float().contiguous()
        ctx.save_for_backward(x_pairs_c)
        ctx.orig_dtype = x_pairs.dtype
        return ops.clifford_coupling_forward(x_pairs_c).to(x_pairs.dtype)

    @staticmethod
    def backward(ctx, grad_features: torch.Tensor):
        (x_pairs_c,) = ctx.saved_tensors
        grad_x = ops.clifford_coupling_backward(x_pairs_c, grad_features.float().contiguous())
        return grad_x.to(ctx.orig_dtype)


def clifford_coupling(x_pairs: torch.Tensor) -> torch.Tensor:
    return _CliffordCoupling.apply(x_pairs)


class _FusedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        phase_angles: torch.Tensor,
        log_magnitude: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(phase_angles, log_magnitude, token_ids)
        return ops.fused_embedding_forward(phase_angles, log_magnitude, token_ids)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        phase_angles, log_magnitude, token_ids = ctx.saved_tensors
        grad_phase, grad_log_mag = ops.fused_embedding_backward(
            phase_angles, log_magnitude, token_ids, grad_output.contiguous()
        )
        return grad_phase, grad_log_mag, None


def fused_embedding(
    phase_angles: torch.Tensor,
    log_magnitude: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    return _FusedEmbedding.apply(phase_angles, log_magnitude, token_ids)


class _FusedPearlNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
        out, rms_inv = ops.fused_norm_forward(x, scale, float(eps))
        ctx.save_for_backward(x, scale, rms_inv)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, scale, rms_inv = ctx.saved_tensors
        grad_x, grad_scale = ops.fused_norm_backward(
            grad_output.contiguous(), x, scale, rms_inv, ctx.eps
        )
        return grad_x, grad_scale, None


def fused_pearl_norm(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return _FusedPearlNorm.apply(x, scale, eps)


class _FusedRadialGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        out = ops.fused_gelu_forward(x, float(eps))
        ctx.save_for_backward(x)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_x = ops.fused_gelu_backward(grad_output.contiguous(), x, ctx.eps)
        return grad_x, None


def fused_radial_gelu(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return _FusedRadialGELU.apply(x, eps)


class _FusedMixerStage(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        angles: torch.Tensor,
        coupling: torch.Tensor,
        raw_scales: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        outs = ops.fused_mixer_forward(x, angles, coupling, raw_scales, float(eps))
        x_out, save_feat_norm, save_inv_norm, save_theta, flat_x = outs
        ctx.save_for_backward(
            flat_x, angles, coupling, raw_scales, save_feat_norm, save_inv_norm, save_theta
        )
        ctx.eps = float(eps)
        ctx.out_shape = list(x.shape)
        ctx.out_dtype = x.dtype
        return x_out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            flat_x,
            angles,
            coupling,
            raw_scales,
            save_feat_norm,
            save_inv_norm,
            save_theta,
        ) = ctx.saved_tensors
        grads = ops.fused_mixer_backward(
            grad_out.contiguous(),
            flat_x,
            angles,
            coupling,
            raw_scales,
            save_feat_norm,
            save_inv_norm,
            save_theta,
            ctx.eps,
            ctx.out_shape,
        )
        grad_x, grad_angles, grad_coupling, grad_raw_scales = grads
        if grad_x.dtype != ctx.out_dtype:
            grad_x = grad_x.to(ctx.out_dtype)
        return grad_x, grad_angles, grad_coupling, grad_raw_scales, None


def fused_mixer_stage(
    x: torch.Tensor,
    angles: torch.Tensor,
    coupling: torch.Tensor,
    raw_scales: torch.Tensor,
    eps: float = 1e-2,
) -> torch.Tensor:
    return _FusedMixerStage.apply(x, angles, coupling, raw_scales, eps)


class _FusedZLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z_loss, lse, n_valid, flat_logits = ops.fused_zloss_forward(logits, mask)
        flat_mask = mask.reshape(-1).to(torch.int32)
        ctx.save_for_backward(flat_logits, lse, flat_mask, n_valid)
        ctx.out_shape = list(logits.shape)
        ctx.out_dtype = logits.dtype
        return z_loss

    @staticmethod
    def backward(ctx, grad_z_loss: torch.Tensor):
        flat_logits, lse, flat_mask, n_valid = ctx.saved_tensors
        grad_logits = ops.fused_zloss_backward(
            grad_z_loss.contiguous(), flat_logits, lse, flat_mask, n_valid, ctx.out_shape
        )
        if grad_logits.dtype != ctx.out_dtype:
            grad_logits = grad_logits.to(ctx.out_dtype)
        return grad_logits, None


def fused_zloss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _FusedZLoss.apply(logits, mask)


__all__ = [
    "apply_fused_coupling",
    "apply_fused_ffn",
    "apply_fused_mor",
    "clear_kernel_caches",
    "clifford_coupling",
    "fused_embedding",
    "fused_mixer_stage",
    "fused_pearl_norm",
    "fused_radial_gelu",
    "fused_zloss",
    "layers",
]
