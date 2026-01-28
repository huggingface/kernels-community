import torch
import torch.nn as nn

from ._ops import ops


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        if not x.is_contiguous():
            x = x.contiguous()
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.silu_and_mul(out, x)
        return out

class Silu(nn.Module):
    """An activation function for SiLU.

    The function computes x -> silu(x).

    Shapes:
        x: (num_tokens, d) or (batch_size, seq_len, d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        ops.silu(out, x)
        return out

class Gelu(nn.Module):
    """An activation function for GELU.

    The function computes x -> gelu(x).

    Shapes:
        x: (num_tokens, d) or (batch_size, seq_len, d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        ops.gelu(out, x)
        return out

class GeluTanh(nn.Module):
    """An activation function for GELU with `tanh` approximation.

    The function computes x -> gelu_tanh(x).

    Shapes:
        x: (num_tokens, d) or (batch_size, seq_len, d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        ops.gelu_tanh(out, x)
        return out


class MulAndSilu(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> x[:d] * silu(x[d:]) where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.mul_and_silu(out, x)
        return out


class GeluAndMul(nn.Module):
    """An activation function for GeGLU.

    The function computes x -> GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        if not x.is_contiguous():
            x = x.contiguous()
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.gelu_and_mul(out, x)
        return out


class GeluTanhAndMul(nn.Module):
    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        if not x.is_contiguous():
            x = x.contiguous()
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.gelu_tanh_and_mul(out, x)
        return out


class FatreluAndMul(nn.Module):
    """An activation function for FATReLU.

    The function computes x -> FATReLU(x[:d]) * x[d:] where
    d = x.shape[-1] // 2.
    This is used in openbmb/MiniCPM-S-1B-sft.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor):
        if not x.is_contiguous():
            x = x.contiguous()
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.fatrelu_and_mul(out, x, self.threshold)
        return out


class FastGELU(nn.Module):
    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        ops.gelu_fast(out, x)
        return out


class NewGELU(nn.Module):
    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        ops.gelu_new(out, x)
        return out


class QuickGELU(nn.Module):
    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        ops.gelu_quick(out, x)
        return out
