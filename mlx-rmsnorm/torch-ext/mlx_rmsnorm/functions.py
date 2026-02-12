import torch
from ._ops import ops

def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    weight = weight.view(-1)
    output = torch.zeros_like(x)
    ops.launch_forward_kernel(x, weight, output, epsilon)
    output = output.view(original_shape)
    return output

def rmsnorm_backward(x: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    weight = weight.view(-1)
    grad_output = grad_output.view(-1)
    grad_input = torch.zeros_like(x)
    grad_weight = torch.zeros_like(weight)
    ops.launch_backward_kernel(x, weight, grad_output, grad_input, grad_weight, epsilon)
    grad_input = grad_input.view(original_shape)
    grad_weight = grad_weight.view(original_shape)
    return grad_input, grad_weight