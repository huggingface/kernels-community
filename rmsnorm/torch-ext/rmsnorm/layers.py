import torch
from ._ops import ops

class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, variance_epsilon):
        ctx.variance_epsilon = variance_epsilon
        output, rstd = ops.apply_rms_norm(hidden_states, weight, variance_epsilon)
        ctx.save_for_backward(hidden_states, weight, output, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight, output, rstd = ctx.saved_tensors
        grads = ops.apply_rms_norm_backward(
            grad_output,
            hidden_states,
            weight,
            output,
            rstd,
            ctx.variance_epsilon,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1]
        )
        return grads[0], grads[1], None

class RMSNorm(torch.nn.Module):
    """
    RMSNorm module that uses the optimized LigerRMSNormFunction.
    
    Args:
        hidden_size (int): The size of the hidden dimension.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.
        offset (float, optional): Offset value to shift the weight tensor. Defaults to 0.0.
        casting_mode (str, optional): The casting mode to use. Defaults to "llama".
        in_place (bool, optional): Whether to modify dY in-place to store dX during backward. Defaults to True.
    """
    

    weight: torch.Tensor
    variance_epsilon: float
    
    def forward(self, hidden_states):
        """
        Apply RMS normalization to the input tensor.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (B, T, H) or (BxT, H)
            
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        return RMSNormFunction.apply(
            hidden_states,
            self.weight, 
            self.variance_epsilon,
        )
    
__all__ = ["RMSNorm"]
