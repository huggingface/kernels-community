"""
Type definitions for TriMul task.

Input: Tuple of (input_tensor, mask, weights, config)
  - input_tensor: Input tensor of shape [batch_size, seq_len, seq_len, dim]
  - mask: Mask tensor of shape [batch_size, seq_len, seq_len]
  - weights: Dictionary containing model weights
  - config: Dictionary containing model configuration parameters

Output: Output tensor of shape [batch_size, seq_len, seq_len, dim]
"""

import torch
from typing import Tuple, Dict, Any

# Input type: (input_tensor, mask, weights, config)
input_t = Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]

# Output type: output tensor
output_t = torch.Tensor