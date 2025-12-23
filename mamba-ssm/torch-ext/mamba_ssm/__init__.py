__version__ = "2.2.4"

from .ops import selective_state_update, mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
from .ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from .modules import Mamba, Mamba2
from .models import MambaLMHeadModel

falcon_mamba_inner_fn = mamba_inner_fn
__all__ = [
    "selective_scan_fn", "mamba_inner_fn", "falcon_mamba_inner_fn", "selective_state_update", "mamba_chunk_scan_combined", "mamba_split_conv1d_scan_combined",
    "Mamba", "Mamba2", "MambaLMHeadModel",
]
