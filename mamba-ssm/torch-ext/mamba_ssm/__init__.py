__version__ = "2.2.4"

from . import layers
from .models import MambaLMHeadModel
from .modules import Mamba, Mamba2
from .ops import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update
from .ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn


falcon_mamba_inner_fn = mamba_inner_fn
__all__ = [
    # wrappers
    "layers",
    # originals
    "selective_scan_fn", "mamba_inner_fn", "falcon_mamba_inner_fn", "selective_state_update", "mamba_chunk_scan_combined", "mamba_split_conv1d_scan_combined",
    "Mamba", "Mamba2", "MambaLMHeadModel",
]
