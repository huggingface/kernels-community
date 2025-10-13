import torch
from . import _flash_attn_2b4ec51_dirty
ops = torch.ops._flash_attn_2b4ec51_dirty

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"_flash_attn_2b4ec51_dirty::{op_name}"