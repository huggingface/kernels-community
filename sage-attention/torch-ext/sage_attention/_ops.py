import torch
from . import _sage_attention_bb220cf_dirty
ops = torch.ops._sage_attention_bb220cf_dirty

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"_sage_attention_bb220cf_dirty::{op_name}"