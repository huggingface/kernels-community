# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "kernels==0.17.2",
#     "torch==2.13.0",
# ]
# ///

from kernels import get_kernel

kernel_module = get_kernel("kernels-community/flash-attn2", version=2)
flash_attn_func = kernel_module.flash_attn_func
