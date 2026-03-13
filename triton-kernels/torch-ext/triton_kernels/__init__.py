# Make sure to add this in the build folder as this won't build if we put that here

# from . import matmul_ogs, tensor_details, numerics_details, tensor, swiglu, routing

# __all__ = ["matmul_ogs" , "tensor_details", "numerics_details", "tensor", "swiglu", "routing"]

# Then, run the following code to build the kernels: 
# docker run --rm \
#   -v $(pwd):/app \
#   -w /app \
#   ghcr.io/huggingface/kernel-builder:main
