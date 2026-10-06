import os

# Use the CI backends by default, while allowing local overrides.
os.environ.setdefault("EINOPS_TEST_BACKENDS", "torch,numpy")
