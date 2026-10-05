import os

# Upstream's test suite refuses to run without EINOPS_TEST_BACKENDS. Default it
# to the frameworks that are installed, so `pytest` works out of the box.
if "EINOPS_TEST_BACKENDS" not in os.environ:
    available = []
    for name in ["torch", "jax", "numpy", "tensorflow"]:
        try:
            __import__(name)
            available.append(name)
        except ImportError:
            pass
    os.environ["EINOPS_TEST_BACKENDS"] = ",".join(available)
