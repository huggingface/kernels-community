"""Optional causal-conv1d dependency.

The faster fused short-convolution is provided by the separately-installed
``causal_conv1d`` package (e.g. ``pip install causal-conv1d``). When it is
installed we use it; otherwise every symbol degrades to ``None`` and callers
fall back to the pure-PyTorch / Triton implementations.
"""

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None
