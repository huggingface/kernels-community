"""The kernel against a reference implementation of the same recurrence.

The reference is transformers' `torch_recurrent_gated_delta_rule` written out, so a pass means the
kernel is a drop-in for the model's own fallback -- not merely self-consistent. Random inputs are
enough: every element of the recurrence is exercised by one step, and several steps in a row are what
catch a wrong stride.
"""

import os
import sys

import pytest
import torch


DEV = "mps"
LIB = os.environ.get("GGML_ATTN_LOCAL_LIB")

if LIB:  # a library built outside the nix build, loaded directly
    torch.ops.load_library(LIB)
    ops = getattr(torch.ops, os.path.basename(LIB).removesuffix(".so"))
else:
    # A packaged build, resolved through `kernels` -- which is what a consumer does, so the tests
    # exercise the same loading path rather than a layout detail. A variant directory is not a
    # package, so a bare `import` cannot find it.
    from pathlib import Path

    try:
        from kernels import get_local_kernel

        ops = get_local_kernel(Path(__file__).resolve().parent.parent, "metal")
    except Exception as error:  # pragma: no cover
        pytest.skip(f"no kernel to test ({error}); set GGML_ATTN_LOCAL_LIB or build one",
                    allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def call(q, k, v, g, beta, state):
    """`(out, final_state)`, whichever way the kernel was loaded.

    The raw op returns one flat allocation -- the outputs then the final state -- because two returns
    of a custom op may not alias each other. The packaged wrapper splits it; against a locally built
    library there is no wrapper, so the split happens here.
    """
    result = ops.gated_delta_net(q, k, v, g, beta, state)
    if isinstance(result, tuple):
        return result
    n_seqs, n_tokens, n_heads, head_dim = v.shape
    n_out = n_seqs * n_tokens * n_heads * head_dim
    return (
        result[:n_out].view(n_seqs, n_tokens, n_heads, head_dim),
        result[n_out:].view(n_seqs, n_heads, head_dim, head_dim),
    )


def reference(q, k, v, g, beta, state):
    """transformers' torch recurrence, kept in (n_seqs, n_tokens, n_heads, head_dim) order.

    `state` arrives in the kernel's [value][key] layout, so it is transposed into the [key][value]
    order this recurrence builds and transposed back on the way out. That transpose is the whole
    difference between the two conventions, and doing it here is what makes the comparison a real
    test of the kernel rather than of the layout.
    """
    q, k, v = (t.transpose(1, 2).float() for t in (q, k, v))
    g, beta = (t.transpose(1, 2).float() for t in (g, beta))
    n_seqs, n_heads, n_tokens, head_dim = k.shape
    scale = head_dim**-0.5
    q = q * scale
    out = torch.zeros(n_seqs, n_heads, n_tokens, head_dim, dtype=v.dtype, device=v.device)
    s = state.float().transpose(-1, -2).contiguous()
    for i in range(n_tokens):
        q_t, k_t, v_t = q[:, :, i], k[:, :, i], v[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        s = s * g_t
        kv_mem = torch.matmul(k_t.unsqueeze(-2), s).squeeze(-2)
        delta = (v_t - kv_mem) * beta_t
        s = s + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = torch.matmul(q_t.unsqueeze(-2), s).squeeze(-2)
    return out.transpose(1, 2), s.transpose(-1, -2)


def inputs(n_seqs=1, n_tokens=1, n_heads=4, head_dim=128, seed=0):
    gen = torch.Generator().manual_seed(seed)
    shape = (n_seqs, n_tokens, n_heads, head_dim)

    def rnd(*s):
        return torch.randn(*s, generator=gen).to(DEV)

    q, k, v = rnd(*shape), rnd(*shape), rnd(*shape)
    # q and k are l2-normalised by the model before the rule, so normalise here too: it is the range
    # the kernel actually sees, and an unnormalised k makes the recurrence diverge over many steps.
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    g = -rnd(n_seqs, n_tokens, n_heads).abs()  # log-domain decay, negative
    beta = rnd(n_seqs, n_tokens, n_heads).sigmoid()
    state = rnd(n_seqs, n_heads, head_dim, head_dim) * 0.1
    return q, k, v, g, beta, state


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("n_tokens", [1, 4])
@pytest.mark.parametrize("n_heads", [1, 4, 32])
def test_matches_reference(head_dim, n_tokens, n_heads):
    if not ops.supports_gated_delta_net(head_dim):
        pytest.skip(f"no kernel for head_dim {head_dim}")
    args = inputs(n_tokens=n_tokens, n_heads=n_heads, head_dim=head_dim)
    out, state = call(*args)
    ref_out, ref_state = reference(*args)
    torch.mps.synchronize()

    scale = max(ref_out.abs().max().item(), 1e-6)
    assert (out - ref_out).abs().max().item() / scale < 2e-5, "outputs diverge"
    s_scale = max(ref_state.abs().max().item(), 1e-6)
    assert (state - ref_state).abs().max().item() / s_scale < 2e-5, "final state diverges"


def test_state_is_carried_between_calls():
    """Feeding the returned state back must match running the tokens in one go."""
    head_dim, n_heads = 128, 8
    if not ops.supports_gated_delta_net(head_dim):
        pytest.skip("no kernel")
    q, k, v, g, beta, state = inputs(n_tokens=4, n_heads=n_heads, head_dim=head_dim)

    whole, whole_state = call(q, k, v, g, beta, state)
    s = state
    steps = []
    for i in range(4):
        o, s = call(
            q[:, i : i + 1], k[:, i : i + 1], v[:, i : i + 1],
            g[:, i : i + 1], beta[:, i : i + 1], s,
        )
        steps.append(o)
    stepwise = torch.cat(steps, dim=1)
    torch.mps.synchronize()

    scale = max(whole.abs().max().item(), 1e-6)
    assert (whole - stepwise).abs().max().item() / scale < 2e-5
    assert (whole_state - s).abs().max().item() / max(whole_state.abs().max().item(), 1e-6) < 2e-5


def test_unsupported_head_dim_is_reported_not_faulted():
    assert not ops.supports_gated_delta_net(48 + 1)
    assert not ops.supports_gated_delta_net(160)  # nsg 5, not instantiated upstream


def test_compiles_without_a_graph_break():
    """The op must carry a fake, or a caller cannot compile through it."""
    if LIB:
        pytest.skip("fakes live in the packaged python wrapper, not in the raw library")
    head_dim = 128
    if not ops.supports_gated_delta_net(head_dim):
        pytest.skip("no kernel")
    args = inputs(n_tokens=1, n_heads=8, head_dim=head_dim)
    fn = torch.compile(lambda *a: ops.gated_delta_net(*a), fullgraph=True)  # the wrapper
    out, state = fn(*args)
    torch.mps.synchronize()
    assert out.shape == args[2].shape


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
