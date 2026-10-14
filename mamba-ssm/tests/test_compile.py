import kernels
import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def layers():
    return kernels.get_kernel("kernels-community/mamba-ssm", version=4).layers


def rand(*shape, dtype, requires_grad=True, scale=1.0):
    return (torch.randn(*shape, device="cuda", dtype=dtype) * scale).requires_grad_(requires_grad)


def neg_rand(*shape, dtype=torch.float32):
    """Strictly negative leaf tensor, as the `A` of a stable SSM has to be."""
    return (-rand(*shape, dtype=dtype).detach().abs()).requires_grad_()


def assert_close(actual, expected, dtype):
    torch.testing.assert_close(
        actual, expected, rtol=3e-2 if dtype == torch.bfloat16 else 2e-3,
        atol=3e-3 if dtype == torch.bfloat16 else 2e-5,
    )


def check_compiled(fn, args, dtype, backward=True, **kwargs):
    """Compare `fn` compiled with `fullgraph=True` against eager on fresh copies of `args`.

    Inputs are compared as well as outputs, since some ops update a state argument in place.
    """
    compiled = torch.compile(fn, fullgraph=True)
    try:
        # Reuse the compiled callable to exercise cached launches as well as autotuning.
        for _ in range(2):
            results = []
            for f in (compiled, fn):
                inputs = [
                    t.detach().clone().requires_grad_(t.requires_grad) if torch.is_tensor(t) else t
                    for t in args
                ]
                outputs = f(*inputs, **kwargs)
                if backward:
                    # Only the first output is differentiated: returned states do not support gradients.
                    (outputs[0] if isinstance(outputs, tuple) else outputs).float().square().mean().backward()
                tensors = [t for t in inputs if torch.is_tensor(t)]
                results.append((outputs, tensors, [t.grad for t in tensors]))
            assert_close(*results, dtype)
    finally:
        torch._dynamo.reset()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_D", [False, True])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("has_delta_bias", [False, True])
def test_selective_scan_compile(layers, dtype, has_D, has_z, has_delta_bias):
    torch.manual_seed(0)
    # More than one native scan chunk, including an incomplete final chunk.
    batch, dim, dstate, seqlen = 2, 8, 16, 2051
    args = (
        rand(batch, dim, seqlen, dtype=dtype),  # hidden_states
        rand(batch, dim, seqlen, dtype=dtype),  # dt
        neg_rand(dim, dstate),  # A
        rand(batch, dstate, seqlen, dtype=dtype),  # B
        rand(batch, dstate, seqlen, dtype=dtype),  # C
        rand(dim, dtype=torch.float32) if has_D else None,  # D
        rand(batch, dim, seqlen, dtype=dtype) if has_z else None,  # z
        rand(dim, dtype=torch.float32) if has_delta_bias else None,  # delta_bias
    )
    check_compiled(
        layers.selective_scan_fn(), args, dtype, delta_softplus=True, return_last_state=True,
    )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["mamba", "mamba2"])
def test_selective_state_update_compile(layers, dtype, layout):
    torch.manual_seed(0)
    batch, nheads, headdim, dstate, ngroups = 2, 4, 16, 16, 1
    update = layers.selective_state_update()
    if layout == "mamba":
        dim = nheads * headdim
        args = (
            rand(batch, dim, dstate, dtype=dtype, requires_grad=False),  # state, updated in place
            rand(batch, dim, dtype=dtype, requires_grad=False),  # hidden_states
            rand(batch, dim, dtype=dtype, requires_grad=False),  # dt
            neg_rand(dim, dstate).detach(),  # A
            rand(batch, dstate, dtype=dtype, requires_grad=False),  # B
            rand(batch, dstate, dtype=dtype, requires_grad=False),  # C
            rand(dim, dtype=torch.float32, requires_grad=False),  # D
            rand(dim, dtype=torch.float32, requires_grad=False),  # dt_bias
            rand(batch, dim, dtype=dtype, requires_grad=False),  # z
        )

        def fn(state, x, dt, A, B, C, D, dt_bias, z):
            return update(state, x, dt, A, B, C, D=D, dt_bias=dt_bias, dt_softplus=True, z=z)

    else:
        args = (
            rand(batch, nheads, headdim, dstate, dtype=dtype, requires_grad=False),  # state, updated in place
            rand(batch, nheads, headdim, dtype=dtype, requires_grad=False),  # hidden_states
            rand(batch, nheads, 1, dtype=dtype, requires_grad=False),  # dt
            neg_rand(nheads, 1, 1).detach(),  # A
            rand(batch, ngroups, dstate, dtype=dtype, requires_grad=False),  # B
            rand(batch, ngroups, dstate, dtype=dtype, requires_grad=False),  # C
            rand(nheads, 1, dtype=torch.float32, requires_grad=False),  # D
            rand(nheads, 1, dtype=torch.float32, requires_grad=False),  # dt_bias
        )

        def fn(state, x, dt, A, B, C, D, dt_bias):
            # Per-head parameters are broadcast over the head dimension, as transformers does.
            return update(
                state, x, dt.expand(-1, -1, headdim), A.expand(-1, headdim, dstate), B, C,
                D=D.expand(-1, headdim), dt_bias=dt_bias.expand(-1, headdim), dt_softplus=True,
            )

    check_compiled(fn, args, dtype, backward=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_initial_states", [False, True])
def test_mamba_chunk_scan_combined_compile(layers, dtype, has_initial_states):
    torch.manual_seed(0)
    batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size = 2, 33, 4, 16, 16, 1, 16
    args = (
        rand(batch, seqlen, nheads, headdim, dtype=dtype),  # hidden_states
        rand(batch, seqlen, nheads, dtype=dtype),  # dt
        neg_rand(nheads),  # A
        rand(batch, seqlen, ngroups, dstate, dtype=dtype),  # B
        rand(batch, seqlen, ngroups, dstate, dtype=dtype),  # C
        rand(nheads, dtype=dtype),  # D
        # Step sizes of a trained model are small; unit-scale steps make the state blow up and the
        # autotuned reduction order then dominates the fp32 comparison.
        (rand(nheads, dtype=dtype).detach() - 4).requires_grad_(),  # dt_bias
        rand(batch, nheads, headdim, dstate, dtype=dtype) if has_initial_states else None,
    )

    def fn(x, dt, A, B, C, D, dt_bias, initial_states):
        return layers.mamba_chunk_scan_combined()(
            x, dt, A, B, C, chunk_size=chunk_size, D=D, dt_bias=dt_bias,
            initial_states=initial_states, dt_softplus=True, return_final_states=True,
        )

    check_compiled(fn, args, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
# The second configuration gives an in-projection width that is a multiple of 8, the first does not.
@pytest.mark.parametrize("nheads,headdim", [(4, 16), (8, 8)])
def test_mamba_split_conv1d_scan_combined_compile(layers, dtype, nheads, headdim):
    torch.manual_seed(0)
    batch, seqlen, dstate, ngroups, chunk_size, width, d_model = 2, 33, 16, 1, 16, 4, 32
    dim = nheads * headdim
    conv_dim = dim + 2 * ngroups * dstate
    args = (
        rand(batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads, dtype=dtype),  # zxbcdt
        rand(conv_dim, width, dtype=dtype, scale=0.5),  # conv1d_weight
        rand(conv_dim, dtype=dtype),  # conv1d_bias
        (rand(nheads, dtype=dtype).detach() - 4).requires_grad_(),  # dt_bias, see above
        neg_rand(nheads),  # A
        rand(nheads, dtype=dtype),  # D
        rand(dim, dtype=dtype),  # rmsnorm_weight
        rand(d_model, dim, dtype=dtype, scale=0.1),  # outproj_weight
        rand(d_model, dtype=dtype),  # outproj_bias
    )

    def fn(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, rmsnorm_weight, outproj_weight, outproj_bias):
        return layers.mamba_split_conv1d_scan_combined()(
            zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size=chunk_size,
            activation="silu", rmsnorm_weight=rmsnorm_weight, rmsnorm_eps=1e-5,
            outproj_weight=outproj_weight, outproj_bias=outproj_bias, headdim=headdim,
            ngroups=ngroups, norm_before_gate=False,
        )

    check_compiled(fn, args, dtype)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_conv1d_compile(layers, dtype):
    torch.manual_seed(0)
    batch, seqlen, dim, width = 2, 33, 64, 4
    # Channel-last input triggers the original non-contiguous out= graph break.
    args = (
        rand(batch, seqlen, dim, dtype=dtype),  # hidden_states in channel-last layout
        rand(dim, width, dtype=dtype),  # weight
        rand(dim, dtype=dtype),  # bias
    )

    def fn(projection, weight, bias):
        return layers.causal_conv1d_fn()(
            projection[..., :dim].transpose(1, 2), weight, bias, activation="silu",
        )

    check_compiled(fn, args, dtype)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seqlen", [None, 3])
def test_causal_conv1d_update_compile(layers, dtype, seqlen):
    torch.manual_seed(0)
    batch, dim, width = 2, 64, 4
    args = (
        rand(*(batch, dim) + ((seqlen,) if seqlen else ()), dtype=dtype, requires_grad=False),  # hidden_states
        rand(batch, dim, width, dtype=dtype, requires_grad=False),  # conv_state, updated in place
        rand(dim, width, dtype=dtype, requires_grad=False),  # weight
        rand(dim, dtype=dtype, requires_grad=False),  # bias
    )
    check_compiled(layers.causal_conv1d_update(), args, dtype, backward=False, activation="silu")
