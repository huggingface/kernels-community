import pytest
import torch
import torch.nn.functional as F

import esmfold2_trimul


# Channel counts the fused GEMMs support: powers of two >= 64. ESMFold2 uses 128.
CHANNELS = [64, 128]
DIRECTIONS = ["outgoing", "incoming"]

_WEIGHT_KEYS = (
    "norm_in_weight",
    "norm_in_bias",
    "p_in_weight",
    "g_in_weight",
    "norm_out_weight",
    "norm_out_bias",
    "p_out_weight",
    "g_out_weight",
)


def _weights(c_z, device):
    """bf16 parameters matching the layer's split of ESMFold2's proj_bundle."""

    def g(*shape, scale):
        return (torch.randn(*shape, device=device) * scale).to(torch.bfloat16)

    return dict(
        norm_in_weight=g(c_z, scale=0.2),
        norm_in_bias=g(c_z, scale=0.1),
        p_in_weight=g(2 * c_z, c_z, scale=0.1),
        g_in_weight=g(2 * c_z, c_z, scale=0.1),
        norm_out_weight=g(c_z, scale=0.2),
        norm_out_bias=g(c_z, scale=0.1),
        p_out_weight=g(c_z, c_z, scale=0.1),
        g_out_weight=g(c_z, c_z, scale=0.1),
    )


def _reference(pair, direction, residual, drop_mask, *, mask, eps=1e-5, **w):
    """fp32 PyTorch reference for ``residual + drop_mask * TriMul(pair)``."""
    B, L_row, L_col, c_z = pair.shape
    f = lambda t: None if t is None else t.float()

    x = F.layer_norm(f(pair), (c_z,), f(w["norm_in_weight"]), f(w["norm_in_bias"]), eps)

    delta = torch.sigmoid(x @ f(w["g_in_weight"]).T) * (x @ f(w["p_in_weight"]).T)
    if mask is not None:
        delta = delta * f(mask).reshape(B, L_row, L_col, 1)

    a, b_t = delta.permute(3, 0, 1, 2).split(c_z, dim=0)
    if direction == "outgoing":
        y = torch.einsum("dbik,dbjk->dbij", a, b_t)
    else:
        y = torch.einsum("dbki,dbkj->dbij", a, b_t)

    y = F.layer_norm(
        y.permute(1, 2, 3, 0),
        (c_z,),
        f(w["norm_out_weight"]),
        f(w["norm_out_bias"]),
        eps,
    )

    out = torch.sigmoid(x @ f(w["g_out_weight"]).T) * (y @ f(w["p_out_weight"]).T)
    if drop_mask is not None:
        out = out * f(drop_mask)
    return f(residual) + out


def _assert_matches_reference(B, L, c_z, direction, use_mask, use_drop_mask):
    device = torch.device("cuda")
    torch.manual_seed(0)
    w = _weights(c_z, device)

    pair = (torch.randn(B, L, L, c_z, device=device)).to(torch.bfloat16)
    residual = (torch.randn(B, L, L, c_z, device=device)).to(torch.bfloat16)
    mask = torch.rand(B, L, L, device=device).to(torch.bfloat16) if use_mask else None
    drop_mask = (
        (torch.randn(B, 1, L, c_z, device=device)).to(torch.bfloat16)
        if use_drop_mask
        else None
    )

    with torch.no_grad():
        got = esmfold2_trimul.triangle_multiplicative_update_with_residual(
            pair, direction, residual, drop_mask, mask=mask, **w
        )
    want = _reference(pair, direction, residual, drop_mask, mask=mask, **w)

    assert got.shape == pair.shape
    assert got.dtype == torch.bfloat16
    # Everything downstream of the input LayerNorm runs in bf16, so compare on
    # mean relative error rather than elementwise bf16 ulps.
    rel = (got.float() - want).abs().mean() / want.abs().mean().clamp_min(1e-6)
    assert rel < 0.01, f"mean relative error {rel:.3e} vs fp32 reference"


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("c_z", CHANNELS)
def test_matches_reference(direction, c_z):
    _assert_matches_reference(2, 24, c_z, direction, use_mask=False, use_drop_mask=False)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_drop_mask", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_masks(use_mask, use_drop_mask):
    _assert_matches_reference(
        2, 24, 128, "outgoing", use_mask=use_mask, use_drop_mask=use_drop_mask
    )


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("L", [2, 7, 33, 64])
def test_sequence_lengths(L):
    """L is unconstrained: the row dim is masked in every kernel.

    B == L == 1 is excluded: it makes M == 1, which Triton specializes to a
    constexpr and the LayerNorm kernel's ``M.to(tl.int64)`` then rejects.
    """
    _assert_matches_reference(1, L, 64, "outgoing", use_mask=True, use_drop_mask=False)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deterministic():
    """A repeated call must be bit-identical; drift means an out-of-bounds read."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    c_z = 128
    w = _weights(c_z, device)
    pair = (torch.randn(2, 24, 24, c_z, device=device)).to(torch.bfloat16)
    residual = (torch.randn(2, 24, 24, c_z, device=device)).to(torch.bfloat16)

    outs = []
    for _ in range(3):
        # Churn the allocator so foreign memory would differ between runs.
        del_me = torch.randn(4_000_000, device=device)
        del del_me
        with torch.no_grad():
            outs.append(
                esmfold2_trimul.triangle_multiplicative_update_with_residual(
                    pair, "outgoing", residual, None, mask=None, **w
                ).clone()
            )
    for other in outs[1:]:
        assert torch.equal(outs[0], other)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("c_z", [16, 32, 96, 192])
def test_rejects_unsupported_channels(c_z):
    """Unsupported c_z must raise, not silently read out of bounds."""
    pair = torch.zeros(1, 4, 4, c_z, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="power of two"):
        esmfold2_trimul.triangle_multiplicative_update_with_residual(
            pair,
            "outgoing",
            pair.clone(),
            None,
            mask=None,
            **{k: torch.zeros(1, dtype=torch.bfloat16) for k in _WEIGHT_KEYS},
        )


@pytest.mark.kernels_ci
def test_layer_is_exposed_for_kernels():
    """`kernels` resolves Hub layers as `<module>.layers.<layer_name>`."""
    layer = getattr(
        esmfold2_trimul.layers, "ESMFold2TriangleMultiplication", None
    )
    assert layer is not None
    assert issubclass(layer, torch.nn.Module)
    # `kernels` requires layers to be stateless: no constructor of their own.
    assert layer.__init__ is torch.nn.Module.__init__
