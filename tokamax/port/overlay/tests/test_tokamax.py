import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import kernels
import numpy as np
import pytest

tokamax = kernels.get_kernel("kernels-community/tokamax", version=1)

# Tokamax runtime dependencies that this kernel drops or vendors. JAX and
# torch_tpu import absl and immutabledict themselves, so those are not listed.
DROPPED = ("einshape", "flax", "jaxtyping", "pydantic", "qwix", "xprof")

LOAD_WITHOUT_DROPPED = """
import sys


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {dropped!r}:
            raise ModuleNotFoundError(f"{{name}} is blocked", name=name)
        return None


sys.meta_path.insert(0, Block())

import kernels

tokamax = kernels.get_kernel("kernels-community/tokamax", version=1)
print(tokamax.__version__)
"""


def on_tpu() -> bool:
    try:
        return jax.devices()[0].platform == "tpu"
    except RuntimeError:
        return False


requires_tpu = pytest.mark.skipif(not on_tpu(), reason="requires a TPU")


@pytest.mark.kernels_ci
def test_loads_without_dropped_dependencies():
    # A fresh interpreter, so that nothing imported by pytest or other tests
    # can mask an import of a dropped dependency.
    proc = subprocess.run(
        [sys.executable, "-c", LOAD_WITHOUT_DROPPED.format(dropped=DROPPED)],
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[-1] == tokamax.__version__


@pytest.mark.kernels_ci
def test_public_api():
    assert tokamax.__version__ == "0.0.17"
    for name in tokamax.__all__:
        assert hasattr(tokamax, name), name


def ragged_dot_inputs(num_groups, m, k, n, dtype=jnp.bfloat16):
    k0, k1 = jax.random.split(jax.random.key(0))
    lhs = jax.random.normal(k0, (m, k), jnp.float32).astype(dtype)
    rhs = jax.random.normal(k1, (num_groups, k, n), jnp.float32).astype(dtype)
    # Uneven groups that do not fill all rows.
    sizes = np.linspace(0, 2 * m // num_groups, num_groups).astype(np.int32)
    sizes[-1] = 0
    return lhs, rhs, jnp.asarray(sizes)


def ragged_dot_reference(lhs, rhs, group_sizes):
    return jax.lax.ragged_dot(
        lhs.astype(jnp.float32),
        rhs.astype(jnp.float32),
        group_sizes,
        precision=jax.lax.Precision.HIGHEST,
    )


def assert_ragged_close(actual, expected, group_sizes, **tol):
    # Rows past the last group are left undefined by the kernels.
    rows = int(group_sizes.sum())
    np.testing.assert_allclose(actual[:rows], expected[:rows], **tol)


@pytest.mark.kernels_ci
@requires_tpu
def test_ragged_dot_gmm():
    lhs, rhs, group_sizes = ragged_dot_inputs(4, 512, 256, 256)
    out = tokamax.ragged_dot.gmm(
        lhs,
        rhs,
        group_sizes,
        precision=(jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        out_dtype=jnp.float32,
    )
    expected = ragged_dot_reference(lhs, rhs, group_sizes)
    assert_ragged_close(out, expected, group_sizes, atol=1e-2, rtol=1e-2)


@pytest.mark.kernels_ci
@requires_tpu
@pytest.mark.parametrize("as_qarray", [False, True])
def test_ragged_dot_gmm_int8_rhs(as_qarray):
    lhs, rhs, group_sizes = ragged_dot_inputs(8, 512, 256, 512)
    # One scale per group and output column, as in the upstream tests.
    tiled_axes = {0: 1, 1: 256, 2: 1}
    if as_qarray:
        # Quantized inside the kernel; exercises the vendored immutabledict.
        rhs_q = tokamax.quantization.AsQArray(rhs, jnp.int8, tiled_axes=tiled_axes)
        # gmm quantizes inside jit, which rounds differently from eager
        # quantization on TPU, so quantize the reference the same way.
        rhs_dq = tokamax.dequantize(jax.jit(lambda x: x.as_qarray())(rhs_q))
    else:
        rhs_q = tokamax.quantize(rhs, jnp.int8, tiled_axes=tiled_axes)
        assert isinstance(rhs_q, tokamax.QArray)
        rhs_dq = tokamax.dequantize(rhs_q)
    out = tokamax.ragged_dot.gmm(
        lhs,
        rhs_q,
        group_sizes,
        precision=(jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        out_dtype=jnp.float32,
    )
    expected = ragged_dot_reference(lhs, rhs_dq, group_sizes)
    # Tolerance of the upstream quantized tests.
    assert_ragged_close(out, expected, group_sizes, atol=0.4, rtol=0.1)


@pytest.mark.kernels_ci
@requires_tpu
def test_gmm_v2():
    lhs, rhs, group_sizes = ragged_dot_inputs(4, 512, 256, 256)
    out = tokamax.gmm_v2.gmm_v2(
        lhs, rhs, group_sizes, preferred_element_type=jnp.float32
    )
    expected = ragged_dot_reference(lhs, rhs, group_sizes)
    assert_ragged_close(out, expected, group_sizes, atol=1e-2, rtol=1e-2)


@pytest.mark.kernels_ci
@requires_tpu
def test_splash_attention_causal():
    num_heads, seq_len, head_dim = 2, 512, 128
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    shape = (num_heads, seq_len, head_dim)
    q = jax.random.normal(kq, shape, jnp.float32) / np.sqrt(head_dim)
    k = jax.random.normal(kk, shape, jnp.float32)
    v = jax.random.normal(kv, shape, jnp.float32)

    mask = tokamax.splash_attention_mask.CausalMask((seq_len, seq_len))
    attention = tokamax.splash_attention_kernel.make_splash_mha_single_device(mask)
    out = attention(q, k, v)

    logits = jnp.einsum("hqd,hkd->hqk", q, k, precision="highest")
    causal = jnp.tril(jnp.ones((seq_len, seq_len), bool))
    weights = jax.nn.softmax(jnp.where(causal, logits, -jnp.inf), axis=-1)
    expected = jnp.einsum("hqk,hkd->hqd", weights, v, precision="highest")
    np.testing.assert_allclose(out, expected, atol=2e-2, rtol=2e-2)
