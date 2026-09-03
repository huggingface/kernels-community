# Cheap representative subset of the test suite, run in CI via
# `pytest -m kernels_ci`. Each test calls the full test function with a small
# set of parameters.
import pytest

from . import test_compaction, test_matmul, test_mxfp, test_routing, test_swiglu
from .test_matmul import opt_flags_scope  # noqa: F401  (imported to register the fixture)
from .test_tensor_details import test_layout_blackwell, test_layout_hopper


@pytest.mark.kernels_ci
def test_matmul_ragged_ci(device, opt_flags_scope, fresh_knobs):
    # Ragged MoE with gather+scatter, output gammas, and a block_m constraint.
    test_matmul.test_op(m=256, n=256, k=256, split_k=1, do_gather=True, do_scatter=True, fused_scatter=False,
                        has_y_gammas=True, is_persistent=False, n_expts_tot=4, n_expts_act=2, n_expt_shards=1,
                        mode="ragged", act_dtype_str="float16", weight_dtype_str="float16", block_m=128,
                        hbm_swizzling=False, epilogue_subtile=None, device=device, opt_flags_scope=opt_flags_scope,
                        fresh_knobs=fresh_knobs)


@pytest.mark.kernels_ci
def test_matmul_mxfp4_ci(device, opt_flags_scope, fresh_knobs):
    # Dense bf16 activations with mxfp4 weights.
    test_matmul.test_op(m=16, n=256, k=256, split_k=1, do_gather=False, do_scatter=False, fused_scatter=False,
                        has_y_gammas=False, is_persistent=False, n_expts_tot=1, n_expts_act=1, n_expt_shards=1,
                        mode="plain", act_dtype_str="bfloat16", weight_dtype_str="mxfloat4_e2m1", block_m=16,
                        hbm_swizzling=False, epilogue_subtile=None, device=device, opt_flags_scope=opt_flags_scope,
                        fresh_knobs=fresh_knobs)


@pytest.mark.kernels_ci
def test_fused_act_ci(device, opt_flags_scope):
    test_matmul.test_fused_act(m=256, n=256, k=256, mode="ragged", split_k=1, do_gather=False, do_scatter=True,
                               fused_scatter=False, is_persistent=False, epilogue_subtile=1, swiglu_alpha=1.0,
                               swiglu_limit=1.2, device=device, opt_flags_scope=opt_flags_scope)


@pytest.mark.kernels_ci
def test_set_idle_sms_ci(opt_flags_scope):
    # opt_flags_scope cleans up the global idle_sms constraint that
    # test_set_idle_sms leaves behind.
    test_matmul.test_set_idle_sms()


@pytest.mark.kernels_ci
def test_routing_ci(device):
    test_routing.test_op(n_tokens_pad=371, n_tokens_raw=None, n_expts_tot=128, n_expts_act=32, sm_first=True,
                         use_expt_indx=False, device=device)


@pytest.mark.kernels_ci
def test_routing_expt_indx_ci(device):
    test_routing.test_op(n_tokens_pad=1152, n_tokens_raw=911, n_expts_tot=1500, n_expts_act=8, sm_first=False,
                         use_expt_indx=True, device=device)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("limit", [1e-2, 10])
def test_swiglu_ci(limit, device):
    test_swiglu.test_op(1311, 4352, limit, device)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("n_tokens, n_cols, k, p", [(8192, 64, 4, 0.5), (131, 128, 16, 0.0)])
def test_compaction_ci(n_tokens, n_cols, k, p, device):
    test_compaction.test_compaction(n_tokens, n_cols, k, p, device)


@pytest.mark.kernels_ci
def test_mxfp_rounding_ci():
    test_mxfp.test_mxfp4_rounding_cases("bfloat16")


@pytest.mark.kernels_ci
def test_mxfp_quant_dequant_ci():
    test_mxfp.test_mxfp_quant_dequant("float4_e2m1", "bfloat16")


@pytest.mark.kernels_ci
def test_mxfp_casting_ci():
    test_mxfp.test_mxfp_casting((10, 254, 60), 0, "float4_e2m1", "bfloat16",
                                test_mxfp.DequantScaleRoundingMode.ROUND_DOWN)


@pytest.mark.kernels_ci
def test_layout_blackwell_ci():
    test_layout_blackwell.test_mxfp4_scale_roundtrip((10, 254, 60))


@pytest.mark.kernels_ci
def test_layout_hopper_ci():
    test_layout_hopper.test_mxfp4_value_roundtrip((16, 32), False, 0, 2)
    test_layout_hopper.test_mxfp4_scale_roundtrip((256, 64), 0, 4)


@pytest.mark.kernels_ci
def test_layout_hopper_upcast_ci():
    # The original test carries this as a skipif mark, which is bypassed when
    # calling the test function directly.
    if not (test_layout_hopper.is_cuda() and test_layout_hopper.cuda_capability_geq(9)):
        pytest.skip("Only supported on cuda with capability >= 9")
    test_layout_hopper.test_upcast_mxfp4_to_bf16()
