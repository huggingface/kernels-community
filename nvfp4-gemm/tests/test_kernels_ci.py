"""Fast subset of `test_nvfp4_gemm.py` for the kernels-community CI runner."""

import pytest

import test_nvfp4_gemm as _nvfp4_gemm

pytestmark = pytest.mark.kernels_ci

requires_blackwell = _nvfp4_gemm.requires_blackwell


def test_swizzled_sf_shape_math():
    _nvfp4_gemm.test_swizzled_sf_shape_math()


def test_quantize_reference_emulation():
    _nvfp4_gemm.test_quantize_reference_emulation()


@requires_blackwell
@pytest.mark.parametrize("m", [1, 64])
def test_pack_and_gemm_matches_emulation(m):
    _nvfp4_gemm.test_pack_and_gemm_matches_emulation(m)


@requires_blackwell
def test_gemv_matches_dequant():
    _nvfp4_gemm.test_gemv_matches_dequant(4096, 6656, 1)


@requires_blackwell
def test_gemv_swiglu_matches_dequant():
    _nvfp4_gemm.test_gemv_swiglu_matches_dequant(1)


def test_hub_layer_is_pure():
    _nvfp4_gemm.test_hub_layer_is_pure()
