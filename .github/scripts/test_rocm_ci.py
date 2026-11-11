import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import rocm_ci  # noqa: E402

CI_TESTS = [
    "torch210-cpu-x86_64-linux",
    "torch210-rocm70-x86_64-linux",
    "torch214-rocm72-x86_64-linux",
    "torch215-cu132-x86_64-linux",
    "torch215-rocm100-x86_64-linux",
    "torch215-rocm714-x86_64-linux",
]

CPP_ROCM = {"kernel": {"k": {"language": "cpp-rocm", "rocm-archs": ["gfx90a", "gfx942"]}}}
GFX1201_ONLY = {"kernel": {"k": {"language": "cpp-rocm", "rocm-archs": ["gfx1201"]}}}


def probe(rocm=True, backend_ci_test=False, ci_tests=CI_TESTS):
    return {"rocm": rocm, "backendCiTest": backend_ci_test, "ciTests": ci_tests}


def test_prefers_backend_ci_tests():
    attr, _ = rocm_ci.choose(CPP_ROCM, probe(backend_ci_test=True), set())
    assert attr == "backendCiTests.rocm"


def test_fallback_picks_newest_torch_then_oldest_rocm():
    # 7.14 is older than 10.0 even though 714 > 100.
    assert rocm_ci.select_variant(CI_TESTS) == "torch215-rocm714-x86_64-linux"
    attr, message = rocm_ci.choose(CPP_ROCM, probe(), set())
    assert attr == "ciTests.torch215-rocm714-x86_64-linux"
    assert "backendCiTests" in message


def test_rocm_version_decoding():
    assert rocm_ci._rocm_version("70") == (7, 0)
    assert rocm_ci._rocm_version("714") == (7, 14)
    assert rocm_ci._rocm_version("100") == (10, 0)


def test_skips_without_rocm_build():
    assert rocm_ci.choose({}, probe(rocm=False), set())[0] is None


def test_skips_when_filtered_out():
    assert rocm_ci.choose(CPP_ROCM, probe(), {"cuda"})[0] is None
    assert rocm_ci.choose(CPP_ROCM, probe(), {"cuda", "rocm"})[0] is not None


def test_skips_archs_without_runner_gpu():
    attr, message = rocm_ci.choose(GFX1201_ONLY, probe(backend_ci_test=True), set())
    assert attr is None
    assert "gfx942" in message


def test_default_archs_and_triton_kernels_run():
    assert rocm_ci.runs_on_runner({"kernel": {"k": {"language": "cpp-rocm"}}})
    assert rocm_ci.runs_on_runner({})


def test_old_kernel_builder_skips_with_hint():
    attr, message = rocm_ci.choose(CPP_ROCM, probe(ci_tests=None), set())
    assert attr is None
    assert "update_flakes" in message
