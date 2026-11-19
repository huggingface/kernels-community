import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent / "workflows" / "generate-build-matrix.py"
)

SPEC = importlib.util.spec_from_file_location("generate_build_matrix", SCRIPT)
gbm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gbm
SPEC.loader.exec_module(gbm)


OVERRIDES = {
    "cuda": {
        "max-jobs": 2,
        "cores": 16,
        "aarch64-linux": {"max-jobs": 1},
    },
    "xpu": {"aarch64-linux": {"cores": 4}},
}


def test_defaults_without_overrides():
    assert gbm.resolve_concurrency({}, "cuda", "x86_64-linux") == {
        "max_jobs": gbm.DEFAULT_MAX_JOBS,
        "cores": gbm.DEFAULT_CORES,
    }


def test_backend_overrides_apply_to_every_arch():
    assert gbm.resolve_concurrency(OVERRIDES, "cuda", "x86_64-linux") == {
        "max_jobs": 2,
        "cores": 16,
    }


def test_arch_overrides_take_precedence_per_key():
    assert gbm.resolve_concurrency(OVERRIDES, "cuda", "aarch64-linux") == {
        "max_jobs": 1,
        "cores": 16,
    }


def test_arch_overrides_fall_back_to_defaults():
    assert gbm.resolve_concurrency(OVERRIDES, "xpu", "aarch64-linux") == {
        "max_jobs": gbm.DEFAULT_MAX_JOBS,
        "cores": 4,
    }
    assert gbm.resolve_concurrency(OVERRIDES, "xpu", "x86_64-linux") == {
        "max_jobs": gbm.DEFAULT_MAX_JOBS,
        "cores": gbm.DEFAULT_CORES,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"cuda": {"aarch64": {"max-jobs": 1}}},
        {"cuda": {"max_jobs": 1}},
        {"cuda": {"aarch64-linux": {"aarch64-linux": {"cores": 1}}}},
        {"cuda": {"aarch64-linux": 1}},
        {"cuda": {"cores": "12"}},
        {"cuda": {"cores": True}},
        {"cude": {"cores": 12}},
        {"cuda": []},
    ],
)
def test_invalid_overrides_are_rejected(overrides):
    with pytest.raises(SystemExit):
        gbm.validate_kernel_overrides("k", overrides)


def test_repo_overrides_are_valid():
    for kernel, overrides in gbm.load_concurrency_overrides().items():
        gbm.validate_kernel_overrides(kernel, overrides)


def test_matrix_uses_per_arch_overrides(monkeypatch, capsys):
    monkeypatch.setattr(gbm, "load_concurrency_overrides", lambda: {"k": OVERRIDES})
    monkeypatch.setattr(
        sys, "argv", ["generate-build-matrix.py", '["cuda"]', '["cuda"]']
    )
    monkeypatch.setenv("KERNEL", "k")
    monkeypatch.delenv("BACKENDS_FILTER", raising=False)
    gbm.main()
    by_arch = {
        entry["arch"]: (entry["max_jobs"], entry["cores"])
        for entry in json.loads(capsys.readouterr().out)["include"]
    }
    assert by_arch == {"x86_64-linux": (2, 16), "aarch64-linux": (1, 16)}
