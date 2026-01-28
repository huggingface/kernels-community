import pytest
import triton

def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_knobs(monkeypatch):
    try:
        _ver_str = getattr(triton, "__version__", "0.0.0").split("+")[0]
        _parts = _ver_str.split(".")
        _ver_tuple = tuple(int(p) for p in (_parts + ["0", "0", "0"])[:3])
    except Exception:
        _ver_tuple = (0, 0, 0)

    from triton._internal_testing import _fresh_knobs_impl
    if _ver_tuple > (3, 4, 0):
        fresh_function, reset_function = _fresh_knobs_impl()
    else:
        fresh_function, reset_function = _fresh_knobs_impl(monkeypatch)

    try:
        yield fresh_function()
    finally:
        reset_function()
