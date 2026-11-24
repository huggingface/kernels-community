"""CPU-level tests for the FlyDSL kernel.

FlyDSL is repackaged from the upstream release wheel. Compiling and running
kernels needs a supported AMD GPU and ROCm, so these tests only check that
the kernel is loadable and that its embedded MLIR toolchain works. No GPU
work is launched.
"""

import importlib

import kernels
import pytest

flydsl = kernels.get_kernel("kernels-sdk/flydsl", version=1)

pytestmark = pytest.mark.kernels_ci


def import_submodule(name):
    return importlib.import_module(f"{flydsl.__name__}.{name}")


def test_version():
    assert isinstance(flydsl.__version__, str)
    assert flydsl.__version__


def test_mlir_context_and_parse():
    ir = import_submodule("_mlir.ir")

    with ir.Context() as ctx:
        module = ir.Module.parse("module {}", ctx)
        assert "module" in str(module)


def test_fly_dialect_importable():
    fly = import_submodule("_mlir._mlir_libs._mlirDialectsFly")
    assert fly is not None


def test_compiler_key():
    jit_function = import_submodule("compiler.jit_function")

    key = jit_function._flydsl_key_cached(False, "", (), "")
    assert key.startswith(f"flydsl:{flydsl.__version__}-")


def test_native_libraries_found():
    libraries = import_submodule("runtime._libraries")

    dirs = libraries._distribution_dirs()
    assert any((d / "libfly_rocm_aot_runtime.a").is_file() for d in dirs)


@pytest.mark.parametrize(
    "path",
    [
        "compiler",
        "compiler.backends.rocm",
        "compiler.protocol",
        "expr",
        "expr.gpu",
        "expr.meta",
        "expr.rocdl",
        "expr.typing",
        "expr.utils.arith",
        "runtime.device",
        "utils.env",
        "_mlir.ir",
        "_mlir.extras.types",
        "_mlir._mlir_libs._mlirDialectsFly",
        "_mlir.dialects.arith",
        "_mlir.dialects.builtin",
        "_mlir.dialects.fly",
        "_mlir.dialects.gpu",
        "_mlir.dialects.llvm",
        "_mlir.dialects.rocdl",
    ],
)
def test_submodules_are_attributes(path):
    # A kernel that depends on FlyDSL cannot `import flydsl.<x>`; it uses
    # attribute access, which requires the submodules to be loaded.
    module = flydsl
    for part in path.split("."):
        module = getattr(module, part)
