import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_kernel_test_versions.py")
SPEC = importlib.util.spec_from_file_location("check_kernel_test_versions", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def write_kernel(root, *, version=2, repo_id="kernels-community/example", nested=False):
    kernel = root / "example"
    manifest_root = kernel / "src" if nested else kernel
    manifest_root.mkdir(parents=True)
    (kernel / "tests").mkdir()
    (manifest_root / "build.toml").write_text(
        f"""[general]
version = {version}

[general.hub]
repo-id = "{repo_id}"
"""
    )
    return kernel


def test_matching_module_call(tmp_path):
    kernel = write_kernel(tmp_path)
    (kernel / "tests" / "test_example.py").write_text(
        "import kernels as loader\n"
        'example = loader.get_kernel("kernels-community/example", version=2)\n'
    )

    assert checker.check_repository(tmp_path) == ([], 1, 1)


def test_matching_import_with_nested_manifest(tmp_path):
    kernel = write_kernel(tmp_path, nested=True)
    (kernel / "tests" / "helper.py").write_text(
        "from kernels import get_kernel\n"
        'example = get_kernel("kernels-community/example", version=2)\n'
    )

    assert checker.check_repository(tmp_path) == ([], 1, 1)


def test_reports_stale_version(tmp_path):
    kernel = write_kernel(tmp_path, version=3)
    test = kernel / "tests" / "test_example.py"
    test.write_text(
        "import kernels\n"
        'example = kernels.get_kernel("kernels-community/example", version=2)\n'
    )

    problems, checked, suites = checker.check_repository(tmp_path)

    assert (checked, suites) == (1, 1)
    assert len(problems) == 1
    assert problems[0].path == test
    assert problems[0].line == 2
    assert "version=3" in problems[0].message


def test_reports_missing_version(tmp_path):
    kernel = write_kernel(tmp_path)
    (kernel / "tests" / "test_example.py").write_text(
        "from kernels import get_kernel\n"
        'example = get_kernel("kernels-community/example")\n'
    )

    problems, checked, suites = checker.check_repository(tmp_path)

    assert (checked, suites) == (1, 1)
    assert len(problems) == 1
    assert 'get_kernel("kernels-community/example", version=2)' in problems[0].message


def test_ignores_comments_unrelated_methods_and_suites_without_calls(tmp_path):
    kernel = write_kernel(tmp_path)
    (kernel / "tests" / "test_example.py").write_text(
        """# kernels.get_kernel("kernels-community/example", version=1)
class Timer:
    def get_kernel(self):
        return None

Timer().get_kernel()
"""
    )

    assert checker.check_repository(tmp_path) == ([], 0, 0)


def test_skips_kernel_without_tests(tmp_path):
    kernel = tmp_path / "example"
    kernel.mkdir()
    (kernel / "build.toml").write_text("this is not valid TOML")

    assert checker.check_repository(tmp_path) == ([], 0, 0)
