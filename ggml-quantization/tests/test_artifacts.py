"""The shipped artifacts are real binaries.

A `.so` committed while its working copy was an unsmudged LFS pointer stores the pointer text as the
object: a ~130-byte file that fails to `dlopen`. It has happened once, silently, and it is invisible
to every other test because loading such a build looks like "no build for this platform".
"""

from pathlib import Path

import pytest


BUILD = Path(__file__).resolve().parent.parent / "build"
ARTIFACTS = sorted(BUILD.glob("*/*.so")) if BUILD.is_dir() else []


@pytest.mark.skipif(not ARTIFACTS, reason="no built artifacts in this checkout")
@pytest.mark.parametrize("so", ARTIFACTS, ids=lambda p: p.parent.name)
def test_artifact_is_a_binary_not_an_lfs_pointer(so):
    head = so.read_bytes()[:64]
    assert not head.startswith(b"version https://git-lfs"), (
        f"{so.relative_to(BUILD)} is an LFS pointer ({so.stat().st_size} bytes), not a library. "
        "It was committed while unsmudged; restore it from the last commit whose pointer records a "
        "real size, and re-commit."
    )
    assert so.stat().st_size > 100_000, f"{so.relative_to(BUILD)} is only {so.stat().st_size} bytes"


@pytest.mark.skipif(not ARTIFACTS, reason="no built artifacts in this checkout")
def test_the_build_for_this_platform_loads():
    """A build that ships for this machine must import; "will not load" is not "not built".

    Without this, a corrupt artifact makes every other test skip — the suite reports green for a
    package that cannot be used at all.
    """
    import platform

    import torch
    from kernels import get_local_kernel

    arch = "aarch64" if platform.machine() in ("arm64", "aarch64") else "x86_64"
    system = "darwin" if platform.system() == "Darwin" else "linux"
    backend = "metal" if system == "darwin" else "cuda"
    major, minor = torch.__version__.split(".")[:2]
    match = [
        d for d in sorted(BUILD.iterdir())
        if d.name.startswith(f"torch{major}{minor}-") and backend in d.name and arch in d.name
    ]
    if not match:
        pytest.skip(f"no torch{major}{minor} {backend} {arch} build in this checkout")

    try:
        get_local_kernel(BUILD.parent, backend)
    except Exception as error:  # noqa: BLE001
        pytest.fail(f"{match[0].name} ships but will not load: {error}")
