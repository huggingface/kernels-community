import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import hub_pr_upload_args as hub  # noqa: E402

COMMUNITY = "kernels-community"
STAGING = "kernels-staging"

COMMUNITY_TOML = b'[general.hub]\nrepo-id = "kernels-community/mykernel"\n'
EXTERNAL_TOML = b'[general.hub]\nrepo-id = "MiniMaxAI/msa"\n'


def write_kernel(tmp_path, kernel, toml):
    (tmp_path / kernel).mkdir()
    (tmp_path / kernel / "build.toml").write_bytes(toml)


# The kernel is resolved from the working directory, not from the script's own
# location: workflows run the helpers from a separate checkout of the default
# branch while the kernel comes from the PR checkout.
def test_kernel_resolved_from_working_directory(tmp_path, monkeypatch):
    write_kernel(tmp_path, "msa", EXTERNAL_TOML)
    monkeypatch.chdir(tmp_path)
    assert hub.external_repo_id("msa", COMMUNITY) == "MiniMaxAI/msa"
    assert hub.repo_id("msa", COMMUNITY) == "MiniMaxAI/msa"


def test_missing_build_toml_falls_back_to_prefix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert hub.external_repo_id("newkernel", COMMUNITY) == ""
    assert hub.repo_id("newkernel", STAGING) == "kernels-staging/newkernel"


def test_community_repo_id_is_not_external(tmp_path, monkeypatch):
    write_kernel(tmp_path, "mykernel", COMMUNITY_TOML)
    monkeypatch.chdir(tmp_path)
    assert hub.external_repo_id("mykernel", COMMUNITY) == ""
    assert hub.repo_id("mykernel", COMMUNITY) == "kernels-community/mykernel"


def test_repo_id_without_hub_section(tmp_path, monkeypatch):
    write_kernel(tmp_path, "mykernel", b'[general]\nname = "mykernel"\n')
    monkeypatch.chdir(tmp_path)
    assert hub.repo_id("mykernel", COMMUNITY) == "kernels-community/mykernel"


def test_community_opens_a_pull_request_against_an_external_repo(tmp_path, monkeypatch):
    write_kernel(tmp_path, "msa", EXTERNAL_TOML)
    monkeypatch.chdir(tmp_path)
    assert hub.upload_args("msa", COMMUNITY) == "--create-pr"


def test_community_uploads_its_own_kernels_by_repo_id(tmp_path, monkeypatch):
    write_kernel(tmp_path, "mykernel", COMMUNITY_TOML)
    monkeypatch.chdir(tmp_path)
    assert hub.upload_args("mykernel", COMMUNITY) == "--repo-id kernels-community/mykernel"


# Regression: `/kernel-bot build-and-stage` on a kernel whose build.toml points
# at a vendor org used to emit a bare --create-pr, which drops --repo-id and
# lets the uploader fall back to build.toml -- so a staging build tried to open
# a pull request against sgl-project/sgl-flash-attn3 (branch pr-<n>, which does
# not exist there, with a token that cannot write to it).
def test_staging_never_targets_the_vendor_repo(tmp_path, monkeypatch):
    write_kernel(tmp_path, "sgl-flash-attn3", b'[general.hub]\nrepo-id = "sgl-project/sgl-flash-attn3"\n')
    monkeypatch.chdir(tmp_path)
    assert hub.external_repo_id("sgl-flash-attn3", STAGING) == ""
    assert hub.repo_id("sgl-flash-attn3", STAGING) == "kernels-staging/sgl-flash-attn3"
    assert hub.upload_args("sgl-flash-attn3", STAGING) == "--repo-id kernels-staging/sgl-flash-attn3"


# Invariants over every real kernel: assert properties, not exact per-kernel output.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


@pytest.fixture
def kernels():
    # external_repo_id resolves build.toml relative to cwd.
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield sorted(
            name
            for name in os.listdir(REPO_ROOT)
            # Ported kernels keep build.toml in <kernel>/src.
            if os.path.exists(os.path.join(REPO_ROOT, name, "build.toml"))
            or os.path.exists(os.path.join(REPO_ROOT, name, "src", "build.toml"))
        )
    finally:
        os.chdir(cwd)


def test_kernels_were_discovered(kernels):
    # Guards against the sweep silently testing nothing (wrong cwd, etc.).
    assert len(kernels) > 0


def test_no_kernel_leaks_out_of_the_staging_org(kernels):
    for kernel in kernels:
        assert hub.repo_id(kernel, STAGING) == f"{STAGING}/{kernel}"
        assert hub.upload_args(kernel, STAGING) == f"--repo-id {STAGING}/{kernel}"
