import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import hub_pr_upload_args as hub  # noqa: E402

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
    assert hub.external_repo_id("msa") == "MiniMaxAI/msa"
    assert hub.repo_id("msa", "kernels-community") == "MiniMaxAI/msa"


def test_missing_build_toml_falls_back_to_prefix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert hub.external_repo_id("newkernel") == ""
    assert hub.repo_id("newkernel", "kernels-staging") == "kernels-staging/newkernel"


def test_community_repo_id_is_not_external(tmp_path, monkeypatch):
    write_kernel(tmp_path, "mykernel", COMMUNITY_TOML)
    monkeypatch.chdir(tmp_path)
    assert hub.external_repo_id("mykernel") == ""
    assert hub.repo_id("mykernel", "kernels-community") == "kernels-community/mykernel"


def test_repo_id_without_hub_section(tmp_path, monkeypatch):
    write_kernel(tmp_path, "mykernel", b'[general]\nname = "mykernel"\n')
    monkeypatch.chdir(tmp_path)
    assert hub.repo_id("mykernel", "kernels-community") == "kernels-community/mykernel"
