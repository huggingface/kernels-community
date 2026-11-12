import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import autotune


@pytest.fixture
def autotune_run(tmp_path, monkeypatch):
    targets = [
        {"kernel": "example", "name": "l4", "gpu": "NVIDIA L4"},
        {"kernel": "example", "name": "a100", "gpu": "NVIDIA A100"},
    ]
    jobs = []
    for job_id, target in enumerate(targets, start=1):
        jobs.append(
            {
                "id": job_id,
                "name": f"autotune / Autotune (example/{target['name']})",
                "conclusion": "success",
                "steps": [],
                "html_url": f"https://github.com/example/repo/actions/jobs/{job_id}",
            }
        )
        artifact = tmp_path / f"artifacts/autotune-example.{target['name']}"
        artifact.mkdir(parents=True)
        (artifact / "source.json").write_text('{"status": "changed"}')

    monkeypatch.setenv("GITHUB_REPOSITORY", "example/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setenv("GPU_JOBS", json.dumps(targets))
    monkeypatch.setenv("SKIPPED_TARGETS", "[]")
    monkeypatch.setattr(
        autotune.subprocess,
        "check_output",
        lambda command: json.dumps([{"jobs": jobs}]),
    )
    return tmp_path


def test_collect_groups_changed_gpus_by_kernel(autotune_run):
    autotune.collect(autotune_run / "results.json", autotune_run / "artifacts")

    outputs = (autotune_run / "outputs").read_text().splitlines()
    assert 'kernels=["example"]' in outputs
    assert "has_changes=true" in outputs
    results = json.loads((autotune_run / "results.json").read_text())
    assert [result["name"] for result in results] == ["l4", "a100"]
    assert [result["status"] for result in results] == ["changed", "changed"]


def test_collect_skips_pr_creation_when_configs_are_unchanged(autotune_run):
    for target in ("l4", "a100"):
        artifact = autotune_run / f"artifacts/autotune-example.{target}"
        (artifact / "source.json").write_text('{"status": "no_changes"}')

    autotune.collect(autotune_run / "results.json", autotune_run / "artifacts")

    outputs = (autotune_run / "outputs").read_text().splitlines()
    assert "kernels=[]" in outputs
    assert "has_changes=false" in outputs
    results = json.loads((autotune_run / "results.json").read_text())
    assert [result["status"] for result in results] == ["no_changes", "no_changes"]


@pytest.fixture
def autotuner_pr():
    return {
        "user": {"login": "github-actions[bot]"},
        "head": {
            "repo": {"full_name": "example/repo"},
            "ref": "autotune/example/abc123",
        },
        "body": "<!-- kernels-community-autotune -->\n\nUpdated configs.",
    }


def test_autotuner_pr_is_recognized(autotuner_pr):
    assert autotune.is_autotune_pr(autotuner_pr, "example/repo")


def test_pr_without_autotune_marker_is_not_recognized(autotuner_pr):
    autotuner_pr["body"] = "Updated configs."

    assert not autotune.is_autotune_pr(autotuner_pr, "example/repo")


def test_human_pr_with_autotune_marker_is_not_recognized(autotuner_pr):
    autotuner_pr["user"]["login"] = "maintainer"

    assert not autotune.is_autotune_pr(autotuner_pr, "example/repo")


def test_flat_configs_replace_only_the_selected_gpu(tmp_path):
    target = {
        "name": "l4",
        "gpu": "NVIDIA L4",
    }
    source, destination = tmp_path / "artifact", tmp_path / "configs"
    source.mkdir()
    destination.mkdir()
    stale = "N=1024,device_name=NVIDIA_L4.json"
    fresh = "N=2048,device_name=NVIDIA_L4.json"
    other = "N=1024,device_name=NVIDIA_A100.json"
    (destination / stale).write_text('{"num_warps": 4}')
    (destination / other).write_text('{"num_warps": 8}')
    (source / fresh).write_text('{"num_warps": 16}')

    changed = autotune.replace_target_configs(source, destination, target)

    assert set(changed) == {destination / stale, destination / fresh}
    assert not (destination / stale).exists()
    assert (destination / fresh).read_text() == (source / fresh).read_text()
    assert (destination / other).read_text() == '{"num_warps": 8}'


def test_flat_config_artifact_rejects_another_gpu_before_deleting(tmp_path):
    target = {
        "name": "l4",
        "gpu": "NVIDIA L4",
    }
    source, destination = tmp_path / "artifact", tmp_path / "configs"
    source.mkdir()
    destination.mkdir()
    name = "N=1024,device_name=NVIDIA_L4.json"
    (source / name).write_text("{}")
    (source / "N=1024,device_name=NVIDIA_A100.json").write_text("{}")
    (destination / name).write_text('{"keep": true}')

    with pytest.raises(ValueError, match="unexpected configs"):
        autotune.replace_target_configs(source, destination, target)
    assert (destination / name).read_text() == '{"keep": true}'


def test_publish_stages_flat_files_in_source_configs(tmp_path, monkeypatch):
    manifest = Path("flash-attn-ops/autotune.toml").read_text()
    build = Path("flash-attn-ops/build.toml").read_text()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/repo")
    kernel = Path("flash-attn-ops")
    kernel.mkdir()
    (kernel / "autotune.toml").write_text(manifest)
    (kernel / "build.toml").write_text(build)
    artifacts = Path("artifacts")
    artifact = artifacts / "autotune-flash-attn-ops.l4"
    (artifact / "NVIDIA L4").mkdir(parents=True)
    name = "op=layer_norm_fwd,N=1024,dtype=bfloat16,device_name=NVIDIA_L4.json"
    (artifact / "NVIDIA L4" / name).write_text('{"512": []}')
    (artifact / "source.json").write_text(
        json.dumps(
            {
                "repo_id": "kernels-community/flash-attn-ops",
                "source_sha": "source",
                "status": "changed",
            }
        )
    )
    results = Path("results.json")
    results.write_text(
        json.dumps([{"kernel": "flash-attn-ops", "name": "l4", "status": "changed"}])
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=1 if command == ["git", "diff", "--cached", "--quiet"] else 0
        )

    monkeypatch.setattr(autotune.subprocess, "run", run)
    monkeypatch.setattr(
        autotune.subprocess,
        "check_output",
        lambda command, **kwargs: (
            "source\n" if command[:2] == ["git", "rev-parse"] else "[]"
        ),
    )
    autotune.publish(
        kernel, artifacts, results, "https://example.com/run", dry_run=True
    )

    expected = kernel / "torch-ext/flash_attn_ops/configs" / name
    assert expected.read_text() == '{"512": []}'
    assert ["git", "add", "--", str(expected)] in commands
    assert not (expected.parent / "NVIDIA L4").exists()
    assert not any(
        command[:2] in (["git", "push"], ["gh", "pr"]) for command in commands
    )
