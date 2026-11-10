import json

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
